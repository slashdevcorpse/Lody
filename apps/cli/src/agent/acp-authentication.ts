import type { ChildProcess } from 'child_process';
import os from 'os';
import spawn from 'cross-spawn';
import { z } from 'zod';
import type { AuthMethod } from '@agentclientprotocol/sdk';
import type {
  AgentConfigCliType,
  BuiltinCliType,
  BuiltinRuntimeOverrides,
  CustomAcpLaunchSpec,
} from '@lody/shared';
import {
  getManagedBuiltinRuntimeByAgentType,
  hasBuiltinEnvAuthentication,
  isManagedBuiltinAgentType,
} from '@lody/shared';

import { withoutElectronBootstrapCredentials } from '@/electron-bootstrap-env';
import type { Logger } from '@/utils/logger';
import { formatErrorMessage } from '@/utils/format-error';
import { BuiltinAuthenticationOutputParser } from './acp-authentication-output';
import { shutdownLocalAcpAgent } from './acp-runner';
import { getLoginShellEnv } from './login-shell-env';
import {
  acquireAccountProfileUse,
  acquireAccountProfileAuthentication,
  accountProfileAuthenticationArgs,
  resolveAccountProfileEnv,
} from './account-profiles';
import { withAcpSessionStartSlot } from './acp-session-start-gate';
import {
  mergeACPProcessEnv,
  mergeLoginShellEnv,
  resolveBuiltinAuthenticationProcessLaunch,
  type ResolvedACPProcessLaunch,
  withDefaultAcpPathEntries,
} from './setting';

export type AcpAuthenticationProgressEvent =
  | { status: 'starting' }
  | {
      status: 'authorization';
      authorizationUrl: string;
      userCode?: string;
      acceptsAuthorizationCode?: boolean;
      expiresInSeconds?: number;
    }
  | { status: 'output'; stream: 'stdout' | 'stderr'; output: string }
  | { status: 'authenticated' }
  | { status: 'cancelled' }
  | { status: 'error'; error: string };

export type AcpAuthenticationResult =
  | {
      success: true;
      disposition: 'authenticated' | 'cancelled' | 'not-running' | 'input-accepted';
    }
  | { success: false; disposition: 'error'; error: string };

// Finish before the UI/RPC 300s deadline, leaving enough time for graceful
// termination, SIGKILL escalation, and the final response to travel back.
const DEFAULT_AUTHENTICATION_TIMEOUT_MS = 285_000;
const DEFAULT_TERMINATION_GRACE_MS = 3_000;
const DEFAULT_STATUS_PROBE_TIMEOUT_MS = 15_000;

const BUILTIN_AUTH_METHODS = {
  kimi: [
    {
      id: 'login',
      name: 'Kimi Code',
      description: 'Sign in with Kimi Code',
      type: 'terminal',
      args: ['--login'],
    },
  ],
  grok: [
    {
      id: 'xai-device-login',
      name: 'xAI',
      description: 'Sign in with an xAI account',
      type: 'terminal',
      args: ['login', '--device-auth'],
    },
  ],
  claude: [
    {
      id: 'claude-ai-login',
      name: 'Claude subscription',
      description: 'Sign in with a Claude Pro, Max, Team, or Enterprise subscription',
      type: 'terminal',
      args: ['auth', 'login', '--claudeai'],
    },
  ],
  codex: [
    {
      id: 'chat-gpt',
      name: 'ChatGPT',
      description: 'Sign in with a ChatGPT account',
    },
  ],
} satisfies Record<BuiltinCliType, readonly AuthMethod[]>;

type RunningAuthentication = {
  releaseAccountLease?: () => void;
  child?: ChildProcess;
  requestId: string;
  agentType: string;
  cancelled: boolean;
  timedOut: boolean;
  terminating: boolean;
  acceptsAuthorizationCode: boolean;
  authorizationCodeSubmitted: boolean;
};

type AcpAuthenticationManagerOptions = {
  authenticationTimeoutMs?: number;
  terminationGraceMs?: number;
  spawnProcess?: typeof spawn;
  resolveLoginShellEnv?: typeof getLoginShellEnv;
};

export type BuiltinAuthenticationProbeResult =
  | { status: 'authenticated'; identity?: string }
  | { status: 'unauthenticated'; authMethods: readonly AuthMethod[] }
  | { status: 'unknown' };

export type ProbeBuiltinAuthenticationOptions = {
  cliType: AgentConfigCliType;
  agentType: string;
  runtimeOverrides?: BuiltinRuntimeOverrides;
  env?: NodeJS.ProcessEnv;
  accountProfileId?: string;
  profilesRoot?: string;
  accountStatusOnly?: boolean;
  onManagedRuntimeProgress?: Parameters<
    typeof resolveBuiltinAuthenticationProcessLaunch
  >[0]['onManagedRuntimeProgress'];
  logger: Logger;
  signal?: AbortSignal;
  statusProbeTimeoutMs?: number;
  spawnProcess?: typeof spawn;
  resolveLoginShellEnv?: typeof getLoginShellEnv;
};

function getBuiltinDisplayName(agentType: string): string {
  return getManagedBuiltinRuntimeByAgentType(agentType)?.displayName ?? agentType;
}

function formatAuthenticationExitError(
  agentType: BuiltinCliType,
  displayName: string,
  exitCode: number | null
): string {
  const base = `${displayName} authentication exited with code ${exitCode ?? 'unknown'}`;
  if (agentType !== 'codex') return base;
  return `${base}. Make sure device-code login is enabled in your ChatGPT security settings or workspace permissions, then try again.`;
}

async function buildAuthenticationProcessEnv(options: {
  launch: ResolvedACPProcessLaunch;
  agentType: string;
  env?: NodeJS.ProcessEnv;
  accountProfileId?: string;
  profilesRoot?: string;
  accountStatusOnly?: boolean;
  resolveLoginShellEnv: typeof getLoginShellEnv;
}): Promise<NodeJS.ProcessEnv> {
  const loginShellEnv = await options.resolveLoginShellEnv();
  const baseEnv: NodeJS.ProcessEnv = {
    ...process.env,
    ...options.env,
    NO_COLOR: '1',
  };
  delete baseEnv.FORCE_COLOR;
  const merged = withoutElectronBootstrapCredentials(
    withDefaultAcpPathEntries(
      mergeACPProcessEnv(options.launch, mergeLoginShellEnv(baseEnv, loginShellEnv)),
      options.agentType
    )
  );
  return resolveAccountProfileEnv({ ...options, cliType: 'builtin', env: merged });
}

/**
 * Uses the provider's official status command to distinguish missing local
 * credentials from an ACP startup failure. Kimi and Grok have no equivalent
 * lightweight status command. Codex's status command only describes its OpenAI
 * credential store and cannot account for custom model providers, so those ACP
 * adapters remain the source of truth.
 */
export async function probeBuiltinAuthentication(
  options: ProbeBuiltinAuthenticationOptions
): Promise<BuiltinAuthenticationProbeResult> {
  const release = acquireAccountProfileUse(options);
  try {
    return await probeBuiltinAuthenticationWithAccountLease(options);
  } finally {
    release();
  }
}

async function probeBuiltinAuthenticationWithAccountLease(
  options: ProbeBuiltinAuthenticationOptions
): Promise<BuiltinAuthenticationProbeResult> {
  options.signal?.throwIfAborted();
  if (options.cliType !== 'builtin' || !isManagedBuiltinAgentType(options.agentType)) {
    return { status: 'unknown' };
  }
  if (
    options.agentType === 'kimi' ||
    options.agentType === 'grok' ||
    (options.agentType === 'codex' && !options.accountStatusOnly)
  ) {
    return { status: 'unknown' };
  }
  const launch = await resolveBuiltinAuthenticationProcessLaunch({
    cliType: options.cliType,
    agentType: options.agentType,
    runtimeOverrides: options.runtimeOverrides,
    action: 'status',
    onManagedRuntimeProgress: options.onManagedRuntimeProgress,
    signal: options.signal,
  });
  options.signal?.throwIfAborted();
  if (!launch) return { status: 'unknown' };

  const env = await buildAuthenticationProcessEnv({
    launch,
    agentType: options.agentType,
    env: options.env,
    accountProfileId: options.accountProfileId,
    profilesRoot: options.profilesRoot,
    resolveLoginShellEnv: options.resolveLoginShellEnv ?? getLoginShellEnv,
  });
  options.signal?.throwIfAborted();
  if (hasBuiltinEnvAuthentication(options.agentType, env)) {
    return { status: 'unknown' };
  }
  if (options.agentType === 'codex' && options.accountStatusOnly) {
    return withAcpSessionStartSlot(
      { label: 'account-status', logger: options.logger, abortSignal: options.signal },
      () => probeCodexAccount(options, launch, env)
    );
  }
  const child = (options.spawnProcess ?? spawn)(launch.command, launch.args, {
    cwd: os.homedir(),
    env,
    stdio: options.accountStatusOnly ? ['ignore', 'pipe', 'ignore'] : 'ignore',
    windowsHide: true,
  });
  let statusOutput = '';
  child.stdout?.on('data', (chunk: Buffer) => {
    if (statusOutput.length < 16_384)
      statusOutput += chunk.toString('utf8').slice(0, 16_384 - statusOutput.length);
  });
  const timeoutMs = Math.max(1, options.statusProbeTimeoutMs ?? DEFAULT_STATUS_PROBE_TIMEOUT_MS);
  const exit = await new Promise<{
    aborted?: boolean;
    code: number | null;
    error?: unknown;
    timedOut?: boolean;
  }>((resolve) => {
    let settled = false;
    let timeoutHandle: ReturnType<typeof setTimeout> | undefined;
    const finish = (result: {
      aborted?: boolean;
      code: number | null;
      error?: unknown;
      timedOut?: boolean;
    }): void => {
      if (settled) return;
      settled = true;
      if (timeoutHandle) clearTimeout(timeoutHandle);
      options.signal?.removeEventListener('abort', handleAbort);
      resolve(result);
    };
    const handleAbort = (): void => {
      try {
        child.kill('SIGKILL');
      } catch {
        // The process may have exited between cancellation and the kill call.
      }
      finish({ aborted: true, code: null });
    };
    options.signal?.addEventListener('abort', handleAbort, { once: true });
    if (options.signal?.aborted) {
      handleAbort();
      return;
    }
    timeoutHandle = setTimeout(() => {
      try {
        child.kill('SIGKILL');
      } catch {
        // The process may have exited between the timeout and kill call.
      }
      finish({ code: null, timedOut: true });
    }, timeoutMs);
    timeoutHandle.unref?.();
    child.once('error', (error) => finish({ code: null, error }));
    child.once('exit', (code) => finish({ code }));
  });

  if (exit.aborted) {
    throw new DOMException('ACP authentication probe was cancelled', 'AbortError');
  }
  if (exit.timedOut) {
    options.logger.debug(
      `[acp-auth] ${getBuiltinDisplayName(options.agentType)} status probe timed out; falling back to ACP`
    );
    return { status: 'unknown' };
  }
  if (exit.error !== undefined) {
    throw new Error(
      `${getBuiltinDisplayName(options.agentType)} authentication status failed: ${formatErrorMessage(exit.error)}`
    );
  }
  const parsed = z
    .object({ loggedIn: z.boolean(), email: z.string().max(320).optional() })
    .safeParse(
      (() => {
        try {
          return JSON.parse(statusOutput);
        } catch {
          return null;
        }
      })()
    );
  if (options.accountStatusOnly && exit.code === 0 && (!parsed.success || !parsed.data.loggedIn)) {
    return { status: 'unknown' };
  }
  return exit.code === 0
    ? {
        status: 'authenticated',
        ...(parsed.success && parsed.data.email ? { identity: parsed.data.email } : {}),
      }
    : {
        status: 'unauthenticated',
        authMethods: BUILTIN_AUTH_METHODS[options.agentType],
      };
}

/** Official app-server account/read, without refreshing tokens or creating a thread. */
async function probeCodexAccount(
  options: ProbeBuiltinAuthenticationOptions,
  launch: ResolvedACPProcessLaunch,
  env: NodeJS.ProcessEnv
): Promise<BuiltinAuthenticationProbeResult> {
  const child = (options.spawnProcess ?? spawn)(
    launch.command,
    accountProfileAuthenticationArgs(options, ['app-server']),
    {
      cwd: os.homedir(),
      env,
      stdio: ['pipe', 'pipe', 'ignore'],
      windowsHide: true,
    }
  );
  let timer: ReturnType<typeof setTimeout> | undefined;
  let onAbort = () => {};
  try {
    return await new Promise<BuiltinAuthenticationProbeResult>((resolve) => {
      let finished = false;
      let buffer = '';
      let receivedBytes = 0;
      const finish = (result: BuiltinAuthenticationProbeResult) => {
        if (finished) return;
        finished = true;
        resolve(result);
      };
      const send = (value: unknown) => child.stdin?.write(`${JSON.stringify(value)}\n`);
      onAbort = () => finish({ status: 'unknown' });
      options.signal?.addEventListener('abort', onAbort, { once: true });
      timer = setTimeout(onAbort, options.statusProbeTimeoutMs ?? DEFAULT_STATUS_PROBE_TIMEOUT_MS);
      timer.unref?.();
      child.once('error', onAbort);
      child.once('exit', onAbort);
      child.stdin?.on('error', onAbort);
      child.stdout?.on('data', (chunk: Buffer) => {
        if (finished) return;
        receivedBytes += chunk.length;
        buffer += chunk.toString('utf8');
        if (receivedBytes > 65_536) {
          onAbort();
          return;
        }
        let newline: number;
        while ((newline = buffer.indexOf('\n')) !== -1) {
          const line = buffer.slice(0, newline);
          buffer = buffer.slice(newline + 1);
          let value: unknown;
          try {
            value = JSON.parse(line);
          } catch {
            onAbort();
            return;
          }
          const message = z
            .object({
              id: z.number().optional(),
              result: z.unknown().optional(),
              error: z.unknown().optional(),
            })
            .safeParse(value);
          if (!message.success) continue;
          if (message.data.error !== undefined) {
            onAbort();
            return;
          }
          if (message.data.id === 1) {
            send({ method: 'initialized', params: {} });
            send({ id: 2, method: 'account/read', params: { refreshToken: false } });
          } else if (message.data.id === 2) {
            const account = z
              .object({
                account: z
                  .object({ type: z.string(), email: z.string().max(320).optional() })
                  .nullable(),
              })
              .safeParse(message.data.result);
            if (!account.success) {
              onAbort();
              return;
            }
            finish(
              account.data.account
                ? {
                    status: 'authenticated',
                    ...(account.data.account.email ? { identity: account.data.account.email } : {}),
                  }
                : { status: 'unauthenticated', authMethods: BUILTIN_AUTH_METHODS.codex }
            );
          }
        }
      });
      if (options.signal?.aborted) {
        onAbort();
        return;
      }
      send({
        id: 1,
        method: 'initialize',
        params: { clientInfo: { name: 'lody-account-status', version: '1' }, capabilities: {} },
      });
    });
  } finally {
    clearTimeout(timer);
    options.signal?.removeEventListener('abort', onAbort);
    await shutdownLocalAcpAgent({
      agentProcess: child,
      logger: options.logger,
      sessionLabel: 'account-status',
    });
  }
}

export class AcpAuthenticationManager {
  // Login slots are isolated by provider and account profile.
  private readonly runningByAgentType = new Map<string, RunningAuthentication>();
  private readonly authenticationTimeoutMs: number;
  private readonly terminationGraceMs: number;
  private readonly spawnProcess: typeof spawn;
  private readonly resolveLoginShellEnv: typeof getLoginShellEnv;

  constructor(
    private readonly logger: Logger,
    options: AcpAuthenticationManagerOptions = {}
  ) {
    this.authenticationTimeoutMs = Math.max(
      1,
      options.authenticationTimeoutMs ?? DEFAULT_AUTHENTICATION_TIMEOUT_MS
    );
    this.terminationGraceMs = Math.max(
      1,
      options.terminationGraceMs ?? DEFAULT_TERMINATION_GRACE_MS
    );
    this.spawnProcess = options.spawnProcess ?? spawn;
    this.resolveLoginShellEnv = options.resolveLoginShellEnv ?? getLoginShellEnv;
  }

  async authenticate(options: {
    requestId: string;
    cliType: AgentConfigCliType;
    agentType: string;
    customAcp?: CustomAcpLaunchSpec;
    runtimeOverrides?: BuiltinRuntimeOverrides;
    env?: Record<string, string>;
    accountProfileId?: string;
    profilesRoot?: string;
    onAccountLeaseReleased?: () => void;
    onProgress?: (event: AcpAuthenticationProgressEvent) => void;
  }): Promise<AcpAuthenticationResult> {
    if (options.cliType !== 'builtin' || !isManagedBuiltinAgentType(options.agentType)) {
      return {
        success: false,
        disposition: 'error',
        error: `Authentication is not supported for ${options.agentType}`,
      };
    }

    const displayName = getBuiltinDisplayName(options.agentType);
    const agentType: BuiltinCliType = options.agentType;
    const accountKey = JSON.stringify([agentType, options.accountProfileId ?? 'system-default']);

    if (this.runningByAgentType.has(accountKey)) {
      return {
        success: false,
        disposition: 'error',
        error: `${displayName} authentication is already running`,
      };
    }

    const running: RunningAuthentication = {
      requestId: options.requestId,
      agentType,
      cancelled: false,
      timedOut: false,
      terminating: false,
      acceptsAuthorizationCode: false,
      authorizationCodeSubmitted: false,
    };
    // Reserve the slot before any async launch preparation. This makes
    // concurrent starts and cancellation deterministic even before spawn.
    this.runningByAgentType.set(accountKey, running);

    let timeoutHandle: ReturnType<typeof setTimeout> | undefined;
    const interruptedResult = (): AcpAuthenticationResult | null => {
      if (running.cancelled) {
        options.onProgress?.({ status: 'cancelled' });
        return { success: true, disposition: 'cancelled' };
      }
      if (running.timedOut) {
        const error = `${displayName} authentication timed out. Please try again.`;
        options.onProgress?.({ status: 'error', error });
        return { success: false, disposition: 'error', error };
      }
      return null;
    };

    timeoutHandle = setTimeout(() => {
      if (running.cancelled) return;
      running.timedOut = true;
      if (!running.child && this.runningByAgentType.get(accountKey) === running) {
        this.runningByAgentType.delete(accountKey);
        running.releaseAccountLease?.();
      }
      this.terminateAuthentication(options.agentType, running, 'timed out');
    }, this.authenticationTimeoutMs);
    timeoutHandle.unref?.();

    try {
      const releaseProfileAuthentication = acquireAccountProfileAuthentication(options);
      let leaseReleased = false;
      running.releaseAccountLease = () => {
        if (leaseReleased) return;
        leaseReleased = true;
        releaseProfileAuthentication();
        options.onAccountLeaseReleased?.();
      };
      const launch = await resolveBuiltinAuthenticationProcessLaunch({
        cliType: options.cliType,
        agentType: options.agentType,
        runtimeOverrides: options.runtimeOverrides,
        action: 'login',
      });
      if (!launch) {
        throw new Error(`${displayName} authentication is unavailable`);
      }
      const launchInterruption = interruptedResult();
      if (launchInterruption) return launchInterruption;

      const env = await buildAuthenticationProcessEnv({
        launch,
        agentType: options.agentType,
        env: options.env,
        accountProfileId: options.accountProfileId,
        profilesRoot: options.profilesRoot,
        resolveLoginShellEnv: this.resolveLoginShellEnv,
      });
      const preparationInterruption = interruptedResult();
      if (preparationInterruption) return preparationInterruption;

      options.onProgress?.({ status: 'starting' });
      const startingInterruption = interruptedResult();
      if (startingInterruption) return startingInterruption;
      const child = this.spawnProcess(
        launch.command,
        accountProfileAuthenticationArgs(options, launch.args),
        {
          cwd: os.homedir(),
          env,
          stdio: ['pipe', 'pipe', 'pipe'],
          detached: process.platform !== 'win32',
          windowsHide: true,
        }
      );
      running.child = child;
      child.stdin?.on('error', (error: unknown) => {
        this.logger.debug(
          `[acp-auth] ${displayName} authorization input failed: ${formatErrorMessage(error)}`
        );
      });

      const outputParser = new BuiltinAuthenticationOutputParser(agentType);

      const emitOutput = (stream: 'stdout' | 'stderr', chunk: unknown): void => {
        const output = String(chunk);
        if (output.length === 0) return;
        const authorization = outputParser.push(output);
        if (authorization) {
          running.acceptsAuthorizationCode = authorization.acceptsAuthorizationCode === true;
          options.onProgress?.({ status: 'authorization', ...authorization });
        }
        // Retained temporarily for older renderer versions. Current UI consumes
        // the structured authorization event and does not render terminal text.
        options.onProgress?.({
          status: 'output',
          stream,
          output: output.slice(0, 16_384),
        });
      };
      child.stdout?.on('data', (chunk) => emitOutput('stdout', chunk));
      child.stderr?.on('data', (chunk) => emitOutput('stderr', chunk));

      const exit = await new Promise<{ code: number | null; error?: unknown }>((resolve) => {
        let settled = false;
        const finish = (result: { code: number | null; error?: unknown }): void => {
          if (settled) return;
          settled = true;
          resolve(result);
        };
        child.once('error', (error) => finish({ code: null, error }));
        child.once('exit', (code) => finish({ code }));
      });

      const processInterruption = interruptedResult();
      if (processInterruption) return processInterruption;
      if (exit.error !== undefined || exit.code !== 0) {
        const error =
          exit.error !== undefined
            ? formatErrorMessage(exit.error)
            : formatAuthenticationExitError(agentType, displayName, exit.code);
        options.onProgress?.({ status: 'error', error });
        return { success: false, disposition: 'error', error };
      }

      options.onProgress?.({ status: 'authenticated' });
      return { success: true, disposition: 'authenticated' };
    } catch (error) {
      const interruption = interruptedResult();
      if (interruption) return interruption;
      const message = formatErrorMessage(error);
      options.onProgress?.({ status: 'error', error: message });
      return { success: false, disposition: 'error', error: message };
    } finally {
      running.releaseAccountLease?.();
      if (timeoutHandle) {
        clearTimeout(timeoutHandle);
      }
      if (this.runningByAgentType.get(accountKey) === running) {
        this.runningByAgentType.delete(accountKey);
      }
    }
  }

  cancel(agentType: string, requestId: string): AcpAuthenticationResult {
    const entry = [...this.runningByAgentType.entries()].find(
      ([, item]) => item.agentType === agentType && item.requestId === requestId
    );
    const running = entry?.[1];
    if (!running || running.requestId !== requestId) {
      return { success: true, disposition: 'not-running' };
    }

    running.cancelled = true;
    if (!running.child && entry !== undefined) {
      this.runningByAgentType.delete(entry[0]);
      running.releaseAccountLease?.();
    }
    this.terminateAuthentication(agentType, running, 'cancelled');
    return { success: true, disposition: 'cancelled' };
  }

  submitAuthorizationCode(
    agentType: string,
    requestId: string,
    authorizationCode: string
  ): AcpAuthenticationResult {
    const entry = [...this.runningByAgentType.entries()].find(
      ([, item]) => item.agentType === agentType && item.requestId === requestId
    );
    const running = entry?.[1];
    if (!running || running.requestId !== requestId) {
      return { success: true, disposition: 'not-running' };
    }
    if (!running.acceptsAuthorizationCode) {
      return {
        success: false,
        disposition: 'error',
        error: `${getBuiltinDisplayName(agentType)} is not waiting for an authorization code`,
      };
    }
    if (running.authorizationCodeSubmitted) {
      return {
        success: false,
        disposition: 'error',
        error: `${getBuiltinDisplayName(agentType)} authorization code was already submitted`,
      };
    }

    const normalizedCode = authorizationCode.trim();
    if (
      normalizedCode.length === 0 ||
      normalizedCode.length > 4096 ||
      normalizedCode.includes('\n') ||
      normalizedCode.includes('\r')
    ) {
      return { success: false, disposition: 'error', error: 'Invalid authorization code' };
    }
    const stdin = running.child?.stdin;
    if (!stdin || !stdin.writable || stdin.destroyed) {
      return {
        success: false,
        disposition: 'error',
        error: `${getBuiltinDisplayName(agentType)} is no longer accepting authorization input`,
      };
    }

    try {
      running.authorizationCodeSubmitted = true;
      stdin.end(`${normalizedCode}\n`);
      return { success: true, disposition: 'input-accepted' };
    } catch (error) {
      running.authorizationCodeSubmitted = false;
      return {
        success: false,
        disposition: 'error',
        error: formatErrorMessage(error),
      };
    }
  }

  private terminateAuthentication(
    agentType: string,
    running: RunningAuthentication,
    reason: 'cancelled' | 'timed out'
  ): void {
    if (running.terminating || !running.child) return;
    running.terminating = true;
    void shutdownLocalAcpAgent({
      agentProcess: running.child,
      logger: this.logger,
      sessionLabel: `acp-auth:${agentType}:${reason}`,
      exitTimeoutMs: this.terminationGraceMs,
    }).catch((error: unknown) => {
      this.logger.debug(
        `[acp-auth] Failed to terminate authentication process: ${formatErrorMessage(error)}`
      );
    });
  }
}
