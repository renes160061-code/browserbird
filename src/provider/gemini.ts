/** @fileoverview Gemini provider: arg building and streaming via the Google GenAI SDK. */

import type { AgentConfig } from '../core/types.ts';
import type { StreamEvent, ToolImage } from './stream.ts';

export interface SpawnOptions {
  message: string;
  sessionId?: string;
  agent: AgentConfig;
  mcpConfigPath?: string;
  timezone?: string;
  globalTimeoutMs?: number;
  extraEnv?: Record<string, string>;
  docsPrompt?: string;
}

// ---------------------------------------------------------------------------
// buildCommand is kept for interface compatibility with spawn.ts.
// For Gemini we don't shell out to a CLI — runGemini() drives the SDK
// directly. buildCommand returns a sentinel that spawn.ts will never use
// because spawnProvider short-circuits to runGemini when the binary is
// "gemini-sdk".
// ---------------------------------------------------------------------------

interface ProviderCommand {
  binary: string;
  args: string[];
  cwd?: string;
  env?: Record<string, string>;
}

export function buildCommand(options: SpawnOptions): ProviderCommand {
  const apiKey = process.env['GEMINI_API_KEY'] ?? '';
  return {
    binary: 'gemini-sdk', // sentinel — not actually spawned
    args: [],
    env: apiKey ? { GEMINI_API_KEY: apiKey } : {},
  };
}

// ---------------------------------------------------------------------------
// Core streaming runner
// ---------------------------------------------------------------------------

/**
 * Runs a single-turn (or multi-turn via history) Gemini request and yields
 * StreamEvents that are compatible with the rest of the BrowserBird pipeline.
 *
 * Gemini does not have a persistent session concept like Claude Code's
 * --resume flag, so we store conversation history in memory and re-send it
 * on each call within the same session.  The in-memory map is keyed by the
 * browserbird sessionId.
 */

interface GeminiMessage {
  role: 'user' | 'model';
  parts: Array<{ text: string }>;
}

// In-memory history per session (survives process lifetime only).
const sessionHistory = new Map<string, GeminiMessage[]>();

export async function* runGemini(
  options: SpawnOptions,
  signal: AbortSignal,
): AsyncGenerator<StreamEvent> {
  const apiKey = options.extraEnv?.['GEMINI_API_KEY'] ?? process.env['GEMINI_API_KEY'];
  if (!apiKey) {
    yield { type: 'error', error: 'GEMINI_API_KEY is not set' };
    return;
  }

  // Lazy-import so the SDK is only required at runtime.
  const { GoogleGenerativeAI } = await import('@google/generative-ai');
  const genAI = new GoogleGenerativeAI(apiKey);

  const model = options.agent.model ?? 'gemini-2.0-flash';

  const systemParts: string[] = [];
  if (options.agent.systemPrompt) systemParts.push(options.agent.systemPrompt);
  if (options.docsPrompt) systemParts.push(options.docsPrompt);
  if (options.timezone) {
    systemParts.push(
      `System timezone: ${options.timezone}. All cron expressions and scheduled times use this timezone.`,
    );
  }
  const systemInstruction = systemParts.join('\n\n') || undefined;

  const genModel = genAI.getGenerativeModel({
    model,
    ...(systemInstruction ? { systemInstruction } : {}),
  });

  // Build or extend history for this session.
  const historyKey = options.sessionId ?? '';
  const history: GeminiMessage[] = sessionHistory.get(historyKey) ?? [];

  // Emit a synthetic init event so downstream code (channel.ts etc.) can
  // read the sessionId and model.
  yield {
    type: 'init',
    sessionId: historyKey || 'gemini-session',
    model,
    apiKeySource: 'env',
  };

  const startMs = Date.now();
  let fullText = '';
  let tokensIn = 0;
  let tokensOut = 0;

  try {
    const chat = genModel.startChat({ history });

    const result = await chat.sendMessageStream(options.message, { signal } as RequestInit);

    for await (const chunk of result.stream) {
      if (signal.aborted) break;
      const text = chunk.text();
      if (text) {
        fullText += text;
        yield { type: 'text_delta', delta: text };
      }
    }

    const response = await result.response;
    tokensIn = response.usageMetadata?.promptTokenCount ?? 0;
    tokensOut = response.usageMetadata?.candidatesTokenCount ?? 0;

    // Persist the new exchange to history.
    history.push({ role: 'user', parts: [{ text: options.message }] });
    history.push({ role: 'model', parts: [{ text: fullText }] });
    sessionHistory.set(historyKey, history);

    yield {
      type: 'completion',
      subtype: 'success',
      result: fullText,
      sessionId: historyKey || 'gemini-session',
      isError: false,
      tokensIn,
      tokensOut,
      cacheCreationTokens: 0,
      cacheReadTokens: 0,
      costUsd: 0, // Gemini pricing varies; set to 0 unless you add a cost calc
      durationMs: Date.now() - startMs,
      numTurns: Math.ceil(history.length / 2),
    };
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    yield { type: 'error', error: message };
    yield {
      type: 'completion',
      subtype: 'error_during_execution',
      result: '',
      sessionId: historyKey || 'gemini-session',
      isError: true,
      tokensIn,
      tokensOut,
      cacheCreationTokens: 0,
      cacheReadTokens: 0,
      costUsd: 0,
      durationMs: Date.now() - startMs,
      numTurns: Math.ceil(history.length / 2),
    };
  }
}

/** Clears the in-memory history for a given session (call on session expiry). */
export function clearSessionHistory(sessionId: string): void {
  sessionHistory.delete(sessionId);
}

// ---------------------------------------------------------------------------
// parseStreamLine — kept for interface compatibility but is a no-op for
// Gemini because we never parse subprocess stdout here.
// ---------------------------------------------------------------------------
export function parseStreamLine(_line: string): StreamEvent[] {
  return [];
}
