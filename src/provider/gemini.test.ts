/** @fileoverview Tests for the Gemini provider helpers. */

import { describe, it } from 'node:test';
import { deepStrictEqual, strictEqual } from 'node:assert';
import { parseStreamLine, buildCommand } from './gemini.ts';
import type { AgentConfig } from '../core/types.ts';

function makeAgent(overrides?: Partial<AgentConfig>): AgentConfig {
  return {
    id: 'test',
    name: 'Test',
    model: 'gemini-2.0-flash',
    maxTurns: 10,
    systemPrompt: '',
    channels: ['*'],
    ...overrides,
  };
}

describe('gemini buildCommand', () => {
  it('returns gemini-sdk sentinel binary', () => {
    const cmd = buildCommand({ message: 'test', agent: makeAgent() });
    strictEqual(cmd.binary, 'gemini-sdk');
  });

  it('returns empty args array', () => {
    const cmd = buildCommand({ message: 'test', agent: makeAgent() });
    deepStrictEqual(cmd.args, []);
  });

  it('includes GEMINI_API_KEY in env when set', () => {
    const original = process.env['GEMINI_API_KEY'];
    process.env['GEMINI_API_KEY'] = 'test-key-123';
    const cmd = buildCommand({ message: 'test', agent: makeAgent() });
    strictEqual(cmd.env?.['GEMINI_API_KEY'], 'test-key-123');
    if (original === undefined) delete process.env['GEMINI_API_KEY'];
    else process.env['GEMINI_API_KEY'] = original;
  });

  it('returns empty env when GEMINI_API_KEY is not set', () => {
    const original = process.env['GEMINI_API_KEY'];
    delete process.env['GEMINI_API_KEY'];
    const cmd = buildCommand({ message: 'test', agent: makeAgent() });
    deepStrictEqual(cmd.env, {});
    if (original !== undefined) process.env['GEMINI_API_KEY'] = original;
  });
});

describe('gemini parseStreamLine', () => {
  it('always returns empty array (no-op for SDK-based provider)', () => {
    strictEqual(parseStreamLine('').length, 0);
    strictEqual(parseStreamLine('{"type":"system"}').length, 0);
    strictEqual(parseStreamLine('not json').length, 0);
  });
});
