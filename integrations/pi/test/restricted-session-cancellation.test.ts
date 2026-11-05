import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const createAgentSessionMock = vi.hoisted(() => vi.fn());

vi.mock("@earendil-works/pi-coding-agent", async () => {
	const actual = await vi.importActual<typeof import("@earendil-works/pi-coding-agent")>("@earendil-works/pi-coding-agent");
	return {
		...actual,
		createAgentSession: createAgentSessionMock,
		DefaultResourceLoader: class { async reload(): Promise<void> {} },
		getAgentDir: () => "/tmp/cadgenbench-restricted-session-test",
		SessionManager: { create: () => ({}) },
		SettingsManager: { inMemory: () => ({}) },
	};
});

import { runRestrictedSession } from "../src/drawing-reflection.js";
import { RunController, type ResolvedModel } from "../src/runner.js";
import type { SubagentRunUsage } from "../src/types.js";

const workspaces: string[] = [];
beforeEach(() => { createAgentSessionMock.mockReset(); });
afterEach(async () => {
	vi.restoreAllMocks();
	vi.unstubAllEnvs();
	vi.useRealTimers();
	for (const workspace of workspaces.splice(0)) await rm(workspace, { recursive: true, force: true });
});

function deferred<T>() {
	let resolve!: (value: T) => void;
	let reject!: (reason: Error) => void;
	const promise = new Promise<T>((resolvePromise, rejectPromise) => { resolve = resolvePromise; reject = rejectPromise; });
	return { promise, resolve, reject };
}

async function fixture() {
	const workspace = await mkdtemp(join(tmpdir(), "restricted-session-cancellation-"));
	workspaces.push(workspace);
	const cancellation = new AbortController();
	const controller = new RunController();
	const addListener = vi.spyOn(cancellation.signal, "addEventListener");
	const removeListener = vi.spyOn(cancellation.signal, "removeEventListener");
	const deleteSession = vi.spyOn(controller, "delete");
	const unsubscribe = vi.fn();
	const session = {
		sessionId: "repair-session-cancellation",
		getActiveToolNames: () => ["submit_repair"],
		getSessionStats: () => ({
			sessionFile: undefined,
			tokens: { input: 7, output: 4, cacheRead: 1, cacheWrite: 1, total: 13 },
			cost: 0.01,
			toolCalls: 2,
		}),
		subscribe: vi.fn(() => unsubscribe),
		prompt: vi.fn(async (_text: string, _options?: unknown): Promise<void> => {}),
		abort: vi.fn(async (): Promise<void> => {}),
		dispose: vi.fn(),
	};
	createAgentSessionMock.mockResolvedValue({ session });
	let submitted: { candidate_id: string } | null = null;
	const onUsage = vi.fn(async (_usage: SubagentRunUsage): Promise<void> => {});
	const run = () => runRestrictedSession({
		role: "geometry_repair", workspace, outputDir: "drawing_feedback/geometry-repair/test",
		resolvedModel: { model: { id: "test", provider: "test" }, runtime: {} } as unknown as ResolvedModel,
		thinkingLevel: "low", controller, signal: cancellation.signal,
		systemPrompt: "Verify one suspected difference and submit a structured repair result.",
		prompt: "Check the mounting foot count.", images: [], toolNames: ["submit_repair"], tools: [],
		getSubmitted: () => submitted,
		activeTimeoutMs: 60_000, wallTimeoutMs: 120_000,
		resultPath: "drawing_feedback/geometry-repair/test/result.json", onUsage,
	});
	const expectCleaned = () => {
		expect(addListener).toHaveBeenCalledOnce();
		const listener = addListener.mock.calls[0]![1];
		expect(addListener).toHaveBeenCalledWith("abort", listener, { once: true });
		expect(removeListener).toHaveBeenCalledExactlyOnceWith("abort", listener);
		expect(unsubscribe).toHaveBeenCalledOnce();
		expect(session.dispose).toHaveBeenCalledOnce();
		expect(deleteSession).toHaveBeenCalledExactlyOnceWith(session);
		expect(vi.getTimerCount()).toBe(0);
	};
	const expectInterruptedUsage = () => {
		expect(onUsage).toHaveBeenCalledOnce();
		const usage = onUsage.mock.calls[0]![0];
		expect(usage).toMatchObject({
			role: "geometry_repair", status: "interrupted", accounting_id: session.sessionId,
			tokens: { input: 7, output: 4, cache_read: 1, cache_write: 1, total: 13 },
			cost: 0.01, tool_calls: 2,
		});
		expect(usage).not.toHaveProperty("result_path");
	};
	return { cancellation, controller, session, onUsage, run, expectCleaned, expectInterruptedUsage,
		setSubmitted: (value: { candidate_id: string }) => { submitted = value; } };
}

describe("runRestrictedSession cancellation", () => {
	it("uses the multiplied active deadline and reports the real elapsed time", async () => {
		vi.useFakeTimers();
		vi.stubEnv("CADGENBENCH_SUBAGENT_TIMEOUT_MULTIPLIER", "3");
		const f = await fixture();
		const started = deferred<void>();
		const pending = deferred<void>();
		f.session.prompt.mockImplementation(async () => { started.resolve(); await pending.promise; });
		f.session.abort.mockImplementation(async () => { pending.reject(new Error("SDK deadline aborted")); });

		const execution = f.run();
		const rejection = expect(execution).rejects.toThrow("exhausted 3 minutes of active Agent time");
		await started.promise;
		await vi.advanceTimersByTimeAsync(60_000);
		expect(f.session.abort).not.toHaveBeenCalled();
		await vi.advanceTimersByTimeAsync(120_000);
		await rejection;

		expect(f.session.abort).toHaveBeenCalledOnce();
		expect(f.onUsage.mock.calls[0]![0]).toMatchObject({ status: "timeout", duration_s: 180 });
		f.expectCleaned();
	});

	it("aborts an active SDK session on the tool signal and records interrupted usage without a submit reminder", async () => {
		vi.useFakeTimers();
		const f = await fixture();
		const started = deferred<void>();
		const pending = deferred<void>();
		f.session.prompt.mockImplementation(async () => { started.resolve(); await pending.promise; });
		f.session.abort.mockImplementation(async () => { pending.reject(new Error("SDK stream aborted")); });

		const execution = f.run();
		const rejection = expect(execution).rejects.toThrow("SDK stream aborted");
		await started.promise;
		f.cancellation.abort("Main cancelled this repair");
		await rejection;

		expect(f.session.abort).toHaveBeenCalledOnce();
		expect(f.session.prompt).toHaveBeenCalledOnce();
		expect(f.controller.interrupted).toBe(false);
		f.expectInterruptedUsage();
		f.expectCleaned();
	});

	it("does not prompt the model when its tool signal was already aborted", async () => {
		vi.useFakeTimers();
		const f = await fixture();
		f.cancellation.abort("cancelled before launch");

		await expect(f.run()).rejects.toThrow("interrupted");

		expect(f.session.prompt).not.toHaveBeenCalled();
		f.expectInterruptedUsage();
		f.expectCleaned();
	});

	it.each(["resolve", "reject"])("does not accept a submitted value produced while an aborted SDK prompt settles by %s", async (settle) => {
		vi.useFakeTimers();
		const f = await fixture();
		const started = deferred<void>();
		const pending = deferred<void>();
		f.session.prompt.mockImplementation(async () => { started.resolve(); await pending.promise; });
		f.session.abort.mockImplementation(async () => {
			f.setSubmitted({ candidate_id: "late-candidate" });
			if (settle === "resolve") pending.resolve();
			else pending.reject(new Error("SDK cancelled after late submission"));
		});

		const execution = f.run();
		const rejection = expect(execution).rejects.toThrow(settle === "resolve" ? "interrupted" : "SDK cancelled after late submission");
		await started.promise;
		f.cancellation.abort();
		await rejection;

		expect(f.session.abort).toHaveBeenCalledOnce();
		expect(f.session.prompt).toHaveBeenCalledOnce();
		f.expectInterruptedUsage();
		f.expectCleaned();
	});

	it("does not start another submit reminder after cancellation during the first reminder", async () => {
		vi.useFakeTimers();
		const f = await fixture();
		const reminderStarted = deferred<void>();
		const pending = deferred<void>();
		f.session.prompt.mockResolvedValueOnce(undefined).mockImplementationOnce(async () => {
			reminderStarted.resolve();
			await pending.promise;
		});
		f.session.abort.mockImplementation(async () => { pending.reject(new Error("SDK reminder aborted")); });

		const execution = f.run();
		const rejection = expect(execution).rejects.toThrow("SDK reminder aborted");
		await reminderStarted.promise;
		f.cancellation.abort();
		await rejection;

		expect(f.session.prompt).toHaveBeenCalledTimes(2);
		expect(f.session.prompt.mock.calls[1]![0]).toContain("You have not submitted the required structured result");
		f.expectInterruptedUsage();
		f.expectCleaned();
	});

	it("removes the signal listener on successful completion so later cancellation cannot abort a disposed session", async () => {
		vi.useFakeTimers();
		const f = await fixture();
		f.session.prompt.mockImplementation(async () => { f.setSubmitted({ candidate_id: "verified-candidate" }); });

		const result = await f.run();
		expect(result.value).toEqual({ candidate_id: "verified-candidate" });
		expect(f.onUsage).toHaveBeenCalledOnce();
		expect(result.usage).toMatchObject({ status: "success", result_path: "drawing_feedback/geometry-repair/test/result.json" });
		f.expectCleaned();
		f.cancellation.abort();
		expect(f.session.abort).not.toHaveBeenCalled();
	});
});
