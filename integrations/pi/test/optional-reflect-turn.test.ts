import { createHash } from "node:crypto";

import type { AssistantMessage, ImageContent } from "@earendil-works/pi-ai";
import type { AgentSession } from "@earendil-works/pi-coding-agent";
import { describe, expect, it, vi } from "vitest";

import { InfrastructureError } from "../src/errors.js";
import { hasCompletedOptionalReflectTurn, runOptionalReflectTurn } from "../src/optional-reflect-turn.js";
import type { AgentJob, AgentResult } from "../src/types.js";

function assistant(stopReason: AssistantMessage["stopReason"] = "stop"): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "text", text: "Finished reviewing the comparison." }],
		api: "openai-completions",
		provider: "test",
		model: "test",
		usage: {
			input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason,
		timestamp: Date.now(),
	};
}

function setup() {
	const job = {
		continuation: {
			kind: "main_self_reflect_inspect",
			id: "optional-reflect-test",
			prompt: "Review the drawing and inspect the final formal CAD.",
		},
	} as AgentJob;
	const result = {} as AgentResult;
	const controller = new AbortController();
	const messages: AgentSession["messages"] = [assistant()];
	const prompt = vi.fn(async () => { messages.push(assistant()); });
	const session: Pick<AgentSession, "prompt" | "messages"> = { prompt, messages };
	const checkpoints: AgentResult[] = [];
	const persist = vi.fn(async () => { checkpoints.push(structuredClone(result)); });
	const onCompleted = vi.fn(async () => undefined);
	const options = {
		job, result, session, prompt: "Continue reviewing the final CAD.",
		signal: controller.signal, persist, onCompleted,
	};
	return { ...options, options, controller, messages, prompt, checkpoints };
}

describe("Optional Reflect turn checkpoint", () => {
	it("tracks completion for direct no-new-images reflection without aligned-view evidence", async () => {
		const context = setup();
		context.job.continuation!.kind = "main_self_reflect";
		context.job.continuation!.no_new_images = true;
		await runOptionalReflectTurn(context.options);
		expect(hasCompletedOptionalReflectTurn(context.job, context.result)).toBe(true);
		expect(context.prompt).toHaveBeenCalledWith(context.options.prompt, { expandPromptTemplates: false });
	});

	it("persists a started continuation before prompting and completes only after reconciliation", async () => {
		const context = setup();
		const image: ImageContent = { type: "image", data: "test", mimeType: "image/png" };
		context.prompt.mockImplementation(async () => {
			expect(context.checkpoints).toHaveLength(1);
			expect(context.checkpoints[0]?.optional_reflect_turn?.status).toBe("running");
			expect(context.checkpoints[0]?.applied_continuation?.id).toBe(context.job.continuation!.id);
			expect(hasCompletedOptionalReflectTurn(context.job, context.result)).toBe(false);
			context.messages.push(assistant());
		});
		context.onCompleted.mockImplementation(async () => {
			expect(context.result.optional_reflect_turn?.status).toBe("running");
			expect(context.persist).toHaveBeenCalledOnce();
		});
		await runOptionalReflectTurn({ ...context.options, images: [image] });
		expect(context.prompt).toHaveBeenCalledWith(context.options.prompt, {
			expandPromptTemplates: false, images: [image],
		});
		expect(context.onCompleted).toHaveBeenCalledOnce();
		expect(context.checkpoints.map((checkpoint) => checkpoint.optional_reflect_turn?.status)).toEqual([
			"running", "completed",
		]);
		expect(context.result.optional_reflect_turn?.completed_at).toBeDefined();
		expect(hasCompletedOptionalReflectTurn(context.job, context.result)).toBe(true);
	});

	it("leaves a durable running checkpoint when execution dies during its first prompt", async () => {
		const context = setup();
		const crash = new Error("simulated process termination");
		context.prompt.mockImplementation(async () => { throw crash; });
		await expect(runOptionalReflectTurn(context.options)).rejects.toBe(crash);
		expect(context.checkpoints).toHaveLength(1);
		const durable = context.checkpoints[0]!;
		expect(durable.optional_reflect_turn?.status).toBe("running");
		expect(durable.applied_continuation?.id).toBe(context.job.continuation!.id);
		expect(hasCompletedOptionalReflectTurn(context.job, durable)).toBe(false);
	});

	it("rejects an interrupt even when Pi resolves prompt with a final stop message", async () => {
		const context = setup();
		context.prompt.mockImplementation(async () => {
			context.messages.push(assistant());
			context.controller.abort();
		});
		await expect(runOptionalReflectTurn(context.options)).rejects.toThrow("interrupted");
		expect(context.result.optional_reflect_turn?.status).toBe("running");
		expect(context.onCompleted).not.toHaveBeenCalled();
		expect(context.persist).toHaveBeenCalledOnce();
	});

	it.each(["error", "aborted", "length", "toolUse"] as const)(
		"rejects Pi prompt resolving with %s rather than a completed decision",
		(stopReason) => {
			const context = setup();
			context.prompt.mockImplementation(async () => { context.messages.push(assistant(stopReason)); });
			return expect(runOptionalReflectTurn(context.options)).rejects.toThrow(
				new InfrastructureError(`Optional Reflect turn did not finish normally: ${stopReason}`),
			).then(() => {
				expect(context.result.optional_reflect_turn?.status).toBe("running");
				expect(context.onCompleted).not.toHaveBeenCalled();
				expect(context.persist).toHaveBeenCalledOnce();
			});
		},
	);

	it.each([true, false])("requires a new assistant message (prior assistant exists: %s)", async (priorExists) => {
		const context = setup();
		if (!priorExists) context.messages.length = 0;
		context.prompt.mockImplementation(async () => undefined);
		await expect(runOptionalReflectTurn(context.options)).rejects.toThrow("no new final assistant message");
		expect(context.result.optional_reflect_turn?.status).toBe("running");
		expect(context.onCompleted).not.toHaveBeenCalled();
	});

	it("keeps preflight failures incomplete", async () => {
		const context = setup();
		const failure = new Error("No configured model credential");
		context.prompt.mockRejectedValue(failure);
		await expect(runOptionalReflectTurn(context.options)).rejects.toBe(failure);
		expect(context.result.optional_reflect_turn?.status).toBe("running");
		expect(context.onCompleted).not.toHaveBeenCalled();
	});

	it("keeps failed evidence reconciliation incomplete despite a normal assistant stop", async () => {
		const context = setup();
		const failure = new InfrastructureError("Inspection evidence does not match the session");
		context.onCompleted.mockRejectedValue(failure);
		await expect(runOptionalReflectTurn(context.options)).rejects.toBe(failure);
		expect(context.result.optional_reflect_turn?.status).toBe("running");
		expect(context.result.optional_reflect_turn?.completed_at).toBeUndefined();
		expect(context.persist).toHaveBeenCalledOnce();
	});

	it("restores in-memory running state when saving completion fails", async () => {
		const context = setup();
		const failure = new Error("completion checkpoint write failed");
		context.persist.mockImplementationOnce(async () => {
			context.checkpoints.push(structuredClone(context.result));
		}).mockRejectedValueOnce(failure);
		await expect(runOptionalReflectTurn(context.options)).rejects.toBe(failure);
		expect(context.onCompleted).toHaveBeenCalledOnce();
		expect(context.persist).toHaveBeenCalledTimes(2);
		expect(context.result.optional_reflect_turn?.status).toBe("running");
		expect(context.result.optional_reflect_turn?.completed_at).toBeUndefined();
		expect(hasCompletedOptionalReflectTurn(context.job, context.result)).toBe(false);
	});

	it("does not prompt if saving the running checkpoint fails", async () => {
		const context = setup();
		context.persist.mockRejectedValue(new Error("checkpoint unavailable"));
		await expect(runOptionalReflectTurn(context.options)).rejects.toThrow("checkpoint unavailable");
		expect(context.prompt).not.toHaveBeenCalled();
		expect(context.onCompleted).not.toHaveBeenCalled();
		expect(context.result.optional_reflect_turn?.status).toBe("running");
	});

	it("completes a retry after an interrupted turn and keeps its continuation identity", async () => {
		const context = setup();
		context.prompt.mockImplementationOnce(async () => {
			context.messages.push(assistant("aborted"));
			context.controller.abort();
		});
		await expect(runOptionalReflectTurn(context.options)).rejects.toThrow("interrupted");
		const applied = context.result.applied_continuation;
		await runOptionalReflectTurn({ ...context.options, signal: new AbortController().signal });
		expect(context.result.applied_continuation).toEqual(applied);
		expect(context.checkpoints.map((checkpoint) => checkpoint.optional_reflect_turn?.status)).toEqual([
			"running", "running", "completed",
		]);
		expect(hasCompletedOptionalReflectTurn(context.job, context.result)).toBe(true);
	});

	it.each(["missing", "running", "other-id", "other-prompt", "other-kind"] as const)(
		"rejects a %s completion checkpoint",
		async (caseName) => {
			const context = setup();
			await runOptionalReflectTurn(context.options);
			if (caseName === "missing") delete context.result.optional_reflect_turn;
			else if (caseName === "running") context.result.optional_reflect_turn!.status = "running";
			else if (caseName === "other-id") context.result.optional_reflect_turn!.continuation_id = "another";
			else if (caseName === "other-prompt") {
				context.result.optional_reflect_turn!.prompt_sha256 = createHash("sha256").update("another").digest("hex");
			} else context.job.continuation!.kind = "main_self_reflect";
			expect(hasCompletedOptionalReflectTurn(context.job, context.result)).toBe(false);
		},
	);
});
