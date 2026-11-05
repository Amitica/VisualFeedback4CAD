import { createHash } from "node:crypto";

import type { ImageContent } from "@earendil-works/pi-ai";
import type { AgentSession } from "@earendil-works/pi-coding-agent";

import { InfrastructureError } from "./errors.js";
import { isoNow } from "./files.js";
import type { AgentJob, AgentResult } from "./types.js";

export function hasCompletedOptionalReflectTurn(job: AgentJob, result: AgentResult): boolean {
	const continuation = job.continuation;
	const turn = result.optional_reflect_turn;
	return (continuation?.kind === "main_self_reflect_inspect" ||
		(continuation?.kind === "main_self_reflect" && continuation.no_new_images === true)) &&
		turn?.continuation_id === continuation.id &&
		turn.prompt_sha256 === createHash("sha256").update(continuation.prompt).digest("hex") &&
		turn.status === "completed";
}

export async function runOptionalReflectTurn(options: {
	job: AgentJob;
	result: AgentResult;
	session: Pick<AgentSession, "prompt" | "messages">;
	prompt: string;
	images?: ImageContent[];
	signal: AbortSignal;
	persist: () => Promise<void>;
	onCompleted: () => Promise<void>;
}): Promise<void> {
	const { job, result, session, signal } = options;
	const continuation = job.continuation;
	if (continuation?.kind !== "main_self_reflect_inspect" &&
		!(continuation?.kind === "main_self_reflect" && continuation.no_new_images === true)) {
		throw new InfrastructureError("Optional Reflect turn requires a main_self_reflect_inspect continuation");
	}
	const assertRunning = (): void => {
		if (signal.aborted) throw new Error("interrupted");
	};
	assertRunning();
	const promptSha256 = createHash("sha256").update(continuation.prompt).digest("hex");
	const runningTurn = {
		continuation_id: continuation.id,
		prompt_sha256: promptSha256,
		status: "running" as const,
		started_at: isoNow(),
	};
	result.optional_reflect_turn = runningTurn;
	// Applied means the continuation has started; only a completed turn can finish it.
	if (result.applied_continuation?.id !== continuation.id ||
		result.applied_continuation.prompt_sha256 !== promptSha256) {
		result.applied_continuation = {
			id: continuation.id,
			prompt_sha256: promptSha256,
			applied_at: runningTurn.started_at,
		};
	}
	await options.persist();
	assertRunning();
	const previousAssistant = session.messages.findLast((message) => message.role === "assistant");
	await session.prompt(options.prompt, {
		expandPromptTemplates: false,
		...(options.images !== undefined ? { images: options.images } : {}),
	});
	assertRunning();
	const finalAssistant = session.messages.findLast((message) => message.role === "assistant");
	if (!finalAssistant || finalAssistant === previousAssistant) {
		throw new InfrastructureError("Optional Reflect turn produced no new final assistant message");
	}
	if (finalAssistant.stopReason !== "stop") {
		throw new InfrastructureError(
			`Optional Reflect turn did not finish normally: ${finalAssistant.stopReason}` +
				(finalAssistant.errorMessage ? ` (${finalAssistant.errorMessage})` : ""),
		);
	}
	await options.onCompleted();
	assertRunning();
	result.optional_reflect_turn = { ...runningTurn, status: "completed", completed_at: isoNow() };
	try {
		await options.persist();
	} catch (error) {
		// A failed completion checkpoint must not authorize acceptance in this process.
		result.optional_reflect_turn = runningTurn;
		throw error;
	}
}
