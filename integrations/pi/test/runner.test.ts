import { createHash } from "node:crypto";
import { describe, expect, it, vi } from "vitest";
import type { AgentSession } from "@earendil-works/pi-coding-agent";

import { InfrastructureError } from "../src/errors.js";
import { normalizeCadToolArguments } from "../src/cad-tool.js";
import { loadSystemPrompt } from "../src/prompts.js";
import {
	applyProviderCredentialCompatibility,
	assertImageCapableModel,
	assertContinuation,
	continuationAccepted,
	driveUntilValid,
	dimensionRefreshFeedback,
	RunController,
	unchangedContinuationPrompt,
} from "../src/runner.js";
import {
	ACTIVE_AGENT_TOOLS,
	agentToolsForMode,
	BASE_AGENT_TOOLS,
	GEOMETRY_REPAIR_PROFILE,
	SHAPE_FEEDBACK_PROFILE,
	CAUSAL_REPAIR_PROFILE,
	DUAL_ROI_CROP_PROFILE,
	DIMENSION_ANNOTATION_PROFILE,
	TARGET_ANNOTATION_PROFILE,
	type AgentJob,
	type AgentResult,
	type ArtifactPayload,
	type CandidateInspection,
	type MainViewReflectSummary,
	type PiContinuationKind,
} from "../src/types.js";

function artifact(status: ArtifactPayload["status"], valid = false): ArtifactPayload {
	return {
		status,
		path: valid ? "output.step" : null,
		sha256: valid ? "step-sha" : null,
		validation: valid
			? {
					is_valid: true,
					is_watertight: true,
					topology_errors: [],
					solid_count: 1,
					shell_count: 1,
					face_count: 6,
					volume: 1,
					bbox: {},
				}
			: null,
		validation_error: null,
		render_path: null,
		render_error: null,
	};
}

function inspection(
	status: ArtifactPayload["status"],
	options: { valid?: boolean; formal?: boolean; reason?: string } = {},
): CandidateInspection {
	return {
		artifact: artifact(status, options.valid ?? false),
		formal_execution_current: options.formal ?? false,
		reason: options.reason ?? "test inspection",
		source_sha256: "source-sha",
	};
}

function continuationFixture(kind: PiContinuationKind = "main_self_reflect_inspect") {
	const baseline = inspection("valid", { valid: true, formal: true });
	const job = {
		continuation: {
			kind,
			id: "reflection-test",
			prompt: "Inspect the final CAD before finishing.",
			baseline_source_sha256: baseline.source_sha256!,
			baseline_step_sha256: baseline.artifact.sha256!,
		},
	} as AgentJob;
	const summary: MainViewReflectSummary = {
		version: 1,
		kind: "main_view_reflect",
		locked_view_manifest_sha256: "manifest-sha",
		baseline_source_sha256: baseline.source_sha256!,
		baseline_step_sha256: baseline.artifact.sha256!,
		audit_manifest_path: "drawing_feedback/main-view-reflect/audit.json",
		inspection_count: 1,
		crop_count: 0,
		inspected_formal_states: [{ source_sha256: baseline.source_sha256!, step_sha256: baseline.artifact.sha256! }],
		cropped_formal_states: [],
	};
	const result = {
		applied_continuation: {
			id: job.continuation!.id,
			prompt_sha256: "prompt-sha",
			applied_at: "2026-09-06T00:00:00Z",
		},
		optional_reflect_turn: {
			continuation_id: job.continuation!.id,
			prompt_sha256: createHash("sha256").update(job.continuation!.prompt).digest("hex"),
			status: "completed",
			started_at: "2026-09-06T00:00:00Z",
			completed_at: "2026-09-06T00:01:00Z",
		},
		main_view_reflection: summary,
	} as AgentResult;
	return { job, result, summary, baseline };
}

describe("persistent Agent tool set", () => {
	it("keeps baseline system rules and the cheat sheet while exposing repair tools without duplicate workflow instructions", async () => {
		const prompt = await loadSystemPrompt(
			new URL("../../..", import.meta.url).pathname,
			new URL("..", import.meta.url).pathname,
			"geometry-repair",
		);
		expect(prompt).toContain("`inspect_comparison`, `repair_geometry`, and `adopt_shape_candidate`");
		expect(prompt).toContain("Maintain `/workspace/solution.py`");
		expect(prompt).toContain("# build123d cheat sheet");
		expect(prompt).not.toContain("No drawing-analysis or reflection SubAgent is available");
		expect(prompt).not.toContain("preview_shape_feedback");
		expect(prompt).not.toContain("evaluation_id");
		const baseline = await loadSystemPrompt(
			new URL("../../..", import.meta.url).pathname,
			new URL("..", import.meta.url).pathname,
			"base-only",
		);
		for (const line of baseline.split("\n").filter(line => line && !line.startsWith("Available tools") && !line.startsWith("- Continuously edit"))) {
			expect(prompt).toContain(line);
		}
	});
	it("accepts the standalone atomic repair profile and rejects auxiliary profile mixing", () => {
		const { job } = continuationFixture();
		job.agent_mode = "base-only";
		Object.assign(job.continuation!, {
			baseline_source_sha256: "a".repeat(64), baseline_step_sha256: "b".repeat(64),
			locked_view_manifest_path: "drawing_feedback/locked.json", locked_view_manifest_sha256: "c".repeat(64),
			visual_review_profile: GEOMETRY_REPAIR_PROFILE,
		});
		expect(() => assertContinuation(job)).not.toThrow();
		for (const [field, value] of [
			["causal_repair_profile", CAUSAL_REPAIR_PROFILE],
			["dual_roi_crop_profile", DUAL_ROI_CROP_PROFILE],
			["target_annotation_profile", TARGET_ANNOTATION_PROFILE],
		]) {
			const mixed = structuredClone(job);
			Object.assign(mixed.continuation!, { [field!]: value });
			expect(() => assertContinuation(mixed)).toThrow("Atomic geometry repair cannot be combined");
		}
		job.continuation!.visual_review_profile = SHAPE_FEEDBACK_PROFILE;
		expect(() => assertContinuation(job)).not.toThrow();
		job.continuation!.dual_roi_crop_profile = DUAL_ROI_CROP_PROFILE;
		expect(() => assertContinuation(job)).toThrow("Aligned shape feedback cannot be combined");
	});
	it("enables all seven native tools plus formal execution and 2D self-reflection", () => {
		expect(ACTIVE_AGENT_TOOLS).toEqual([
			"read",
			"write",
			"edit",
			"bash",
			"grep",
			"find",
			"ls",
			"execute_build123d",
			"self_reflect_2d_drawing",
		]);
	});

	it("removes the project drawing SubAgent entrypoint in base-only mode", () => {
		expect(BASE_AGENT_TOOLS).toEqual([
			"read",
			"write",
			"edit",
			"bash",
			"grep",
			"find",
			"ls",
			"execute_build123d",
		]);
		expect(agentToolsForMode("base-only")).toEqual(BASE_AGENT_TOOLS);
		expect(agentToolsForMode("base-only")).not.toContain("self_reflect_2d_drawing");
		expect(agentToolsForMode(undefined)).toEqual(ACTIVE_AGENT_TOOLS);
	});

	it("loads a base-only prompt that neither advertises nor requests the drawing SubAgent", async () => {
		const repositoryRoot = new URL("../../..", import.meta.url).pathname;
		const integrationRoot = new URL("..", import.meta.url).pathname;
		const prompt = await loadSystemPrompt(repositoryRoot, integrationRoot, "base-only");
		expect(prompt).not.toContain("self_reflect_2d_drawing");
		expect(prompt).not.toContain("restricted search and reflection subagents");
		expect(prompt).toContain("No drawing-analysis or reflection SubAgent is available");
	});

	it("normalizes only the legacy code argument and leaves other malformed arguments for schema rejection", () => {
		expect(normalizeCadToolArguments({})).toEqual({});
		expect(normalizeCadToolArguments({ code: "legacy source" })).toEqual({});
		expect(normalizeCadToolArguments({ unexpected: true })).toEqual({ unexpected: true });
		expect(normalizeCadToolArguments({ code: "legacy", extra: true })).toEqual({ code: "legacy", extra: true });
		expect(normalizeCadToolArguments(null)).toBeNull();
	});
});

describe("run-until-valid controller", () => {
	it("aborts the main session and both nested subagent sessions exactly once", async () => {
		const controller = new RunController();
		const sessions = Array.from({ length: 3 }, () => ({ abort: vi.fn(async () => undefined) })) as unknown as AgentSession[];
		for (const session of sessions) controller.add(session);

		await controller.interrupt("SIGINT");
		await controller.interrupt("duplicate SIGINT");

		expect(controller.interrupted).toBe(true);
		expect(controller.interruptReason).toBe("SIGINT");
		expect(controller.signal.aborted).toBe(true);
		for (const session of sessions) expect(session.abort).toHaveBeenCalledOnce();
	});

	it("continues until a valid artifact also has current formal source and STEP hashes", async () => {
		const promptInitial = vi.fn(async () => undefined);
		const promptContinue = vi.fn(async () => undefined);
		const staleValid = inspection("valid", {
			valid: true,
			formal: false,
			reason: "output.step was generated or replaced through bash",
		});
		const currentValid = inspection("valid", { valid: true, formal: true });
		const inspect = vi
			.fn<() => Promise<CandidateInspection>>()
			.mockResolvedValueOnce(staleValid)
			.mockResolvedValueOnce(currentValid);
		const onInspection = vi.fn(async () => undefined);

		const result = await driveUntilValid({
			initialPromptPending: true,
			promptInitial,
			promptContinue,
			inspect,
			onInspection,
			isInterrupted: () => false,
		});

		expect(result).toEqual(currentValid);
		expect(promptInitial).toHaveBeenCalledOnce();
		expect(promptContinue).toHaveBeenCalledOnce();
		expect(promptContinue).toHaveBeenCalledWith(staleValid);
		expect(onInspection).toHaveBeenCalledTimes(2);
	});

	it("resume accepts an already-current formal candidate without replaying the image prompt", async () => {
		const promptInitial = vi.fn(async () => undefined);
		const promptContinue = vi.fn(async () => undefined);
		await driveUntilValid({
			initialPromptPending: false,
			promptInitial,
			promptContinue,
			inspect: vi.fn(async () => inspection("valid", { valid: true, formal: true })),
			onInspection: vi.fn(async () => undefined),
			isInterrupted: () => false,
		});
		expect(promptInitial).not.toHaveBeenCalled();
		expect(promptContinue).not.toHaveBeenCalled();
	});

	it("keeps a Reflection continuation running until both source and STEP differ from baseline", async () => {
		const baseline = inspection("valid", { valid: true, formal: true });
		baseline.source_sha256 = "baseline-source";
		baseline.artifact.sha256 = "baseline-step";
		const revised = inspection("valid", { valid: true, formal: true });
		revised.source_sha256 = "revised-source";
		revised.artifact.sha256 = "revised-step";
		const promptContinue = vi.fn(async () => undefined);
		const inspect = vi
			.fn<() => Promise<CandidateInspection>>()
			.mockResolvedValueOnce(baseline)
			.mockResolvedValueOnce(revised);

		const result = await driveUntilValid({
			initialPromptPending: true,
			promptInitial: async () => undefined,
			promptContinue,
			inspect,
			onInspection: async () => undefined,
			isInterrupted: () => false,
			isAccepted: (candidate) =>
				candidate.formal_execution_current &&
				candidate.source_sha256 !== "baseline-source" &&
				candidate.artifact.sha256 !== "baseline-step",
		});

		expect(result).toEqual(revised);
		expect(promptContinue).toHaveBeenCalledOnce();
		expect(promptContinue).toHaveBeenCalledWith(baseline);
	});

	it("propagates infrastructure failures and honors interruption before prompting", async () => {
		await expect(
			driveUntilValid({
				initialPromptPending: true,
				promptInitial: async () => {
					throw new InfrastructureError("provider failed");
				},
				promptContinue: async () => undefined,
				inspect: async () => inspection("missing"),
				onInspection: async () => undefined,
				isInterrupted: () => false,
			}),
		).rejects.toThrow("provider failed");
		const prompt = vi.fn(async () => undefined);
		await expect(
			driveUntilValid({
				initialPromptPending: true,
				promptInitial: prompt,
				promptContinue: async () => undefined,
				inspect: async () => inspection("missing"),
				onInspection: async () => undefined,
				isInterrupted: () => true,
			}),
		).rejects.toThrow("interrupted");
		expect(prompt).not.toHaveBeenCalled();
	});

	it("honors an interruption between a completed turn and its continuation", async () => {
		let interrupted = false;
		const promptContinue = vi.fn(async () => undefined);
		await expect(
			driveUntilValid({
				initialPromptPending: true,
				promptInitial: async () => undefined,
				promptContinue,
				inspect: async () => {
					interrupted = true;
					return inspection("invalid", { formal: true });
				},
				onInspection: async () => undefined,
				isInterrupted: () => interrupted,
			}),
		).rejects.toThrow("interrupted");
		expect(promptContinue).not.toHaveBeenCalled();
	});
});

describe("Optional Reflect completion evidence", () => {
	it("accepts unchanged direct no-new-images output only after a completed matching reflection", () => {
		const { job, result, baseline } = continuationFixture();
		job.continuation!.kind = "main_self_reflect";
		job.continuation!.no_new_images = true;
		result.applied_continuation!.prompt_sha256 = result.optional_reflect_turn!.prompt_sha256;
		expect(continuationAccepted(job, result, baseline)).toBe(true);
		result.optional_reflect_turn!.status = "running";
		expect(continuationAccepted(job, result, baseline)).toBe(false);
		result.optional_reflect_turn!.status = "completed";
		expect(continuationAccepted(job, result, { ...baseline, formal_execution_current: false })).toBe(false);
	});

	it("accepts an unchanged baseline after inspecting its formal STEP without requiring a crop", () => {
		const { job, result, baseline, summary } = continuationFixture();
		expect(summary.crop_count).toBe(0);
		expect(summary.cropped_formal_states).toEqual([]);
		expect(continuationAccepted(job, result, baseline)).toBe(true);
	});

	it("requires a fresh comparison when an edit changes the STEP", () => {
		const { job, result, baseline, summary } = continuationFixture();
		const revised = {
			...baseline,
			source_sha256: "revised-source",
			artifact: { ...baseline.artifact, sha256: "revised-step" },
		};
		expect(continuationAccepted(job, result, revised)).toBe(false);
		summary.inspected_formal_states.push({ source_sha256: "revised-source", step_sha256: "revised-step" });
		summary.inspection_count += 1;
		expect(continuationAccepted(job, result, revised)).toBe(true);
	});

	it("reuses STEP-bound comparison evidence after a source-only change is formally executed", () => {
		const { job, result, baseline } = continuationFixture();
		const sourceOnlyEdit = { ...baseline, source_sha256: "comment-only-source" };
		expect(continuationAccepted(job, result, { ...sourceOnlyEdit, formal_execution_current: false })).toBe(false);
		expect(continuationAccepted(job, result, sourceOnlyEdit)).toBe(true);
	});

	it.each(["missing", "failed", "stale", "crop-only"] as const)(
		"rejects %s comparison evidence even for valid formal CAD",
		(evidence) => {
			const { job, result, baseline, summary } = continuationFixture();
			if (evidence === "missing") delete result.main_view_reflection;
			else if (evidence === "stale") {
				summary.inspected_formal_states = [{ source_sha256: "older-source", step_sha256: "older-step" }];
			} else {
				// Attempt counters include failed calls; only successful inspection records establish evidence.
				summary.inspection_count = evidence === "failed" ? 4 : 0;
				summary.inspected_formal_states = [];
				if (evidence === "crop-only") {
					summary.crop_count = 1;
					summary.cropped_formal_states = [{ source_sha256: baseline.source_sha256!, step_sha256: baseline.artifact.sha256! }];
				}
			}
			expect(continuationAccepted(job, result, baseline)).toBe(false);
		},
	);

	it.each(["solution.py changed since formal execution", "output.step replaced since formal execution"])(
		"rejects inspected CAD when %s",
		(reason) => {
			const { job, result, baseline } = continuationFixture();
			expect(continuationAccepted(job, result, {
				...baseline,
				formal_execution_current: false,
				reason,
			})).toBe(false);
		},
	);

	it.each(["invalid-status", "invalid-shape", "not-watertight", "missing-validation"] as const)(
		"rejects a final STEP with %s despite current comparison evidence",
		(failure) => {
			const { job, result, baseline } = continuationFixture();
			if (failure === "invalid-status") baseline.artifact.status = "invalid";
			else if (failure === "invalid-shape") baseline.artifact.validation!.is_valid = false;
			else if (failure === "not-watertight") baseline.artifact.validation!.is_watertight = false;
			else baseline.artifact.validation = null;
			expect(continuationAccepted(job, result, baseline)).toBe(false);
		},
	);

	it.each(["missing", "different"] as const)("requires the continuation to have been applied (%s id)", (id) => {
		const { job, result, baseline } = continuationFixture();
		if (id === "missing") delete result.applied_continuation;
		else result.applied_continuation!.id = "another-continuation";
		expect(continuationAccepted(job, result, baseline)).toBe(false);
	});

	it("requests final STEP inspection without demanding a code change or local crop", () => {
		const { job } = continuationFixture();
		const prompt = unchangedContinuationPrompt(job);
		expect(prompt).toContain("inspect_comparison");
		expect(prompt).toContain("STEP");
		expect(prompt).not.toContain("dual_roi_crop");
		expect(prompt).not.toContain("has not changed both");
		expect(prompt).not.toContain("until the revised source and STEP are both changed");
	});

	it("continues the controller until the final STEP has successful inspection evidence", async () => {
		const { job, result, baseline, summary } = continuationFixture();
		summary.inspected_formal_states = [];
		let continuationCalls = 0;
		const promptContinue = vi.fn(async () => {
			if (++continuationCalls > 1) throw new Error("The inspected unchanged STEP should have been accepted");
			summary.inspected_formal_states = [{ source_sha256: baseline.source_sha256!, step_sha256: baseline.artifact.sha256! }];
		});
		const accepted = await driveUntilValid({
			initialPromptPending: false,
			promptInitial: vi.fn(async () => undefined),
			promptContinue,
			inspect: async () => baseline,
			onInspection: async () => undefined,
			isInterrupted: () => false,
			isAccepted: (candidate) => continuationAccepted(job, result, candidate),
		});
		expect(promptContinue).toHaveBeenCalledExactlyOnceWith(baseline);
		expect(accepted).toBe(baseline);
	});

	it.each(["main_self_reflect", "reflection_report"] as const)(
		"preserves the source-and-STEP change requirement for %s",
		(kind) => {
			const { job, result, baseline } = continuationFixture(kind);
			expect(continuationAccepted(job, result, baseline)).toBe(false);
			expect(continuationAccepted(job, result, { ...baseline, source_sha256: "revised-source" })).toBe(false);
			expect(continuationAccepted(job, result, {
				...baseline,
				artifact: { ...baseline.artifact, sha256: "revised-step" },
			})).toBe(false);
			expect(continuationAccepted(job, result, {
				...baseline,
				source_sha256: "revised-source",
				artifact: { ...baseline.artifact, sha256: "revised-step" },
			})).toBe(true);
			expect(unchangedContinuationPrompt(job)).toContain("has not changed both solution.py and output.step");
		},
	);

	it("preserves the global and local comparison requirements for Main View Reflect", () => {
		const { job, result, baseline, summary } = continuationFixture("main_view_reflect");
		expect(continuationAccepted(job, result, baseline)).toBe(false);
		summary.crop_count = 1;
		summary.cropped_formal_states = [{ source_sha256: baseline.source_sha256!, step_sha256: baseline.artifact.sha256! }];
		expect(continuationAccepted(job, result, baseline)).toBe(true);
		summary.inspected_formal_states = [];
		expect(continuationAccepted(job, result, baseline)).toBe(false);
		expect(unchangedContinuationPrompt(job)).toContain("dual_roi_crop");
	});
});

describe("model preflight", () => {
	it("rejects a text-only model before a session starts", () => {
		expect(() => assertImageCapableModel({ provider: "test", id: "text", input: ["text"] })).toThrow(
			"does not declare image input support",
		);
		expect(() => assertImageCapableModel({ provider: "test", id: "vision", input: ["text", "image"] })).not.toThrow();
	});

	it("uses KIMI_API_KEY as a moonshotai-cn fallback without overriding an explicit key", () => {
		const fallbackEnvironment: NodeJS.ProcessEnv = { KIMI_API_KEY: "kimi-platform-key" };
		applyProviderCredentialCompatibility("moonshotai-cn", fallbackEnvironment);
		expect(fallbackEnvironment.MOONSHOT_API_KEY).toBe("kimi-platform-key");

		const explicitEnvironment: NodeJS.ProcessEnv = {
			KIMI_API_KEY: "kimi-platform-key",
			MOONSHOT_API_KEY: "explicit-moonshot-key",
		};
		applyProviderCredentialCompatibility("moonshotai-cn", explicitEnvironment);
		expect(explicitEnvironment.MOONSHOT_API_KEY).toBe("explicit-moonshot-key");

		const unrelatedEnvironment: NodeJS.ProcessEnv = { KIMI_API_KEY: "kimi-coding-key" };
		applyProviderCredentialCompatibility("kimi-coding", unrelatedEnvironment);
		expect(unrelatedEnvironment.MOONSHOT_API_KEY).toBeUndefined();
	});
});


describe("dimension refresh feedback after formal execution", () => {
	function setup() {
		const { job, result, baseline } = continuationFixture();
		job.continuation!.dimension_annotation_profile = DIMENSION_ANNOTATION_PROFILE;
		job.continuation!.locked_view_manifest_sha256 = "manifest-sha";
		result.dimension_annotations = {
			profile: DIMENSION_ANNOTATION_PROFILE, state_path: "dimensions/state.json", revision_id: "previous",
			source_sha256: baseline.source_sha256!, step_sha256: baseline.artifact.sha256!, locked_manifest_sha256: "manifest-sha",
			status: "completed", regions: [{ region_id: "front", status: "annotated" }],
		};
		return { job, result, baseline, controller: new RunController(), persistResult: vi.fn(async () => undefined) };
	}
	it("returns no extra feedback after a successful refresh", async () => {
		const state = setup();
		expect(await dimensionRefreshFeedback({ ...state, refresh: async () => undefined })).toBeUndefined();
		expect(state.result.dimension_annotations!.status).toBe("completed");
		expect(state.persistResult).not.toHaveBeenCalled();
	});
	it("returns a retry instruction and invalidates completion when dimension refresh fails", async () => {
		const state = setup();
		expect(continuationAccepted(state.job, state.result, state.baseline)).toBe(true);
		const message = await dimensionRefreshFeedback({ ...state, refresh: async () => { throw new InfrastructureError("Dimension SubAgent timed out"); } });
		expect(message).toContain("Formal CAD execution succeeded");
		expect(message).toContain("Dimension SubAgent timed out");
		expect(message).toContain("Retry inspect_comparison");
		expect(state.result.dimension_annotations!.status).toBe("failed");
		expect(state.persistResult).toHaveBeenCalledOnce();
		expect(continuationAccepted(state.job, state.result, state.baseline)).toBe(false);
	});
	it("describes an initialization failure without claiming that CAD was just executed", async () => {
		const state = setup();
		const message = await dimensionRefreshFeedback({ ...state, phase: "before_reflection", refresh: async () => { throw new Error("measurements unavailable"); } });
		expect(message).toContain("Initial dimension preparation failed before Main reflection");
		expect(message).not.toContain("Formal CAD execution succeeded");
		expect(message).toContain("Retry inspect_comparison");
		expect(continuationAccepted(state.job, state.result, state.baseline)).toBe(false);
	});
	it("keeps missing dimension state incomplete when preparation fails before the first revision", async () => {
		const state = setup();
		delete state.result.dimension_annotations;
		await dimensionRefreshFeedback({ ...state, refresh: async () => { throw new Error("Current CAD inspection unavailable"); } });
		expect(state.result.dimension_annotations).toBeUndefined();
		expect(continuationAccepted(state.job, state.result, state.baseline)).toBe(false);
	});
	it.each(["interrupt", "abort_error"] as const)("propagates %s instead of converting cancellation into a warning", async (mode) => {
		const state = setup();
		const error = new Error("stop");
		if (mode === "interrupt") await state.controller.interrupt();
		else error.name = "AbortError";
		await expect(dimensionRefreshFeedback({ ...state, refresh: async () => { throw error; } })).rejects.toBe(error);
		expect(state.persistResult).not.toHaveBeenCalled();
	});
	it("does not hide an inability to persist the failed status", async () => {
		const state = setup();
		state.persistResult.mockRejectedValue(new Error("disk unavailable"));
		await expect(dimensionRefreshFeedback({ ...state, refresh: async () => { throw new Error("dimension failure"); } })).rejects.toThrow("disk unavailable");
	});
});
