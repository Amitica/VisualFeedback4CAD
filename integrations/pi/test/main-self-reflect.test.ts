import { createHash } from "node:crypto";

import { describe, expect, it, vi } from "vitest";

import { applyStructuredEdits } from "../src/causal-repair.js";
import { GEOMETRY_REPAIR_COMPLETION_INSTRUCTION } from "../src/geometry-repair-instructions.js";
import {
	applyCodeAblation,
	CODE_CONSTRUCTION_CONTEXT_TYPE,
	injectPersistentCodeConstructionRegistry,
	type CodeConstructionState,
} from "../src/code-construction.js";
import {
	assertSelfReflectAuxiliaryState,
	auxiliaryStateFields,
	CaseTimeoutError,
	classifySelfReflectSource,
	MAIN_SELF_REFLECT_INSPECT_KIND,
	mainSelfReflectInspectPrompt,
	mainSelfReflectNoNewImagesPrompt,
	mainSelfReflectPrompt,
	withIsolatedTimeout,
} from "../src/main-self-reflect.js";
import { unchangedContinuationPrompt } from "../src/runner.js";
import {
	CAUSAL_REPAIR_AGENT_TOOLS,
	CAUSAL_REPAIR_PROFILE,
	CODE_CONSTRUCTION_AGENT_TOOLS,
	CODE_CONSTRUCTION_PROFILE,
	DIMENSION_ANNOTATION_PROFILE,
	DUAL_ROI_CROP_PROFILE,
	GEOMETRY_REPAIR_PROFILE,
	GEOMETRY_REPAIR_TOOLS,
	mainSelfReflectInspectAgentTools,
	MAIN_SELF_REFLECT_INSPECT_AGENT_TOOLS,
	REPAIR_PREVIEW_AGENT_TOOLS,
	REPAIR_PREVIEW_PROFILE,
	SHAPE_FEEDBACK_PROFILE,
	SHAPE_FEEDBACK_TOOLS,
	TARGET_ANNOTATION_AGENT_TOOLS,
	TARGET_ANNOTATION_PROFILE,
	VISUAL_FINDING_AGENT_TOOLS,
	VISUAL_FINDING_PROFILE,
	VISUAL_DIFF_AGENT_TOOLS,
	VISUAL_DIFF_PROFILE,
	VISUAL_REVIEW_AGENT_TOOLS,
	VISUAL_REVIEW_PROFILE,
	visualReviewAgentTools,
	type AgentJob,
	type AgentResult,
	type SubagentRunUsage,
} from "../src/types.js";

function bareResult(overrides: Partial<AgentResult> = {}): AgentResult {
	return {
		fixture_id: "102",
		status: "valid",
		agent_mode: "base-only",
		provider: "moonshotai-cn",
		model: "kimi-k3",
		thinking_level: "high",
		pi_version: "0.84.1",
		container_image: "test",
		container_image_digest: "sha256:test",
		started_at: "2026-08-15T00:00:00.000Z",
		updated_at: "2026-08-15T00:00:00.000Z",
		duration_s: 0,
		last_turn: 1,
		artifact: null,
		tokens: { input: 0, output: 0, cache_read: 0, cache_write: 0, total: 0 },
		cost: 0,
		tool_calls: 0,
		...overrides,
	};
}

function repairUsageResult(): AgentResult {
	const result = bareResult();
	const run: SubagentRunUsage = {
		role: "geometry_repair", accounting_id: "repair-1", tokens: result.tokens, cost: 0, tool_calls: 1,
		started_at: result.started_at, finished_at: result.updated_at, duration_s: 1, status: "success",
	};
	return { ...result, subagent_runs: [run], recorded_subagent_usage_ids: [run.accounting_id], subagent_usage: { tokens: run.tokens, cost: run.cost, tool_calls: run.tool_calls } };
}

describe("self-reflect auxiliary state isolation", () => {
	it("allows dimension measurements only under the explicitly enabled profile", () => {
		const result = repairUsageResult();
		result.subagent_runs![0]!.role = "dimension_annotation";
		expect(() => assertSelfReflectAuxiliaryState(result, undefined, DIMENSION_ANNOTATION_PROFILE)).not.toThrow();
		expect(() => assertSelfReflectAuxiliaryState(result, SHAPE_FEEDBACK_PROFILE, DIMENSION_ANNOTATION_PROFILE)).not.toThrow();
		expect(() => assertSelfReflectAuxiliaryState(result, SHAPE_FEEDBACK_PROFILE)).toThrow("state is forbidden");
		expect(() => assertSelfReflectAuxiliaryState(result, GEOMETRY_REPAIR_PROFILE)).toThrow("Only geometry_repair");
	});
	it.each(["reflection", "view_search", "segmentation", "geometry_repair"] as const)("rejects %s records in the dimension-only profile", (role) => {
		const result = repairUsageResult();
		result.subagent_runs![0]!.role = role;
		expect(() => assertSelfReflectAuxiliaryState(result, SHAPE_FEEDBACK_PROFILE, DIMENSION_ANNOTATION_PROFILE)).toThrow("Only dimension_annotation");
	});
	it("rejects drawing reflection state and incomplete dimension usage accounting", () => {
		const result = repairUsageResult();
		result.subagent_runs![0]!.role = "dimension_annotation";
		result.drawing_feedback = {} as NonNullable<AgentResult["drawing_feedback"]>;
		expect(() => assertSelfReflectAuxiliaryState(result, undefined, DIMENSION_ANNOTATION_PROFILE)).toThrow("Drawing reflection state is forbidden");
		delete result.drawing_feedback;
		delete result.subagent_usage;
		expect(() => assertSelfReflectAuxiliaryState(result, undefined, DIMENSION_ANNOTATION_PROFILE)).toThrow("incomplete or duplicated");
	});
	it("prompts Main to inspect refreshed dimensions while retaining responsibility for repair", () => {
		const prompt = mainSelfReflectInspectPrompt("146", undefined, undefined, undefined, undefined, undefined, SHAPE_FEEDBACK_PROFILE, undefined, DIMENSION_ANNOTATION_PROFILE);
		expect(prompt).toContain("before you begin reflection and after every successful formal execution");
		expect(prompt).toContain("First call inspect_comparison({ region_id })");
		expect(prompt).not.toContain("first call execute_build123d({})");
		expect(mainSelfReflectInspectPrompt("146")).toContain("First call inspect_comparison({ region_id })");
		expect(prompt).toContain("you remain responsible for identifying discrepancies, editing solution.py");
		expect(prompt).toContain("never reuse an old face or edge ID");
		expect(prompt).toContain("preview_shape_feedback");
		expect(mainSelfReflectInspectPrompt("146")).not.toContain("dimension-annotation SubAgent");
	});

	it("permits repair records only in the new profile and retains strict baseline guards", () => {
		expect(() => assertSelfReflectAuxiliaryState(bareResult(), GEOMETRY_REPAIR_PROFILE)).not.toThrow();
		expect(() => assertSelfReflectAuxiliaryState(repairUsageResult(), GEOMETRY_REPAIR_PROFILE)).not.toThrow();
		for (const profile of [undefined, SHAPE_FEEDBACK_PROFILE, REPAIR_PREVIEW_PROFILE]) {
			expect(() => assertSelfReflectAuxiliaryState(repairUsageResult(), profile)).toThrow("state is forbidden");
		}
	});
	it.each(["reflection", "view_search", "segmentation"] as const)("rejects a foreign %s subagent in the repair profile", (role) => {
		const result = repairUsageResult();
		result.subagent_runs![0]!.role = role;
		expect(() => assertSelfReflectAuxiliaryState(result, GEOMETRY_REPAIR_PROFILE)).toThrow("Only geometry_repair");
	});
	it.each(["orphan_usage", "missing_usage", "duplicate_run", "duplicate_id", "unknown_id"])("rejects %s in restored usage", (failure) => {
		const result = repairUsageResult();
		if (failure === "orphan_usage") delete result.subagent_runs;
		if (failure === "missing_usage") delete result.subagent_usage;
		if (failure === "duplicate_run") result.subagent_runs!.push(result.subagent_runs![0]!);
		if (failure === "duplicate_id") result.recorded_subagent_usage_ids!.push("repair-1");
		if (failure === "unknown_id") result.recorded_subagent_usage_ids = ["unknown"];
		expect(() => assertSelfReflectAuxiliaryState(result, GEOMETRY_REPAIR_PROFILE)).toThrow();
	});
});

function selfReflectJob(): AgentJob {
	return {
		protocol_version: 5,
		operation: "run_fixture",
		run_id: "test",
		agent_mode: "base-only",
		fixture: {
			id: "102",
			description: "test",
			inputFiles: ["input.png"],
			inputType: "png",
			taskType: "generation",
			sourceDir: "/tmp/102",
		},
		staged_paths: ["inputs/00-input.png"],
		provider: "moonshotai-cn",
		model: "kimi-k3",
		thinking_level: "high",
		pi_version: "0.84.1",
		container_image: "test",
		container_image_digest: "sha256:test",
		continuation: {
			kind: "main_self_reflect",
			id: "main-self-reflect-102",
			prompt: mainSelfReflectPrompt("102"),
			image_paths: [],
			baseline_source_sha256: "a".repeat(64),
			baseline_step_sha256: "b".repeat(64),
		},
	};
}

describe("Main-Agent-only self-reflection contract", () => {
	it("delegates one suspected difference to an independent repair agent and keeps Preview out of Main", () => {
		const prompt = mainSelfReflectInspectPrompt("103", undefined, undefined, undefined, undefined, undefined, GEOMETRY_REPAIR_PROFILE);
		expect(prompt).toContain("independently verify it against the drawing");
		expect(prompt).toContain("identify one part or group of related features in `part`");
		expect(prompt).toContain("provide the corresponding `view_ids`");
		expect(prompt).toContain("submit only one atomic problem per call");
		expect(prompt).toContain("In `suspected_difference`, describe what the drawing shows");
		for (const type of ["count", "size", "position", "orientation", "profile", "connectivity"]) {
			expect(prompt).toContain(`**\`${type}\`:`);
		}
		expect(prompt).toContain("if both diameter and depth are wrong, submit them separately");
		expect(prompt).toContain("allow associated changes necessary to achieve it");
		expect(prompt).toContain("compare Current and Preview against the target drawing");
		expect(prompt).toContain("whether other correct geometry has been damaged");
		expect(prompt).toContain("call `repair_geometry` again, or discard the candidate");
		expect(prompt).not.toMatch(/[\u3400-\u9fff]/u);
		expect(prompt).not.toContain("four-panel");
		expect(prompt).not.toContain("Target + Current");
		expect(prompt.endsWith(GEOMETRY_REPAIR_COMPLETION_INSTRUCTION)).toBe(true);
		expect(prompt).not.toContain("The final model may remain unchanged from the baseline");
		expect(prompt).not.toContain("Otherwise, finish with the current model");
		expect(prompt).not.toContain("preview_shape_feedback");
		expect(prompt).not.toContain("evaluation_id");
		expect(prompt).not.toContain("directly improve the live solution.py");
		const baseline = mainSelfReflectInspectPrompt("103");
		const baselineParts = baseline.split("\n\n");
		const directEditParagraph = baselineParts.find(paragraph => paragraph.includes("directly improve the live solution.py"))!;
		const [beforeDelegation, afterDelegation] = baseline.split(directEditParagraph);
		expect(prompt.startsWith(beforeDelegation!)).toBe(true);
		const restoredCompletion = prompt.replace(GEOMETRY_REPAIR_COMPLETION_INSTRUCTION, afterDelegation!.trimStart());
		expect(restoredCompletion.endsWith(afterDelegation!)).toBe(true);
		expect(restoredCompletion.slice(0, beforeDelegation!.length) + directEditParagraph + restoredCompletion.slice(-afterDelegation!.length)).toBe(baseline);
		expect(visualReviewAgentTools(GEOMETRY_REPAIR_PROFILE)).toEqual(GEOMETRY_REPAIR_TOOLS);
		expect(GEOMETRY_REPAIR_TOOLS).toEqual(["repair_geometry", "adopt_shape_candidate"]);
		expect(visualReviewAgentTools(SHAPE_FEEDBACK_PROFILE)).toEqual(SHAPE_FEEDBACK_TOOLS);
		expect(visualReviewAgentTools()).toEqual([]);
	});
	it("retains atomic completion requirements when unchanged formal CAD needs a follow-up", () => {
		const atomic = selfReflectJob();
		atomic.continuation = {
			...atomic.continuation!,
			kind: MAIN_SELF_REFLECT_INSPECT_KIND,
			visual_review_profile: GEOMETRY_REPAIR_PROFILE,
			prompt: mainSelfReflectInspectPrompt("102", undefined, undefined, undefined, undefined, undefined, GEOMETRY_REPAIR_PROFILE),
			locked_view_manifest_path: "locked/view_manifest.json",
			locked_view_manifest_sha256: "c".repeat(64),
		};
		const prompt = unchangedContinuationPrompt(atomic);
		expect(prompt).toContain(GEOMETRY_REPAIR_COMPLETION_INSTRUCTION);
		expect(prompt).toMatch(/inspect[^.]*aligned 2D comparisons/iu);
		expect(prompt).toContain("execute_build123d");
		expect(prompt).toContain("valid and watertight");
		expect(prompt).toMatch(/3D render/iu);
		expect(prompt).toMatch(/until[^.]*eliminated/iu);
		expect(prompt).not.toContain("Otherwise, finish with the current model");
		expect(prompt).not.toContain("The final model may remain unchanged from the baseline");
	});
	it("supplies only a generic self-review instruction and requires formal changed output", () => {
		const prompt = mainSelfReflectPrompt("102");
		expect(prompt).toContain("exact existing Pi session");
		expect(prompt).toContain("first call execute_build123d({})");
		expect(prompt).toContain("directly improve the live solution.py");
		expect(prompt).toContain("both solution.py and output.step differ");
		expect(prompt).toContain("valid and watertight");
		for (const forbidden of [
			"Reflection report",
			"self_reflect_2d_drawing",
			"View Search",
			"segmentation",
			"comparison",
			"manifest",
			"finding",
		]) {
			expect(prompt).not.toContain(forbidden);
		}
	});

	it("declares no continuation images so only the job's original drawing pages are attached", () => {
		expect(selfReflectJob().continuation?.image_paths).toEqual([]);
	});

	it("keeps the historical prompt intact and requires final aligned 2D review only for Inspect reflection", () => {
		const original = selfReflectJob();
		const inspect = selfReflectJob();
		inspect.continuation = {
			...inspect.continuation!,
			kind: MAIN_SELF_REFLECT_INSPECT_KIND,
			prompt: mainSelfReflectInspectPrompt("102"),
			locked_view_manifest_path: "locked/view_manifest.json",
			locked_view_manifest_sha256: "c".repeat(64),
		};
		expect(original.continuation?.prompt).toContain("No additional images or external analysis are supplied");
		expect(original.continuation?.prompt).not.toContain("inspect_comparison");
		expect(inspect.continuation.prompt).not.toContain("No additional images or external analysis are supplied");
		expect(inspect.continuation.prompt).toContain("Use inspect_comparison({ region_id })");
		expect(inspect.continuation.prompt).toContain("Before finishing, inspect the aligned 2D comparisons of the final formal CAD in the views relevant to your edits.");
		expect(inspect.continuation.prompt).toContain("If you made no edits, inspect the views relevant to the suspected discrepancy.");
		expect(inspect.continuation.prompt).toContain("Continue if these comparisons reveal a clear, actionable mismatch.");
		expect(inspect.continuation.prompt).toContain("The final model may remain unchanged from the baseline.");
		expect(inspect.continuation.prompt).not.toContain("both solution.py and output.step differ");
		expect(inspect.continuation.prompt).not.toContain("preview_shape_feedback");
		expect(inspect.continuation.prompt).not.toContain("calibrate_drawing_scale");
		expect(inspect.continuation.prompt).toContain("tool description lists the available region_id values");
		expect(inspect.continuation.prompt).not.toContain("inspect that same region_id again after executing the changed source");
		expect(inspect.continuation.prompt).not.toContain("inspect the key region_id values again before finishing");
		expect(inspect.continuation.prompt).not.toContain("intended errors have been resolved");
		expect(inspect.continuation.prompt).not.toContain("did not introduce any new errors");
		expect(inspect.continuation.image_paths).toEqual([]);
		expect(MAIN_SELF_REFLECT_INSPECT_AGENT_TOOLS).toEqual([
			"inspect_comparison",
		]);
		expect(unchangedContinuationPrompt(inspect)).not.toBe(unchangedContinuationPrompt(original));
		expect(unchangedContinuationPrompt(inspect)).not.toContain("changed, valid, and watertight");
	});

	it("adds causal Patch verification only for the explicit profile", () => {
		const causalIncrement = "When you are ready to modify the formal `solution.py` from visual evidence but need to verify which repair is more appropriate, you may first use `compare_repair_hypotheses`. Its two or three candidates should represent materially different repair hypotheses rather than different parameters for the same change. The candidates are only hypotheses and may all be wrong. Judge whether each candidate makes the Current CAD closer to the original engineering drawing and whether all candidates preserve the same error. If none genuinely improves the model, reject them all and form a new hypothesis. You may use the tool again for different problems or hypotheses. After selecting a repair, modify the formal `solution.py` and call `execute_build123d({})`.";
		const unchanged = mainSelfReflectInspectPrompt("102");
		const causal = mainSelfReflectInspectPrompt("102", CAUSAL_REPAIR_PROFILE);
		expect(unchanged).not.toContain("compare_repair_hypotheses");
		expect(causal).toContain(causalIncrement);
		expect(causal.replace(`\n\n${causalIncrement}`, "")).toBe(unchanged);
		expect(causal).not.toContain("inspect that same region_id again after executing the changed source");
		expect(causal).not.toContain("inspect the key region_id values again before finishing");
		expect(causal).toContain("materially different repair hypotheses");
		expect(causal).toContain("closer to the original engineering drawing");
		expect(causal).toContain("all candidates preserve the same error");
		expect(causal).toContain("reject them all and form a new hypothesis");
		expect(causal).toContain("use the tool again for different problems or hypotheses");
		expect(causal).not.toMatch(/[\u3400-\u9fff]/u);
		expect(causal).not.toContain("use inspect_comparison to verify the formal result");
		expect(CAUSAL_REPAIR_AGENT_TOOLS).toEqual(["compare_repair_hypotheses"]);
	});

	it("adds optional Dual ROI Crop only for the explicit profile", () => {
		const dualRoiIncrement = "Use `dual_roi_crop` to enlarge corresponding local structures on the two sides and inspect the Current CAD's shape, size, position, orientation, connectivity, and feature correctness. You may inspect new corresponding ROIs for different fine-grained differences. After local inspection, continue examining the complete view before deciding whether and how to modify the code.";
		const unchanged = mainSelfReflectInspectPrompt("103");
		const dualRoi = mainSelfReflectInspectPrompt("103", undefined, DUAL_ROI_CROP_PROFILE);
		expect(unchanged).not.toContain("dual_roi_crop");
		expect(dualRoi).toContain(dualRoiIncrement);
		expect(dualRoi.replace(`\n\n${dualRoiIncrement}`, "")).toBe(unchanged);
		expect(dualRoi).not.toContain("compare_repair_hypotheses");
		expect(dualRoi).not.toContain("inspect that same region_id again after executing the changed source");
		expect(dualRoi).not.toContain("inspect the key region_id values again before finishing");
		expect(dualRoi).toContain("Use `dual_roi_crop`");
		expect(dualRoi).toContain("new corresponding ROIs");
		expect(dualRoi).toContain("continue examining the complete view");
		expect(dualRoi).not.toMatch(/[\u3400-\u9fff]/u);
		expect(mainSelfReflectInspectAgentTools()).toEqual([
			"inspect_comparison",
		]);
		expect(mainSelfReflectInspectAgentTools(DUAL_ROI_CROP_PROFILE)).toEqual([
			"inspect_comparison",
			"dual_roi_crop",
		]);
	});

	it("adds persistent Target grounding only for the explicit profile", () => {
		const unchanged = mainSelfReflectInspectPrompt("103");
		const annotated = mainSelfReflectInspectPrompt(
			"103",
			undefined,
			undefined,
			undefined,
			undefined,
			undefined,
			undefined,
			TARGET_ANNOTATION_PROFILE,
		);
		expect(unchanged).not.toContain("annotate_target");
		expect(annotated).toContain("To ground visual differences and code edits in precisely identified Target structures");
		expect(annotated).toContain("use `annotate_target` to add persistent numbered marks to geometric structures in the Target engineering drawing");
		expect(annotated).toContain("Each number provides a visual reference");
		expect(annotated).toContain("its name and description convey its semantics");
		expect(annotated).toContain("Inspect the returned image");
		expect(annotated).toContain("refer to each marked structure as `[number] name`");
		expect(annotated).toContain("in subsequent reasoning");
		expect(annotated).toContain("Handle only one atomic Target-drawing discrepancy per iteration");
		expect(annotated).toContain("you must inspect the updated comparison and verify that the result is closer to the Target engineering drawing");
		expect(annotated).toContain("Before finishing, inspect the aligned 2D comparisons of the final formal CAD");
		expect(annotated).toContain("The final model may remain unchanged from the baseline.");
		expect(annotated).not.toContain("both `solution.py` and `output.step` differ from the preserved baseline");
		expect(unchanged).not.toContain("all important Target-drawing discrepancies you can identify");
		expect(unchanged).toContain("Iterate with execute_build123d({}) and inspect every formal render");
		expect(annotated).not.toContain("you may use `annotate_target`");
		expect(annotated).not.toContain("you must give that structure");
		expect(annotated).not.toContain("update_visual_findings");
		expect(annotated).not.toContain("construction_ref");
		expect(TARGET_ANNOTATION_AGENT_TOOLS).toEqual(["annotate_target"]);
		expect(mainSelfReflectInspectAgentTools(
			undefined,
			undefined,
			TARGET_ANNOTATION_PROFILE,
		)).toEqual(["inspect_comparison", "annotate_target"]);
	});

	it("adds mandatory English code-ablation grounding only for the explicit profile", () => {
		const unchanged = mainSelfReflectInspectPrompt("103");
		const highlighted = mainSelfReflectInspectPrompt(
			"103",
			undefined,
			undefined,
			undefined,
			CODE_CONSTRUCTION_PROFILE,
		);
		expect(unchanged).not.toContain("inspect_code_ablation");
		const increment = highlighted.split("\n\n")
			.find((paragraph) => paragraph.startsWith("After finding any important unresolved visual difference"));
		expect(increment).toBeDefined();
		expect(highlighted.replace(`\n\n${increment}`, "")).toBe(unchanged);
		expect(highlighted).toContain("Continue CADGenBench fixture 103 in this exact existing Pi session.");
		expect(highlighted).not.toContain("dual_roi_crop");
		expect(highlighted).not.toContain("compare_repair_hypotheses");
		expect(highlighted).not.toContain("update_visual_findings");
		expect(highlighted).toContain("you must use `inspect_code_ablation`");
		expect(highlighted).toContain("count, shape, extent, orientation, relative position, connectivity");
		expect(highlighted).toContain("spatial relationship to adjacent structures");
		expect(highlighted).toContain("must inspect another, more appropriate code block");
		expect(highlighted).toContain("not a segmentation of the target structure");
		expect(highlighted).toContain("inspect the 2D comparison again");
		expect(highlighted).not.toContain("calibrate_drawing_scale");
		expect(highlighted).not.toContain("Use the tool selectively");
		expect(highlighted).not.toMatch(/[\u3400-\u9fff]/u);
		expect(CODE_CONSTRUCTION_AGENT_TOOLS).toEqual(["inspect_code_ablation"]);
		expect(mainSelfReflectInspectAgentTools(undefined, CODE_CONSTRUCTION_PROFILE)).toEqual([
			"inspect_comparison",
			"inspect_code_ablation",
		]);
		const withFindings = mainSelfReflectInspectPrompt(
			"103",
			undefined,
			undefined,
			VISUAL_FINDING_PROFILE,
			CODE_CONSTRUCTION_PROFILE,
		);
		expect(withFindings).toContain("before recording that finding or editing the related code");
		expect(withFindings).toContain("Put it in the finding's `current_construction_refs` only after confirming the target correspondence");
		expect(withFindings).toContain("A recorded reference is not cleared when its code changes");
		expect(withFindings).toContain("Reinspect that construction by passing its `construction_ref`");
		expect(withFindings).toContain("You may add or remove references when updating a finding");
		expect(withFindings).toContain("may initially be empty and has no fixed finding count");
		expect(withFindings).toContain("never recreate or clear it at the start of a new iteration");
		expect(withFindings).toContain("`target_geometry`");
		expect(withFindings).toContain("`current_geometry`");
		expect(withFindings).toContain("`difference`");
		expect(withFindings).toContain("Update the same finding when its evidence changes");
		expect(withFindings).toContain("delete it only when its entire difference is resolved or unsupported");
		expect(withFindings).toContain("inspect the 2D comparison again and reconcile the existing findings before making another edit");
		expect(withFindings).toContain("call `inspect_code_ablation` with its existing `construction_ref`");
		expect(withFindings).toContain("trace it to a visible endpoint in the target drawing");
		expect(withFindings).toContain("is not evidence of a break");
		expect(withFindings).toContain("When the drawing supports a clear, actionable discrepancy, directly improve the live solution.py; do not stop at an explanation");
		expect(withFindings).toContain("The final model may remain unchanged from the baseline.");
		expect(withFindings).not.toContain("both solution.py and output.step differ from the preserved baseline");
		expect(withFindings).not.toContain("remains active only while its exact registered whole-line code block");
		expect(withFindings).not.toContain("registration evidence from subsequent model context");
		expect(withFindings).not.toContain("Fictitious formatting example unrelated to the current fixture");
		expect(withFindings).not.toContain("Not yet confirmed");
		expect(withFindings).not.toContain("right-side tall bracket");
		expect(withFindings).not.toContain("rear mounting supports");
		expect(withFindings).not.toContain("source revision 2d1175855a8e");
		expect(withFindings).not.toContain("This isolated inspection does not modify");
		expect(withFindings).not.toMatch(/[\u3400-\u9fff]/u);
	});

	it("adds one-step visual diffs without natural-language finding fields", () => {
		const prompt = mainSelfReflectInspectPrompt(
			"103",
			undefined,
			undefined,
			undefined,
			CODE_CONSTRUCTION_PROFILE,
			VISUAL_DIFF_PROFILE,
		);
		expect(prompt).toContain("Use `visual_diff` instead of natural-language visual findings");
		expect(prompt).toContain("pass only its construction_ref and the corresponding Target ROI");
		expect(prompt).toContain("do not replace it with a prose geometry description");
		expect(prompt).toContain("No relation, difference tags, proposal, confirmation, or resolution step");
		expect(prompt).not.toContain("`target_geometry`");
		expect(prompt).not.toContain("`current_geometry`");
		expect(prompt).not.toContain("`difference`");
		expect(VISUAL_DIFF_AGENT_TOOLS).toEqual(["visual_diff"]);
	});

	it("draws isolated executable geometry hypotheses without claiming target truth", () => {
		const prompt = mainSelfReflectInspectPrompt(
			"103",
			undefined,
			undefined,
			undefined,
			undefined,
			undefined,
			VISUAL_REVIEW_PROFILE,
		);
		expect(prompt).toContain("call `visualize_code_hypothesis`");
		expect(prompt).toContain("prepared a set of exact edits for `solution.py`");
		expect(prompt).toContain("local structure in the Target drawing as the ROI");
		expect(prompt).toContain("matching `oldText/newText`");
		expect(prompt).toContain("Compare the two returned panels directly");
		expect(prompt).toContain("Edit Preview on the right");
		expect(prompt).toContain("Target structure on the left");
		expect(prompt).toContain("otherwise revise the code and preview it again");
		expect(prompt).not.toContain("inspect_code_ablation");
		expect(prompt).not.toContain("construction_ref");
		expect(VISUAL_REVIEW_AGENT_TOOLS).toEqual(["visualize_code_hypothesis"]);
	});

	it("adds only the multi-candidate B-rep preview instruction for its profile", () => {
		const unchanged = mainSelfReflectInspectPrompt("103");
		const preview = mainSelfReflectInspectPrompt(
			"103",
			undefined,
			undefined,
			undefined,
			undefined,
			undefined,
			REPAIR_PREVIEW_PROFILE,
		);
		expect(preview).toContain("call `preview_repair_candidates`");
		expect(preview).toContain("two or three distinct code-edit hypotheses");
		expect(preview).toContain("reported b0/b1/b2 changes");
		expect(preview).toContain("The candidates may all be wrong");
		expect(preview).not.toContain("visualize_code_hypothesis");
		expect(preview.replace(/\n\nWhen you have identified[\s\S]*?`execute_build123d\(\{\}\)`\./u, "")).toBe(unchanged);
		expect(REPAIR_PREVIEW_AGENT_TOOLS).toEqual(["preview_repair_candidates"]);
	});

	it("adds only the approved aligned shape feedback paragraph for its explicit profile", () => {
		const unchanged = mainSelfReflectInspectPrompt("103");
		const preview = mainSelfReflectInspectPrompt(
			"103", undefined, undefined, undefined, undefined, undefined, SHAPE_FEEDBACK_PROFILE,
		);
		const increment = "When a discrepancy in the aligned 2D comparisons suggests possible code changes, you may use `preview_shape_feedback` before modifying the formal CAD. Submit `candidates` as an array containing exactly one candidate per call, with only an `id` and one to eight `edits`. Select one to three views that show the relevant geometry. The edits are applied to a temporary copy of the Current source, never a previous unadopted preview. Inspect one three-panel comparison per view, arranged horizontally as Target | Current | Preview. Target shows the engineering drawing; Current and Preview show separate 2D CAD projections using the same camera, 2D transform, and display scale. The image contains only the three column labels and actual views, preserving original drawing annotations. Compare both CAD projections against Target, checking whether the edit improves the intended geometry and preserves geometry that is already correct. If you judge that the candidate reduces the geometric discrepancy between the current CAD and the target engineering drawing, call `adopt_shape_candidate` with that candidate's `id` from the most recent preview as `candidate_id`, then call `execute_build123d({})` and inspect the final formal CAD with `inspect_comparison`. Otherwise, revise the edit and preview again, or keep Current. A preview does not replace the final formal CAD inspection.";
		expect(preview).toContain(increment);
		expect(preview).toContain("Submit `candidates` as an array containing exactly one candidate per call");
		expect(preview).not.toContain("hypothesis");
		expect(preview).toContain("you may use `preview_shape_feedback`");
		expect(preview).toContain("Otherwise, revise the edit and preview again, or keep Current");
		expect(preview).toContain("then call `execute_build123d({})` and inspect the final formal CAD with `inspect_comparison`");
		expect(preview).toContain("The image contains only the three column labels and actual views");
		expect(preview).toContain("one to eight `edits`");
		expect(preview).toContain("Select one to three views");
		expect(preview).toContain("Inspect one three-panel comparison per view, arranged horizontally as Target | Current | Preview");
		expect(preview).not.toContain("one to three entries");
		expect(preview).not.toContain("applied independently");
		expect(preview).toContain("Current and Preview show separate 2D CAD projections using the same camera, 2D transform, and display scale");
		expect(preview).toContain("Compare both CAD projections against Target");
		expect(preview).not.toContain("four-panel");
		expect(preview).not.toContain("overlays");
		expect(preview).not.toContain("BRep difference");
		expect(preview).not.toContain("contour pairing");
		expect(preview).not.toContain("yellow marks");
		expect(preview).toContain("call `adopt_shape_candidate` with that candidate's `id` from the most recent preview as `candidate_id`");
		expect(preview).not.toContain("target-line support");
		expect(preview).not.toContain("Target + Current");
		expect(preview).not.toContain("Target + Preview");
		expect(preview).not.toContain("needed for one change");
		expect(preview.replace(`\n\n${increment}`, "")).toBe(unchanged);
		expect(preview).not.toContain("preview_repair_candidates");
		expect(preview).not.toContain("visualize_code_hypothesis");
		expect(SHAPE_FEEDBACK_PROFILE).toBe("aligned-shape-feedback-v1");
		expect(SHAPE_FEEDBACK_TOOLS).toEqual(["preview_shape_feedback", "adopt_shape_candidate"]);
		for (const profile of [undefined, VISUAL_REVIEW_PROFILE, REPAIR_PREVIEW_PROFILE]) {
			const otherPrompt = mainSelfReflectInspectPrompt(
				"103", undefined, undefined, undefined, undefined, undefined, profile,
			);
			expect(otherPrompt).not.toContain("preview_shape_feedback");
			expect(otherPrompt).not.toContain("adopt_shape_candidate");
		}
	});

	it("injects one persistent code-construction registry with source-bound references", () => {
		const state: CodeConstructionState = {
			fixtureId: "103",
			profile: CODE_CONSTRUCTION_PROFILE,
			lockedViewManifestSha256: "a".repeat(64),
			auditPath: "drawing_feedback/code-ablation/audit.json",
			startedAt: "2026-08-30T00:00:00.000Z",
			constructions: [{
				construction_ref: "C001",
				created_at: "2026-08-30T00:00:00.000Z",
				formal_source_sha256: "b".repeat(64),
				formal_step_sha256: "c".repeat(64),
				code_block: "for spec in supports:\n    part += make_support(spec)",
				code_block_sha256: "d".repeat(64),
				line_start: 40,
				line_end: 41,
			}],
			activeConstructionRefs: ["C001"],
			inspections: [],
		};
		const injected = injectPersistentCodeConstructionRegistry([
			{ role: "user", content: "original" },
			{ role: "custom", customType: CODE_CONSTRUCTION_CONTEXT_TYPE, content: "stale" },
		], state) as Array<{ customType?: string; content?: string }>;
		const registry = injected.filter(({ customType }) => customType === CODE_CONSTRUCTION_CONTEXT_TYPE);
		expect(registry).toHaveLength(1);
		expect(registry[0]?.content).toContain("C001 | status=current | source=bbbbbbbbbbbb | lines=40-41");
		expect(registry[0]?.content).toContain("for spec in supports:");
		expect(registry[0]?.content).not.toContain("stale");
	});

	it("ablates exactly one complete source block with an indentation-preserving pass", () => {
		const source = [
			"model = base()",
			"for x in (-5, 5):",
			"    foot = make_foot(x)",
			"    model = model + foot",
			"export(model)",
		].join("\n");
		const block = [
			"for x in (-5, 5):",
			"    foot = make_foot(x)",
			"    model = model + foot",
		].join("\n");
		const ablated = applyCodeAblation(source, block);
		expect(ablated.source).toBe([
			"model = base()",
			"pass  # ablated by inspect_code_ablation",
			"export(model)",
		].join("\n"));
		expect(ablated.lineStart).toBe(2);
		expect(ablated.lineEnd).toBe(4);
		expect(() => applyCodeAblation(source, "model = base")).toThrow("complete source lines");
	});

	it("adds only one persistent natural-language visual memory tool for the explicit profile", () => {
		const baseInspect = "Use inspect_comparison({ region_id }) to examine the aligned 2D comparison between the input drawing and the CAD generated by your execution, as needed during reflection and for the final review described below. The tool description lists the available region_id values.";
		const persistentInspect = "Use inspect_comparison({ region_id }) to examine the aligned 2D comparison between the input drawing and the CAD generated by your execution. Inspect one region_id image at a time and examine it before requesting another Inspect or Crop image. The tool description lists the available region_id values.";
		const inspectOnlyPersistentInspect = "Use inspect_comparison({ region_id }) to examine the aligned 2D comparison between the input drawing and the CAD generated by your execution. Inspect one region_id image at a time and examine it before requesting another Inspect image. The tool description lists the available region_id values.";
		const baseIdentify = "Identify any missing, extra, incorrectly shaped, incorrectly dimensioned, or misaligned geometry yourself. When the drawing supports a clear, actionable discrepancy, directly improve the live solution.py; do not stop at an explanation, plan, or separate analysis report. Preserve geometry that is already correct.";
		const baseCompletion = "Before finishing, inspect the aligned 2D comparisons of the final formal CAD in the views relevant to your edits. If you made no edits, inspect the views relevant to the suspected discrepancy. Continue if these comparisons reveal a clear, actionable mismatch. Otherwise, finish with the current model, ensuring that the final STEP is valid and watertight. The final model may remain unchanged from the baseline.";
		const dualRoi = mainSelfReflectInspectPrompt("103", undefined, DUAL_ROI_CROP_PROFILE);
		const persistent = mainSelfReflectInspectPrompt(
			"103",
			undefined,
			DUAL_ROI_CROP_PROFILE,
			VISUAL_FINDING_PROFILE,
		);
		const inspectOnlyPersistent = mainSelfReflectInspectPrompt(
			"103",
			undefined,
			undefined,
			VISUAL_FINDING_PROFILE,
		);
		const increment = persistent.split("\n\n")
			.find((paragraph) => paragraph.startsWith("Use `update_visual_findings`"));
		const inspectOnlyIncrement = inspectOnlyPersistent.split("\n\n")
			.find((paragraph) => paragraph.startsWith("Use `update_visual_findings`"));
		expect(increment).toBeDefined();
		expect(inspectOnlyIncrement).toBeDefined();
		expect(persistent).toContain(increment!);
		expect(
			persistent
				.replace(`\n\n${increment!}`, "")
				.replace(persistentInspect, baseInspect),
		).toBe(dualRoi);
		expect(persistent).toContain("Inspect one region_id image at a time");
		expect(persistent.indexOf("`dual_roi_crop`")).toBeLessThan(
			persistent.indexOf("`update_visual_findings`"),
		);
		expect(persistent).toContain("may initially be empty and has no fixed finding count");
		expect(persistent).toContain("never recreate or clear it at the start of a new iteration");
		expect(persistent).toContain("Reconcile existing findings before adding new ones");
		expect(persistent).not.toContain("Fictitious formatting example unrelated to the current fixture");
		expect(persistent).not.toContain("Maintain five independent");
		expect(persistent).not.toContain("Keep this format:");
		expect(persistent).not.toContain("The latest findings are automatically retained");
		expect(persistent).not.toContain("(5) Overall assessment:");
		expect(persistent).not.toContain("old_text=");
		expect(persistent).not.toContain("complete findings paragraph");
		expect(persistent).toContain("`target_geometry`");
		expect(persistent).toContain("`current_geometry`");
		expect(persistent).toContain("`difference`");
		expect(persistent).toContain("delete it only when its entire difference is resolved or unsupported");
		expect(persistent).not.toContain("right-side tall bracket");
		expect(persistent).not.toContain("six mounting");
		expect(persistent).not.toContain("15°");
		expect(persistent).toContain(baseIdentify);
		expect(persistent).toContain(baseCompletion);
		expect(persistent).not.toContain("calibrate_drawing_scale");
		expect(persistent).not.toMatch(/[\u3400-\u9fff]/u);
		expect(dualRoi).toContain(baseCompletion);
		expect(persistent).not.toContain("replace the complete findings paragraph as new content");
		expect(persistent).not.toContain("pending_revalidation");
		expect(persistent).not.toContain("manage_visual_findings");
		expect(inspectOnlyPersistent).toContain(inspectOnlyIncrement!);
		expect(inspectOnlyPersistent).toContain(inspectOnlyPersistentInspect);
		expect(inspectOnlyPersistent).not.toContain("dual_roi_crop");
		expect(inspectOnlyPersistent).not.toContain("Crop");
		expect(mainSelfReflectInspectAgentTools()).toEqual([
			"inspect_comparison",
		]);
		expect(VISUAL_FINDING_AGENT_TOOLS).toEqual(["update_visual_findings"]);
		expect(unchangedContinuationPrompt(selfReflectJob())).toContain("changed, valid, and watertight");
	});

	it("applies exact non-overlapping repair edits and rejects ambiguous or overlapping patches", () => {
		const source = "radius = 10\nheight = 20\n";
		expect(applyStructuredEdits(source, [
			{ old_text: "radius = 10", new_text: "radius = 12" },
			{ old_text: "height = 20", new_text: "height = 18" },
		])).toBe("radius = 12\nheight = 18\n");
		expect(() => applyStructuredEdits("x = 1\nx = 1\n", [
			{ old_text: "x = 1", new_text: "x = 2" },
		])).toThrow("must occur exactly once");
		expect(() => applyStructuredEdits(source, [
			{ old_text: "radius = 10", new_text: "radius = 11" },
			{ old_text: "10\nheight", new_text: "12\nheight" },
		])).toThrow("overlap");
	});

	it("detects every persisted auxiliary-agent result field", () => {
		expect(auxiliaryStateFields(bareResult())).toEqual([]);
		expect(auxiliaryStateFields(bareResult({
			subagent_usage: { tokens: bareResult().tokens, cost: 0, tool_calls: 0 },
			subagent_runs: [],
			recorded_subagent_usage_ids: [],
			drawing_feedback: {} as NonNullable<AgentResult["drawing_feedback"]>,
		}))).toEqual([
			"drawing_feedback",
			"subagent_usage",
			"subagent_runs",
			"recorded_subagent_usage_ids",
		]);
	});

	it("uses a self-review follow-up instead of external findings when output is unchanged", () => {
		const prompt = unchangedContinuationPrompt(selfReflectJob());
		expect(prompt).toContain("Re-examine the original drawings");
		expect(prompt).toContain("changed, valid, and watertight");
		expect(prompt).not.toContain("supplied Reflection findings");
	});

	it("accepts only a verified self-reflect continuation as a chained source", () => {
		const job = selfReflectJob();
		const originalJob = { ...job };
		delete originalJob.continuation;
		const promptSha256 = createHash("sha256").update(job.continuation!.prompt).digest("hex");
		expect(classifySelfReflectSource(originalJob, bareResult())).toBe("original");
		expect(classifySelfReflectSource(job, bareResult({
			applied_continuation: {
				id: job.continuation!.id,
				prompt_sha256: promptSha256,
				applied_at: "2026-08-21T00:00:00.000Z",
			},
		}))).toBe("main_self_reflect");
		expect(classifySelfReflectSource(job, bareResult({
			applied_continuation: {
				id: "another-continuation",
				prompt_sha256: promptSha256,
				applied_at: "2026-08-21T00:00:00.000Z",
			},
		}))).toBe("incompatible");
	});

	it("interrupts only the timed-out operation", async () => {
		vi.useFakeTimers();
		const onTimeout = vi.fn(async () => undefined);
		const pending = new Promise<string>(() => undefined);
		const outcome = withIsolatedTimeout({ operation: pending, timeoutMs: 5_000, onTimeout });
		const rejection = expect(outcome).rejects.toBeInstanceOf(CaseTimeoutError);
		await vi.advanceTimersByTimeAsync(5_000);
		await rejection;
		expect(onTimeout).toHaveBeenCalledOnce();
		vi.useRealTimers();
	});

	it("starts independent cases concurrently and keeps a peer failure isolated", async () => {
		let active = 0;
		let peak = 0;
		const starts: number[] = [];
		const run = async (id: number): Promise<number> => {
			starts.push(Date.now());
			active += 1;
			peak = Math.max(peak, active);
			await new Promise((resolve) => setTimeout(resolve, 20));
			active -= 1;
			if (id === 2) throw new Error("isolated failure");
			return id;
		};
		const settled = await Promise.allSettled([1, 2, 3].map(async (id) => await run(id)));
		expect(peak).toBe(3);
		expect(Math.max(...starts) - Math.min(...starts)).toBeLessThan(15);
		expect(settled.map(({ status }) => status)).toEqual(["fulfilled", "rejected", "fulfilled"]);
	});
});


describe("direct reflection round wording", () => {
	it("preserves the original Initial prompt by default", () => {
		const prompt = mainSelfReflectNoNewImagesPrompt("146");
		expect(prompt).toBe(mainSelfReflectNoNewImagesPrompt("146", false));
		expect(prompt).toContain("this exact existing Initial Pi session");
		expect(prompt).toContain("preserved the Initial source, STEP, render and session");
		expect(prompt).toContain("keeping Initial unchanged is allowed");
		expect(prompt).not.toContain("previous completed reflection round");
	});
	it("uses the prior completed round and full same-session history without extra images or forced edits", () => {
		const prompt = mainSelfReflectNoNewImagesPrompt("146", true);
		expect(prompt).toContain("this exact existing Pi session, retaining the full history");
		expect(prompt).toContain("previous completed reflection round under /workspace/baseline/");
		expect(prompt).toContain("keeping the previous round's output unchanged is allowed");
		expect(prompt).not.toContain("keeping Initial unchanged");
		expect(prompt).not.toContain("preserved the Initial source");
		expect(prompt).toContain("No new images are attached or provided by tools");
		expect(prompt).toContain("only when they address a supported discrepancy");
		expect(prompt).toContain("formally bound, valid and watertight");
		expect(prompt).not.toContain("must change");
	});
});
