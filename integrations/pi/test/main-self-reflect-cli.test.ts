import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";

import { describe, expect, it, vi } from "vitest";

import { DockerAgentRuntime } from "../src/docker-worker.js";
import { main, selfReflectCompletionAccepted } from "../src/main-self-reflect-continuation-cli.js";
import { mainSelfReflectInspectPrompt, mainSelfReflectPrompt } from "../src/main-self-reflect.js";
import {
	CAUSAL_REPAIR_PROFILE,
	CODE_CONSTRUCTION_PROFILE,
	DIMENSION_ANNOTATION_PROFILE,
	DUAL_ROI_CROP_PROFILE,
	GEOMETRY_REPAIR_PROFILE,
	JOB_FILE,
	REPAIR_PREVIEW_PROFILE,
	SHAPE_FEEDBACK_PROFILE,
	TARGET_ANNOTATION_PROFILE,
	VISUAL_DIFF_PROFILE,
	VISUAL_FINDING_PROFILE,
	VISUAL_REVIEW_PROFILE,
	type AgentJob,
	type AgentResult,
} from "../src/types.js";

import { makePng } from "./png-fixture.js";

describe("Self Reflect CLI extended wall-clock budget", () => {
	it.each(["360", "1440"])("accepts %s minutes before the output-directory guard without launching Docker", async value => {
		const daemon = vi.spyOn(DockerAgentRuntime.prototype, "daemonVersion").mockRejectedValue(new Error("Unexpected Docker"));
		try {
			await expect(main(["/tmp/unused-source", "--output-dir", new URL("../src", import.meta.url).pathname,
				"--image", "unused-test-image", "--timeout-minutes", value])).rejects.toThrow("--output-dir must be empty");
			expect(daemon).not.toHaveBeenCalled();
		} finally { daemon.mockRestore(); }
	});
	it.each(["0", "1441", "1.5", "NaN"])("rejects invalid wall-clock budget %s", async value => {
		await expect(main(["/tmp/unused-source", "--output-dir", "/tmp/unused-output", "--image", "unused-test-image",
			"--timeout-minutes", value])).rejects.toThrow("integer from 1 to 1440");
	});
});

function completion(kind: "main_self_reflect" | "main_self_reflect_inspect" = "main_self_reflect_inspect") {
	const source = "a".repeat(64);
	const step = "b".repeat(64);
	const prompt = kind === "main_self_reflect_inspect" ? mainSelfReflectInspectPrompt("146") : mainSelfReflectPrompt("146");
	const job: AgentJob = {
		protocol_version: 5, operation: "run_fixture", run_id: "test", agent_mode: "base-only",
		fixture: { id: "146", description: "test", inputFiles: ["input.png"], inputType: "png", taskType: "generation", sourceDir: "." },
		staged_paths: ["input.png"], provider: "ucloud", model: "gpt-5.6-sol", thinking_level: "high",
		pi_version: "0.84.1", container_image: "test", container_image_digest: "sha256:test",
		continuation: {
			kind, id: "continuation-146", prompt, image_paths: [],
			baseline_source_sha256: source, baseline_step_sha256: step,
		},
	};
	const result: AgentResult = {
		fixture_id: "146", provider: job.provider, model: job.model, thinking_level: "high",
		pi_version: job.pi_version, container_image: "test", container_image_digest: "sha256:test",
		started_at: "2026-09-06T00:00:00Z", updated_at: "2026-09-06T00:00:00Z", duration_s: 0, last_turn: 1,
		tokens: { input: 0, output: 0, cache_read: 0, cache_write: 0, total: 0 }, cost: 0, tool_calls: 1,
		status: "valid", agent_mode: "base-only",
		applied_continuation: {
			id: job.continuation!.id,
			prompt_sha256: createHash("sha256").update(prompt).digest("hex"),
			applied_at: "2026-09-06T00:00:00Z",
		},
		...(kind === "main_self_reflect_inspect" ? {
			optional_reflect_turn: {
				continuation_id: job.continuation!.id,
				prompt_sha256: createHash("sha256").update(prompt).digest("hex"),
				status: "completed" as const,
				started_at: "2026-09-06T00:00:00Z",
				completed_at: "2026-09-06T00:01:00Z",
			},
		} : {}),
		last_executed_source_sha256: source,
		last_executed_step_sha256: step,
		artifact: {
			status: "valid", sha256: step, path: "output.step", validation_error: null, render_path: "iso.png", render_error: null,
			validation: { is_valid: true, is_watertight: true, topology_errors: [], solid_count: 1, shell_count: 1, face_count: 6, volume: 1, bbox: {} },
		},
		main_view_reflection: {
			version: 1, kind: "main_view_reflect", inspection_count: 1, crop_count: 0,
			locked_view_manifest_sha256: "c".repeat(64), baseline_source_sha256: source, baseline_step_sha256: step,
			audit_manifest_path: "drawing_feedback/audit.json",
			inspected_formal_states: [{ source_sha256: source, step_sha256: step }],
			cropped_formal_states: [],
		},
	};
	const after = {
		source_sha256: source, step_sha256: step,
		formal_executions_added: 0, is_valid: true, is_watertight: true,
	};
	return { job, result, after };
}

describe("Self Reflect CLI completion gate", () => {
	it("accepts unchanged no-new-images Initial only after a completed reflection with current formal bindings", () => {
		const { job, result, after } = completion();
		job.continuation!.kind = "main_self_reflect";
		job.continuation!.no_new_images = true;
		expect(selfReflectCompletionAccepted(job, result, after)).toBe(true);
		result.optional_reflect_turn!.status = "running";
		expect(selfReflectCompletionAccepted(job, result, after)).toBe(false);
		result.optional_reflect_turn!.status = "completed";
		result.last_executed_step_sha256 = "d".repeat(64);
		expect(selfReflectCompletionAccepted(job, result, after)).toBe(false);
	});

	it("rejects no-new-images combined with alignment before launching Docker", async () => {
		await expect(main(["/tmp/unused-source", "--output-dir", "/tmp/unused-output", "--image", "unused",
			"--no-new-images", "--inspect-view-search-root", "/tmp/views"])).rejects.toThrow("cannot be combined");
	});

	it.each(["valid", "missing", "running", "failed", "old_source", "old_step", "wrong_manifest", "wrong_profile"] as const)(
		"checks final dimension annotation state: %s", (state) => {
			const { job, result, after } = completion();
			job.continuation!.dimension_annotation_profile = DIMENSION_ANNOTATION_PROFILE;
			job.continuation!.locked_view_manifest_sha256 = "c".repeat(64);
			result.dimension_annotations = {
				profile: DIMENSION_ANNOTATION_PROFILE, state_path: "dimensions/state.json", revision_id: "revision-1",
				status: "completed", source_sha256: after.source_sha256, step_sha256: after.step_sha256,
				locked_manifest_sha256: job.continuation!.locked_view_manifest_sha256,
				regions: [{ region_id: "front", status: "annotated" }],
			};
			if (state === "missing") delete result.dimension_annotations;
			else if (state === "running" || state === "failed") result.dimension_annotations.status = state;
			else if (state === "old_source") result.dimension_annotations.source_sha256 = "d".repeat(64);
			else if (state === "old_step") result.dimension_annotations.step_sha256 = "d".repeat(64);
			else if (state === "wrong_manifest") result.dimension_annotations.locked_manifest_sha256 = "d".repeat(64);
			else if (state === "wrong_profile") Object.assign(result.dimension_annotations, { profile: "unknown" });
			expect(selfReflectCompletionAccepted(job, result, after)).toBe(state === "valid");
		},
	);

	it.each([undefined, REPAIR_PREVIEW_PROFILE, SHAPE_FEEDBACK_PROFILE, GEOMETRY_REPAIR_PROFILE])("accepts an unchanged inspected baseline without a new execution or preview for profile %s", (profile) => {
		const { job, result, after } = completion();
		if (profile) job.continuation!.visual_review_profile = profile;
		expect(selfReflectCompletionAccepted(job, result, after)).toBe(true);
	});

	it("accepts changed source with the same inspected STEP when that source has been formally executed", () => {
		const { job, result, after } = completion();
		after.source_sha256 = "c".repeat(64);
		result.last_executed_source_sha256 = after.source_sha256;
		expect(selfReflectCompletionAccepted(job, result, after)).toBe(true);
	});

	it.each(["missing", "running", "wrong_id", "wrong_prompt", "error", "aborted"] as const)(
		"rejects a %s reflection turn even when formal CAD and final inspection are accepted",
		(kind) => {
			const { job, result, after } = completion();
			if (kind === "missing") delete result.optional_reflect_turn;
			else if (kind === "wrong_id") result.optional_reflect_turn!.continuation_id = "previous-continuation";
			else if (kind === "wrong_prompt") result.optional_reflect_turn!.prompt_sha256 = "c".repeat(64);
			else if (kind === "running") {
				result.optional_reflect_turn!.status = "running";
				delete result.optional_reflect_turn!.completed_at;
			} else {
				// Persisted results are parsed from JSON, so reject unsupported failure statuses too.
				Object.assign(result.optional_reflect_turn!, { status: kind });
			}
			expect(selfReflectCompletionAccepted(job, result, after)).toBe(false);
		},
	);

	it.each(["missing", "old_step", "crop_only"] as const)("rejects %s evidence even when an inspection count is present", (kind) => {
		const { job, result, after } = completion();
		if (kind === "missing") delete result.main_view_reflection;
		else if (kind === "old_step") result.main_view_reflection!.inspected_formal_states[0]!.step_sha256 = "c".repeat(64);
		else {
			result.main_view_reflection!.cropped_formal_states = result.main_view_reflection!.inspected_formal_states;
			result.main_view_reflection!.inspected_formal_states = [];
		}
		expect(selfReflectCompletionAccepted(job, result, after)).toBe(false);
	});

	it.each(["source", "step", "artifact"] as const)("rejects a stale formal %s despite matching visual evidence", (kind) => {
		const { job, result, after } = completion();
		if (kind === "source") result.last_executed_source_sha256 = "c".repeat(64);
		else if (kind === "step") result.last_executed_step_sha256 = "c".repeat(64);
		else result.artifact!.sha256 = "c".repeat(64);
		expect(selfReflectCompletionAccepted(job, result, after)).toBe(false);
	});

	it.each(["is_valid", "is_watertight"] as const)("rejects failed %s validation after final inspection", (field) => {
		const { job, result, after } = completion();
		after[field] = false;
		expect(selfReflectCompletionAccepted(job, result, after)).toBe(false);
	});

	it("requires the matching applied continuation", () => {
		const { job, result, after } = completion();
		result.applied_continuation!.id = "previous-continuation";
		expect(selfReflectCompletionAccepted(job, result, after)).toBe(false);
	});

	it("rejects an invalid artifact even when the reported run and copied validation flags say valid", () => {
		const { job, result, after } = completion();
		result.artifact!.status = "invalid";
		expect(selfReflectCompletionAccepted(job, result, after)).toBe(false);
	});

	it("preserves the original non-Inspect gate: fresh execution and both changed hashes", () => {
		const { job, result, after } = completion("main_self_reflect");
		delete result.main_view_reflection;
		expect(selfReflectCompletionAccepted(job, result, after)).toBe(false);
		after.source_sha256 = "c".repeat(64);
		after.step_sha256 = "d".repeat(64);
		expect(selfReflectCompletionAccepted(job, result, after)).toBe(false);
		after.formal_executions_added = 1;
		expect(selfReflectCompletionAccepted(job, result, after)).toBe(true);
		after.step_sha256 = job.continuation!.baseline_step_sha256;
		expect(selfReflectCompletionAccepted(job, result, after)).toBe(false);
	});

	it("documents final inspection and unchanged-baseline acceptance in CLI help", async () => {
		const output = vi.spyOn(process.stdout, "write").mockImplementation(() => true);
		try {
			expect(await main(["--help"])).toBe(0);
			const help = output.mock.calls.map(([chunk]) => String(chunk)).join("");
			expect(help).toContain("Agent must inspect the final formal");
			expect(help).toContain("The model may remain unchanged from the baseline.");
			expect(help).toContain("original changed-source/STEP completion gate is preserved.");
			expect(help).toContain("--thinking-level <off|minimal|low|medium|high>");
			expect(help).toContain("--thinking-level defaults to high");
			expect(help).toContain(SHAPE_FEEDBACK_PROFILE);
			expect(help).toContain("only preview_shape_feedback and adopt_shape_candidate");
			expect(help).toContain("exactly one candidate per call in one to three aligned views");
			expect(help).toContain("Keep the candidates array");
			expect(help).toContain("exactly one entry containing only id and one to eight edits");
			expect(help).toContain("The edits are applied to a temporary copy of the Current source");
			expect(help).toContain("one three-panel card per view");
			expect(help).not.toContain("one to three candidates");
			expect(help).not.toContain("applied independently to the same Current source");
			expect(help).toContain("Target | Current | Preview");
			expect(help).toContain("Current and Preview are separate monochrome CAD 2D projections");
			expect(help).toContain("The panels share one display scale");
			expect(help).not.toContain("Target + Current");
			expect(help).toContain("same camera and 2D transform");
			expect(help).not.toContain("BRep highlights");
			expect(help).not.toContain("contour pairing");
			expect(help).toContain("To adopt a candidate from the most recent preview, call adopt_shape_candidate");
			expect(help).toContain("with its candidate_id. It applies the verified edits to solution.py only");
			expect(help).toContain("rejects stale previews after the formal source changes");
			expect(help).toContain("A preview does not\nreplace formal execution or the final inspect_comparison requirement.");
			expect(help).not.toContain("target-line support");
			expect(help).not.toContain("one isolated code change");
		} finally {
			output.mockRestore();
		}
	});
});

describe("Self Reflect CLI visual review profiles", () => {
	const argumentsFor = (profile: string, extra: string[] = []) => [
		"/tmp/cadgenbench-unused-source",
		// An existing nonempty directory stops valid parsing before Docker or output writes.
		"--output-dir", new URL("../src", import.meta.url).pathname,
		"--image", "unused-test-image",
		"--model-profile", "gpt-5.6-sol",
		"--visual-review-profile", profile,
		...extra,
	];
	const lockedViews = ["--inspect-view-search-root", "/tmp/cadgenbench-unused-views"];

	it.each([VISUAL_REVIEW_PROFILE, REPAIR_PREVIEW_PROFILE, SHAPE_FEEDBACK_PROFILE, GEOMETRY_REPAIR_PROFILE])(
		"accepts %s through argument validation without starting Docker",
		async (profile) => {
			const daemon = vi.spyOn(DockerAgentRuntime.prototype, "daemonVersion").mockRejectedValue(new Error("Unexpected Docker call"));
			try {
				await expect(main(argumentsFor(profile, lockedViews))).rejects.toThrow("--output-dir must be empty");
				expect(daemon).not.toHaveBeenCalled();
			} finally {
				daemon.mockRestore();
			}
		},
	);

	it("requires locked views for aligned shape feedback", async () => {
		await expect(main(argumentsFor(SHAPE_FEEDBACK_PROFILE))).rejects.toThrow(
			"--visual-review-profile requires --inspect-view-search-root",
		);
	});

	it("rejects unknown visual review profiles and names the supported shape profile", async () => {
		await expect(main(argumentsFor("unsupported-preview", lockedViews))).rejects.toThrow(
			`--visual-review-profile must be ${VISUAL_REVIEW_PROFILE}, ${REPAIR_PREVIEW_PROFILE}, ${SHAPE_FEEDBACK_PROFILE}, or ${GEOMETRY_REPAIR_PROFILE}`,
		);
	});

	it.each([
		{ flag: "--causal-repair-profile", value: CAUSAL_REPAIR_PROFILE, error: `${SHAPE_FEEDBACK_PROFILE} cannot be combined` },
		{ flag: "--dual-roi-crop-profile", value: DUAL_ROI_CROP_PROFILE, error: `${SHAPE_FEEDBACK_PROFILE} cannot be combined` },
		{ flag: "--target-annotation-profile", value: TARGET_ANNOTATION_PROFILE, error: `${SHAPE_FEEDBACK_PROFILE} cannot be combined` },
		{ flag: "--code-construction-profile", value: CODE_CONSTRUCTION_PROFILE, error: "--visual-review-profile replaces --code-construction-profile" },
		{ flag: "--visual-finding-profile", value: VISUAL_FINDING_PROFILE, error: "--visual-review-profile replaces --visual-finding-profile" },
	])("rejects mixing aligned shape feedback with $flag", async ({ flag, value, error }) => {
		await expect(main(argumentsFor(SHAPE_FEEDBACK_PROFILE, [...lockedViews, flag, value]))).rejects.toThrow(error);
	});

	it("rejects mixing aligned shape feedback with visual diff and ablation", async () => {
		await expect(main(argumentsFor(SHAPE_FEEDBACK_PROFILE, [
			...lockedViews,
			"--visual-diff-profile", VISUAL_DIFF_PROFILE,
			"--code-construction-profile", CODE_CONSTRUCTION_PROFILE,
		]))).rejects.toThrow("--visual-review-profile replaces --visual-finding-profile and --visual-diff-profile");
	});

	it("rejects two preview profiles in one invocation", async () => {
		await expect(main(argumentsFor(SHAPE_FEEDBACK_PROFILE, [
			...lockedViews, "--visual-review-profile", REPAIR_PREVIEW_PROFILE,
		]))).rejects.toThrow("Option --visual-review-profile was provided more than once");
	});

	it("requires locked views for atomic repair", async () => {
		await expect(main(argumentsFor(GEOMETRY_REPAIR_PROFILE))).rejects.toThrow("--visual-review-profile requires --inspect-view-search-root");
	});

	it.each([
		["--causal-repair-profile", CAUSAL_REPAIR_PROFILE],
		["--dual-roi-crop-profile", DUAL_ROI_CROP_PROFILE],
		["--target-annotation-profile", TARGET_ANNOTATION_PROFILE],
	])("rejects mixing atomic repair with %s", async (flag, profile) => {
		await expect(main(argumentsFor(GEOMETRY_REPAIR_PROFILE, [...lockedViews, flag, profile]))).rejects.toThrow(`${GEOMETRY_REPAIR_PROFILE} cannot be combined`);
	});

	it("documents restricted child preview and explicit adoption identifiers", async () => {
		const write = vi.spyOn(process.stdout, "write").mockReturnValue(true);
		try {
			await main(["--help"]);
			const help = write.mock.calls.map(([value]) => String(value)).join("");
			expect(help).toContain(GEOMETRY_REPAIR_PROFILE);
			expect(help).toContain("preview_shape_feedback is not exposed to the Main Agent");
			expect(help).toContain("exactly one candidate per Preview call");
			expect(help).toContain("ends the repair stage with\nsubmit_geometry_repair");
			expect(help).toContain("There is no preview-call count limit");
			expect(help).toContain("timeouts and cancellation still apply");
			expect(help).not.toContain("four preview calls per repair task");
			expect(help).toContain("evaluation_id and candidate_id");
		} finally { write.mockRestore(); }
	});
});


describe("Self Reflect CLI dimension annotation profile", () => {
	const args = (extra: string[] = []) => [
		"/tmp/cadgenbench-unused-source", "--output-dir", new URL("../src", import.meta.url).pathname,
		"--image", "unused-test-image", "--model-profile", "gpt-5.6-sol",
		"--dimension-annotation-profile", DIMENSION_ANNOTATION_PROFILE, ...extra,
	];
	const views = ["--inspect-view-search-root", "/tmp/cadgenbench-unused-views"];
	it("requires locked aligned views", async () => {
		await expect(main(args())).rejects.toThrow("--dimension-annotation-profile requires --inspect-view-search-root");
	});
	it.each([{ profile: [] }, { profile: ["--visual-review-profile", SHAPE_FEEDBACK_PROFILE] }])("accepts dimensions with compatible profiles $profile without launching Docker", async ({ profile }) => {
		const daemon = vi.spyOn(DockerAgentRuntime.prototype, "daemonVersion").mockRejectedValue(new Error("Unexpected Docker call"));
		try {
			await expect(main(args([...views, ...profile]))).rejects.toThrow("--output-dir must be empty");
			expect(daemon).not.toHaveBeenCalled();
		} finally { daemon.mockRestore(); }
	});
	it("rejects an unknown dimension profile", async () => {
		await expect(main(args(views).map((value) => value === DIMENSION_ANNOTATION_PROFILE ? "unknown" : value))).rejects.toThrow(`--dimension-annotation-profile must be ${DIMENSION_ANNOTATION_PROFILE}`);
	});
	it.each([
		["--visual-review-profile", GEOMETRY_REPAIR_PROFILE],
		["--visual-review-profile", REPAIR_PREVIEW_PROFILE],
		["--visual-review-profile", VISUAL_REVIEW_PROFILE],
		["--causal-repair-profile", CAUSAL_REPAIR_PROFILE],
		["--dual-roi-crop-profile", DUAL_ROI_CROP_PROFILE],
		["--target-annotation-profile", TARGET_ANNOTATION_PROFILE],
		["--visual-finding-profile", VISUAL_FINDING_PROFILE],
		["--code-construction-profile", CODE_CONSTRUCTION_PROFILE],
		["--visual-diff-profile", VISUAL_DIFF_PROFILE],
	])("rejects an incompatible %s=%s", async (flag, value) => {
		await expect(main(args([...views, flag, value]))).rejects.toThrow(`${DIMENSION_ANNOTATION_PROFILE} may only be combined with ${SHAPE_FEEDBACK_PROFILE}`);
	});
	it("documents automatic refresh and revision-local entity IDs", async () => {
		const write = vi.spyOn(process.stdout, "write").mockReturnValue(true);
		try {
			await main(["--help"]);
			const help = write.mock.calls.map(([value]) => String(value)).join("");
			expect(help).toContain("--dimension-annotation-profile aligned-dimension-subagent-v1");
			expect(help).toContain("after every successful formal execution");
			expect(help).toContain("one STEP revision and cannot be reused");
		} finally { write.mockRestore(); }
	});
});


it.each([
	{ fixed: true, sourceThinking: "high", selectedThinking: undefined, mismatch: false },
	{ fixed: false, sourceThinking: "high", selectedThinking: undefined, mismatch: false },
	...(["off", "minimal", "low", "medium", "high"] as const).map((level) => ({
		fixed: true, sourceThinking: level, selectedThinking: level, mismatch: false,
	})),
	{ fixed: true, sourceThinking: "high", selectedThinking: "low", mismatch: true },
	{ fixed: true, sourceThinking: "low", selectedThinking: undefined, mismatch: true },
])("prepares dimensions with a fixed view and matching source thinking: %j", async ({ fixed, sourceThinking, selectedThinking, mismatch }) => {
	const root = await mkdtemp(join(tmpdir(), "self-reflect-dimensions-"));
	const source = join(root, "source", "146");
	const viewRoot = join(root, "views");
	const view = join(viewRoot, "146");
	const output = join(root, "output");
	const daemon = vi.spyOn(DockerAgentRuntime.prototype, "daemonVersion").mockResolvedValue("test-daemon");
	const digest = vi.spyOn(DockerAgentRuntime.prototype, "imageDigest").mockResolvedValue("sha256:test");
	const prepare = vi.spyOn(DockerAgentRuntime.prototype, "prepareFixture").mockResolvedValue();
	const run = vi.spyOn(DockerAgentRuntime.prototype, "runFixture").mockRejectedValue(new Error("Test stops before any model invocation"));
	const stdout = vi.spyOn(process.stdout, "write").mockReturnValue(true);
	const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
	try {
		const put = async (base: string, path: string, content: string | Buffer) => {
			await mkdir(dirname(join(base, path)), { recursive: true });
			await writeFile(join(base, path), content);
		};
		const sourceText = "from build123d import Box\n";
		const stepText = "ISO-10303-21;\n";
		const hash = (text: string) => createHash("sha256").update(text).digest("hex");
		const { job, result } = completion();
		delete job.continuation;
		delete result.applied_continuation;
		delete result.optional_reflect_turn;
		delete result.main_view_reflection;
		job.thinking_level = sourceThinking;
		result.thinking_level = sourceThinking;
		result.session_file = "sessions/session.jsonl";
		result.last_executed_source_sha256 = hash(sourceText);
		result.last_executed_step_sha256 = hash(stepText);
		const oldJob = JSON.stringify(job);
		const oldResult = JSON.stringify(result);
		await Promise.all([
			put(source, JOB_FILE, oldJob), put(source, "agent_result.json", oldResult),
			put(source, "solution.py", sourceText), put(source, "output.step", stepText),
			put(source, "sessions/session.jsonl", "{}\n"), put(source, "input.png", makePng(200, 160)),
			put(source, "turns/0001/iso.png", makePng(200, 160)),
			put(view, "solution.py", sourceText), put(view, "output.step", stepText),
			put(view, "drawing_feedback/locked/anchor.step", stepText),
		]);
		const selected = {
			candidate_id: "front-1", render_path: "drawing_feedback/locked/render.png",
			aligned_render_path: "drawing_feedback/locked/aligned.png", comparison_path: "drawing_feedback/locked/comparison.png",
			view_spec: { name: "front", direction: [0, 0, 1], up: [0, 1, 0] },
			score: 0.5, raw_score: 0.5, eligible: true, failed_gates: [], scoring_mode: "capacity_transport",
			capacity_transport: { version: 1 },
			transform: { scale_px_per_unit: 1, rotation_deg: 0, center_x_px: 100, center_y_px: 80, model_center_x: 0, model_center_y: 0, reflected: false },
			alignment: { objective: 0.5, mode: "capacity_transport", scale_at_bound: false, rotation_at_limit: false, translation_at_limit: false },
		};
		const manifestPath = "drawing_feedback/locked/view_manifest.json";
		const manifest = JSON.stringify({
			version: 3, scorer_profile: "capacity_transport_v1", input_sha256: "a".repeat(64),
			source_sha256: hash(sourceText), step_sha256: hash(stepText),
			...(fixed ? { projection_frame: { policy: "fixed", anchor_step_path: "drawing_feedback/locked/anchor.step", anchor_step_sha256: hash(stepText) } } : {}),
			regions: [{ region_id: "front", page: 0, roi: [0, 0, 500, 500], crop_path: "drawing_feedback/locked/crop.png", view_type: "orthographic", status: "matched", selected, alternatives: [{ ...selected, candidate_id: "rear-2", view_spec: { name: "rear", direction: [0, 0, -1], up: [0, 1, 0] } }], margin: 0.1, confidence: "high", note: "front" }],
		});
		await put(view, manifestPath, manifest);
		await Promise.all(["drawing_feedback/locked/crop.png", selected.render_path, selected.aligned_render_path, selected.comparison_path].map(async (path) => await put(view, path, makePng(200, 160))));
		await put(view, "view_search_only_result.json", JSON.stringify({
			status: "completed", fixture_id: "146", source_workspace: source, reflection_agent_started: false,
			agent_roles: ["view_search"], manifest_path: manifestPath, source_sha256: hash(sourceText), step_sha256: hash(stepText),
		}));
		const command = [
			source, "--output-dir", output, "--image", "test", "--model-profile", "gpt-5.6-sol",
			"--inspect-view-search-root", viewRoot, "--visual-review-profile", SHAPE_FEEDBACK_PROFILE,
			"--dimension-annotation-profile", DIMENSION_ANNOTATION_PROFILE,
			...(selectedThinking ? ["--thinking-level", selectedThinking] : []),
		];
		if (mismatch) {
			await expect(main(command)).rejects.toThrow(`gpt-5.6-sol ${selectedThinking ?? "high"} run`);
			expect(run).not.toHaveBeenCalled();
			expect(prepare).not.toHaveBeenCalled();
			expect(await readFile(join(source, JOB_FILE), "utf8")).toBe(oldJob);
			expect(await readFile(join(source, "agent_result.json"), "utf8")).toBe(oldResult);
			return;
		}
		if (!fixed) {
			await expect(main(command)).rejects.toThrow("Dimension annotations require a calibrated fixed View Search manifest");
			expect(run).not.toHaveBeenCalled();
			expect(prepare).not.toHaveBeenCalled();
			expect(await readFile(join(view, manifestPath), "utf8")).toBe(manifest);
			return;
		}
		expect(await main(command)).toBe(2);
		const savedJob = JSON.parse(await readFile(join(output, "run/146", JOB_FILE), "utf8")) as AgentJob;
		const contract = JSON.parse(await readFile(join(output, "experiment_contract.json"), "utf8"));
		const copiedResult = JSON.parse(await readFile(join(output, "run/146/agent_result.json"), "utf8"));
		const baselineManifest = JSON.parse(await readFile(join(output, "run/146/baseline/manifest.json"), "utf8"));
		const summary = JSON.parse(await readFile(join(output, "continuation_summary.json"), "utf8"));
		expect(savedJob.thinking_level).toBe(sourceThinking);
		expect(copiedResult.thinking_level).toBe(sourceThinking);
		expect(contract.thinking_level).toBe(sourceThinking);
		expect(summary.thinking_level).toBe(sourceThinking);
		expect(baselineManifest.baseline_thinking_level).toBe(sourceThinking);
		expect(baselineManifest.continuation_thinking_level).toBe(sourceThinking);
		expect(await readFile(join(output, "run/146/baseline/agent_result.json"), "utf8")).toBe(oldResult);
		expect(savedJob.continuation?.kind).toBe("main_self_reflect_inspect");
		expect(savedJob.continuation?.visual_review_profile).toBe(SHAPE_FEEDBACK_PROFILE);
		expect(savedJob.continuation?.dimension_annotation_profile).toBe(DIMENSION_ANNOTATION_PROFILE);
		expect(savedJob.continuation?.prompt).toContain("dimension-annotation SubAgent");
		expect(savedJob.run_id).toContain("shape-feedback-dimensions");
		expect(contract.dimension_annotation_profile).toBe(DIMENSION_ANNOTATION_PROFILE);
		expect(contract.allowed_subagent_roles).toEqual(["dimension_annotation"]);
		expect(contract.acceptance.allowed_auxiliary_agent_roles).toEqual(["dimension_annotation"]);
		expect(contract.acceptance.final_dimension_annotation_required).toBe(true);
		expect(contract.allowed_tools).toContain("preview_shape_feedback");
		expect(contract.allowed_tools).not.toContain("repair_geometry");
		expect(run).toHaveBeenCalledOnce();
		expect(await readFile(join(source, JOB_FILE), "utf8")).toBe(oldJob);
		expect(await readFile(join(source, "agent_result.json"), "utf8")).toBe(oldResult);
		expect(await readFile(join(source, "sessions/session.jsonl"), "utf8")).toBe("{}\n");
		expect(await readFile(join(source, "solution.py"), "utf8")).toBe(sourceText);
		expect(await readFile(join(view, manifestPath), "utf8")).toBe(manifest);
	} finally {
		for (const mock of [daemon, digest, prepare, run, stdout, stderr]) mock.mockRestore();
		await rm(root, { recursive: true, force: true });
	}
});

it.each(["xhigh", "LOW", "invalid", ""])("rejects unsupported thinking level %j before Docker or model calls", async (level) => {
	const daemon = vi.spyOn(DockerAgentRuntime.prototype, "daemonVersion").mockResolvedValue("test-daemon");
	try {
		await expect(main(["/unused/source", "--output-dir", "/unused/output", "--image", "test", "--thinking-level", level]))
			.rejects.toThrow("--thinking-level must be one of off, minimal, low, medium, high");
		expect(daemon).not.toHaveBeenCalled();
	} finally { daemon.mockRestore(); }
});
