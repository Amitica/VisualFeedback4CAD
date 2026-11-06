import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { DockerAgentRuntime } from "../src/docker-worker.js";
import { chainedContinuationSuffix, main } from "../src/main-self-reflect-continuation-cli.js";
import { assertContinuation, lockedViewOriginHashes } from "../src/runner.js";
import { DIMENSION_ANNOTATION_PROFILE, JOB_FILE, type AgentJob, type AgentResult } from "../src/types.js";
import { makePng } from "./png-fixture.js";

const roots: string[] = [];
afterEach(async () => { vi.restoreAllMocks(); for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });
const hash = (value: string) => createHash("sha256").update(value).digest("hex");
const readJson = async <T>(path: string): Promise<T> => JSON.parse(await readFile(path, "utf8")) as T;
async function put(root: string, path: string, value: string | Buffer) { await mkdir(dirname(join(root, path)), { recursive: true }); await writeFile(join(root, path), value); }

async function fixture(direct = false) {
	const root = await mkdtemp(join(tmpdir(), "chained-fixed-views-")); roots.push(root);
	const source = join(root, "r1", "146");
	const sourceText = "# repaired round 1 source\n";
	const stepText = "ROUND-1-STEP\n";
	const originalSource = hash("Original Initial source");
	const originalStepText = "ORIGINAL-INITIAL-STEP\n";
	const originalStep = hash(originalStepText);
	const manifestPath = "drawing_feedback/original-fixed-frame/view_manifest.json";
	const anchorPath = "drawing_feedback/original-fixed-frame/anchor.step";
	const selected = {
		candidate_id: "front-1", render_path: "drawing_feedback/original-fixed-frame/render.png",
		aligned_render_path: "drawing_feedback/original-fixed-frame/aligned.png", comparison_path: "drawing_feedback/original-fixed-frame/comparison.png",
		view_spec: { name: "front", direction: [0, 0, 1], up: [0, 1, 0] },
		score: 0.5, raw_score: 0.5, eligible: true, failed_gates: [], scoring_mode: "capacity_transport",
		capacity_transport: { version: 1 },
		transform: { scale_px_per_unit: 4, rotation_deg: 3, center_x_px: 101, center_y_px: 80, model_center_x: 7, model_center_y: -9, reflected: false },
		alignment: { objective: 0.5, mode: "capacity_transport", scale_at_bound: false, rotation_at_limit: false, translation_at_limit: false },
	};
	const manifest = JSON.stringify({
		version: 3, scorer_profile: "capacity_transport_v1", input_sha256: "a".repeat(64), source_sha256: originalSource, step_sha256: originalStep,
		projection_frame: { policy: "fixed", anchor_step_path: anchorPath, anchor_step_sha256: originalStep },
		regions: [{ region_id: "front", page: 0, roi: [0, 0, 500, 500], crop_path: "drawing_feedback/original-fixed-frame/crop.png", view_type: "orthographic", status: "matched", selected, alternatives: [{ ...selected, candidate_id: "back-2", view_spec: { name: "back", direction: [0, 0, -1], up: [0, 1, 0] } }], margin: 0.1, confidence: "high", note: "front" }],
	});
	const job: AgentJob = {
		protocol_version: 5, operation: "run_fixture", run_id: "r1", agent_mode: "base-only",
		fixture: { id: "146", description: "test", inputFiles: ["input.png"], inputType: "png", taskType: "generation", sourceDir: "." },
		staged_paths: ["input.png"], provider: "ucloud", model: "gpt-5.6-sol", thinking_level: "low", pi_version: "0.84.1", container_image: "test", container_image_digest: "sha256:test",
		continuation: {
			kind: direct ? "main_self_reflect" : "main_self_reflect_inspect", id: direct ? "r1-direct" : "r1-dimensions", prompt: "Existing R1 reflection prompt", image_paths: [],
			baseline_source_sha256: originalSource, baseline_step_sha256: originalStep,
			...(direct ? { no_new_images: true } : { dimension_annotation_profile: DIMENSION_ANNOTATION_PROFILE, locked_view_manifest_path: manifestPath, locked_view_manifest_sha256: hash(manifest) }),
		},
	};
	const result = {
		fixture_id: "146", provider: job.provider, model: job.model, thinking_level: "low", pi_version: job.pi_version, container_image: "test", container_image_digest: "sha256:test",
		started_at: "2026-11-05T00:00:00Z", updated_at: "2026-11-05T00:01:00Z", duration_s: 60, last_turn: 1,
		tokens: { input: 5, output: 5, cache_read: 0, cache_write: 0, total: 10 }, cost: 0, tool_calls: 1, status: "valid", agent_mode: "base-only", session_file: "sessions/session.jsonl",
		applied_continuation: { id: job.continuation!.id, prompt_sha256: hash(job.continuation!.prompt), applied_at: "2026-11-05T00:00:00Z" },
		optional_reflect_turn: { continuation_id: job.continuation!.id, prompt_sha256: hash(job.continuation!.prompt), status: "completed", started_at: "2026-11-05T00:00:00Z", completed_at: "2026-11-05T00:01:00Z" },
		last_executed_source_sha256: hash(sourceText), last_executed_step_sha256: hash(stepText),
		artifact: { status: "valid", sha256: hash(stepText), path: "output.step", validation_error: null, render_path: "turns/0001/iso.png", render_error: null, validation: { is_valid: true, is_watertight: true, topology_errors: [], solid_count: 1, shell_count: 1, face_count: 6, volume: 1, bbox: {} } },
		...(!direct ? {
			dimension_annotations: { profile: DIMENSION_ANNOTATION_PROFILE, state_path: "drawing_feedback/dimension-annotations/prior/state.json", revision_id: "prior", status: "completed", source_sha256: hash(sourceText), step_sha256: hash(stepText), locked_manifest_sha256: hash(manifest), regions: [{ region_id: "front", status: "annotated" }] },
			subagent_runs: [{ role: "dimension_annotation", accounting_id: "prior-dimensions" }], recorded_subagent_usage_ids: ["prior-dimensions"], subagent_usage: {},
			main_view_reflection: { version: 1, kind: "main_view_reflect", inspection_count: 1, crop_count: 0, locked_view_manifest_sha256: hash(manifest), baseline_source_sha256: originalSource, baseline_step_sha256: originalStep, audit_manifest_path: "drawing_feedback/old-audit.json", inspected_formal_states: [{ source_sha256: hash(sourceText), step_sha256: hash(stepText) }], cropped_formal_states: [] },
		} : {}),
	} as unknown as AgentResult;
	await Promise.all([
		put(source, JOB_FILE, JSON.stringify(job)), put(source, "agent_result.json", JSON.stringify(result)),
		put(source, "solution.py", sourceText), put(source, "output.step", stepText), put(source, "sessions/session.jsonl", '{"type":"session","id":"original-session"}\n'),
		put(source, "input.png", makePng(200, 160)), put(source, "turns/0001/iso.png", makePng(200, 160)),
	]);
	if (!direct) await Promise.all([
		put(source, manifestPath, manifest), put(source, anchorPath, originalStepText), put(source, "drawing_feedback/dimension-annotations/prior/last-completed.json", '{"old_cache":true}'),
		...["drawing_feedback/original-fixed-frame/crop.png", selected.render_path, selected.aligned_render_path, selected.comparison_path].map(path => put(source, path, makePng(200, 160))),
	]);
	const daemon = vi.spyOn(DockerAgentRuntime.prototype, "daemonVersion").mockResolvedValue("test-daemon");
	vi.spyOn(DockerAgentRuntime.prototype, "imageDigest").mockResolvedValue("sha256:test");
	const prepare = vi.spyOn(DockerAgentRuntime.prototype, "prepareFixture").mockResolvedValue();
	const run = vi.spyOn(DockerAgentRuntime.prototype, "runFixture").mockRejectedValue(new Error("Stop before model calls"));
	vi.spyOn(process.stdout, "write").mockReturnValue(true); vi.spyOn(process.stderr, "write").mockReturnValue(true);
	const command = (input: string, output: string) => [input, "--output-dir", output, "--image", "test", "--provider", "ucloud", "--model", "gpt-5.6-sol", "--thinking-level", "low", "--allow-chained-self-reflect", ...(direct ? ["--no-new-images"] : ["--reuse-fixed-view-from-source", "--dimension-annotation-profile", DIMENSION_ANNOTATION_PROFILE])];
	return { root, source, sourceText, stepText, originalSource, originalStep, originalStepText, manifestPath, manifest, anchorPath, job, result, command, daemon, prepare, run };
}

describe("chained fixed-view reflection", () => {
	it("preserves the original physical frame through two unchanged subsequent rounds, while rebinding the live CAD baseline", async () => {
		const f = await fixture();
		const output2 = join(f.root, "r2");
		expect(await main(f.command(f.source, output2))).toBe(2);
		const source2 = join(output2, "run/146");
		const job2 = await readJson<AgentJob>(join(source2, JOB_FILE));
		expect(job2.continuation).toMatchObject({ reuse_fixed_view_from_source: true, baseline_source_sha256: hash(f.sourceText), baseline_step_sha256: hash(f.stepText), view_origin_source_sha256: f.originalSource, view_origin_step_sha256: f.originalStep, previous_continuation_id: f.job.continuation!.id, locked_view_manifest_sha256: hash(f.manifest) });
		expect(() => assertContinuation(job2)).not.toThrow();
		expect(lockedViewOriginHashes(job2)).toEqual({ baselineSourceSha256: f.originalSource, baselineStepSha256: f.originalStep });
		expect(await readFile(join(source2, f.manifestPath), "utf8")).toBe(f.manifest);
		expect(await readFile(join(source2, f.anchorPath), "utf8")).toBe(f.originalStepText);
		await expect(readFile(join(source2, "drawing_feedback/dimension-annotations/prior/last-completed.json"))).rejects.toThrow();
		const result2 = await readJson<AgentResult>(join(source2, "agent_result.json"));
		expect(result2.dimension_annotations).toBeUndefined();
		expect(result2.optional_reflect_turn).toBeUndefined();
		// Complete R2 synthetically with no CAD change to test R3 preparation only.
		const completed2 = structuredClone(f.result);
		completed2.applied_continuation = { id: job2.continuation!.id, prompt_sha256: hash(job2.continuation!.prompt), applied_at: "2026-11-05T01:00:00Z" };
		completed2.optional_reflect_turn = { ...completed2.optional_reflect_turn!, continuation_id: job2.continuation!.id, prompt_sha256: hash(job2.continuation!.prompt) };
		await put(source2, "agent_result.json", JSON.stringify(completed2));
		const output3 = join(f.root, "r3");
		expect(await main(f.command(source2, output3))).toBe(2);
		const job3 = await readJson<AgentJob>(join(output3, "run/146", JOB_FILE));
		expect(job3.continuation!.id).not.toBe(job2.continuation!.id);
		expect(job3.continuation).toMatchObject({ baseline_source_sha256: hash(f.sourceText), baseline_step_sha256: hash(f.stepText), view_origin_source_sha256: f.originalSource, view_origin_step_sha256: f.originalStep, previous_continuation_id: job2.continuation!.id });
		expect(await readFile(join(output3, "run/146", f.manifestPath), "utf8")).toBe(f.manifest);
		expect(await readFile(join(output3, "run/146", f.anchorPath), "utf8")).toBe(f.originalStepText);
		expect(await readFile(join(output3, "run/146/sessions/session.jsonl"), "utf8")).toBe(await readFile(join(f.source, "sessions/session.jsonl"), "utf8"));
		expect(await readFile(join(f.source, "agent_result.json"), "utf8")).toBe(JSON.stringify(f.result));
		expect(f.run).toHaveBeenCalledTimes(2);
	});

	it.each(["incomplete", "stale_dimensions", "other_agent", "tampered_anchor", "tampered_manifest"])("rejects %s before starting a new Agent", async failure => {
		const f = await fixture();
		if (failure === "incomplete") f.result.optional_reflect_turn!.status = "running";
		if (failure === "stale_dimensions") f.result.dimension_annotations!.step_sha256 = "f".repeat(64);
		if (failure === "other_agent") f.result.subagent_runs![0]!.role = "geometry_repair";
		await put(f.source, "agent_result.json", JSON.stringify(f.result));
		if (failure === "tampered_anchor") await put(f.source, f.anchorPath, "different origin");
		if (failure === "tampered_manifest") await put(f.source, f.manifestPath, f.manifest + " ");
		await expect(main(f.command(f.source, join(f.root, "rejected")))).rejects.toThrow();
		expect(f.run).not.toHaveBeenCalled();
		expect(f.prepare).not.toHaveBeenCalled();
	});

	it("chains direct no-new-images with full prior session and a distinct continuation ID", async () => {
		const f = await fixture(true);
		const output = join(f.root, "r2-direct");
		expect(await main(f.command(f.source, output))).toBe(2);
		const job = await readJson<AgentJob>(join(output, "run/146", JOB_FILE));
		expect(job.continuation).toMatchObject({ kind: "main_self_reflect", no_new_images: true, previous_continuation_id: f.job.continuation!.id, image_paths: [] });
		expect(job.continuation!.id).not.toBe(f.job.continuation!.id);
		expect(job.continuation!.view_origin_step_sha256).toBeUndefined();
		expect(await readFile(join(output, "run/146/baseline/session.jsonl"), "utf8")).toBe(await readFile(join(f.source, "sessions/session.jsonl"), "utf8"));
	});

	it("rejects fixed-frame reuse without required explicit flags before Docker", async () => {
		const args = ["/unused/source", "--output-dir", "/unused/output", "--image", "test", "--reuse-fixed-view-from-source"];
		const daemon = vi.spyOn(DockerAgentRuntime.prototype, "daemonVersion");
		await expect(main(args)).rejects.toThrow("requires --allow-chained-self-reflect");
		await expect(main([...args, "--allow-chained-self-reflect"])).rejects.toThrow("requires --dimension-annotation-profile");
		expect(daemon).not.toHaveBeenCalled();
	});

	it("derives different chained identities when prior round changes but CAD does not", () => {
		const r2 = chainedContinuationSuffix("r1", "a".repeat(64), "b".repeat(64));
		const r3 = chainedContinuationSuffix(r2, "a".repeat(64), "b".repeat(64));
		expect(r3).not.toBe(r2);
	});

	it("preserves the legacy manifest baseline for old jobs and rejects partial or unscoped origin bindings", async () => {
		const f = await fixture();
		expect(lockedViewOriginHashes(f.job)).toEqual({ baselineSourceSha256: f.originalSource, baselineStepSha256: f.originalStep });
		const chained = structuredClone(f.job);
		Object.assign(chained.continuation!, { reuse_fixed_view_from_source: true, previous_continuation_id: f.job.continuation!.id, view_origin_source_sha256: f.originalSource });
		expect(() => assertContinuation(chained)).toThrow("valid pair");
		chained.continuation!.view_origin_step_sha256 = f.originalStep;
		expect(() => assertContinuation(chained)).not.toThrow();
		delete chained.continuation!.reuse_fixed_view_from_source;
		expect(() => assertContinuation(chained)).toThrow("chained fixed-view dimension continuation");
	});
});
