import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, dirname, join } from "node:path";

import { generateUnifiedPatch } from "@earendil-works/pi-coding-agent";
import { Value } from "typebox/value";
import { afterEach, describe, expect, it, vi } from "vitest";

import { createAdoptShapeCandidateTool } from "../src/adopt-shape-candidate.js";
import { GEOMETRY_DIFFERENCE_INSTRUCTIONS } from "../src/geometry-repair-instructions.js";
import type { WorkerClient, WorkerRequest } from "../src/local-worker.js";
import { RunController, type ResolvedModel } from "../src/runner.js";
import { createRepairGeometryTool, REPAIR_GEOMETRY_PARAMETERS_SCHEMA, SUBMIT_GEOMETRY_REPAIR_PARAMETERS_SCHEMA } from "../src/repair-geometry.js";
import type { AgentJob, AgentResult, ArtifactPayload, DrawingViewManifest, SubagentRunUsage, WorkerResponse } from "../src/types.js";
import { makePng } from "./png-fixture.js";

interface TestTool {
	name: string;
	description: string;
	execute: (...args: unknown[]) => Promise<TestResult>;
}
interface TestResult {
	content: Array<{ type: string; text?: string; data?: string }>;
	details: Record<string, any>;
}
interface SessionOptions {
	tools: readonly TestTool[];
	toolNames: readonly string[];
	prompt: string;
	systemPrompt: string;
	images: Array<{ type: string; data?: string }>;
	getSubmitted: () => unknown;
	onUsage: (usage: SubagentRunUsage) => Promise<void>;
	signal?: AbortSignal;
}

const roots: string[] = [];
afterEach(async () => {
	vi.useRealTimers();
	for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true });
});

function sha(value: string | Buffer): string { return createHash("sha256").update(value).digest("hex"); }
async function invoke(tool: TestTool, params: object): Promise<TestResult> {
	return tool.execute(`call-${tool.name}`, params, undefined, undefined, {});
}
function selectedTool(options: SessionOptions, name: string): TestTool {
	const tool = options.tools.find((candidate) => candidate.name === name);
	if (!tool) throw new Error(`Missing test tool ${name}; received ${options.tools.map((candidate) => candidate.name).join(", ")}`);
	return tool;
}
function displayedPatch(result: TestResult) {
	const block = result.content[1];
	expect(block?.type).toBe("text");
	const text = block?.text ?? "";
	const reference = /^Patch \(evaluation_id=([^,\n]+), candidate_id=([^)\n]+)\):\n/u.exec(text);
	if (!reference) throw new Error("Main did not receive an adoptable reference with its patch");
	return { text, evaluation_id: reference[1]!, candidate_id: reference[2]! };
}

async function fixture(options: {
	invalidCandidateIds?: string[]; section?: boolean; driftSectionOrigin?: boolean;
	inputCount?: number; preparedPagesPerInput?: number; missingPreparedInput?: number; foreignPreparedSource?: boolean;
	sharedPreparedPaths?: boolean;
} = {}) {
	const workspace = await mkdtemp(join(tmpdir(), "repair-geometry-"));
	roots.push(workspace);
	const source = "mount_count = 5\nmount_x = 10\nexport_step(part, 'output.step')\n";
	const step = "formal STEP";
	const png = makePng(200, 160, 1);
	const sidePng = makePng(200, 160, 4);
	const save = async (path: string, bytes: string | Buffer) => {
		await mkdir(dirname(join(workspace, path)), { recursive: true });
		await writeFile(join(workspace, path), bytes);
	};
	await save("solution.py", source);
	await save("output.step", step);
	await save("reference.png", png);
	await save("reference-side.png", sidePng);
	const originalDrawings: Array<{ source_path: string; path: string; bytes: Buffer }> = [];
	const stagedPaths: string[] = [];
	for (let inputIndex = 0; inputIndex < (options.inputCount ?? 1); inputIndex += 1) {
		const pageCount = options.preparedPagesPerInput ?? 1;
		const sourcePath = `inputs/${String(inputIndex).padStart(2, "0")}-input.${pageCount > 1 ? "pdf" : "png"}`;
		stagedPaths.push(sourcePath);
		if (pageCount > 1) await save(sourcePath, `Synthetic multi-page document ${inputIndex}`);
		for (let pageIndex = 0; pageIndex < pageCount; pageIndex += 1) {
			const bytes = makePng(360, 280, 80 + inputIndex * 10 + pageIndex);
			const path = pageCount === 1 ? sourcePath : `prepared-originals/input-${inputIndex}-page-${pageIndex}.png`;
			await save(path, bytes);
			originalDrawings.push({ source_path: sourcePath, path, bytes });
		}
	}
	await save("calibration.step", "immutable Direct STEP");
	const artifact = (hash: string, path = "output.step"): ArtifactPayload => ({
		status: "valid", path, sha256: hash,
		validation: { is_valid: true, is_watertight: true, topology_errors: [], solid_count: 1, shell_count: 1, face_count: 6, volume: 1, bbox: {} },
		validation_error: null, render_path: null, render_error: null,
	});
	const manifest = {
		version: 4, scorer_profile: "capacity_transport_v1", input_sha256: sha(png), source_sha256: sha(source), step_sha256: sha("immutable Direct STEP"),
		projection_frame: { policy: "fixed", anchor_step_path: "calibration.step", anchor_step_sha256: sha("immutable Direct STEP") },
		regions: ["front", "side"].map((region_id) => ({
			region_id, page: 0, roi: [0, 0, 1000, 1000], crop_path: region_id === "front" ? "reference.png" : "reference-side.png", view_type: "orthographic", status: "matched",
			selected: {
				candidate_id: `selected-${region_id}`,
				render_path: "reference.png", aligned_render_path: "reference.png", comparison_path: "reference.png", score: 0.5, eligible: true, failed_gates: [],
				view_spec: { name: region_id, direction: region_id === "front" ? [0, -1, 0] : [1, 0, 0], up: [0, 0, 1] },
				transform: { scale_px_per_unit: 2, rotation_deg: -1, center_x_px: 100, center_y_px: 80, model_center_x: 0, model_center_y: -3, reflected: false },
			}, alternatives: [], margin: 1, confidence: "high", note: "",
		})),
	} as unknown as DrawingViewManifest;
	if (options.section) {
		manifest.regions[0]!.view_type = "section";
		manifest.regions[0]!.selected!.view_spec.section_offset_fraction = 0;
		manifest.regions[0]!.selected!.view_spec.section_keep = "front";
	}
	const result = {
		fixture_id: "103", status: "running", provider: "test", model: "test", thinking_level: "low",
		pi_version: "0.84.1", container_image: "test", container_image_digest: "sha256:test",
		started_at: "2026-09-08T00:00:00Z", updated_at: "2026-09-08T00:00:00Z", duration_s: 0, last_turn: 1,
		last_executed_source_sha256: sha(source), last_executed_step_sha256: sha(step), artifact: artifact(sha(step)),
		tokens: { input: 0, output: 0, cache_read: 0, cache_write: 0, total: 0 }, cost: 0, tool_calls: 0,
	} as AgentResult;
	const job = {
		protocol_version: 5, operation: "run_fixture", run_id: "repair-geometry-test",
		fixture: { id: "103", description: "tray drawing", inputFiles: stagedPaths.map(path => basename(path).replace(/^\d+-/u, "")), inputType: "image", taskType: "generation", sourceDir: "fixtures/103" },
		staged_paths: stagedPaths, provider: "test", model: "test", thinking_level: "low",
		pi_version: "0.84.1", container_image: "test", container_image_digest: "sha256:test",
	} as AgentJob;
	const requests: WorkerRequest[] = [];
	const executions: string[] = [];
	const hashFile = async (path: unknown) => sha(await readFile(join(workspace, String(path))));
	const worker: WorkerClient = {
		async request<T extends WorkerResponse>(_workspace: string, request: WorkerRequest): Promise<T> {
			requests.push(request);
			if (request.operation === "inspect") return { protocol_version: 5, operation: "inspect", artifact: artifact(sha(step)) } as unknown as T;
			if (request.operation === "prepare_images") {
				const selected = (request.paths as string[]).flatMap(sourcePath => originalDrawings
					.filter(item => item.source_path === sourcePath && sourcePath !== stagedPaths[options.missingPreparedInput ?? -1])
				);
				const images = await Promise.all(selected.map(async (item, index) => {
					const path = options.sharedPreparedPaths ? `prepared_inputs/${String(index).padStart(2, "0")}.png` : item.path;
					if (options.sharedPreparedPaths) await save(path, item.bytes);
					return { source_path: options.foreignPreparedSource ? "unrequested.png" : item.source_path, path, mime_type: "image/png", width: 360, height: 280, original_width: 360, original_height: 280 };
				}));
				return { protocol_version: 5, operation: "prepare_images", images } as unknown as T;
			}
			if (request.operation === "execute_repair_candidate") {
				const candidateSource = await readFile(join(workspace, String(request.source_path)), "utf8");
				executions.push(candidateSource);
				const invalid = options.invalidCandidateIds?.includes(basename(dirname(String(request.source_path))));
				const candidatePath = join(String(request.output_dir), "output.step");
				const candidateBytes = `STEP ${candidateSource}`;
				if (!invalid) await save(candidatePath, candidateBytes);
				return {
					protocol_version: 5, operation: request.operation, candidate_id: request.candidate_id, source_path: request.source_path,
					source_sha256: sha(candidateSource), formal_source_sha256: sha(source), formal_step_sha256: sha(step), formal_unchanged: true,
					execution: { success: !invalid, exit_code: invalid ? 1 : 0, timed_out: false, stdout: "", stderr: invalid ? "invalid geometry" : "", files_produced: {}, duration_s: 0.1 },
					artifact: invalid ? { ...artifact(sha(step)), status: "missing", path: null, sha256: null } : artifact(sha(candidateBytes), candidatePath),
				} as unknown as T;
			}
			if (request.operation === "compare_candidate_topology") {
				const topology = { b0: 1, b1: 5, b2: 0, chi_surface: -8, n_components: 1, n_triangles: 100, n_vertices: 50, linear_deflection_mm: 0.1 };
				return {
					protocol_version: 5, operation: request.operation, status: "ok", current_step_path: "output.step", current_step_sha256: sha(step), current_topology: topology, current_error: null,
					candidates: await Promise.all((request.candidates as Array<{ candidate_id: string; step_path: string }>).map(async (candidate) => ({
						...candidate, step_sha256: await hashFile(candidate.step_path), status: "ok", topology, topology_delta: { b0: 0, b1: 0, b2: 0 }, error: null,
					}))),
				} as unknown as T;
			}
			if (request.operation === "render_fixed_projection") {
				const bytes = makePng(200, 160, request.step_path === "output.step" ? 2 : 3 + executions.length);
				const path = join(String(request.output_dir), "aligned-render.png");
				await save(path, bytes);
				return {
					protocol_version: 5, operation: request.operation, status: "ok", region_id: request.region_id, step_path: request.step_path,
					step_sha256: await hashFile(request.step_path), view_spec: request.view_spec, transform: request.transform,
					projection_anchor_step_sha256: await hashFile(request.projection_anchor_step_path), projection_origin: [0, 0, 0],
					...((request.view_spec as { section_offset_fraction?: number }).section_offset_fraction !== undefined ? { section_origin: options.driftSectionOrigin && request.step_path !== "output.step" ? [0, 1, 0] : [0, 0, 0] } : {}),
					render_path: path, sha256: sha(bytes), width: 200, height: 160, mime_type: "image/png",
				} as unknown as T;
			}
			if (request.operation === "compose_shape_feedback") {
				const path = join(String(request.output_dir), "shape-feedback.png");
				const bytes = makePng(400, 300, 20 + executions.length);
				await save(path, bytes);
				const candidates = await Promise.all((request.candidates as Array<{ candidate_id: string; render_path: string }>).map(async (candidate) => ({ candidate_id: candidate.candidate_id, render_sha256: await hashFile(candidate.render_path) })));
				return {
					protocol_version: 5, operation: request.operation, status: "ok", region_id: request.region_id,
					card_path: path, sha256: sha(bytes), width: 400, height: 300, card_layout: "three_panel_row", card_count: candidates.length,
					cards: candidates.map(({ candidate_id }) => ({ candidate_id, card_path: path, sha256: sha(bytes), width: 400, height: 300, mime_type: "image/png", card_layout: "three_panel_row", panel_count: 3, panel_titles: ["Target", "Current", "Preview"] })),
					reference_sha256: await hashFile(request.reference_path), current_render_sha256: await hashFile(request.current_render_path),
					candidates, target_roi: request.target_roi, files: { card: path }, diagnostics: {},
				} as unknown as T;
			}
			if (request.operation === "compose_reflection_inspection") {
				const path = join(String(request.output_dir), "comparison.png");
				const bytes = makePng(400, 200, request.region_id === "side" ? 61 : 60);
				await save(path, bytes);
				return { protocol_version: 5, operation: request.operation, status: "ok", region_id: request.region_id, mode: request.mode, context_path: path, width: 400, height: 200, sha256: { context: sha(bytes) } } as unknown as T;
			}
			throw new Error(`Unexpected worker operation ${request.operation}`);
		},
	};
	return { workspace, source, step, png, sidePng, originalDrawings, save, manifest, result, job, worker, requests, executions, controller: new RunController(), resolvedModel: { runtime: {}, model: {} } as unknown as ResolvedModel, usages: [] as SubagentRunUsage[] };
}

type Fixture = Awaited<ReturnType<typeof fixture>>;
const request = { part: "tray mounting feet", difference_type: "count", suspected_difference: "The current tray may be missing a mounting foot.", view_ids: ["front"] };
const repairSummary = "Added the missing low mounting foot with its connecting arm and hole; the preview matches its target position.";

function candidate(id: string, x = 10) {
	return {
		views: [{ region_id: "front" }],
		candidates: [{ id, edits: [
			{ oldText: "mount_count = 5", newText: "mount_count = 6" },
			...(x === 10 ? [] : [{ oldText: "mount_x = 10", newText: `mount_x = ${x}` }]),
		] }],
	};
}
async function start(f: Fixture, script: (session: SessionOptions) => Promise<void>, params: object = request, signal?: AbortSignal) {
	const sessions: SessionOptions[] = [];
	const usage = {
		role: "geometry_repair", accounting_id: "repair-test-session", status: "success",
		started_at: "2026-09-08T00:00:00Z", finished_at: "2026-09-08T00:00:01Z", duration_s: 1,
		tokens: { input: 100, output: 20, cache_read: 0, cache_write: 0, total: 120 }, cost: 0.1, tool_calls: 3,
	} as SubagentRunUsage;
	const runSession = async (session: SessionOptions) => {
		sessions.push(session);
		await script(session);
		await session.onUsage(usage);
		return { value: session.getSubmitted(), usage };
	};
	const tool = createRepairGeometryTool({
		workspace: f.workspace, fixture: f.job.fixture, stagedPaths: f.job.staged_paths, worker: f.worker, controller: f.controller,
		lockedManifest: f.manifest, resolvedModel: f.resolvedModel, thinkingLevel: "low", getResult: () => f.result,
		onUsage: async (value) => { f.usages.push(value); },
		runSession: runSession as NonNullable<Parameters<typeof createRepairGeometryTool>[0]["runSession"]>,
	});
	const result = await (tool as unknown as TestTool).execute("repair-call", params, signal, undefined, {});
	return { result, sessions, tool };
}
async function preview(session: SessionOptions, id: string, x = 10) {
	return invoke(selectedTool(session, "preview_shape_feedback"), candidate(id, x));
}
async function submit(session: SessionOptions, params: object) {
	return invoke(selectedTool(session, "submit_geometry_repair"), { summary: "No acceptable candidate was submitted.", ...params });
}
async function propose(session: SessionOptions, result: TestResult, id: string) {
	return submit(session, {
		status: "proposed", evaluation_id: result.details.evaluation_id, candidate_id: id,
		summary: repairSummary,
	});
}

describe("repair_geometry", () => {
	it("accepts one part and one difference type without requiring code or a target value", () => {
		expect(Value.Check(REPAIR_GEOMETRY_PARAMETERS_SCHEMA, request)).toBe(true);
		expect(Value.Check(REPAIR_GEOMETRY_PARAMETERS_SCHEMA, { ...request, view_ids: ["front", "side"] })).toBe(true);
		for (const params of [
			{ ...request, difference_type: ["count", "size"] },
			{ ...request, difference_type: "anything" },
			{ ...request, part: "" }, { ...request, suspected_difference: "" },
			{ ...request, view_ids: [] }, { ...request, view_ids: ["front"], patch: "arbitrary code" },
			{ ...request, view_ids: ["front", "side", "top"] },
		]) expect(Value.Check(REPAIR_GEOMETRY_PARAMETERS_SCHEMA, params)).toBe(false);
	});

	it.each(["premise_rejected", "insufficient_evidence", "repair_failed", "needs_split"] as const)("finishes %s despite placeholder references and returns evidence without a candidate", async (status) => {
		const f = await fixture();
		const summary = {
			premise_rejected: "Both Target and Current contain six feet, so no count repair is required.",
			insufficient_evidence: "The target foot is occluded, so the suspected count discrepancy cannot be verified.",
			repair_failed: "Missing foot confirmed; connection route remains ambiguous.",
			needs_split: "The request combines independent count and size changes; delegate them separately.",
		}[status];
		const references = {
			premise_rejected: { evaluation_id: null, candidate_id: "" },
			insufficient_evidence: { evaluation_id: " ", candidate_id: "none" },
			repair_failed: { evaluation_id: "0123456789abcdef-1234abcd", candidate_id: "previous" },
			needs_split: { evaluation_id: "unused", candidate_id: null },
		}[status];
		const { result } = await start(f, async (session) => {
			const args = { status, summary, ...references };
			expect(Value.Check(SUBMIT_GEOMETRY_REPAIR_PARAMETERS_SCHEMA, args)).toBe(true);
			const receipt = await submit(session, args);
			expect(receipt.content).toEqual([{ type: "text", text: "Result submitted. This repair task is finished." }]);
			expect(session.getSubmitted()).toMatchObject({ status, summary });
		});
		expect(result.details.status).toBe(status);
		expect(result.details.summary).toBe(summary);
		expect(result.details).not.toHaveProperty("assessment");
		const audit = JSON.parse(await readFile(join(f.workspace, result.details.audit_path), "utf8"));
		for (const field of ["evaluation_id", "candidate_id", "patch_path"]) {
			expect(result.details).not.toHaveProperty(field);
			expect(audit.result).not.toHaveProperty(field);
		}
		expect(f.executions).toHaveLength(0);
		expect(result.content).toEqual([
			{ type: "text", text: summary },
			expect.objectContaining({ type: "image", data: makePng(400, 200, 60).toString("base64") }),
		]);
		expect(await readFile(join(f.workspace, "solution.py"), "utf8")).toBe(f.source);
		expect(await readFile(join(f.workspace, "output.step"), "utf8")).toBe(f.step);
	});

	it("returns only the summary, complete multi-edit patch, and unchanged final preview images", async () => {
		const f = await fixture();
		let finalPreview: TestResult | undefined;
		const { result, sessions, tool } = await start(f, async (session) => {
			const requestsBefore = f.requests.length;
			await expect(invoke(selectedTool(session, "preview_shape_feedback"), {
				...candidate("obsolete_roi", 14),
				views: [{ region_id: "front", target_roi: { x_min: 100, y_min: 200, x_max: 700, y_max: 900 } }],
			})).rejects.toThrow(/region_id|target_roi/iu);
			expect(f.requests).toHaveLength(requestsBefore);
			finalPreview = await preview(session, "six_feet", 14);
			await propose(session, finalPreview, "six_feet");
		});
		expect(tool.executionMode).toBe("sequential");
		expect(result.details).toMatchObject({ status: "proposed", evaluation_id: finalPreview!.details.evaluation_id, candidate_id: "six_feet" });
		expect(result.details.summary).toBe(repairSummary);
		expect(result.details).not.toHaveProperty("assessment");
		const proposedSource = f.source.replace("mount_count = 5", "mount_count = 6").replace("mount_x = 10", "mount_x = 14");
		expect(f.executions).toEqual([proposedSource]);
		expect(result.content[0]).toEqual({ type: "text", text: repairSummary });
		const patch = displayedPatch(result);
		expect(patch).toMatchObject({ evaluation_id: finalPreview!.details.evaluation_id, candidate_id: "six_feet" });
		expect(patch.text).toContain(generateUnifiedPatch("solution.py", f.source, proposedSource));
		expect(result.content.slice(2).every((item) => item.type === "image")).toBe(true);
		expect(result.content.filter((item) => item.type === "image").map((item) => item.data)).toEqual(finalPreview!.content.filter((item) => item.type === "image").map((item) => item.data));
		expect(finalPreview!.content.filter((item) => item.type === "text").map((item) => item.text).join("\n")).not.toMatch(/\b(?:b[012]|topology|ROI)\b/iu);
		expect(finalPreview!.details.topology.current_topology).toMatchObject({ b0: 1, b1: 5, b2: 0 });
		expect(f.requests.filter(({ operation }) => operation === "compose_shape_feedback").map(({ target_roi }) => target_roi)).toEqual([[0, 0, 1000, 1000]]);
		expect(sessions).toHaveLength(1);
		expect(f.usages).toHaveLength(1);
		expect(await readFile(join(f.workspace, "solution.py"), "utf8")).toBe(f.source);
		expect(await readFile(join(f.workspace, "output.step"), "utf8")).toBe(f.step);
	});

	it.each(["count", "size"] as const)("delivers the %s repair boundary and projection acceptance instructions to the restricted child", async (differenceType) => {
		const f = await fixture();
		const { sessions } = await start(f, async (session) => {
			await submit(session, { status: "repair_failed", summary: "No candidate." });
		}, differenceType === "count" ? request : {
			...request, part: "mounting holes", difference_type: differenceType,
			suspected_difference: "The mounting-hole diameter may be too small.",
		});
		const session = sessions[0]!;
		expect(new Set(session.toolNames)).toEqual(new Set(["read_repair_input", "inspect_comparison", "preview_shape_feedback", "submit_geometry_repair"]));
		expect(session.images.length).toBeGreaterThanOrEqual(2);
		expect(session.prompt).toContain(f.source);
		expect(session.prompt).toContain(GEOMETRY_DIFFERENCE_INSTRUCTIONS[differenceType]);
		expect(session.prompt).not.toContain(GEOMETRY_DIFFERENCE_INSTRUCTIONS[differenceType === "count" ? "size" : "count"]);
		if (differenceType === "size") expect(session.prompt).toMatch(/one dimension[\s\S]*diameter and depth[\s\S]*separately/iu);
		expect(session.systemPrompt).toMatch(/required geometry[^.]*specified location[^.]*Preview/iu);
		expect(session.systemPrompt).toMatch(/judgment[^.]*actual projections/iu);
		const previewDescription = selectedTool(session, "preview_shape_feedback").description;
		expect(previewDescription).not.toMatch(/adopt the change/iu);
		expect(previewDescription).toContain("submit_geometry_repair");
		expect(previewDescription).toMatch(/Main Agent decides whether to adopt/iu);
	});

	it("attaches the complete original drawing before selected crop evidence without requiring a read call", async () => {
		const f = await fixture();
		const { result } = await start(f, async (session) => {
			expect(session.images.map(image => image.data)).toEqual([
				f.originalDrawings[0]!.bytes.toString("base64"),
				f.png.toString("base64"), makePng(400, 200, 60).toString("base64"),
			]);
			expect(session.prompt).toContain("inputs/00-input.png");
			await submit(session, { status: "repair_failed", summary: "No candidate was attempted." });
		});
		const preparation = f.requests.filter(({ operation }) => operation === "prepare_images");
		expect(preparation).toHaveLength(1);
		expect(preparation[0]).toMatchObject({ paths: ["inputs/00-input.png"], max_edge: 2000 });
		const audit = JSON.parse(await readFile(join(f.workspace, result.details.audit_path), "utf8"));
		expect(audit.original_drawings.map(({ source_path, sha256 }: { source_path: string; sha256: string }) => ({ source_path, sha256 }))).toEqual(f.originalDrawings.map(({ source_path, bytes }) => ({ source_path, sha256: sha(bytes) })));
		for (const [index, original] of audit.original_drawings.entries()) {
			expect(original.path).toMatch(/baseline\/original-drawings\/\d+\.png$/u);
			expect(await readFile(join(f.workspace, original.path))).toEqual(f.originalDrawings[index]!.bytes);
		}
	});

	it("preserves every prepared page from all drawing inputs before all selected view pairs", async () => {
		const f = await fixture({ inputCount: 2, preparedPagesPerInput: 2 });
		const { result } = await start(f, async (session) => {
			expect(session.images.map(image => image.data)).toEqual([
				...f.originalDrawings.map(image => image.bytes.toString("base64")),
				f.png.toString("base64"), makePng(400, 200, 60).toString("base64"),
				f.sidePng.toString("base64"), makePng(400, 200, 61).toString("base64"),
			]);
			for (const path of f.job.staged_paths) expect(session.prompt).toContain(path);
			await submit(session, { status: "repair_failed", summary: "No candidate was attempted." });
		}, { ...request, view_ids: ["front", "side"] });
		const preparation = f.requests.filter(({ operation }) => operation === "prepare_images");
		expect(preparation).toHaveLength(1);
		expect(preparation[0]!.paths).toEqual(f.job.staged_paths);
		const audit = JSON.parse(await readFile(join(f.workspace, result.details.audit_path), "utf8"));
		expect(audit.original_drawings.map(({ source_path, sha256 }: { source_path: string; sha256: string }) => ({ source_path, sha256 }))).toEqual(f.originalDrawings.map(({ source_path, bytes }) => ({ source_path, sha256: sha(bytes) })));
		for (const [index, original] of audit.original_drawings.entries()) expect(await readFile(join(f.workspace, original.path))).toEqual(f.originalDrawings[index]!.bytes);
	});

	it.each([
		{ inputCount: 1, missingPreparedInput: 0 },
		{ inputCount: 2, missingPreparedInput: 1 },
		{ foreignPreparedSource: true },
	])("rejects missing or foreign original evidence before starting the child: %j", async (options) => {
		const f = await fixture(options);
		let started = false;
		await expect(start(f, async () => { started = true; })).rejects.toThrow(/original|drawing|source|prepared/iu);
		expect(started).toBe(false);
		expect(f.executions).toHaveLength(0);
	});

	it("keeps original snapshots intact when an on-demand read overwrites shared prepared-image slots", async () => {
		const f = await fixture({ inputCount: 2, sharedPreparedPaths: true });
		const { result } = await start(f, async (session) => {
			expect(session.images[0]!.data).toBe(f.originalDrawings[0]!.bytes.toString("base64"));
			const reread = await invoke(selectedTool(session, "read_repair_input"), { kind: "drawing", page: 1 });
			expect(reread.content.find(({ type }) => type === "image")!.data).toBe(f.originalDrawings[1]!.bytes.toString("base64"));
			await submit(session, { status: "repair_failed", summary: "No candidate was attempted." });
		});
		expect(await readFile(join(f.workspace, "prepared_inputs/00.png"))).toEqual(f.originalDrawings[1]!.bytes);
		const audit = JSON.parse(await readFile(join(f.workspace, result.details.audit_path), "utf8"));
		for (const [index, original] of audit.original_drawings.entries()) expect(await readFile(join(f.workspace, original.path))).toEqual(f.originalDrawings[index]!.bytes);
	});

	it("requires a nonblank bounded summary for every final status", async () => {
		const f = await fixture();
		await start(f, async (session) => {
			expect(session.toolNames).not.toContain("record_geometry_assessment");
			const valid = await preview(session, "valid");
			for (const status of ["proposed", "premise_rejected", "insufficient_evidence", "repair_failed", "needs_split"]) {
				for (const summary of [undefined, "", " ", "x".repeat(2001), 42]) {
					await expect(invoke(selectedTool(session, "submit_geometry_repair"), {
						status, ...(summary === undefined ? {} : { summary }),
						...(status === "proposed" ? { evaluation_id: valid.details.evaluation_id, candidate_id: "valid" } : {}),
					})).rejects.toThrow();
				}
			}
			await propose(session, valid, "valid");
		});
		expect(f.executions).toHaveLength(1);
	});

	it("allows more than four previews and ends the repair when the child submits", async () => {
		const f = await fixture();
		const { result } = await start(f, async (session) => {
			let latest: TestResult | undefined;
			for (let i = 0; i < 6; i += 1) {
				latest = await preview(session, `attempt_${i}`, 12 + i);
				expect(session.getSubmitted()).toBeNull();
			}
			await propose(session, latest!, "attempt_5");
			expect(session.getSubmitted()).toMatchObject({ status: "proposed", candidate_id: "attempt_5" });
			await expect(preview(session, "after_submission", 20)).rejects.toThrow(/already submitted/iu);
		});
		expect(f.executions).toHaveLength(6);
		const audit = JSON.parse(await readFile(join(f.workspace, result.details.audit_path), "utf8"));
		expect(audit.max_previews).toBeNull();
		expect(audit.preview_attempts).toHaveLength(6);
	});

	it("returns and adopts the displayed earlier candidate patch without leaking later attempts", async () => {
		const f = await fixture();
		let best: TestResult | undefined;
		let later: TestResult | undefined;
		const { result } = await start(f, async (session) => {
			best = await preview(session, "best", 12);
			later = await preview(session, "worse", 30);
			await propose(session, best, "best");
		});
		expect(result.details).toMatchObject({ evaluation_id: best!.details.evaluation_id, candidate_id: "best" });
		const returned = result.content.filter((item) => item.type === "image").map((item) => item.data);
		expect(returned).toEqual(best!.content.filter((item) => item.type === "image").map((item) => item.data));
		expect(returned).not.toEqual(later!.content.filter((item) => item.type === "image").map((item) => item.data));
		expect(result.content[0]).toEqual({ type: "text", text: repairSummary });
		const patch = displayedPatch(result);
		const selectedSource = f.source.replace("mount_count = 5", "mount_count = 6").replace("mount_x = 10", "mount_x = 12");
		const selectedPatch = generateUnifiedPatch("solution.py", f.source, selectedSource);
		expect(patch).toMatchObject({ evaluation_id: best!.details.evaluation_id, candidate_id: "best" });
		expect(patch.text).toContain(selectedPatch);
		expect(patch.text).not.toContain("mount_x = 30");
		expect(result.content.slice(2).every((item) => item.type === "image")).toBe(true);
		expect(result.details).not.toHaveProperty("previews");
		expect(result.details).not.toHaveProperty("assessment_history");
		const audit = JSON.parse(await readFile(join(f.workspace, result.details.audit_path), "utf8"));
		expect(audit.preview_attempts).toHaveLength(2);
		expect(audit).not.toHaveProperty("assessment_history");
		expect(audit.preview_attempts.every((attempt: object) => !("assessment_revision" in attempt))).toBe(true);
		expect(audit.result).toMatchObject({ candidate_id: "best", evaluation_id: best!.details.evaluation_id });
		expect(await readFile(join(f.workspace, audit.baseline_path, "solution.py"), "utf8")).toBe(f.source);
		const adoption = await invoke(createAdoptShapeCandidateTool({
			workspace: f.workspace, lockedManifest: f.manifest, controller: f.controller,
		}) as unknown as TestTool, { evaluation_id: patch.evaluation_id, candidate_id: patch.candidate_id });
		expect(adoption.details).toMatchObject({ status: "applied", evaluation_id: patch.evaluation_id, candidate_id: "best", patch: selectedPatch });
		expect(await readFile(join(f.workspace, "solution.py"), "utf8")).toBe(selectedSource);
		expect(await readFile(join(f.workspace, "output.step"), "utf8")).toBe(f.step);
	});

	it("bases each complete patch on the same Current instead of chaining prior previews", async () => {
		const f = await fixture();
		await start(f, async (session) => {
			await preview(session, "first", 12);
			const second = await preview(session, "second", 14);
			await propose(session, second, "second");
		});
		expect(f.executions).toHaveLength(2);
		expect(f.executions[1]).toBe(f.source.replace("mount_count = 5", "mount_count = 6").replace("mount_x = 10", "mount_x = 14"));
		const projections = f.requests.filter(({ operation }) => operation === "render_fixed_projection");
		for (const projection of projections) {
			expect(projection.projection_anchor_step_path).toBe("calibration.step");
			expect(projection.transform).toEqual(f.manifest.regions[0]!.selected!.transform);
		}
		expect(f.requests.some(({ operation }) => operation === "rank_drawing_views")).toBe(false);
	});

	it("can directly revise an unsuccessful candidate and summarize the selected repair", async () => {
		const f = await fixture({ invalidCandidateIds: ["bad_position"] });
		const { result } = await start(f, async (session) => {
			await preview(session, "bad_position", 30);
			const fixed = await preview(session, "corrected_position", 12);
			await propose(session, fixed, "corrected_position");
		});
		expect(result.details.summary).toBe(repairSummary);
		expect(result.details).not.toHaveProperty("assessment");
		expect(f.executions).toHaveLength(2);
		expect(f.executions.every((source) => source.includes("mount_count = 6"))).toBe(true);
	});

	it.each(["solution.py", "output.step", "calibration.step"])("rejects a preview after %s changes during delegation", async (path) => {
		const f = await fixture();
		await expect(start(f, async (session) => {
			await f.save(path, "changed concurrently");
			await expect(preview(session, "stale")).rejects.toThrow(/chang|stale|anchor|hash/iu);
			await submit(session, { status: "repair_failed", summary: "Current changed." });
		})).rejects.toThrow(/chang|stale|anchor|hash/iu);
		expect(f.executions).toHaveLength(0);
	});

	it("rejects a proposed result without a successfully executed and displayed candidate", async () => {
		const f = await fixture({ invalidCandidateIds: ["invalid"] });
		await start(f, async (session) => {
			const invalid = await preview(session, "invalid");
			await expect(propose(session, invalid, "invalid")).rejects.toThrow();
			await expect(submit(session, {
				status: "proposed", evaluation_id: "0123456789abcdef-1234abcd", candidate_id: "unknown", summary: "No preview exists for this reference.",
			})).rejects.toThrow();
			await submit(session, { status: "repair_failed", summary: "All attempted candidates failed execution." });
		});
	});

	it("can reject the premise after a successful preview while discarding its references and patch", async () => {
		const f = await fixture();
		const summary = "The drawing evidence does not support adding another foot to Current.";
		let attempted: TestResult | undefined;
		const { result } = await start(f, async (session) => {
			attempted = await preview(session, "unneeded_foot");
			await expect(submit(session, { status: "repair_failed", changes: ["Changed the feet."] })).rejects.toThrow();
			expect(session.getSubmitted()).toBeNull();
			await submit(session, {
				status: "premise_rejected", summary,
				evaluation_id: attempted.details.evaluation_id, candidate_id: "unneeded_foot",
			});
		});
		expect(result.details.status).toBe("premise_rejected");
		for (const field of ["evaluation_id", "candidate_id", "patch_path"]) expect(result.details).not.toHaveProperty(field);
		expect(result.content).toEqual([
			{ type: "text", text: summary },
			expect.objectContaining({ type: "image", data: makePng(400, 200, 60).toString("base64") }),
		]);
		expect(result.content.filter((item) => item.type === "image")).not.toEqual(attempted!.content.filter((item) => item.type === "image"));
		expect(f.executions).toHaveLength(1);
		expect(await readFile(join(f.workspace, "solution.py"), "utf8")).toBe(f.source);
		expect(await readFile(join(f.workspace, "output.step"), "utf8")).toBe(f.step);
	});

	it("rejects the removed assessment and detailed report fields on submission", async () => {
		const f = await fixture();
		await start(f, async (session) => {
			const valid = await preview(session, "valid");
			const ref = { status: "proposed", evaluation_id: valid.details.evaluation_id, candidate_id: "valid" };
			for (const fields of [
				{ assessment: { target_observation: "Six feet", current_observation: "Five feet", evidence: "Drawing", repair_target: "Add foot" } },
				{ changes: ["Added foot"] }, { preview_observations: "Foot matches" }, { remaining: [] },
				{ repair_target: "Add foot" }, { target_observation: "Six feet" }, { evidence: "Drawing" },
				{ verdict: "confirmed" }, { revision_reason: "Updated interpretation" },
			]) await expect(submit(session, { ...ref, ...fields })).rejects.toThrow();
			await propose(session, valid, "valid");
		});
	});

	it("rejects a view outside the locked manifest before starting a sub-session", async () => {
		const f = await fixture();
		let started = false;
		await expect(start(f, async () => { started = true; }, { ...request, view_ids: ["unknown"] })).rejects.toThrow();
		expect(started).toBe(false);
	});

	it.each(["evaluation_id", "candidate_id"] as const)("keeps the session open for invalid proposed %s until a real preview reference is supplied", async (field) => {
		const f = await fixture();
		const { result } = await start(f, async (session) => {
			const valid = await preview(session, "valid");
			for (const invalid of [undefined, null, "", " ", "none", "unused", "../bad"]) {
				const args: Record<string, unknown> = { status: "proposed", summary: repairSummary, evaluation_id: valid.details.evaluation_id, candidate_id: "valid" };
				if (invalid === undefined) delete args[field];
				else args[field] = invalid;
				expect(Value.Check(SUBMIT_GEOMETRY_REPAIR_PARAMETERS_SCHEMA, args)).toBe(true);
				await expect(submit(session, args)).rejects.toThrow();
				expect(session.getSubmitted()).toBeNull();
			}
			const receipt = await propose(session, valid, "valid");
			expect(receipt.content).toEqual([{ type: "text", text: "Result submitted. This repair task is finished." }]);
			expect(session.getSubmitted()).toMatchObject({ status: "proposed", evaluation_id: valid.details.evaluation_id, candidate_id: "valid" });
		});
		expect(result.details).toMatchObject({ status: "proposed", candidate_id: "valid" });
		expect(f.executions).toHaveLength(1);
		expect(await readFile(join(f.workspace, "solution.py"), "utf8")).toBe(f.source);
		expect(await readFile(join(f.workspace, "output.step"), "utf8")).toBe(f.step);
	});

	it("does not accept a candidate from a different repair even when Current has not changed", async () => {
		const f = await fixture();
		let previous: TestResult | undefined;
		await start(f, async (session) => {
			previous = await preview(session, "external");
			await propose(session, previous, "external");
		});
		const { result } = await start(f, async (session) => {
			await expect(propose(session, previous!, "external")).rejects.toThrow(/this repair/iu);
			await submit(session, { status: "repair_failed", summary: "No candidate was tested in this repair." });
		});
		expect(result.details.status).toBe("repair_failed");
		expect(f.executions).toHaveLength(1);
	});

	it("lets the child read the full drawing and inspect other locked views before using them", async () => {
		const f = await fixture();
		await start(f, async (session) => {
			const source = await invoke(selectedTool(session, "read_repair_input"), { kind: "source" });
			expect(source.content[0]!.text).toBe(f.source);
			const requestsBeforeRead = f.requests.length;
			const drawing = await invoke(selectedTool(session, "read_repair_input"), { kind: "drawing", page: 0 });
			expect(drawing.content.some((item) => item.type === "image")).toBe(true);
			expect(drawing.details.path).toBe("inputs/00-input.png");
			expect(f.requests.slice(requestsBeforeRead).find(({ operation }) => operation === "prepare_images")!.paths).toEqual(["inputs/00-input.png"]);
			await expect(invoke(selectedTool(session, "read_repair_input"), { kind: "drawing", page: 9 })).rejects.toThrow(/page/iu);
			await expect(invoke(selectedTool(session, "preview_shape_feedback"), { ...candidate("uninspected"), views: [{ region_id: "side" }] })).rejects.toThrow(/inspect/iu);
			await invoke(selectedTool(session, "inspect_comparison"), { region_id: "side" });
			const auxiliary = await invoke(selectedTool(session, "preview_shape_feedback"), { ...candidate("auxiliary_only"), views: [{ region_id: "side" }] });
			await expect(propose(session, auxiliary, "auxiliary_only")).rejects.toThrow(/task view/iu);
			const viewed = await invoke(selectedTool(session, "preview_shape_feedback"), { ...candidate("side_checked"), views: [{ region_id: "front" }, { region_id: "side" }] });
			await propose(session, viewed, "side_checked");
		});
		expect(f.executions).toHaveLength(2);
	});

	it("does not reinterpret a missing structured submission as a semantic verdict", async () => {
		const f = await fixture();
		await expect(start(f, async () => {})).rejects.toThrow(/structured|submit/iu);
		expect(await readFile(join(f.workspace, "solution.py"), "utf8")).toBe(f.source);
	});

	it("rejects an in-memory calibration change during a child session", async () => {
		const f = await fixture();
		await expect(start(f, async (session) => {
			f.manifest.regions[0]!.selected!.transform!.rotation_deg = 4;
			await preview(session, "changed_frame");
		})).rejects.toThrow(/calibration changed/iu);
		expect(f.executions).toHaveLength(0);
	});

	it("requires evidence for every original task view in the final selected candidate", async () => {
		const f = await fixture();
		const { result } = await start(f, async (session) => {
			const frontOnly = await preview(session, "incomplete");
			await expect(propose(session, frontOnly, "incomplete")).rejects.toThrow(/task view/iu);
			const complete = await invoke(selectedTool(session, "preview_shape_feedback"), { ...candidate("complete"), views: [{ region_id: "front" }, { region_id: "side" }] });
			await propose(session, complete, "complete");
		}, { ...request, view_ids: ["front", "side"] });
		expect(result.content.filter(({ type }) => type === "image")).toHaveLength(2);
	});

	it("can repair against a calibrated physical section using the same cut plane for Current and Preview", async () => {
		const f = await fixture({ section: true });
		const { result } = await start(f, async (session) => {
			const section = await preview(session, "section_repair");
			await propose(session, section, "section_repair");
		});
		expect(result.details.status).toBe("proposed");
		const projections = f.requests.filter(({ operation }) => operation === "render_fixed_projection");
		expect(projections.length).toBeGreaterThanOrEqual(3);
		for (const projection of projections) {
			expect(projection.view_spec).toMatchObject({ section_offset_fraction: 0, section_keep: "front" });
			expect(projection.projection_anchor_step_path).toBe("calibration.step");
		}
	});

	it("rejects a candidate whose physical section plane drifts from Current", async () => {
		const f = await fixture({ section: true, driftSectionOrigin: true });
		await start(f, async (session) => {
			await expect(preview(session, "shifted_section")).rejects.toThrow(/frame|origin|section/iu);
			await submit(session, { status: "repair_failed", summary: "Section plane was not preserved." });
		});
		expect(f.requests.some(({ operation }) => operation === "compose_shape_feedback")).toBe(false);
	});

	it("passes outer cancellation to the restricted session and blocks any subsequent preview", async () => {
		const f = await fixture();
		const cancellation = new AbortController();
		await expect(start(f, async (session) => {
			expect(session.signal?.aborted).toBe(false);
			cancellation.abort();
			expect(session.signal?.aborted).toBe(true);
			await expect(preview(session, "cancelled")).rejects.toThrow(/cancel/iu);
		}, request, cancellation.signal)).rejects.toThrow(/cancel/iu);
		expect(f.executions).toHaveLength(0);
	});
});
