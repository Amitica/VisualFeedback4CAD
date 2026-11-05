import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, dirname, join } from "node:path";
import { Value } from "typebox/value";
import { afterEach, describe, expect, it } from "vitest";
import type { WorkerClient, WorkerRequest } from "../src/local-worker.js";
import { RunController } from "../src/runner.js";
import { createAdoptShapeCandidateTool } from "../src/adopt-shape-candidate.js";
import { createShapeFeedbackTool, formatShapeTopologyFeedback, PREVIEW_SHAPE_FEEDBACK_PARAMETERS_SCHEMA, type ShapeFeedbackAudit } from "../src/shape-feedback.js";
import type { AgentResult, ArtifactPayload, DrawingViewManifest, RepairPreviewTopology, WorkerCandidateTopologyResponse, WorkerResponse } from "../src/types.js";
import { makePng } from "./png-fixture.js";

const roots: string[] = [];
afterEach(async () => { for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });
function sha(value: string | Buffer): string { return createHash("sha256").update(value).digest("hex"); }
interface ToolResult { content: Array<{ type: string; text?: string }>; details: ShapeFeedbackAudit & { audit_path: string } }

async function fixture(options: {
	invalidCandidate?: boolean; invalidCandidateIds?: string[]; driftOrigin?: boolean; wrongFeedbackHash?: boolean;
	projectionFailureIds?: string[]; projectionFailureRegions?: string[]; regionIds?: string[];
	fixedAnchor?: boolean; wrongAnchor?: boolean; allowSections?: boolean;
	wrongFeedback?: "current_hash" | "candidate_hash" | "id" | "count" | "duplicate" | "missing" | "roi" | "protocol";
	sectionMode?: "label" | "camera"; feedbackWarnings?: string[];
	topologyFailure?: "current" | "preview" | "timeout"; topologyUnchanged?: boolean;
	topologyRejection?: boolean; abortTopology?: boolean;
	wrongTopology?: "current_hash" | "candidate_hash" | "id" | "count" | "delta" | "protocol";
} = {}) {
	const workspace = await mkdtemp(join(tmpdir(), "shape-feedback-"));
	roots.push(workspace);
	const source = "width = 10\nheight = 20\n";
	const step = "formal STEP";
	await writeFile(join(workspace, "solution.py"), source);
	await writeFile(join(workspace, "output.step"), step);
	const png = makePng(200, 160, 1);
	await writeFile(join(workspace, "reference.png"), png);
	const artifact = (hash: string, path = "output.step"): ArtifactPayload => ({
		status: "valid", path, sha256: hash,
		validation: { is_valid: true, is_watertight: true, topology_errors: [], solid_count: 1, shell_count: 1, face_count: 6, volume: 1, bbox: {} },
		validation_error: null, render_path: null, render_error: null,
	});
	const transform = { scale_px_per_unit: 2, rotation_deg: -1, center_x_px: 100, center_y_px: 80, model_center_x: 0, model_center_y: -3, reflected: false };
	const manifest = {
		version: 4, scorer_profile: "capacity_transport_v1", input_sha256: sha(png), source_sha256: sha(source), step_sha256: sha(step),
		regions: (options.regionIds ?? ["front", "side"]).map((region_id) => ({
			region_id, page: 0, roi: [0, 0, 1000, 1000], crop_path: "reference.png", view_type: "orthographic", status: "matched",
			selected: { candidate_id: `selected-${region_id}`, view_spec: { name: region_id, direction: region_id === "front" ? [0, -1, 0] : [1, 0, 0], up: [0, 0, 1] }, transform },
			alternatives: [], margin: 1, confidence: "high", note: "",
		})),
	} as unknown as DrawingViewManifest;
	if (options.fixedAnchor) {
		await writeFile(join(workspace, "calibration.step"), "calibrated baseline STEP");
		manifest.step_sha256 = sha("calibrated baseline STEP");
		manifest.projection_frame = { policy: "fixed", anchor_step_path: "calibration.step", anchor_step_sha256: manifest.step_sha256 };
		// The approved ISO roll is deliberately outside the old +/-6 degree re-fit window.
		transform.rotation_deg = 27.253068;
	}
	if (options.sectionMode === "label") manifest.regions[1]!.view_type = "section";
	if (options.sectionMode === "camera") {
		manifest.regions[1]!.selected!.view_spec.section_offset_fraction = 0;
		manifest.regions[1]!.selected!.view_spec.section_keep = "front";
	}
	const result = { last_executed_source_sha256: sha(source), last_executed_step_sha256: sha(step) } as AgentResult;
	const controller = new RunController();
	const requests: WorkerRequest[] = [];
	const save = async (path: string, bytes: Buffer | string) => {
		await mkdir(dirname(join(workspace, path)), { recursive: true });
		await writeFile(join(workspace, path), bytes);
	};
	const hashFile = async (path: unknown) => sha(await readFile(join(workspace, String(path))));
	const worker: WorkerClient = {
		async request<T extends WorkerResponse>(_workspace: string, request: WorkerRequest): Promise<T> {
			requests.push(request);
			if (request.operation === "inspect") return { protocol_version: 5, operation: "inspect", artifact: artifact(sha(step)) } as unknown as T;
			if (request.operation === "execute_repair_candidate") {
				const candidateSource = await readFile(join(workspace, String(request.source_path)), "utf8");
				const invalid = options.invalidCandidate || options.invalidCandidateIds?.includes(basename(dirname(String(request.source_path))));
				const candidatePath = join(String(request.output_dir), "output.step");
				const candidateBytes = `STEP ${candidateSource}`;
				if (!invalid) await save(candidatePath, candidateBytes);
				return {
					protocol_version: 5, operation: "execute_repair_candidate", candidate_id: request.candidate_id, source_path: request.source_path,
					source_sha256: sha(candidateSource), formal_source_sha256: sha(source), formal_step_sha256: sha(step), formal_unchanged: true,
					execution: { success: !invalid, exit_code: invalid ? 1 : 0, timed_out: false, stdout: "", stderr: invalid ? "invalid geometry" : "", files_produced: {}, duration_s: 0.1 },
					artifact: invalid ? { ...artifact(sha(step)), status: "missing", path: null, sha256: null } : artifact(sha(candidateBytes), candidatePath),
				} as unknown as T;
			}
			if (request.operation === "compare_candidate_topology") {
				if (options.abortTopology) {
					await controller.interrupt("user cancellation");
					throw new Error("CAD worker was aborted");
				}
				if (options.topologyRejection) throw new Error("CAD worker returned malformed JSON");
				if (options.topologyFailure === "timeout") return {
					protocol_version: 5, operation: request.operation, status: "error", error: { kind: "timeout", message: "Topology computation timed out" },
				} as unknown as T;
				const topology = (b0: number, b1: number, b2: number): RepairPreviewTopology => ({ b0, b1, b2, chi_surface: 2 * (b0 - b1 + b2), n_components: b0 + b2, n_triangles: 100, n_vertices: 50, linear_deflection_mm: 0.1 });
				const current = options.topologyFailure === "current" ? null : topology(2, 4, 1);
				const response: WorkerCandidateTopologyResponse = {
					protocol_version: options.wrongTopology === "protocol" ? 4 : 5, operation: request.operation, status: "ok", current_step_path: "output.step",
					current_step_sha256: options.wrongTopology === "current_hash" ? "0".repeat(64) : sha(step),
					current_topology: current, current_error: current ? null : "Current mesh is not closed",
					candidates: await Promise.all((request.candidates as Array<{ candidate_id: string; step_path: string }>).map(async (candidate) => {
						const unavailable = options.topologyFailure === candidate.candidate_id;
						const value = unavailable ? null : options.topologyUnchanged ? topology(2, 4, 1) : topology(1, 6, 0);
						return {
							...candidate, candidate_id: options.wrongTopology === "id" ? "unexpected" : candidate.candidate_id,
							step_sha256: options.wrongTopology === "candidate_hash" ? "0".repeat(64) : await hashFile(candidate.step_path),
							status: unavailable ? "error" : "ok", topology: value,
							topology_delta: value && current ? { b0: value.b0 - current.b0 + (options.wrongTopology === "delta" ? 1 : 0), b1: value.b1 - current.b1, b2: value.b2 - current.b2 } : null,
							error: unavailable ? "Candidate mesh is not closed" : null,
						};
					})),
				};
				if (options.wrongTopology === "count") response.candidates.pop();
				return response as unknown as T;
			}
			if (request.operation === "render_fixed_projection") {
				const isCandidate = request.step_path !== "output.step";
				const candidateId = basename(dirname(dirname(String(request.step_path))));
				if (isCandidate && (options.projectionFailureIds?.includes(candidateId) || options.projectionFailureRegions?.includes(String(request.region_id)))) return {
					protocol_version: 5, operation: request.operation, status: "error", error: { kind: "drawing", message: "Candidate projection failed" },
				} as unknown as T;
				const bytes = makePng(200, 160, isCandidate ? 2 + parseInt((await hashFile(request.step_path)).slice(0, 2), 16) % 250 : 1);
				const path = join(String(request.output_dir), "aligned-render.png");
				await save(path, bytes);
				return {
					protocol_version: 5, operation: "render_fixed_projection", status: "ok", region_id: request.region_id,
					step_path: request.step_path, step_sha256: await hashFile(request.step_path), view_spec: request.view_spec,
					transform: request.transform, projection_anchor_step_sha256: options.wrongAnchor ? "0".repeat(64) : await hashFile(request.projection_anchor_step_path),
					projection_origin: isCandidate && options.driftOrigin ? [1, 0, 0] : [0, 0, 0],
					...((request.view_spec as { section_offset_fraction?: number }).section_offset_fraction !== undefined ? { section_origin: [0, 0, 0] } : {}),
					render_path: path, sha256: sha(bytes), width: 200, height: 160, mime_type: "image/png",
				} as unknown as T;
			}
			if (request.operation === "compose_shape_feedback") {
				const path = join(String(request.output_dir), "shape-feedback.png");
				const bytes = makePng(400, 300, 3);
				await save(path, bytes);
				const candidates = await Promise.all((request.candidates as Array<{ candidate_id: string; render_path: string }>).map(async ({ candidate_id, render_path }) => ({
					candidate_id: options.wrongFeedback === "id" ? "unexpected" : candidate_id,
					render_sha256: options.wrongFeedback === "candidate_hash" ? "0".repeat(64) : await hashFile(render_path),
				})));
				if (options.wrongFeedback === "count") candidates.pop();
				if (options.wrongFeedback === "duplicate") candidates[1] = candidates[0]!;
				const cards = await Promise.all(candidates.map(async ({ candidate_id }, index) => {
					const cardPath = index === 0 ? path : join(String(request.output_dir), "candidates", candidate_id, "shape-feedback.png");
					const cardBytes = index === 0 ? bytes : makePng(400, 300, 3 + index);
					await save(cardPath, cardBytes);
					return { candidate_id, card_path: cardPath, sha256: sha(cardBytes), width: 400, height: 300, mime_type: "image/png",
						card_layout: "three_panel_row", panel_count: 3,
						panel_titles: ["Target", "Current", "Preview"] };
				}));
				return {
					protocol_version: options.wrongFeedback === "protocol" ? 4 : 5, operation: "compose_shape_feedback", status: "ok", region_id: request.region_id,
					card_path: path, sha256: sha(bytes), width: 400, height: 300,
					cards, card_count: cards.length, card_layout: "three_panel_row",
					reference_sha256: options.wrongFeedbackHash ? "0".repeat(64) : await hashFile(request.reference_path),
					current_render_sha256: options.wrongFeedback === "current_hash" ? "0".repeat(64) : await hashFile(request.current_render_path),
					candidates: options.wrongFeedback === "missing" ? undefined : candidates,
					target_roi: options.wrongFeedback === "roi" ? [0, 0, 500, 500] : request.target_roi, files: { card: path }, diagnostics: { target_ink_pixels: 100 },
					...(options.feedbackWarnings ? { warnings: options.feedbackWarnings } : {}),
				} as unknown as T;
			}
			throw new Error(`Unexpected operation ${request.operation}`);
		},
	};
	const tool = createShapeFeedbackTool({ workspace, lockedManifest: manifest, worker, controller, getResult: () => result, allowSections: options.allowSections ?? false });
	const invoke = (params: object): Promise<ToolResult> => (tool.execute as (...args: unknown[]) => Promise<ToolResult>)("call", params, undefined, undefined, {});
	const adoptTool = createAdoptShapeCandidateTool({ workspace, lockedManifest: manifest, controller });
	const adopt = (candidateId: string) => (adoptTool.execute as (...args: unknown[]) => Promise<unknown>)("adopt", { candidate_id: candidateId }, undefined, undefined, {});
	return { workspace, source, step, requests, invoke, adopt, manifest, description: tool.description };
}

const candidateEdits = {
	edits: [
		{ oldText: "width = 10", newText: "width = 12" },
		{ oldText: "height = 20", newText: "height = 24" },
	],
};

const singleCandidate = { candidates: [{ id: "preview", ...candidateEdits }] };
const widerCandidate = { id: "wider", edits: [{ oldText: "width = 10", newText: "width = 14" }] };
const threeCandidates = [
	{ id: "wide", ...candidateEdits },
	widerCandidate,
	{ id: "widest", edits: [{ oldText: "width = 10", newText: "width = 16" }] },
];

describe("preview_shape_feedback", () => {
	it("preserves the calibrated frame for changed formal STEP and the preview candidate", async () => {
		const f = await fixture({ fixedAnchor: true });
		const result = await f.invoke({ views: [{ region_id: "front" }, { region_id: "side" }], candidates: [widerCandidate] });
		expect(result.details.status).toBe("completed");
		expect(f.requests.some(({ operation }) => operation === "rank_drawing_views")).toBe(false);
		const projections = f.requests.filter(({ operation }) => operation === "render_fixed_projection");
		expect(projections).toHaveLength(4);
		for (const request of projections) {
			expect(request.projection_anchor_step_path).toBe("calibration.step");
			expect(request.transform).toEqual(f.manifest.regions[0]!.selected!.transform);
		}
		for (const projection of result.details.projections) {
			expect(projection.projection_anchor_step_sha256).toBe(f.manifest.step_sha256);
			expect(projection.projection_anchor_step_sha256).not.toBe(sha(f.step));
		}
		await f.adopt("wider");
		expect(await readFile(join(f.workspace, "solution.py"), "utf8")).toBe("width = 14\nheight = 20\n");
		expect(await readFile(join(f.workspace, "output.step"), "utf8")).toBe(f.step);
	});

	it("rejects a fixed projection returned from the wrong calibration anchor", async () => {
		const f = await fixture({ fixedAnchor: true, wrongAnchor: true });
		await expect(f.invoke({ views: [{ region_id: "front" }], ...singleCandidate })).rejects.toThrow("frozen code-hypothesis frame");
		expect(f.requests.some(({ operation }) => operation === "compose_shape_feedback")).toBe(false);
	});

	it("adopts the selected preview patch without executing or copying its STEP", async () => {
		const f = await fixture();
		const preview = await f.invoke({ views: [{ region_id: "front" }], candidates: [widerCandidate] });
		const callsBeforeAdoption = f.requests.length;
		await f.adopt("wider");
		expect(await readFile(join(f.workspace, "solution.py"), "utf8")).toBe("width = 14\nheight = 20\n");
		expect(await readFile(join(f.workspace, "output.step"), "utf8")).toBe(f.step);
		expect(f.requests).toHaveLength(callsBeforeAdoption);
		expect(preview.content.some(({ text }) => text?.includes("adopt_shape_candidate"))).toBe(true);
	});

	it("uses the most recent preview when candidate IDs are reused", async () => {
		const f = await fixture();
		await f.invoke({ views: [{ region_id: "front" }], ...singleCandidate });
		await f.invoke({ views: [{ region_id: "front" }], candidates: [{
			id: "preview", edits: [{ oldText: "height = 20", newText: "height = 22" }],
		}] });
		await f.adopt("preview");
		expect(await readFile(join(f.workspace, "solution.py"), "utf8")).toBe("width = 10\nheight = 22\n");
	});

	it("does not adopt an older candidate after a newer preview fails", async () => {
		const controls = { wrongFeedbackHash: false };
		const f = await fixture(controls);
		await f.invoke({ views: [{ region_id: "front" }], ...singleCandidate });
		controls.wrongFeedbackHash = true;
		await expect(f.invoke({ views: [{ region_id: "front" }], ...singleCandidate })).rejects.toThrow("mismatched input evidence");
		await expect(f.adopt("preview")).rejects.toThrow();
		expect(await readFile(join(f.workspace, "solution.py"), "utf8")).toBe(f.source);
		expect(await readFile(join(f.workspace, "output.step"), "utf8")).toBe(f.step);
	});

	it("accepts only id and edits on a candidate and rejects obsolete hypothesis input", () => {
		const params = { views: [{ region_id: "front" }], ...singleCandidate };
		expect(Value.Check(PREVIEW_SHAPE_FEEDBACK_PARAMETERS_SCHEMA, params)).toBe(true);
		expect(Value.Check(PREVIEW_SHAPE_FEEDBACK_PARAMETERS_SCHEMA, { ...params, candidates: [{ ...params.candidates[0], hypothesis: "Widen the part" }] })).toBe(false);
		expect(Value.Check(PREVIEW_SHAPE_FEEDBACK_PARAMETERS_SCHEMA, { ...params, hypothesis: "Widen the part" })).toBe(false);
		expect(Value.Check(PREVIEW_SHAPE_FEEDBACK_PARAMETERS_SCHEMA, { views: params.views, ...candidateEdits })).toBe(false);
		expect(Value.Check(PREVIEW_SHAPE_FEEDBACK_PARAMETERS_SCHEMA, { ...params, ...candidateEdits })).toBe(false);
		expect(Value.Check(PREVIEW_SHAPE_FEEDBACK_PARAMETERS_SCHEMA, { ...params, id: "old" })).toBe(false);
	});

	it("rejects old top-level hypothesis and edits before any worker request", async () => {
		const f = await fixture();
		await expect(f.invoke({ views: [{ region_id: "front" }], ...candidateEdits })).rejects.toThrow("top-level hypothesis or edits are not supported");
		expect(f.requests).toHaveLength(0);
	});

	it.each([
		["no candidates", [], "exactly one candidate"],
		["two candidates", threeCandidates.slice(0, 2), "exactly one candidate"],
		["three candidates", threeCandidates, "exactly one candidate"],
		["duplicate IDs", [{ id: "same", ...candidateEdits }, { id: "same", ...candidateEdits }], "exactly one candidate"],
		["path traversal", [{ id: "../escape", ...candidateEdits }], "Candidate ID must start"],
		["absolute path", [{ id: "/escape", ...candidateEdits }], "Candidate ID must start"],
		["empty ID", [{ id: "", ...candidateEdits }], "Candidate ID must start"],
		["long ID", [{ id: "a".repeat(65), ...candidateEdits }], "Candidate ID must start"],
		["no edits", [{ id: "bad", ...candidateEdits, edits: [] }], "one to eight edits"],
		["too many edits", [{ id: "bad", ...candidateEdits, edits: Array.from({ length: 9 }, () => candidateEdits.edits[0]!) }], "one to eight edits"],
		["obsolete hypothesis", [{ id: "bad", ...candidateEdits, hypothesis: "Widen the part" }], "only id and edits"],
	] as const)("rejects %s before any worker request", async (_label, candidates, error) => {
		const f = await fixture();
		const filesBefore = await readdir(f.workspace);
		await expect(f.invoke({ views: [{ region_id: "front" }], candidates })).rejects.toThrow(error);
		expect(f.requests).toHaveLength(0);
		expect(await readdir(f.workspace)).toEqual(filesBefore);
		expect(await readFile(join(f.workspace, "solution.py"), "utf8")).toBe(f.source);
		expect(await readFile(join(f.workspace, "output.step"), "utf8")).toBe(f.step);
	});

	it("enforces candidate and edit limits in the tool schema", () => {
		const params = { views: [{ region_id: "front" }], ...singleCandidate };
		expect(Value.Check(PREVIEW_SHAPE_FEEDBACK_PARAMETERS_SCHEMA, params)).toBe(true);
		for (const candidates of [[], threeCandidates.slice(0, 2), threeCandidates, [{ id: "../escape", ...candidateEdits }], [{ id: "x", ...candidateEdits, edits: [] }], [{ id: "x", ...candidateEdits, edits: Array.from({ length: 9 }, () => candidateEdits.edits[0]!) }]]) {
			expect(Value.Check(PREVIEW_SHAPE_FEEDBACK_PARAMETERS_SCHEMA, { ...params, candidates })).toBe(false);
		}
	});

	it.each(["label", "camera"] as const)("excludes section views identified by %s and rejects them before any worker request", async (sectionMode) => {
		const f = await fixture({ sectionMode });
		expect(f.description).toContain("Available region_id values: front.");
		await expect(f.invoke({ views: [{ region_id: "side" }], ...singleCandidate })).rejects.toThrow("Section view side is unsupported");
		expect(f.requests).toHaveLength(0);
	});

	it("previews a calibrated physical section when enabled, preserving Current and Preview's cut plane", async () => {
		const f = await fixture({ fixedAnchor: true, sectionMode: "camera", allowSections: true });
		expect(f.description).toContain("Available region_id values: front, side.");
		const result = await f.invoke({ views: [{ region_id: "side" }], ...singleCandidate });
		expect(result.details.status).toBe("completed");
		expect(result.details.frames[0]!.sectionOrigin).toEqual([0, 0, 0]);
		const projections = f.requests.filter(({ operation }) => operation === "render_fixed_projection");
		expect(projections).toHaveLength(2);
		for (const projection of projections) {
			expect(projection.projection_anchor_step_path).toBe("calibration.step");
			expect(projection.view_spec).toMatchObject({ section_offset_fraction: 0, section_keep: "front" });
		}
		expect(result.content.filter(({ type }) => type === "image")).toHaveLength(1);
	});

	it("shows empty-view warnings without exposing obsolete ROI terminology and retains original diagnostics", async () => {
		const warnings = ["Current CAD projection is empty in this ROI.", "Candidate CAD projection is empty in this ROI.", "Target ROI contains no ink; do not interpret gray as no change."];
		const f = await fixture({ feedbackWarnings: warnings });
		const result = await f.invoke({ views: [{ region_id: "front" }], ...singleCandidate });
		const cardText = result.content.find(({ type, text }) => type === "text" && text?.startsWith("View: front"))?.text;
		for (const warning of [
			"Current CAD projection is empty in this view.",
			"Candidate CAD projection is empty in this view.",
			"Target view contains no ink; do not interpret gray as no change.",
		]) expect(cardText).toContain(warning);
		expect(cardText).not.toMatch(/\bROI\b/u);
		expect(result.details.feedback[0]!.warnings).toEqual(warnings);
		expect(result.content.filter(({ type }) => type === "image")).toHaveLength(1);
	});

	it("describes separate Target, Current and Preview projections", async () => {
		const f = await fixture();
		const result = await f.invoke({ views: [{ region_id: "front" }], ...singleCandidate });
		const text = result.content.filter(({ type }) => type === "text").map(({ text }) => text).join("\n");
		for (const phrase of ["Left to right: Target | Current | Preview", "Current and Preview are separate monochrome CAD 2D projections"]) {
			expect(f.description).toContain(phrase);
			expect(text).toContain(phrase);
		}
		expect(text).toContain("Candidate: preview");
		expect(f.description).toContain("exactly one candidate");
		expect(f.description).not.toContain("hypothesis");
		expect(f.description).toContain("Images contain only the three");
		expect(f.description).toContain("never from a previous unadopted preview");
		expect(f.description).toContain("Then call execute_build123d");
		expect(f.description).not.toContain("one to three candidate");
		expect(text).toContain("Left to right: Target | Current | Preview.");
		expect(text).not.toMatch(/BRep|material changes|added material|removed material|contour feedback|yellow/);
		for (const request of f.requests.filter(({ operation }) => operation === "compose_shape_feedback")) {
			for (const field of ["current_step_path", "view_spec", "transform", "projection_origin"]) expect(request).not.toHaveProperty(field);
			for (const candidate of request.candidates as object[]) expect(candidate).not.toHaveProperty("step_path");
		}
		expect(result.details.status).toBe("completed");
		expect(result.content.filter(({ type }) => type === "image")).toHaveLength(1);
		await f.adopt("preview");
		expect(await readFile(join(f.workspace, "solution.py"), "utf8")).toBe("width = 12\nheight = 24\n");
		expect(await readFile(join(f.workspace, "output.step"), "utf8")).toBe(f.step);
	});

	it("executes one multi-edit candidate once and shares its STEP across frozen views without changing formal files", async () => {
		const f = await fixture();
		expect(f.description).toContain("- Target: the original engineering-drawing view.");
		expect(f.description).toContain("- Current: the current CAD's 2D projection.");
		expect(f.description).toContain("- Preview: the modified CAD's 2D projection.");
		expect(f.description).not.toContain("Target-line support");
		const result = await f.invoke({ views: [{ region_id: "front" }, { region_id: "side" }], ...singleCandidate });
		expect(f.requests.filter(({ operation }) => operation === "execute_repair_candidate")).toHaveLength(1);
		expect(result.details.candidates).toHaveLength(1);
		expect(result.details.candidates[0]).toMatchObject({ id: "preview", edits: candidateEdits.edits });
		expect(result.details.candidates[0]).not.toHaveProperty("hypothesis");
		const topologyRequests = f.requests.filter(({ operation }) => operation === "compare_candidate_topology");
		expect(topologyRequests).toHaveLength(1);
		expect(topologyRequests[0]!.timeout_s).toBe(150);
		expect(topologyRequests[0]!.candidates).toEqual(result.details.candidates.map((candidate) => ({ candidate_id: candidate.id, step_path: candidate.execution!.artifact.path })));
		const topologyIndex = f.requests.indexOf(topologyRequests[0]!);
		expect(f.requests.slice(0, topologyIndex).filter(({ operation }) => operation === "execute_repair_candidate")).toHaveLength(1);
		expect(f.requests.slice(0, topologyIndex).some(({ operation }) => operation === "render_fixed_projection")).toBe(false);
		expect(f.requests.filter(({ operation }) => operation === "render_fixed_projection")).toHaveLength(4);
		const compositions = f.requests.filter(({ operation }) => operation === "compose_shape_feedback");
		expect(compositions).toHaveLength(2);
		expect(compositions.map(({ candidates }) => (candidates as Array<{ candidate_id: string }>).map(({ candidate_id }) => candidate_id))).toEqual([["preview"], ["preview"]]);
		expect(compositions.map(({ target_roi }) => target_roi)).toEqual([[0, 0, 1000, 1000], [0, 0, 1000, 1000]]);
		for (const request of f.requests.filter(({ operation }) => operation === "render_fixed_projection")) expect(request.projection_anchor_step_path).toBe("output.step");
		expect(result.content.filter(({ type }) => type === "image")).toHaveLength(2);
		expect(result.details.status).toBe("completed");
		expect(new Set(result.details.projections.map(({ step_sha256 }) => step_sha256)).size).toBe(1);
		const audit = JSON.parse(await readFile(join(f.workspace, result.details.audit_path), "utf8"));
		expect(audit.feedback).toHaveLength(2);
		expect(audit.topology).toEqual(result.details.topology);
		const text = result.content.filter(({ type }) => type === "text").map(({ text }) => text).join("\n");
		expect(text).toContain("Preview generated. Views returned: 2/2.");
		expect(text).toContain("Left to right: Target | Current | Preview.");
		expect(text).toContain("One three-panel card per view: Target | Current | Preview.");
		expect(text).not.toContain("Overlays:");
		expect(text).toContain("All panels show the complete aligned input frame, fitted with one shared display scale.");
		expect(text).not.toContain("Target-line support change");
		expect(text).not.toContain("Change map:");
		expect(text).not.toMatch(/Target \+|Top row|Bottom row|dark gray|blue =/);
		expect(text).not.toMatch(/\b(?:b[012]|topology|ROI)\b/iu);
		const topologyText = formatShapeTopologyFeedback(audit.topology);
		expect(topologyText).toContain("Current: b0=2, b1=4, b2=1.");
		expect(topologyText).toContain("Preview Candidate preview: b0=1, b1=6, b2=0; delta (b0=-1, b1=+2, b2=-1).");
		expect(audit.candidates[0]).not.toHaveProperty("hypothesis");
		expect(audit.candidates[0].patch).toContain("+width = 12");
		expect(audit.candidates[0].patch).toContain("+height = 24");
		expect(await readFile(join(f.workspace, audit.candidates[0].source_path), "utf8")).toBe("width = 12\nheight = 24\n");
		expect(await readFile(join(f.workspace, "solution.py"), "utf8")).toBe(f.source);
		expect(await readFile(join(f.workspace, "output.step"), "utf8")).toBe(f.step);
	});

	it("reports an invalid isolated candidate without attempting topology or projections", async () => {
		const f = await fixture({ invalidCandidate: true });
		const result = await f.invoke({ views: [{ region_id: "front" }], ...singleCandidate });
		expect(result.details.status).toBe("partial");
		expect(result.details.candidates.map(({ id, status }) => ({ id, status }))).toEqual([{ id: "preview", status: "invalid" }]);
		expect(result.details.feedback).toHaveLength(0);
		expect(result.details.topology).toBeNull();
		expect(f.requests.some(({ operation }) => operation === "compare_candidate_topology" || operation === "render_fixed_projection")).toBe(false);
		expect(result.content.filter(({ type }) => type === "image")).toHaveLength(0);
		const text = result.content.filter(({ type }) => type === "text").map(({ text }) => text).join("\n");
		expect(text).toContain("Preview unavailable. No view images returned.");
		expect(text).toContain("invalid geometry");
		expect(text).not.toContain("Image panels,");
		expect(text).not.toContain("Overlays:");
		expect(text).not.toContain("Change map:");
		expect(await readFile(join(f.workspace, "solution.py"), "utf8")).toBe(f.source);
		expect(await readFile(join(f.workspace, "output.step"), "utf8")).toBe(f.step);
	});

	it.each([1, 2, 3])("returns one three-panel image for each of %i views of one candidate", async (viewCount) => {
		const regionIds = ["front", "side", "top"].slice(0, viewCount);
		const f = await fixture({ regionIds });
		const params = { views: regionIds.map((region_id) => ({ region_id })), ...singleCandidate };
		expect(Value.Check(PREVIEW_SHAPE_FEEDBACK_PARAMETERS_SCHEMA, params)).toBe(true);
		const result = await f.invoke(params);
		const executions = f.requests.filter(({ operation }) => operation === "execute_repair_candidate");
		expect(executions).toHaveLength(1);
		expect(await Promise.all(result.details.candidates.map(({ source_path }) => readFile(join(f.workspace, source_path), "utf8")))).toEqual([
			"width = 12\nheight = 24\n",
		]);
		for (const execution of executions) {
			expect(execution.formal_source_sha256).toBe(sha(f.source));
			expect(execution.formal_step_sha256).toBe(sha(f.step));
		}
		const renders = f.requests.filter(({ operation }) => operation === "render_fixed_projection");
		expect(renders.filter(({ step_path }) => step_path === "output.step")).toHaveLength(viewCount);
		expect(renders.filter(({ step_path }) => step_path !== "output.step")).toHaveLength(viewCount);
		const compositions = f.requests.filter(({ operation }) => operation === "compose_shape_feedback");
		expect(compositions).toHaveLength(viewCount);
		for (const composition of compositions) {
			const projections = result.details.projections.filter(({ region_id }) => region_id === composition.region_id);
			expect(composition.candidates).toEqual(projections.map(({ candidate_id, render_path }) => ({ candidate_id, render_path, hypothesis: "" })));
			expect(composition).not.toHaveProperty("candidate_id");
			expect(composition).not.toHaveProperty("candidate_render_path");
			expect(f.requests.slice(0, f.requests.indexOf(composition)).filter(({ operation, region_id }) => operation === "render_fixed_projection" && region_id === composition.region_id)).toHaveLength(2);
			const frame = result.details.frames.find(({ region_id }) => region_id === composition.region_id)!;
			expect(composition.current_render_path).toBe(frame.currentRenderPath);
			for (const rendered of renders.filter(({ region_id }) => region_id === composition.region_id)) {
				expect(rendered.projection_anchor_step_path).toBe("output.step");
				expect(rendered.view_spec).toEqual(frame.viewSpec);
				expect(rendered.transform).toEqual(frame.transform);
			}
			for (const projection of projections) {
				expect(projection.camera_sha256).toBe(frame.cameraSha256);
				expect(projection.transform_sha256).toBe(frame.transformSha256);
				expect(projection.projection_origin).toEqual(frame.projectionOrigin);
			}
		}
		expect(result.details.feedback).toHaveLength(viewCount);
		expect(result.content.filter(({ type }) => type === "image")).toHaveLength(viewCount);
		expect(result.details.feedback.map(({ cards }) => cards.map(({ candidate_id }) => candidate_id))).toEqual(regionIds.map(() => ["preview"]));
		expect(result.details.feedback.map(({ candidates }) => candidates.map(({ candidate_id }) => candidate_id))).toEqual(regionIds.map(() => ["preview"]));
		expect(result.details.status).toBe("completed");
		const text = result.content.filter(({ type }) => type === "text").map(({ text }) => text).join("\n");
		expect(text).toContain("Candidate: preview");
		expect(text).toContain("Left to right: Target | Current | Preview.");
		expect(text).not.toMatch(/\b(?:b[012]|topology|ROI)\b/iu);
		expect(text).not.toContain("Overlays:");
		expect(f.description).toContain("call adopt_shape_candidate");
		expect(f.description).toContain("applies the saved code replacements to solution.py");
		expect(await readFile(join(f.workspace, "solution.py"), "utf8")).toBe(f.source);
		expect(await readFile(join(f.workspace, "output.step"), "utf8")).toBe(f.step);
	});

	it("keeps a successful view when the same candidate fails projection in another view", async () => {
		const f = await fixture({ projectionFailureRegions: ["side"] });
		const result = await f.invoke({ views: [{ region_id: "front" }, { region_id: "side" }], ...singleCandidate });
		expect(result.details.status).toBe("partial");
		expect(result.details.candidates.map(({ id, status }) => ({ id, status }))).toEqual([
			{ id: "preview", status: "valid" },
		]);
		expect(result.details.feedback).toHaveLength(1);
		expect(result.details.feedback[0]!.region_id).toBe("front");
		expect(result.details.feedback[0]!.candidates.map(({ candidate_id }) => candidate_id)).toEqual(["preview"]);
		expect(result.content.filter(({ type }) => type === "image")).toHaveLength(1);
		expect(f.requests.filter(({ operation }) => operation === "compose_shape_feedback").map(({ region_id }) => region_id)).toEqual(["front"]);
		const topologyRequest = f.requests.find(({ operation }) => operation === "compare_candidate_topology")!;
		expect((topologyRequest.candidates as Array<{ candidate_id: string }>).map(({ candidate_id }) => candidate_id)).toEqual(["preview"]);
		const text = result.content.filter(({ type }) => type === "text").map(({ text }) => text).join("\n");
		expect(text).toContain("Preview incomplete. Views returned: 1/2.");
		expect(text).toContain("preview/side: Candidate projection failed");
		for (const block of result.content.filter(({ text }) => text?.startsWith("View:"))) expect(block.text).toContain("View: front | Candidate: preview");
		expect(await readFile(join(f.workspace, "solution.py"), "utf8")).toBe(f.source);
		expect(await readFile(join(f.workspace, "output.step"), "utf8")).toBe(f.step);
	});

	it("does not compose a card when all candidate projections fail", async () => {
		const f = await fixture({ projectionFailureIds: ["preview"] });
		const result = await f.invoke({ views: [{ region_id: "front" }], ...singleCandidate });
		expect(result.details.status).toBe("partial");
		expect(result.details.feedback).toHaveLength(0);
		expect(f.requests.some(({ operation }) => operation === "compose_shape_feedback")).toBe(false);
	});

	it("retains zero topology delta in audit evidence without adding it to model-visible feedback", async () => {
		const f = await fixture({ topologyUnchanged: true });
		const result = await f.invoke({ views: [{ region_id: "front" }], ...singleCandidate });
		const text = result.content.filter(({ type }) => type === "text").map(({ text }) => text).join("\n");
		expect(text).not.toMatch(/\b(?:b[012]|topology|ROI)\b/iu);
		const audit = JSON.parse(await readFile(join(f.workspace, result.details.audit_path), "utf8"));
		expect(audit.topology).toEqual(result.details.topology);
		expect(audit.topology.candidates[0].topology_delta).toEqual({ b0: 0, b1: 0, b2: 0 });
		const topologyText = formatShapeTopologyFeedback(audit.topology);
		expect(topologyText).toContain("Preview Candidate preview: b0=2, b1=4, b2=1; delta (b0=+0, b1=+0, b2=+0).");
		expect(topologyText).toContain("Zero deltas mean unchanged topology counts; geometry may still change.");
		expect(result.content.filter(({ type }) => type === "image")).toHaveLength(1);
	});

	it.each(["current_hash", "candidate_hash", "id", "count", "delta", "protocol"] as const)("rejects mismatched topology %s evidence before rendering", async (wrongTopology) => {
		const f = await fixture({ wrongTopology });
		await expect(f.invoke({ views: [{ region_id: "front" }], ...singleCandidate })).rejects.toThrow(/Topology (returned mismatched|delta disagrees|response violated)/);
		expect(f.requests.some(({ operation }) => operation === "render_fixed_projection")).toBe(false);
		expect(await readFile(join(f.workspace, "output.step"), "utf8")).toBe(f.step);
	});

	it("degrades a rejected topology worker request to unavailable evidence and still returns 2D cards", async () => {
		const f = await fixture({ topologyRejection: true });
		const result = await f.invoke({ views: [{ region_id: "front" }, { region_id: "side" }], ...singleCandidate });
		expect(result.details.status).toBe("partial");
		expect(result.content.filter(({ type }) => type === "image")).toHaveLength(2);
		expect(result.details.topology).toMatchObject({ status: "error", error: { message: "Topology worker request failed: CAD worker returned malformed JSON" } });
		const text = result.content.filter(({ type }) => type === "text").map(({ text }) => text).join("\n");
		expect(text).toContain("Preview generated. Views returned: 2/2.");
		expect(text).not.toMatch(/\b(?:b[012]|topology|ROI)\b/iu);
		expect(result.details.errors).toContain("STEP topology unavailable: Topology worker request failed: CAD worker returned malformed JSON");
		expect(await readFile(join(f.workspace, "output.step"), "utf8")).toBe(f.step);
	});

	it("propagates user cancellation of topology without starting 2D rendering", async () => {
		const f = await fixture({ abortTopology: true });
		await expect(f.invoke({ views: [{ region_id: "front" }], ...singleCandidate })).rejects.toThrow("CAD worker was aborted");
		expect(f.requests.some(({ operation }) => operation === "render_fixed_projection")).toBe(false);
		expect(await readFile(join(f.workspace, "output.step"), "utf8")).toBe(f.step);
	});

	it.each(["current", "preview", "timeout"] as const)("keeps 2D cards and audits unavailable %s topology without exposing it or fabricating delta", async (topologyFailure) => {
		const f = await fixture({ topologyFailure });
		const result = await f.invoke({ views: [{ region_id: "front" }, { region_id: "side" }], ...singleCandidate });
		expect(result.details.status).toBe("partial");
		expect(result.content.filter(({ type }) => type === "image")).toHaveLength(2);
		expect(f.requests.filter(({ operation }) => operation === "compare_candidate_topology")).toHaveLength(1);
		const text = result.content.filter(({ type }) => type === "text").map(({ text }) => text).join("\n");
		expect(text).toContain("Preview generated. Views returned: 2/2.");
		expect(text).not.toMatch(/\b(?:b[012]|topology|ROI)\b/iu);
		expect(result.details.errors.length).toBeGreaterThan(0);
		const topologyText = formatShapeTopologyFeedback(result.details.topology!);
		expect(topologyText).not.toContain("delta (b0=");
		if (topologyFailure === "current") {
			expect(topologyText).toContain("Current topology unavailable: Current mesh is not closed");
			expect(topologyText).toContain("Preview Candidate preview: b0=1, b1=6, b2=0; delta unavailable: Current topology unavailable");
		} else if (topologyFailure === "preview") {
			expect(topologyText).toContain("Preview Candidate preview: topology and delta unavailable: Candidate mesh is not closed");
			expect(topologyText).toContain("Current: b0=2, b1=4, b2=1");
		} else {
			expect(topologyText).toContain("STEP topology unavailable: Topology computation timed out");
		}
	});

	it("rejects an origin change before composing misleading feedback", async () => {
		const f = await fixture({ driftOrigin: true });
		await expect(f.invoke({ views: [{ region_id: "front" }], ...singleCandidate })).rejects.toThrow("drifted from frozen view");
		expect(f.requests.some(({ operation }) => operation === "compose_shape_feedback")).toBe(false);
		expect(await readFile(join(f.workspace, "output.step"), "utf8")).toBe(f.step);
	});

	it("rejects feedback computed from different input evidence", async () => {
		const f = await fixture({ wrongFeedbackHash: true });
		await expect(f.invoke({ views: [{ region_id: "front" }], ...singleCandidate })).rejects.toThrow("mismatched input evidence");
	});

	it.each(["current_hash", "candidate_hash", "id", "count", "duplicate", "missing", "roi", "protocol"] as const)("rejects feedback with mismatched %s evidence", async (wrongFeedback) => {
		const f = await fixture({ wrongFeedback });
		await expect(f.invoke({ views: [{ region_id: "front" }], ...singleCandidate })).rejects.toThrow("mismatched input evidence");
		expect(await readFile(join(f.workspace, "solution.py"), "utf8")).toBe(f.source);
		expect(await readFile(join(f.workspace, "output.step"), "utf8")).toBe(f.step);
	});

	it("rejects obsolete ROI inputs and empty edit anchors before any worker request", async () => {
		const f = await fixture();
		for (const target_roi of [
			{ x_min: 0, y_min: 0, x_max: 1000, y_max: 1000 },
			{ x_min: 0, y_min: 0, x_max: 10, y_max: 10 },
		]) {
			const params = { views: [{ region_id: "front", target_roi }], ...singleCandidate };
			expect(Value.Check(PREVIEW_SHAPE_FEEDBACK_PARAMETERS_SCHEMA, params)).toBe(false);
			await expect(f.invoke(params)).rejects.toThrow(/region_id|target_roi/iu);
		}
		await expect(f.invoke({ views: [{ region_id: "front" }], candidates: [{ id: "preview", edits: [{ oldText: "", newText: "x" }] }] })).rejects.toThrow("non-empty oldText");
		expect(f.requests).toHaveLength(0);
	});
});
