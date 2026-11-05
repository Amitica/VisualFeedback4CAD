import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import type { WorkerClient, WorkerRequest } from "../src/local-worker.js";
import { RunController } from "../src/runner.js";
import {
	applyPiStyleEdits,
	createVisualReviewTool,
	restoreVisualReviewState,
	VISUALIZE_CODE_HYPOTHESIS_PARAMETERS_SCHEMA,
} from "../src/visual-review.js";
import {
	VISUAL_REVIEW_PROFILE,
	type AgentResult,
	type DrawingViewManifest,
	type WorkerResponse,
} from "../src/types.js";
import { makePng } from "./png-fixture.js";

interface TestTool {
	name: string;
	label: string;
	description: string;
	execute: (...arguments_: unknown[]) => Promise<unknown>;
}

const roots: string[] = [];

afterEach(async () => {
	for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true });
});

function sha(value: Buffer | string): string {
	return createHash("sha256").update(value).digest("hex");
}

describe("visualize_code_hypothesis", () => {
	it("uses Pi-compatible exact replacements and rejects ambiguity or overlap", () => {
		expect(applyPiStyleEdits("a = 1\nb = 2\n", [
			{ oldText: "a = 1", newText: "a = 3" },
			{ oldText: "b = 2", newText: "b = 4" },
		])).toBe("a = 3\nb = 4\n");
		expect(() => applyPiStyleEdits("x\nx\n", [{ oldText: "x", newText: "y" }])).toThrow("found 2");
		expect(() => applyPiStyleEdits("abcdef", [
			{ oldText: "abc", newText: "x" },
			{ oldText: "bc", newText: "y" },
		])).toThrow("overlap");
	});

	it("executes an isolated edit and returns a render, exact patch, and B-rep facts", async () => {
		const workspace = await mkdtemp(join(tmpdir(), "code-hypothesis-"));
		roots.push(workspace);
		const source = "part = make(10)\nexport(part)\n";
		const sourceSha256 = sha(source);
		const step = Buffer.from("formal-step");
		const stepSha256 = sha(step);
		const png = makePng(200, 160, 4);
		await Promise.all([
			writeFile(join(workspace, "solution.py"), source),
			writeFile(join(workspace, "output.step"), step),
			writeFile(join(workspace, "reference.png"), png),
			writeFile(join(workspace, "current.png"), png),
		]);
		const artifact = {
			status: "valid" as const,
			path: "output.step",
			sha256: stepSha256,
			validation: {
				is_valid: true,
				is_watertight: true,
				topology_errors: [],
				solid_count: 1,
				shell_count: 1,
				face_count: 6,
				volume: 1,
				bbox: {},
			},
			validation_error: null,
			render_path: null,
			render_error: null,
		};
		const result = {
			fixture_id: "103",
			status: "valid",
			agent_mode: "base-only",
			provider: "test",
			model: "test",
			thinking_level: "high",
			pi_version: "0.84.1",
			container_image: "test",
			container_image_digest: "sha256:test",
			started_at: "2026-09-01T00:00:00.000Z",
			updated_at: "2026-09-01T00:00:00.000Z",
			duration_s: 0,
			last_turn: 1,
			artifact,
			last_executed_source_sha256: sourceSha256,
			last_executed_step_sha256: stepSha256,
			tokens: { input: 0, output: 0, cache_read: 0, cache_write: 0, total: 0 },
			cost: 0,
			tool_calls: 0,
		} satisfies AgentResult;
		const transform = {
			scale_px_per_unit: 1,
			rotation_deg: 0,
			center_x_px: 100,
			center_y_px: 80,
			model_center_x: 0,
			model_center_y: 0,
			reflected: false,
		};
		const viewSpec = { name: "iso", direction: [-1, -1, 1], up: [1, 1, 2] };
		const manifest = {
			version: 4,
			scorer_profile: "capacity_transport_v1",
			input_sha256: "a".repeat(64),
			source_sha256: sourceSha256,
			step_sha256: stepSha256,
			regions: [{
				region_id: "iso",
				page: 0,
				roi: [0, 0, 1000, 1000],
				crop_path: "reference.png",
				view_type: "isometric",
				status: "matched",
				selected: {
					candidate_id: "candidate-001",
					view_spec: viewSpec,
					transform,
					aligned_render_path: "current.png",
					comparison_path: "current.png",
				},
				alternatives: [],
				margin: 1,
				confidence: "high",
				note: "",
			}],
		} as unknown as DrawingViewManifest;
		const state = await restoreVisualReviewState({
			workspace,
			fixtureId: "103",
			profile: VISUAL_REVIEW_PROFILE,
			lockedViewManifestSha256: "e".repeat(64),
		});
		const candidateStep = Buffer.from("candidate-step");
		const worker: WorkerClient = {
			async request<T extends WorkerResponse>(_workspace: string, request: WorkerRequest): Promise<T> {
				if (request.operation === "inspect") {
					return { protocol_version: 5, operation: "inspect", artifact } as unknown as T;
				}
				if (request.operation === "execute_repair_candidate") {
					const candidateSource = await readFile(join(workspace, String(request.source_path)), "utf8");
					const candidatePath = join(String(request.output_dir), "output.step");
					await mkdir(dirname(join(workspace, candidatePath)), { recursive: true });
					await writeFile(join(workspace, candidatePath), candidateStep);
					return {
						protocol_version: 5,
						operation: "execute_repair_candidate",
						candidate_id: String(request.candidate_id),
						source_path: String(request.source_path),
						source_sha256: sha(candidateSource),
						formal_source_sha256: sourceSha256,
						formal_step_sha256: stepSha256,
						formal_unchanged: true,
						execution: { success: true, exit_code: 0, timed_out: false, stdout: "", stderr: "", files_produced: {}, duration_s: 1 },
						artifact: { ...artifact, path: candidatePath, sha256: sha(candidateStep) },
					} as unknown as T;
				}
				if (request.operation === "render_fixed_projection") {
					const renderPath = join(String(request.output_dir), "aligned-render.png");
					await mkdir(dirname(join(workspace, renderPath)), { recursive: true });
					await writeFile(join(workspace, renderPath), png);
					const renderedStepSha256 = request.step_path === "output.step" ? stepSha256 : sha(candidateStep);
					return {
						protocol_version: 5,
						operation: "render_fixed_projection",
						status: "ok",
						region_id: "iso",
						step_path: request.step_path,
						step_sha256: renderedStepSha256,
						projection_anchor_step_sha256: stepSha256,
						view_spec: viewSpec,
						transform,
						projection_origin: [0, 0, 0],
						render_path: renderPath,
						width: 200,
						height: 160,
						mime_type: "image/png",
						sha256: sha(png),
					} as T;
				}
				if (request.operation === "compose_code_edit_visualization") {
					const card = makePng(500, 300, 8);
					const cardPath = join(String(request.output_dir), "code-hypothesis.png");
					await mkdir(dirname(join(workspace, cardPath)), { recursive: true });
					await writeFile(join(workspace, cardPath), card);
					return {
						protocol_version: 5,
						operation: "compose_code_edit_visualization",
						status: "ok",
						region_id: "iso",
						hypothesis_id: request.hypothesis_id,
						target_roi: request.target_roi,
						target_pixel_roi: [20, 20, 100, 100],
						visual_encoding: {
							current_cad: "gray",
							removed_by_proposal: "red",
							added_by_proposal: "green",
						},
						geometry: [{
							relation: "added_by_proposal",
							solid_count: 1,
							face_count: 6,
							edge_count: 12,
							volume_mm3: 10,
							surface_area_mm2: 20,
							bbox_mm: { x: 1, y: 2, z: 3, x_min: 0, x_max: 1, y_min: 0, y_max: 2, z_min: 0, z_max: 3 },
							centroid_mm: [0.5, 1, 1.5],
							surface_types: ["plane"],
						}],
						card_path: cardPath,
						width: 500,
						height: 300,
						mime_type: "image/png",
						sha256: sha(card),
					} as T;
				}
				throw new Error(`Unexpected operation ${request.operation}`);
			},
		};
		const tool = createVisualReviewTool({
			workspace,
			state,
			lockedManifest: manifest,
			worker,
			controller: new RunController(),
			getResult: () => result,
		}) as unknown as TestTool;
		expect(tool.name).toBe("visualize_code_hypothesis");
		expect(tool.label).toBe("Preview CAD edit");
		expect(tool.description).toBe("Preview how exact edits to solution.py change the current CAD beside a selected Target drawing ROI. Each edits[].oldText must match exactly once in the current file, and edits must not overlap. The edits run only in an isolated copy. Returns a two-panel Target ROI / Edit Preview image showing visible line changes in one fixed view, together with the bounding-box size and topology counts of 3D geometry added or removed by the edits. The formal solution.py and output.step remain unchanged.");
		expect(Object.keys(VISUALIZE_CODE_HYPOTHESIS_PARAMETERS_SCHEMA.properties)).toEqual(["region_id", "target_roi", "edits"]);
		const response = await tool.execute("call", {
			region_id: "iso",
			target_roi: { x_min: 100, y_min: 100, x_max: 600, y_max: 700 },
			edits: [{ oldText: "make(10)", newText: "make(12)" }],
		}, undefined, undefined, {}) as {
			content: Array<{ type: string; text?: string }>;
			details: { hypothesis: { epistemic_status: string; edits: Array<{ oldText: string; newText: string }>; patch: string } };
		};
		expect(response.content.map(({ type }) => type)).toEqual(["text", "image"]);
		expect(response.content[0]?.text).toContain("Edit preview generated.");
		expect(response.content[0]?.text).toContain("The left panel shows the selected region of the Target drawing.");
		expect(response.content[0]?.text).toContain("gray = visible CAD lines before editing; green = visible lines added by the code edits; red = visible lines removed by the code edits");
		expect(response.content[0]?.text).toContain("Added 3D geometry: XYZ bounding-box size = 1.0 × 2.0 × 3.0 mm. This added geometry consists of 1 solid with 6 faces and 12 edges.");
		expect(response.content[0]?.text).toContain("Removed 3D geometry: none.");
		expect(response.content[0]?.text).not.toContain("hypothesis_id");
		expect(response.content[0]?.text).not.toContain("unified_patch");
		expect(response.details.hypothesis.epistemic_status).toBe("unverified_hypothesis");
		expect(response.details.hypothesis.edits).toEqual([{ oldText: "make(10)", newText: "make(12)" }]);
		expect(response.details.hypothesis.patch).toContain("-part = make(10)");
		expect(response.details.hypothesis.patch).toContain("+part = make(12)");
		expect(JSON.stringify(response.details)).not.toContain("confidence");
		expect(await readFile(join(workspace, "solution.py"), "utf8")).toBe(source);
		expect(await readFile(join(workspace, "output.step"))).toEqual(step);
	});
});
