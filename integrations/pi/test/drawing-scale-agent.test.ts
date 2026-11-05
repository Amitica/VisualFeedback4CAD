import { createHash } from "node:crypto";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import {
	createDrawingScaleAgentTools,
	DRAWING_SCALE_AGENT_SYSTEM_PROMPT,
	DRAWING_SCALE_AGENT_TOOLS,
	drawingScaleAgentPrompt,
	type DrawingScaleAgentState,
} from "../src/drawing-scale-agent.js";
import type { WorkerClient, WorkerRequest } from "../src/local-worker.js";
import { RunController } from "../src/runner.js";
import {
	PROTOCOL_VERSION,
	type ArtifactPayload,
	type DrawingViewManifest,
	type WorkerResponse,
} from "../src/types.js";
import { makePng } from "./png-fixture.js";

interface TestTool {
	name: string;
	execute: (...arguments_: unknown[]) => Promise<unknown>;
}

const workspaces: string[] = [];

afterEach(async () => {
	for (const workspace of workspaces.splice(0)) await rm(workspace, { recursive: true, force: true });
});

function digest(value: Buffer | string): string {
	return createHash("sha256").update(value).digest("hex");
}

async function writePng(workspace: string, path: string, marker: number): Promise<string> {
	const bytes = makePng(320, 240, marker);
	await mkdir(dirname(join(workspace, path)), { recursive: true });
	await writeFile(join(workspace, path), bytes);
	return digest(bytes);
}

function artifact(stepSha256: string): ArtifactPayload {
	return {
		status: "valid",
		path: "output.step",
		sha256: stepSha256,
		validation: {
			is_valid: true,
			is_watertight: true,
			topology_errors: [],
			solid_count: 1,
			shell_count: 1,
			face_count: 12,
			volume: 10,
			bbox: { x: 100, y: 60, z: 20 },
		},
		validation_error: null,
		render_path: null,
		render_error: null,
	};
}

async function setup() {
	const workspace = await mkdtemp(join(tmpdir(), "drawing-scale-agent-"));
	workspaces.push(workspace);
	await writeFile(join(workspace, "output.step"), "step\n");
	const stepSha = digest("step\n");
	for (const [index, regionId] of ["front", "side"].entries()) {
		await writePng(workspace, `evidence/${regionId}.png`, 10 + index);
	}
	const regions = ["front", "side"].map((regionId, index) => ({
		region_id: regionId,
		page: 0,
		roi: [0, 0, 500, 500] as [number, number, number, number],
		crop_path: `evidence/${regionId}.png`,
		view_type: "orthographic" as const,
		status: "matched" as const,
		selected: {
			candidate_id: `candidate-${index}`,
			render_path: "unused.png",
			comparison_path: "unused.png",
			aligned_render_path: "unused.png",
			view_spec: { name: regionId, direction: [0, 0, 1] as [number, number, number], up: [0, 1, 0] as [number, number, number] },
			score: 1,
			raw_score: 1,
			eligible: true,
			failed_gates: [],
			scoring_mode: "capacity_transport" as const,
			capacity_transport: {},
			transform: {
				scale_px_per_unit: index + 1,
				rotation_deg: 0,
				center_x_px: 100,
				center_y_px: 100,
				model_center_x: 0,
				model_center_y: 0,
				reflected: false,
			},
			alignment: {
				objective: 0,
				mode: "capacity_transport" as const,
				scale_at_bound: false,
				rotation_at_limit: false,
				translation_at_limit: false,
			},
		},
		alternatives: [],
		margin: null,
		confidence: "high" as const,
		note: "test",
		width: 320,
		height: 240,
		crop_sha256: "a".repeat(64),
	}));
	const manifest = {
		version: 3,
		scorer_profile: "capacity_transport_v1",
		input_sha256: "b".repeat(64),
		source_sha256: "c".repeat(64),
		step_sha256: stepSha,
		regions,
	} as unknown as DrawingViewManifest;
	const fixedScales: number[] = [];
	const freeScaleByRegion = new Map([["front", 1], ["side", 2]]);
	const worker: WorkerClient = {
		async request<T extends WorkerResponse>(_workspace: string, request: WorkerRequest): Promise<T> {
			if (request.operation === "rank_drawing_views") {
				const regionId = request.region_id as string;
				const output = request.output_dir as string;
				const renderPath = join(output, "render.png");
				const alignedPath = join(output, "aligned.png");
				const comparisonPath = join(output, "comparison.png");
				await Promise.all([
					writePng(workspace, renderPath, 20),
					writePng(workspace, alignedPath, 21),
					writePng(workspace, comparisonPath, 22),
				]);
				const candidate = (request.candidates as Array<{ candidate_id: string; view_spec: object }>)[0]!;
				const scale = freeScaleByRegion.get(regionId)!;
				return {
					protocol_version: PROTOCOL_VERSION,
					operation: "rank_drawing_views",
					status: "ok",
					region_id: regionId,
					step_path: "output.step",
					step_sha256: stepSha,
					rankings: [{
						candidate_id: candidate.candidate_id,
						view_spec: candidate.view_spec,
						render_path: renderPath,
						aligned_render_path: alignedPath,
						comparison_path: comparisonPath,
						score: 1,
						raw_score: 1,
						eligible: true,
						failed_gates: [],
						scoring_mode: "capacity_transport",
						capacity_transport: {},
						transform: {
							scale_px_per_unit: scale,
							rotation_deg: 0,
							center_x_px: 100,
							center_y_px: 100,
							model_center_x: 0,
							model_center_y: 0,
							reflected: false,
						},
						alignment: { objective: 0, mode: "capacity_transport", scale_at_bound: false, rotation_at_limit: false, translation_at_limit: false },
					}],
					margin: null,
					comparison_path: comparisonPath,
					scoring_mode: "capacity_transport",
					scorer_profile: "capacity_transport_v1",
				} as unknown as T;
			}
			if (request.operation === "render_fixed_projection") {
				const output = request.output_dir as string;
				const path = join(output, "fixed.png");
				const sha = await writePng(workspace, path, 30);
				const transform = request.transform as { scale_px_per_unit: number };
				fixedScales.push(transform.scale_px_per_unit);
				return {
					protocol_version: PROTOCOL_VERSION,
					operation: "render_fixed_projection",
					status: "ok",
					region_id: request.region_id,
					step_path: "output.step",
					step_sha256: stepSha,
					view_spec: request.view_spec,
					transform: request.transform,
					render_path: path,
					width: 320,
					height: 240,
					mime_type: "image/png",
					sha256: sha,
				} as T;
			}
			if (request.operation === "compose_reflection_inspection") {
				const output = request.output_dir as string;
				const path = join(output, "overview.png");
				const sha = await writePng(workspace, path, 40);
				return {
					protocol_version: PROTOCOL_VERSION,
					operation: "compose_reflection_inspection",
					status: "ok",
					region_id: request.region_id,
					mode: "coordinate_overview",
					context_path: path,
					width: 320,
					height: 240,
					sha256: { context: sha },
				} as T;
			}
			throw new Error(`Unexpected operation ${request.operation}`);
		},
	};
	const state: DrawingScaleAgentState = {
		source: "from build123d import *\nSECRET_CURRENT_SOURCE = 1",
		sourceSha256: "c".repeat(64),
		stepSha256: stepSha,
		artifact: artifact(stepSha),
		lockedTarget: null,
		previews: [],
		submitted: null,
	};
	const tools = createDrawingScaleAgentTools({
		workspace,
		worker,
		controller: new RunController(),
		manifest,
		pageCount: 1,
		state,
	}) as unknown as TestTool[];
	return { tools, state, fixedScales };
}

async function invoke(tool: TestTool, parameters: object): Promise<unknown> {
	return await tool.execute("test", parameters, undefined, undefined, {});
}

describe("standalone Drawing Scale Agent", () => {
	it("has a target-first prompt and exactly four purpose-built tools", () => {
		expect(DRAWING_SCALE_AGENT_TOOLS).toEqual([
			"lock_target_dimension",
			"inspect_current_cad_dimensions",
			"preview_dimension_match",
			"submit_dimension_match",
		]);
		expect(DRAWING_SCALE_AGENT_SYSTEM_PROMPT).toContain("standalone Drawing Scale Correspondence Agent");
		expect(DRAWING_SCALE_AGENT_SYSTEM_PROMPT).toContain("You must always return one paired dimension; abstention is not allowed");
		expect(DRAWING_SCALE_AGENT_SYSTEM_PROMPT).toContain("strict information barrier");
		expect(DRAWING_SCALE_AGENT_SYSTEM_PROMPT).toContain("collinear dimension chains that meet at a shared datum");
		expect(DRAWING_SCALE_AGENT_SYSTEM_PROMPT).toContain("choose the best semantically related proxy anchor");
		expect(DRAWING_SCALE_AGENT_SYSTEM_PROMPT).toContain("Do not output, estimate, discuss, or add a scale_ratio field");
		const prompt = drawingScaleAgentPrompt({
			fixtureId: "103",
			pagePaths: ["inputs/page-000.png"],
		});
		expect(prompt).toContain("call lock_target_dimension");
		expect(prompt).not.toContain("Current solution.py");
		expect(prompt).not.toContain("SECRET_CURRENT_SOURCE");
	});

	it("measures B-rep dimensions, applies one hidden correction to all views, and omits scale_ratio", async () => {
		const setupResult = await setup();
		expect(setupResult.tools.map(({ name }) => name)).toEqual(DRAWING_SCALE_AGENT_TOOLS);
		await expect(invoke(setupResult.tools[1]!, {})).rejects.toThrow("Lock the target drawing dimension");
		await invoke(setupResult.tools[0]!, {
			target_name: "overall_width",
			target_mm: 80,
			target_description: "Complete formed target width between the two outermost drawing boundary extension lines.",
			source_page: 0,
			orientation: "horizontal",
			extent_scope: "complete_object",
			endpoint_a: "Leftmost exterior boundary endpoint identified by the left extension line.",
			endpoint_b: "Rightmost exterior boundary endpoint identified by the right extension line.",
			dimension_chain_mm: [80],
			derivation: "The single 80 mm dimension line directly joins the two complete-object exterior extension lines.",
		});
		const measured = await invoke(setupResult.tools[1]!, {}) as { details: object; content: Array<{ text: string }> };
		expect(measured.details).toMatchObject({
			bbox_x_extent_mm: 100,
			bbox_y_extent_mm: 60,
			bbox_z_extent_mm: 20,
		});
		expect(measured.content[0]!.text).toContain("SECRET_CURRENT_SOURCE");

		const preview = await invoke(setupResult.tools[2]!, {
			current_axis: "x",
			current_description: "Minimum X exterior endpoint to maximum X exterior endpoint of the complete formed current CAD.",
			match_quality: "exact",
			correspondence_basis: "Both measurements span the complete formed part between the same two exterior width endpoints.",
		}) as { details: object };
		expect(setupResult.fixedScales).toEqual([1.25, 2.5]);
		expect(JSON.stringify(preview.details)).not.toContain("scale_ratio");
		expect(preview.details).toMatchObject({
			preview_ref: "P001",
			target: { value_mm: 80 },
			current: { value_mm: 100, measurement: "bbox_x_extent" },
		});

		await invoke(setupResult.tools[3]!, {
			preview_ref: "P001",
			justification: "The source and every preview preserve the same complete formed-part exterior width endpoints.",
		});
		expect(setupResult.state.submitted?.status).toBe("paired");
		expect(setupResult.state.submitted?.match_quality).toBe("exact");
		expect(JSON.stringify(setupResult.state.submitted)).not.toContain("scale_ratio");
		expect(JSON.stringify(setupResult.state.submitted)).not.toContain("display_scale_multiplier");
	});

	it("must submit a previewed proxy when exact endpoint correspondence is unavailable", async () => {
		const setupResult = await setup();
		await invoke(setupResult.tools[0]!, {
			target_name: "overall_width",
			target_mm: 565,
			target_description: "Complete developed width from the leftmost flat-pattern boundary to the rightmost boundary.",
			source_page: 0,
			orientation: "horizontal",
			extent_scope: "complete_object",
			endpoint_a: "Leftmost developed mounting-arm boundary reached by the 370 mm chain segment.",
			endpoint_b: "Rightmost developed mounting-arm boundary reached by the 195 mm chain segment.",
			dimension_chain_mm: [370, 195],
			derivation: "The 370 mm and 195 mm collinear segments meet at the same central datum, so the complete boundary extent is 565 mm.",
		});
		await invoke(setupResult.tools[1]!, {});
		await invoke(setupResult.tools[2]!, {
			current_axis: "y",
			current_description: "Minimum Y exterior endpoint to maximum Y exterior endpoint of the complete formed current CAD.",
			match_quality: "proxy",
			correspondence_basis: "Both measurements describe the dominant transverse extent associated with the same source part.",
			limitations: "The target is developed while the current CAD is formed, so their physical endpoint positions and state are not identical.",
		});
		await invoke(setupResult.tools[3]!, {
			preview_ref: "P001",
			justification: "This is the best available repeatable scale anchor, with its developed-versus-formed limitation preserved explicitly.",
		});
		expect(setupResult.state.previews).toHaveLength(1);
		expect(setupResult.state.submitted).toMatchObject({
			status: "paired",
			match_quality: "proxy",
			target: { value_mm: 565, dimension_chain_mm: [370, 195] },
			current: { measurement: "bbox_y_extent", value_mm: 60 },
		});
		expect(JSON.stringify(setupResult.state.submitted)).not.toContain("scale_ratio");
	});
});
