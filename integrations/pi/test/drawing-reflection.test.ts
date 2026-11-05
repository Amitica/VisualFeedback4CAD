import { mkdtemp, mkdir, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { describe, expect, it, vi } from "vitest";
import { Value } from "typebox/value";

import { addUsageTotals, emptyUsageTotals } from "../src/context.js";
import {
	assertNormalizedRoi,
	assertCompleteDrawingViewSearch,
	assertLockedTopTwo,
	assertRegionViewSemantics,
	assertSupportedViewSpec,
	createReflectionTools,
	createViewSearchTools,
	COMPARE_REGION_GRID_CELLS_PARAMETERS_SCHEMA,
	DRAWING_VIEW_SPEC_SCHEMA,
	SUBMIT_INITIAL_FINDINGS_PARAMETERS_SCHEMA,
	UPDATE_FINDINGS_PARAMETERS_SCHEMA,
	INSPECT_REGION_PARAMETERS_SCHEMA,
	manifestLooksCurrent,
	makeSearchState,
	NORMALIZED_ROI_SCHEMA,
	RANK_CANDIDATES_PARAMETERS_SCHEMA,
	REFLECTION_ACTIVE_TIMEOUT_MS,
	REFLECTION_WALL_TIMEOUT_MS,
	RENDER_CANDIDATE_PARAMETERS_SCHEMA,
	SELF_REFLECT_PARAMETERS_SCHEMA,
	SUBMIT_REFLECTION_PARAMETERS_SCHEMA,
	SUBMIT_VIEW_RESULT_PARAMETERS_SCHEMA,
	recordSubagentUsage,
	VECTOR3_SCHEMA,
} from "../src/drawing-reflection.js";
import { isInfrastructureWorkerFailure } from "../src/local-worker.js";
import type { WorkerClient } from "../src/local-worker.js";
import { RunController } from "../src/runner.js";
import {
	ACTIVE_AGENT_TOOLS,
	PROTOCOL_VERSION,
	REFLECTION_AGENT_TOOLS,
	SEGMENTATION_AGENT_TOOLS,
	VIEW_SEARCH_AGENT_TOOLS,
	type DrawingRegionManifest,
	type DrawingViewSpec,
	type DrawingViewManifest,
	type NormalizedRoi,
	type AgentResult,
	type SubagentRunUsage,
	type UsageTotals,
} from "../src/types.js";

const CAPACITY_TRANSPORT = {
	version: 1,
	score: 0.8,
	normalized_cost: 0.2,
	distance_scale_px: 3,
	unmatched_cad_cost: 1,
	unmatched_reference_cost: 0.25,
	reference_edge_pixels: 100,
	cad_edge_pixels: 90,
	reference_tokens: 64,
	cad_tokens: 58,
	matched_pairs: 50,
	unmatched_cad_tokens: 8,
	unmatched_reference_tokens: 14,
	total_cost: 12,
	matched_cad_fraction: 0.86,
	matched_reference_fraction: 0.78,
	mean_matched_distance_px: 1.2,
	p90_matched_distance_px: 2.4,
};

function capacityFields(score: number) {
	return {
		raw_score: score,
		eligible: true,
		failed_gates: [] as string[],
		scoring_mode: "capacity_transport" as const,
		capacity_transport: { ...CAPACITY_TRANSPORT, score, normalized_cost: 1 - score },
	};
}

function alignmentFields(alignedRenderPath: string) {
	return {
		aligned_render_path: alignedRenderPath,
		transform: {
			scale_px_per_unit: 1,
			rotation_deg: 0,
			center_x_px: 50,
			center_y_px: 50,
			model_center_x: 0,
			model_center_y: 0,
			reflected: false,
		},
		alignment: {
			objective: 0.1,
			mode: "capacity_transport" as const,
			scale_at_bound: false,
			rotation_at_limit: false,
			translation_at_limit: false,
		},
	};
}

describe("2D reflection tool boundaries", () => {
	it("leaves region inspection unlimited and gives Reflection a long wall-time envelope", async () => {
		expect(REFLECTION_ACTIVE_TIMEOUT_MS).toBe(30 * 60_000);
		expect(REFLECTION_WALL_TIMEOUT_MS).toBe(120 * 60_000);
		const workspace = await mkdtemp(join(tmpdir(), "pi-reflection-unlimited-inspection-"));
		try {
			await mkdir(join(workspace, "drawing_feedback"));
			await writeFile(join(workspace, "drawing_feedback", "crop.png"), "png");
			const manifest: DrawingViewManifest = {
				version: 3,
				scorer_profile: "capacity_transport_v1",
				input_sha256: "input-sha",
				source_sha256: "source-sha",
				step_sha256: "step-sha",
				regions: [
					{
						region_id: "detail",
						page: 0,
						roi: [0, 0, 1000, 1000],
						crop_path: "drawing_feedback/crop.png",
						view_type: "detail",
						status: "error",
						alternatives: [],
						margin: null,
						confidence: "low",
						note: "controller-generated search error",
					},
				],
			};
			const [inspectTool] = createReflectionTools({ workspace, state: { manifest, report: null } });
			for (let call = 1; call <= 20; call += 1) {
				await expect(
					inspectTool!.execute(`inspect-${String(call)}`, { region_id: "detail" }, undefined, undefined, {} as never),
				).resolves.toBeDefined();
			}
		} finally {
			await rm(workspace, { recursive: true, force: true });
		}
	});

	it("keeps exactly nine main tools and restricts both subagents", () => {
		expect(ACTIVE_AGENT_TOOLS).toHaveLength(9);
		expect(SEGMENTATION_AGENT_TOOLS).toEqual(["crop_region", "submit_region_manifest"]);
		expect(VIEW_SEARCH_AGENT_TOOLS).toEqual([
			"render_candidate_view",
			"rank_view_candidates",
			"submit_view_result",
		]);
		expect(REFLECTION_AGENT_TOOLS).toEqual(["inspect_region", "submit_reflection"]);
			expect(REFLECTION_AGENT_TOOLS).not.toContain("render_candidate_view");
			expect(VIEW_SEARCH_AGENT_TOOLS).not.toContain("crop_region");
	});

	it("uses a strict empty public schema", () => {
		expect(Value.Check(SELF_REFLECT_PARAMETERS_SCHEMA, {})).toBe(true);
		expect(Value.Check(SELF_REFLECT_PARAMETERS_SCHEMA, { view: "front" })).toBe(false);
		expect(Value.Check(SELF_REFLECT_PARAMETERS_SCHEMA, null)).toBe(false);
	});

	it("uses provider-compatible homogeneous array schemas for ROI and vectors", () => {
		const roiSchema = NORMALIZED_ROI_SCHEMA as unknown as {
			items?: unknown;
			minItems?: number;
			maxItems?: number;
		};
		const vectorSchema = VECTOR3_SCHEMA as unknown as {
			items?: unknown;
			minItems?: number;
			maxItems?: number;
		};
		expect(Array.isArray(roiSchema.items)).toBe(false);
		expect(roiSchema).toMatchObject({ minItems: 4, maxItems: 4 });
		expect(Array.isArray(vectorSchema.items)).toBe(false);
		expect(vectorSchema).toMatchObject({ minItems: 3, maxItems: 3 });
	});

	it("does not let reflection inspection or submission carry camera parameters", () => {
		expect(Value.Check(INSPECT_REGION_PARAMETERS_SCHEMA, { region_id: "page0-main" })).toBe(true);
		expect(
			Value.Check(INSPECT_REGION_PARAMETERS_SCHEMA, {
				region_id: "page0-main",
				view_spec: { direction: [1, 0, 0] },
			}),
		).toBe(false);
		expect(
			Value.Check(COMPARE_REGION_GRID_CELLS_PARAMETERS_SCHEMA, {
				region_id: "page0-main",
				finding_ids: ["finding-001"],
				reference_cell: "B2",
				cad_cell: "C2",
			}),
		).toBe(true);
		expect(
			Value.Check(COMPARE_REGION_GRID_CELLS_PARAMETERS_SCHEMA, {
				region_id: "page0-main",
				finding_ids: ["finding-001"],
				reference_cell: "B2",
				cad_cell: "C2",
				reference_roi: { x_min: 100, y_min: 100, x_max: 400, y_max: 400 },
			}),
		).toBe(false);
		expect(Value.Check(COMPARE_REGION_GRID_CELLS_PARAMETERS_SCHEMA, {
			region_id: "page0-main",
			finding_ids: ["finding-001"],
			reference_cell: "A1",
			cad_cell: "E1",
		})).toBe(false);
		const summary = {
			should_revise: true,
			overall_summary: "A boss is missing.",
			findings: [{
				region_id: "page0-main",
				obs: "The CAD omits the circular boss shown in the drawing.",
				suggested_change: "Add the circular boss.",
			}],
		};
		expect(Value.Check(SUBMIT_REFLECTION_PARAMETERS_SCHEMA, summary)).toBe(true);
		expect(Value.Check(SUBMIT_REFLECTION_PARAMETERS_SCHEMA, {
			should_revise: true,
			overall_summary: "A boss is missing.",
		})).toBe(false);
		expect(Value.Check(SUBMIT_REFLECTION_PARAMETERS_SCHEMA, { ...summary, direction: [1, 0, 0] })).toBe(false);
		expect(Value.Check(SUBMIT_REFLECTION_PARAMETERS_SCHEMA, { ...summary, next_actions: [] })).toBe(false);
		expect(Value.Check(SUBMIT_REFLECTION_PARAMETERS_SCHEMA, { ...summary, confidence: "high" })).toBe(false);
		expect(Value.Check(SUBMIT_INITIAL_FINDINGS_PARAMETERS_SCHEMA, {
			findings: [{
				region_id: "page0-main",
				obs: "The circular boss is absent.",
				suggested_change: "Add the boss.",
			}],
		})).toBe(true);
		const updateBase = {
			region_id: "page0-main",
			crop_id: "0123456789abcdef0123",
			reviewed_finding_ids: ["finding-001"],
		};
		expect(Value.Check(UPDATE_FINDINGS_PARAMETERS_SCHEMA, {
			...updateBase,
			changes: [{
				action: "add",
				obs: "The circular boss is absent.",
				suggested_change: "Add the boss.",
			}],
		})).toBe(true);
		expect(Value.Check(UPDATE_FINDINGS_PARAMETERS_SCHEMA, {
			...updateBase,
			changes: [],
		})).toBe(true);
		expect(Value.Check(UPDATE_FINDINGS_PARAMETERS_SCHEMA, {
			...updateBase,
			changes: [{ action: "add", region_id: "other", obs: "Missing.", suggested_change: "Add it." }],
		})).toBe(false);
		expect(Value.Check(UPDATE_FINDINGS_PARAMETERS_SCHEMA, {
			...updateBase,
			changes: [{
				action: "update",
				finding_id: "finding-001",
				field: "obs",
				old_text: "too small",
				new_text: "too large",
			}],
		})).toBe(true);
		expect(Value.Check(UPDATE_FINDINGS_PARAMETERS_SCHEMA, {
			...updateBase,
			changes: [{ action: "update", finding_id: "finding-001", field: "obs" }],
		})).toBe(false);
		expect(Value.Check(UPDATE_FINDINGS_PARAMETERS_SCHEMA, {
			...updateBase,
			changes: [{ action: "delete", finding_id: "finding-001" }],
		})).toBe(true);
		expect(Value.Check(UPDATE_FINDINGS_PARAMETERS_SCHEMA, {
			...updateBase,
			changes: [{ action: "delete", finding_id: "finding-001", path: "reflection.json" }],
			path: "reflection.json",
		})).toBe(false);
	});

	it("accepts one to 32 bound-region candidate ids or view specs", () => {
		const specs: Array<Omit<DrawingViewSpec, "name">> = [
			{ direction: [0, -1, 0], up: [0, 0, 1] },
			{ direction: [0, 1, 0], up: [0, 0, 1] },
			{ direction: [-1, 0, 0], up: [0, 0, 1] },
		];
		expect(
			Value.Check(RANK_CANDIDATES_PARAMETERS_SCHEMA, {
				candidate_ids: ["candidate-1", "candidate-2", "candidate-3"],
			}),
		).toBe(true);
		expect(Value.Check(RANK_CANDIDATES_PARAMETERS_SCHEMA, { candidates: specs })).toBe(true);
		expect(
			Value.Check(RANK_CANDIDATES_PARAMETERS_SCHEMA, {
				candidates: [{ name: "front", ...specs[0] }, specs[1], specs[2]],
			}),
		).toBe(false);
		expect(
			Value.Check(RENDER_CANDIDATE_PARAMETERS_SCHEMA, {
				view_spec: { name: "front", ...specs[0] },
			}),
		).toBe(false);
		expect(
			Value.Check(RANK_CANDIDATES_PARAMETERS_SCHEMA, {
				candidate_ids: ["a", "b", "c"],
				candidates: specs,
			}),
		).toBe(false);
		expect(Value.Check(RANK_CANDIDATES_PARAMETERS_SCHEMA, {})).toBe(false);
		expect((RANK_CANDIDATES_PARAMETERS_SCHEMA as unknown as { type?: string }).type).toBe("object");
		expect(
			Value.Check(RANK_CANDIDATES_PARAMETERS_SCHEMA, { candidates: specs.slice(0, 1) }),
		).toBe(true);
		expect(
			Value.Check(RANK_CANDIDATES_PARAMETERS_SCHEMA, {
				candidates: Array.from({ length: 33 }, () => specs[0]),
			}),
		).toBe(false);
		expect(
			Value.Check(RANK_CANDIDATES_PARAMETERS_SCHEMA, {
				region_id: "some-other-region",
				candidates: specs,
			}),
		).toBe(false);
		expect(
			Value.Check(RENDER_CANDIDATE_PARAMETERS_SCHEMA, {
				region_id: "some-other-region",
				view_spec: specs[0],
			}),
		).toBe(false);
	});

	it("describes every View Search tool parameter in the schema sent to the Agent", () => {
		type DocumentedSchema = {
			description?: string;
			properties?: Record<string, DocumentedSchema>;
		};
		const schemas = [
			RENDER_CANDIDATE_PARAMETERS_SCHEMA,
			RANK_CANDIDATES_PARAMETERS_SCHEMA,
			SUBMIT_VIEW_RESULT_PARAMETERS_SCHEMA,
		] as unknown as DocumentedSchema[];
		for (const schema of schemas) {
			expect(schema.description).toBeTruthy();
			for (const parameter of Object.values(schema.properties ?? {})) {
				expect(parameter.description).toBeTruthy();
			}
		}
		const viewProperties = (DRAWING_VIEW_SPEC_SCHEMA as unknown as DocumentedSchema).properties!;
		for (const name of ["direction", "up", "section_offset_fraction", "section_keep"]) {
			expect(viewProperties[name]?.description).toBeTruthy();
		}
		expect(
			Value.Check(SUBMIT_VIEW_RESULT_PARAMETERS_SCHEMA, {
				view_type: "orthographic",
				status: "unsupported",
				selected_candidate_id: "candidate-001",
				confidence: "low",
				note: "not allowed",
			}),
		).toBe(false);
	});

	it("enforces the cumulative 32-candidate limit before calling the worker", async () => {
		const region = {
			region_id: "main",
			page: 0,
			roi: [0, 0, 1000, 1000] as NormalizedRoi,
			crop_path: "feedback/crop.png",
			view_type_hint: "orthographic" as const,
			confidence: "high" as const,
			note: "main",
			width: 400,
			height: 300,
			crop_sha256: "crop-sha",
		};
		const state = makeSearchState(region);
		for (let index = 1; index <= 32; index += 1) {
			const candidateId = `candidate-${String(index).padStart(3, "0")}`;
			state.candidates.set(candidateId, {
				candidateId,
				regionId: region.region_id,
				viewSpec: { name: candidateId, direction: [0, -1, 0], up: [0, 0, 1] },
				renderPath: `feedback/${candidateId}.png`,
			});
		}
		const request = vi.fn(async () => {
			throw new Error("worker must not be called past the cumulative limit");
		});
		const tools = createViewSearchTools({
			worker: { request } as unknown as WorkerClient,
			workspace: "/tmp",
			region,
			outputDir: "feedback/run",
			expectedStepSha256: "step-sha",
			controller: new RunController(),
			state,
		});

		await expect(
			tools[0].execute(
				"call-render",
				{ view_spec: { direction: [1, 0, 0], up: [0, 0, 1] } },
				undefined,
				undefined,
				{} as never,
			),
		).rejects.toThrow("maximum 32 candidates");
		await expect(
			tools[1].execute(
				"call-rank",
				{ candidates: [{ direction: [1, 0, 0], up: [0, 0, 1] }] },
				undefined,
				undefined,
				{} as never,
			),
		).rejects.toThrow("at most 32 cumulative candidates");
		expect(request).not.toHaveBeenCalled();
		expect(state.candidates.size).toBe(32);
	});

	it("sends batch specs in one worker call and records worker-generated candidates", async () => {
		const workspace = await mkdtemp(join(tmpdir(), "pi-batch-rank-"));
		try {
			const artifactPaths = [
				"feedback/render-1.png",
				"feedback/render-2.png",
				"feedback/render-3.png",
				"feedback/compare-1.png",
				"feedback/compare-2.png",
				"feedback/compare-3.png",
				"feedback/aligned-1.png",
				"feedback/aligned-2.png",
				"feedback/aligned-3.png",
			];
			await mkdir(join(workspace, "feedback"), { recursive: true });
			await Promise.all(artifactPaths.map(async (path) => await writeFile(join(workspace, path), "png")));
			const specs: Array<Omit<DrawingViewSpec, "name">> = [
				{ direction: [0, -1, 0], up: [0, 0, 1] },
				{ direction: [0, 1, 0], up: [0, 0, 1] },
				{ direction: [-1, 0, 0], up: [0, 0, 1] },
			];
			const request = vi.fn(async (..._args: unknown[]) => ({
				protocol_version: PROTOCOL_VERSION,
				operation: "rank_drawing_views" as const,
				status: "ok" as const,
				region_id: "main",
				step_path: "output.step",
				step_sha256: "step-sha",
					rankings: specs.map((viewSpec, index) => ({
						candidate_id: `candidate-${String(index + 1).padStart(3, "0")}`,
						score: 0.9 - index * 0.1,
						...capacityFields(0.9 - index * 0.1),
					view_spec: {
						name: `candidate-${String(index + 1).padStart(3, "0")}`,
						...viewSpec,
					},
					render_path: `feedback/render-${index + 1}.png`,
					comparison_path: `feedback/compare-${index + 1}.png`,
					...alignmentFields(`feedback/aligned-${index + 1}.png`),
				})),
					margin: 0.1,
					comparison_path: "feedback/compare-1.png",
					scoring_mode: "capacity_transport" as const,
					scorer_profile: "capacity_transport_v1" as const,
				}));
			const regionManifest: DrawingRegionManifest = {
				version: 1,
				input_sha256: "input-sha",
				regions: [
					{
						region_id: "main",
						page: 0,
						roi: [0, 0, 1000, 1000],
						crop_path: "feedback/crop.png",
						view_type_hint: "orthographic",
						confidence: "high",
						note: "main",
						width: 400,
						height: 300,
						crop_sha256: "crop-sha",
					},
				],
			};
				const region = regionManifest.regions[0]!;
				const state = makeSearchState(region);
				const tools = createViewSearchTools({
				worker: { request } as unknown as WorkerClient,
				workspace,
					region,
				outputDir: "feedback/run",
				expectedStepSha256: "step-sha",
				controller: new RunController(),
				state,
			});

			const ranked = (await tools[1].execute(
				"call-1",
					{ candidates: specs },
				undefined,
				undefined,
				{} as never,
			)) as {
				content: Array<{ type: string; text?: string; data?: string }>;
			};

			expect(request).toHaveBeenCalledOnce();
			const workerRequest = request.mock.calls[0]?.[1] as { candidates: Array<Record<string, unknown>> };
			const workerOptions = request.mock.calls[0]?.[2] as { timeoutMs: number };
			expect(workerRequest.candidates).toHaveLength(3);
			expect(workerOptions.timeoutMs).toBe(60_000 + 180_000 * 3);
			expect(workerRequest.candidates.every((candidate) => !("render_path" in candidate))).toBe(true);
			expect(
				workerRequest.candidates.map((candidate) => (candidate.view_spec as DrawingViewSpec).name),
			).toEqual(["candidate-001", "candidate-002", "candidate-003"]);
			expect(state.candidates.size).toBe(3);
			expect(ranked.content.map(({ type }) => type)).toEqual([
				"text",
				"text",
				"image",
				"text",
				"image",
			]);
			const candidateLabels = ranked.content
				.filter(({ type }) => type === "text")
				.flatMap(({ text }) => text ? [text] : [])
				.slice(1);
			expect(candidateLabels).toHaveLength(2);
			expect(candidateLabels[0]).toContain('"numeric_rank":1');
			expect(candidateLabels[0]).toContain('"candidate_id":"candidate-001"');
			expect(candidateLabels[0]).toContain('"direction":[0,-1,0]');
			expect(candidateLabels[0]).toContain('"up":[0,0,1]');
			expect(candidateLabels[1]).toContain('"numeric_rank":2');
			expect(candidateLabels[1]).toContain('"candidate_id":"candidate-002"');
				const storedRanking = state.ranking;
			expect(storedRanking?.rankings.map(({ candidate_id }) => candidate_id)).toEqual([
				"candidate-001",
				"candidate-002",
				"candidate-003",
			]);
		} finally {
			await rm(workspace, { recursive: true, force: true });
		}
	});

	it("controller-names individual renders and rejects a changed worker echo", async () => {
		const request = vi.fn(async (...args: unknown[]) => {
			const workerRequest = args[1] as { view_spec: DrawingViewSpec };
			return {
				protocol_version: PROTOCOL_VERSION,
				operation: "render_drawing_view" as const,
				status: "ok" as const,
				region_id: "main",
				view_spec: { ...workerRequest.view_spec, name: "changed-by-worker" },
				render_path: "feedback/render.png",
				width: 100,
				height: 100,
				step_sha256: "step-sha",
			};
		});
		const regionManifest: DrawingRegionManifest = {
			version: 1,
			input_sha256: "input-sha",
			regions: [
				{
					region_id: "main",
					page: 0,
					roi: [0, 0, 1000, 1000],
					crop_path: "feedback/crop.png",
					view_type_hint: "orthographic",
					confidence: "high",
					note: "main",
					width: 400,
					height: 300,
					crop_sha256: "crop-sha",
				},
			],
		};
		const region = regionManifest.regions[0]!;
		const state = makeSearchState(region);
		const tools = createViewSearchTools({
			worker: { request } as unknown as WorkerClient,
			workspace: "/tmp",
			region,
			outputDir: "feedback/run",
			expectedStepSha256: "step-sha",
			controller: new RunController(),
			state,
		});

		await expect(
				tools[0].execute(
				"call-render",
				{ view_spec: { direction: [0, -1, 0], up: [0, 0, 1] } },
				undefined,
				undefined,
				{} as never,
			),
		).rejects.toThrow("controller-generated candidate name");
		const workerRequest = request.mock.calls[0]?.[1] as { view_spec: DrawingViewSpec };
		expect(workerRequest.view_spec.name).toBe("candidate-001");
		expect(state.candidates.size).toBe(0);
	});

	it("reserves total call 24 for manifest submission and reports the remaining budget", async () => {
		const request = vi.fn(async () => {
			throw new Error("worker must not be called");
		});
		const regionManifest: DrawingRegionManifest = {
			version: 1,
			input_sha256: "input-sha",
			regions: [
				{
					region_id: "detail",
					page: 0,
					roi: [0, 0, 1000, 1000],
					crop_path: "feedback/crop.png",
					view_type_hint: "detail",
					confidence: "low",
					note: "detail",
					width: 400,
					height: 300,
					crop_sha256: "crop-sha",
				},
			],
		};
		const region = regionManifest.regions[0]!;
		const state = makeSearchState(region);
		state.toolCalls = 23;
		const tools = createViewSearchTools({
			worker: { request } as unknown as WorkerClient,
			workspace: "/tmp",
			region,
			outputDir: "feedback/run",
			expectedStepSha256: "step-sha",
			controller: new RunController(),
			state,
		});
		await expect(
				tools[1].execute(
				"call-work-24",
				{
					candidates: [
						{ direction: [1, 0, 0], up: [0, 0, 1] },
						{ direction: [-1, 0, 0], up: [0, 0, 1] },
						{ direction: [0, 1, 0], up: [0, 0, 1] },
					],
				},
				undefined,
				undefined,
				{} as never,
			),
		).rejects.toThrow("reserved for submit_view_result");
		await expect(
			tools[2].execute(
				"call-submit-24",
				{
					view_type: "detail",
					status: "ambiguous",
					selected_candidate_id: "candidate-001",
					confidence: "low",
					note: "uncertain detail",
				},
				undefined,
				undefined,
				{} as never,
			),
		).rejects.toThrow("Top-2");
		expect(state.toolCalls).toBe(24);
		expect(request).not.toHaveBeenCalled();
	});
});

describe("view candidate validation", () => {
	it("accepts canonical and arbitrary finite perpendicular camera vectors", () => {
		const axis: DrawingViewSpec = { name: "front", direction: [1, 0, 0], up: [0, 0, 1] };
		const iso: DrawingViewSpec = { name: "iso", direction: [1, -1, 1], up: [1, 1, 0] };
		expect(Value.Check(DRAWING_VIEW_SPEC_SCHEMA, { direction: axis.direction, up: axis.up })).toBe(true);
		expect(Value.Check(DRAWING_VIEW_SPEC_SCHEMA, axis)).toBe(false);
		expect(() => assertSupportedViewSpec(axis)).not.toThrow();
		expect(() => assertSupportedViewSpec(iso)).not.toThrow();
		expect(() =>
			assertSupportedViewSpec({ name: "arbitrary", direction: [0.2, 0, 1], up: [0, 1, 0] }),
		).not.toThrow();
	});

	it("rejects zero, non-perpendicular, or incomplete section candidates", () => {
		expect(() => assertSupportedViewSpec({ name: "zero", direction: [0, 0, 0], up: [0, 0, 1] })).toThrow(
			"non-zero",
		);
		expect(() => assertSupportedViewSpec({ name: "bad-up", direction: [1, 0, 0], up: [1, 0, 0] })).toThrow(
			"perpendicular",
		);
		expect(() =>
			assertSupportedViewSpec({
				name: "section",
				direction: [1, 0, 0],
				up: [0, 0, 1],
				section_offset_fraction: 0,
			}),
		).toThrow("both");
	});

	it("validates normalized ROI ordering as well as bounds", () => {
		expect(() => assertNormalizedRoi([0, 10, 900, 1000])).not.toThrow();
		expect(() => assertNormalizedRoi([500, 10, 500, 900])).toThrow("x0 < x1");
		expect(() => assertNormalizedRoi([-1, 10, 500, 900])).toThrow("0..1000");
		expect(() => assertNormalizedRoi([0.5, 10, 500, 900])).toThrow("integers");
	});

	it("allows either eligible Top-2 candidate and derives the alternative", () => {
		expect(assertLockedTopTwo(["a", "b", "c"], "a")).toBe("b");
		expect(assertLockedTopTwo(["a", "b", "c"], "b")).toBe("a");
		expect(() => assertLockedTopTwo(["a", "b", "c"], "c")).toThrow("Top-2");
	});

	it("enforces section semantics while allowing every region type to select Top-2", () => {
		const plain: DrawingViewSpec = { name: "front", direction: [0, -1, 0], up: [0, 0, 1] };
		const diagonal: DrawingViewSpec = {
			name: "iso",
			direction: [1, -1, 1],
			up: [1, 1, 0],
		};
		const section: DrawingViewSpec = {
			name: "front-section",
			direction: [0, -1, 0],
			up: [0, 0, 1],
			section_offset_fraction: 0,
			section_keep: "front",
		};
		expect(() => assertRegionViewSemantics("orthographic", "matched", [plain, diagonal])).not.toThrow();
		expect(() => assertRegionViewSemantics("isometric", "matched", [diagonal, plain])).not.toThrow();
		expect(() => assertRegionViewSemantics("section", "matched", [section, section])).not.toThrow();
		expect(() => assertRegionViewSemantics("detail", "matched", [plain, plain])).not.toThrow();
		expect(() => assertRegionViewSemantics("unknown", "ambiguous", [plain, plain])).not.toThrow();
		expect(() => assertRegionViewSemantics("unknown", "error", [])).not.toThrow();
		expect(() => assertRegionViewSemantics("section", "matched", [section, plain])).not.toThrow();
		expect(() => assertRegionViewSemantics("isometric", "matched", [diagonal, section])).not.toThrow();
		expect(() => assertRegionViewSemantics("section", "matched", [plain, section])).toThrow(
			"section parameters",
		);
		expect(() => assertRegionViewSemantics("isometric", "matched", [section, diagonal])).toThrow("Non-section");
	});

	it("strictly validates capacity manifests, eligible Top-2, and drawing page bounds", () => {
		const candidate = (id: string, direction: DrawingViewSpec["direction"]) => ({
			candidate_id: id,
			score: 0.8,
			...capacityFields(0.8),
			view_spec: { name: id, direction, up: [0, 0, 1] as DrawingViewSpec["up"] },
			render_path: `drawing_feedback/${id}.png`,
			comparison_path: `drawing_feedback/${id}-comparison.png`,
			...alignmentFields(`drawing_feedback/${id}-aligned.png`),
		});
		const manifest: DrawingViewManifest = {
			version: 3,
			scorer_profile: "capacity_transport_v1",
			input_sha256: "input-sha",
			source_sha256: "source-sha",
			step_sha256: "step-sha",
			regions: [
				{
					region_id: "page0-main",
					page: 0,
					roi: [0, 0, 1000, 1000],
					crop_path: "drawing_feedback/crop.png",
					view_type: "orthographic",
					status: "matched",
					selected: candidate("front", [0, -1, 0]),
					alternatives: [candidate("rear", [0, 1, 0])],
					margin: 0.1,
					confidence: "high",
					note: "main view",
				},
			],
		};
		expect(manifestLooksCurrent(manifest, "input-sha", "source-sha", "step-sha", 1)).toBe(true);
		const alternativeSection = structuredClone(manifest);
		Object.assign(alternativeSection.regions[0]!.alternatives[0]!.view_spec, { section_offset_fraction: 0, section_keep: "front" });
		expect(manifestLooksCurrent(alternativeSection, "input-sha", "source-sha", "step-sha", 1)).toBe(true);
		const selectedSection = structuredClone(manifest);
		Object.assign(selectedSection.regions[0]!.selected!.view_spec, { section_offset_fraction: 0, section_keep: "front" });
		expect(manifestLooksCurrent(selectedSection, "input-sha", "source-sha", "step-sha", 1)).toBe(false);
		selectedSection.regions[0]!.view_type = "section";
		expect(manifestLooksCurrent(selectedSection, "input-sha", "source-sha", "step-sha", 1)).toBe(true);
		expect(() => assertCompleteDrawingViewSearch(manifest, "input-sha", "source-sha", "step-sha", 1)).not.toThrow();
		expect(() => assertCompleteDrawingViewSearch(manifest, "input-sha", "changed-source", "step-sha", 1)).toThrow("does not match its source CAD");
		const missingAligned = structuredClone(manifest);
		delete (missingAligned.regions[0]!.selected as { aligned_render_path?: string }).aligned_render_path;
		expect(manifestLooksCurrent(missingAligned, "input-sha", undefined, undefined, 1)).toBe(false);
		expect(manifestLooksCurrent({ ...manifest, regions: [{ ...manifest.regions[0]!, page: 1 }] }, "input-sha", undefined, undefined, 1)).toBe(false);
		const invalidScore = structuredClone(manifest);
		invalidScore.regions[0]!.selected!.score = 1.1;
		expect(manifestLooksCurrent(invalidScore, "input-sha", undefined, undefined, 1)).toBe(false);
		const invalidDetail = structuredClone(manifest);
		invalidDetail.regions[0]!.view_type = "detail";
		expect(manifestLooksCurrent(invalidDetail, "input-sha", undefined, undefined, 1)).toBe(true);
		const errorManifest = structuredClone(manifest);
		delete errorManifest.regions[0]!.selected;
		errorManifest.regions[0]!.alternatives = [];
		errorManifest.regions[0]!.margin = null;
		errorManifest.regions[0]!.status = "error";
		expect(manifestLooksCurrent(errorManifest, "input-sha", undefined, undefined, 1)).toBe(false);
		expect(manifestLooksCurrent(errorManifest, "input-sha", undefined, undefined, 1, true)).toBe(true);
		expect(() => assertCompleteDrawingViewSearch(errorManifest, "input-sha", "source-sha", "step-sha", 1)).toThrow("incomplete for 1 region(s): page0-main: main view");
		const mixedManifest = structuredClone(manifest);
		mixedManifest.regions.push({ ...errorManifest.regions[0]!, region_id: "page0-detail", note: "Independent view search failed: active Agent timeout" });
		expect(() => assertCompleteDrawingViewSearch(mixedManifest, "input-sha", "source-sha", "step-sha", 1)).toThrow("page0-detail: Independent view search failed: active Agent timeout");
	});

	it("rejects artifacts reached through a parent symlink outside the workspace", async () => {
		const workspace = await mkdtemp(join(tmpdir(), "pi-drawing-workspace-"));
		const outside = await mkdtemp(join(tmpdir(), "pi-drawing-outside-"));
		try {
			await writeFile(join(outside, "crop.png"), "png");
			await symlink(outside, join(workspace, "feedback"), "dir");
			const manifest: DrawingViewManifest = {
				version: 3,
				scorer_profile: "capacity_transport_v1",
				input_sha256: "input-sha",
				source_sha256: "source-sha",
				step_sha256: "step-sha",
				regions: [
					{
						region_id: "detail",
						page: 0,
						roi: [0, 0, 1000, 1000],
						crop_path: "feedback/crop.png",
						view_type: "detail",
						status: "error",
						alternatives: [],
						margin: null,
						confidence: "low",
						note: "controller-generated search error",
					},
				],
			};
			const tools = createReflectionTools({
				workspace,
				state: { manifest, report: null },
			});
			await expect(
				tools[0]!.execute("call-1", { region_id: "detail" }, undefined, undefined, {} as never),
			).rejects.toThrow("escaped");
		} finally {
			await rm(workspace, { recursive: true, force: true });
			await rm(outside, { recursive: true, force: true });
		}
	});
});

describe("drawing worker and usage accounting", () => {
	it("treats drawing/timeout status errors as structured tool feedback", () => {
		expect(
			isInfrastructureWorkerFailure(
				{
					protocol_version: PROTOCOL_VERSION,
					operation: "render_drawing_view",
					status: "error",
					error: { kind: "drawing", message: "projection failed" },
				},
				0,
			),
		).toBe(false);
		expect(
			isInfrastructureWorkerFailure(
				{
					protocol_version: PROTOCOL_VERSION,
					status: "error",
					error: { kind: "worker", message: "python crashed" },
				},
				0,
			),
		).toBe(true);
		expect(isInfrastructureWorkerFailure({ protocol_version: PROTOCOL_VERSION }, 1)).toBe(true);
	});

	it("adds subagent tokens, cost, and tool calls without dropping cache fields", () => {
		const first: UsageTotals = {
			tokens: { input: 10, output: 2, cache_read: 3, cache_write: 1, total: 16 },
			cost: 0.1,
			tool_calls: 2,
		};
		const second: UsageTotals = {
			tokens: { input: 20, output: 4, cache_read: 6, cache_write: 2, total: 32 },
			cost: 0.2,
			tool_calls: 3,
		};
		expect(addUsageTotals(undefined, first, second)).toEqual({
			tokens: { input: 30, output: 6, cache_read: 9, cache_write: 3, total: 48 },
			cost: 0.30000000000000004,
			tool_calls: 5,
		});
		expect(emptyUsageTotals().tokens.total).toBe(0);
	});

	it("accounts a persisted subagent session once even if its callback is retried", () => {
		const result = {} as AgentResult;
		const usage: SubagentRunUsage = {
			role: "view_search",
			accounting_id: "session-123",
			started_at: "2026-08-08T00:00:00.000Z",
			finished_at: "2026-08-08T00:00:01.000Z",
			duration_s: 1,
			status: "success",
			result_path: "drawing_feedback/runs/abc/view_manifest.json",
			tokens: { input: 10, output: 2, cache_read: 0, cache_write: 0, total: 12 },
			cost: 0.1,
			tool_calls: 4,
		};
		expect(recordSubagentUsage(result, usage)).toBe(true);
		expect(recordSubagentUsage(result, usage)).toBe(false);
		expect(result.subagent_usage).toEqual({
			tokens: usage.tokens,
			cost: 0.1,
			tool_calls: 4,
		});
		expect(result.subagent_runs).toEqual([usage]);

		const failedRetry: SubagentRunUsage = {
			...usage,
			status: "error",
			error: "manifest rejected",
		};
		expect(recordSubagentUsage(result, failedRetry)).toBe(false);
		expect(result.subagent_runs).toEqual([failedRetry]);
		expect(result.subagent_usage?.tokens.total).toBe(12);
	});
});
