import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Value } from "typebox/value";

const createAgentSessionMock = vi.hoisted(() => vi.fn());

vi.mock("@earendil-works/pi-coding-agent", async () => {
	const actual = await vi.importActual<typeof import("@earendil-works/pi-coding-agent")>(
		"@earendil-works/pi-coding-agent",
	);
	return {
		...actual,
		createAgentSession: createAgentSessionMock,
		DefaultResourceLoader: class {
			async reload(): Promise<void> {}
		},
		getAgentDir: () => "/tmp/cadgenbench-pi-test-agent",
		SessionManager: { create: () => ({}) },
		SettingsManager: { inMemory: () => ({}) },
	};
});

import {
	CROP_REGION_PARAMETERS_SCHEMA,
	createDrawingSegmentationTools,
	createViewSearchTools,
	makeSearchState,
	makeSegmentationState,
	regionManifestLooksCurrent,
	runDrawingSegmentation,
	SUBMIT_REGION_MANIFEST_PARAMETERS_SCHEMA,
	SUBMIT_VIEW_RESULT_PARAMETERS_SCHEMA,
} from "../src/drawing-reflection.js";
import type { WorkerClient, WorkerRequest } from "../src/local-worker.js";
import {
	drawingSegmentationPrompt,
	SEGMENTATION_SYSTEM_PROMPT,
} from "../src/prompts.js";
import type { ResolvedModel } from "../src/runner.js";
import { RunController } from "../src/runner.js";
import {
	PROTOCOL_VERSION,
	SEGMENTATION_AGENT_TOOLS,
	VIEW_SEARCH_AGENT_TOOLS,
	type AgentJob,
	type DrawingRegionManifest,
	type DrawingViewSpec,
	type SubagentRunUsage,
	type WorkerCropDrawingRegionResponse,
	type WorkerResponse,
} from "../src/types.js";

const transportDiagnostics = {
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
import { corruptIhdrCrc, fakeIhdrOnly, makePng, rewritePngDimensions } from "./png-fixture.js";

interface TestTool {
	name: string;
	execute: (...arguments_: unknown[]) => Promise<unknown>;
}

interface SessionCreationOptions {
	tools: string[];
	customTools: TestTool[];
}

interface ToolResult<T> {
	details: T;
}

const workspaces: string[] = [];

beforeEach(() => {
	createAgentSessionMock.mockReset();
});

afterEach(async () => {
	for (const workspace of workspaces.splice(0)) {
		await rm(workspace, { recursive: true, force: true });
	}
});

async function makeWorkspace(prefix: string): Promise<string> {
	const workspace = await mkdtemp(join(tmpdir(), prefix));
	workspaces.push(workspace);
	return workspace;
}

function sha256(bytes: Uint8Array | string): string {
	return createHash("sha256").update(bytes).digest("hex");
}

function job(stagedPaths = ["page-0.png", "page-1.png"]): AgentJob {
	return {
		protocol_version: PROTOCOL_VERSION,
		operation: "run_fixture",
		run_id: "segmentation-test",
		fixture: {
			id: "145",
			description: "Two merged engineering-drawing pages",
			inputFiles: stagedPaths,
			inputType: "image",
			taskType: "generation",
			sourceDir: "fixtures/145",
		},
		staged_paths: stagedPaths,
		provider: "test-provider",
		model: "test-model",
		thinking_level: "low",
		pi_version: "0.84.1",
		container_image: "cadgenbench-pi-agent:0.5.0",
		container_image_digest: "sha256:test",
	};
}

function fakeResolvedModel(): ResolvedModel {
	return { runtime: {}, model: {} } as unknown as ResolvedModel;
}

async function invokeTool<T>(tool: TestTool, callId: string, parameters: object): Promise<ToolResult<T>> {
	return (await tool.execute(callId, parameters, undefined, undefined, {})) as ToolResult<T>;
}

function installSuccessfulSession(options: {
	regions: Array<{
		region_id: string;
		page: number;
		roi: [number, number, number, number];
		view_type_hint: "orthographic" | "isometric" | "section" | "detail" | "unknown";
	}>;
	usage?: { input: number; output: number; total: number; cost: number; toolCalls: number };
	onPromptImages?: (images: Array<{ type: string; data: string; mimeType: string }>) => void;
}): void {
	createAgentSessionMock.mockImplementation(async (creation: SessionCreationOptions) => {
		const unsubscribe = vi.fn();
		return {
			session: {
				sessionId: "segmentation-session",
				getActiveToolNames: () => [...creation.tools],
				getSessionStats: () => ({
					sessionFile: undefined,
					tokens: {
						input: options.usage?.input ?? 12,
						output: options.usage?.output ?? 3,
						cacheRead: 0,
						cacheWrite: 0,
						total: options.usage?.total ?? 15,
					},
					cost: options.usage?.cost ?? 0.02,
					toolCalls: options.usage?.toolCalls ?? options.regions.length + 1,
				}),
				subscribe: vi.fn(() => unsubscribe),
				prompt: vi.fn(async (_prompt: string, promptOptions?: { images?: Array<{ type: string; data: string; mimeType: string }> }) => {
					options.onPromptImages?.(promptOptions?.images ?? []);
					const tools = new Map(creation.customTools.map((tool) => [tool.name, tool]));
					for (const region of options.regions) {
						await invokeTool(tools.get("crop_region")!, `crop-${region.region_id}`, {
							region_id: region.region_id,
							page: region.page,
							roi: region.roi,
						});
					}
					await invokeTool(tools.get("submit_region_manifest")!, "submit-regions", {
						regions: options.regions.map((region) => ({
							region_id: region.region_id,
							view_type_hint: region.view_type_hint,
							confidence: "high",
							note: `${region.region_id} crop`,
						})),
					});
				}),
				abort: vi.fn(async () => undefined),
				dispose: vi.fn(),
			},
		};
	});
}

function makeCropWorker(workspace: string) {
	let call = 0;
	const written = new Map<string, Buffer>();
	const prepared: Buffer[] = [];
	const request = vi.fn(
		async (_workspace: string, workerRequest: WorkerRequest): Promise<WorkerResponse> => {
			if (workerRequest.operation === "prepare_images") {
				const paths = workerRequest.paths as string[];
				const images = [];
				for (const [index, sourcePath] of paths.entries()) {
					const width = 12 + index;
					const height = 8 + index;
					const relativePath = join("prepared_inputs", `${String(index).padStart(2, "0")}.png`);
					const bytes = makePng(width, height, 100 + index);
					await mkdir(dirname(join(workspace, relativePath)), { recursive: true });
					await writeFile(join(workspace, relativePath), bytes);
					prepared[index] = bytes;
					images.push({
						source_path: sourcePath,
						path: relativePath,
						mime_type: "image/png" as const,
						original_width: width * 2,
						original_height: height * 2,
						width,
						height,
					});
				}
				return { protocol_version: PROTOCOL_VERSION, operation: "prepare_images", images };
			}
			if (workerRequest.operation !== "crop_drawing_region") {
				throw new Error(`Unexpected operation ${workerRequest.operation}`);
			}
			call += 1;
			const regionId = String(workerRequest.region_id);
			const width = 100 + call;
			const height = 70 + call;
			const relativePath = join(String(workerRequest.output_dir), "crop.png");
			const bytes = makePng(width, height, call);
			await mkdir(dirname(join(workspace, relativePath)), { recursive: true });
			await writeFile(join(workspace, relativePath), bytes);
			written.set(relativePath, bytes);
			return {
				protocol_version: PROTOCOL_VERSION,
				operation: "crop_drawing_region",
				status: "ok",
				region_id: regionId,
				input_path: String(workerRequest.input_path),
				roi: workerRequest.roi as [number, number, number, number],
				crop_path: relativePath,
				width,
				height,
			};
		},
	);
	return { request, written, prepared };
}

describe("drawing segmentation contracts", () => {
	it("requires every physical view on every page, whole-page coordinates, and unclipped margin", () => {
		const taskPrompt = drawingSegmentationPrompt({
			fixture: job().fixture,
			pagePaths: ["page-0.png", "page-1.png"],
		});
		expect(SEGMENTATION_SYSTEM_PROMPT).toContain("inspect every complete page");
		expect(SEGMENTATION_SYSTEM_PROMPT).toContain("crop every separate region that depicts part geometry");
		expect(SEGMENTATION_SYSTEM_PROMPT).toContain("Never omit a region merely because an identical or similar view appears elsewhere");
		expect(SEGMENTATION_SYSTEM_PROMPT).toContain("two distinct region_ids and two persistent crops");
		expect(SEGMENTATION_SYSTEM_PROMPT).toContain("0 to 1000 over the whole page");
		expect(SEGMENTATION_SYSTEM_PROMPT).toContain("small visible white margin around the part geometry");
		expect(SEGMENTATION_SYSTEM_PROMPT).toContain("check all four edges");
		expect(SEGMENTATION_SYSTEM_PROMPT).toContain("exclude dimension text, dimension lines");
		expect(SEGMENTATION_SYSTEM_PROMPT).toContain("leader lines, explanatory notes");
		expect(SEGMENTATION_SYSTEM_PROMPT).toContain("crop_region again with the same region_id");
		expect(SEGMENTATION_SYSTEM_PROMPT).toContain("Work coverage-first and converge quickly");
		expect(SEGMENTATION_SYSTEM_PROMPT).toContain("Normally use at most two correction crops per region");
		expect(SEGMENTATION_SYSTEM_PROMPT).toContain("call submit_region_manifest immediately");
		expect(SEGMENTATION_SYSTEM_PROMPT).toContain("Crops on the same page must be pairwise disjoint");
		expect(SEGMENTATION_SYSTEM_PROMPT).toContain("their boundaries may touch, but their areas must never overlap");
		expect(taskPrompt).toContain("0=page-0.png, 1=page-1.png");
		expect(taskPrompt).toContain("every separately drawn geometry view");
		expect(taskPrompt).toContain("repeated or similar view in another place is still a separate region");
		expect(taskPrompt).toContain("whole-page 0-1000 coordinate system");
		expect(taskPrompt).toContain("dimension text, dimension lines, extension lines, arrows, leaders");
		expect(taskPrompt).toContain("Never clip real part geometry");
		expect(taskPrompt).toContain("Same-page crop areas must never overlap");
		expect(taskPrompt).toContain("create one initial crop for every view before refining any region");
		expect(taskPrompt).toContain("minor unavoidable annotation noise must not delay submission");
	});

	it("gives segmentation crop/submit only and removes crop from view search", () => {
		expect(SEGMENTATION_AGENT_TOOLS).toEqual(["crop_region", "submit_region_manifest"]);
			expect(VIEW_SEARCH_AGENT_TOOLS).toEqual([
			"render_candidate_view",
			"rank_view_candidates",
				"submit_view_result",
		]);
		expect(VIEW_SEARCH_AGENT_TOOLS).not.toContain("crop_region");
	});

	it("keeps crop and submit schemas strict and controller-owned", () => {
		expect(
			Value.Check(CROP_REGION_PARAMETERS_SCHEMA, {
				region_id: "page0-main",
				page: 0,
				roi: [10, 20, 900, 950],
			}),
		).toBe(true);
		expect(
			Value.Check(CROP_REGION_PARAMETERS_SCHEMA, {
				region_id: "page0-main",
				page: 0,
				roi: [10, 20, 900, 950],
				output_path: "/tmp/injected.png",
			}),
		).toBe(false);

		const submitted = {
			regions: [
				{
					region_id: "page0-main",
					view_type_hint: "orthographic",
					confidence: "high",
					note: "main view",
				},
			],
		};
		expect(Value.Check(SUBMIT_REGION_MANIFEST_PARAMETERS_SCHEMA, submitted)).toBe(true);
		for (const injected of [
			{ page: 0 },
			{ roi: [0, 0, 1000, 1000] },
			{ crop_path: "../../outside.png" },
			{ crop_sha256: "a".repeat(64) },
			{ width: 10, height: 10 },
		]) {
			expect(
				Value.Check(SUBMIT_REGION_MANIFEST_PARAMETERS_SCHEMA, {
					regions: [{ ...submitted.regions[0], ...injected }],
				}),
			).toBe(false);
		}
	});

	it("rejects overlapping same-page crops but permits touching boundaries", async () => {
		const workspace = await makeWorkspace("pi-segmentation-overlap-");
		await writeFile(join(workspace, "page-0.png"), "page zero");
		const worker = makeCropWorker(workspace);
		const state = makeSegmentationState();
		const cacheDir = join("drawing_feedback", "input_cache", "input-sha");
		const tools = createDrawingSegmentationTools({
			worker: worker as unknown as WorkerClient,
			workspace,
			pagePaths: ["page-0.png"],
			workDir: join(cacheDir, "work"),
			canonicalRegionsDir: join(cacheDir, "regions"),
			manifestPath: join(cacheDir, "region_manifest.json"),
			inputSha256: "input-sha",
			controller: new RunController(),
			state,
		});
		await invokeTool(tools[0] as TestTool, "crop-left", {
			region_id: "left-view",
			page: 0,
			roi: [0, 0, 600, 600],
		});
		await invokeTool(tools[0] as TestTool, "crop-right-overlap", {
			region_id: "right-view",
			page: 0,
			roi: [500, 100, 1000, 700],
		});
		const submittedRegions = [
			{
				region_id: "left-view",
				view_type_hint: "orthographic",
				confidence: "high",
				note: "left",
			},
			{
				region_id: "right-view",
				view_type_hint: "orthographic",
				confidence: "high",
				note: "right",
			},
		] as const;
		await expect(
			invokeTool(tools[1] as TestTool, "submit-overlap", { regions: submittedRegions }),
		).rejects.toThrow("left-view and right-view overlap on page 0");
		expect(state.submitted).toBeNull();

		await invokeTool(tools[0] as TestTool, "crop-right-touching", {
			region_id: "right-view",
			page: 0,
			roi: [600, 0, 1000, 600],
		});
		const response = await invokeTool<DrawingRegionManifest>(tools[1] as TestTool, "submit-disjoint", {
			regions: submittedRegions,
		});
		expect(response.details.regions.map(({ roi }) => roi)).toEqual([
			[0, 0, 600, 600],
			[600, 0, 1000, 600],
		]);
	});

	it("persists multiple views from two merged pages and publishes only the latest crop per id", async () => {
		const workspace = await makeWorkspace("pi-segmentation-tools-");
		await writeFile(join(workspace, "page-0.png"), "page zero");
		await writeFile(join(workspace, "page-1.png"), "page one");
		const worker = makeCropWorker(workspace);
		const state = makeSegmentationState();
		const cacheDir = join("drawing_feedback", "input_cache", "input-sha");
		const tools = createDrawingSegmentationTools({
			worker: worker as unknown as WorkerClient,
			workspace,
			pagePaths: ["page-0.png", "page-1.png"],
			workDir: join(cacheDir, "work"),
			canonicalRegionsDir: join(cacheDir, "regions"),
			manifestPath: join(cacheDir, "region_manifest.json"),
			inputSha256: "input-sha",
			controller: new RunController(),
			state,
		});

		await invokeTool(tools[0] as TestTool, "crop-main-draft", {
			region_id: "page0-main",
			page: 0,
			roi: [20, 30, 480, 520],
		});
		await invokeTool(tools[0] as TestTool, "crop-main-revised", {
			region_id: "page0-main",
			page: 0,
			roi: [25, 35, 500, 540],
		});
		await invokeTool(tools[0] as TestTool, "crop-side", {
			region_id: "page0-side",
			page: 0,
			roi: [520, 20, 980, 500],
		});
		await invokeTool(tools[0] as TestTool, "crop-iso", {
			region_id: "page1-iso",
			page: 1,
			roi: [20, 480, 490, 980],
		});
		await invokeTool(tools[0] as TestTool, "crop-bottom", {
			region_id: "page1-bottom",
			page: 1,
			roi: [510, 20, 980, 470],
		});
		await expect(
			invokeTool(tools[1] as TestTool, "submit-duplicate", {
				regions: [
					{
						region_id: "page0-main",
						view_type_hint: "orthographic",
						confidence: "high",
						note: "main",
					},
					{
						region_id: "page0-main",
						view_type_hint: "orthographic",
						confidence: "high",
						note: "duplicate",
					},
				],
			}),
		).rejects.toThrow("Duplicate region page0-main");

		const response = await invokeTool<DrawingRegionManifest>(tools[1] as TestTool, "submit", {
			regions: [
				{
					region_id: "page0-main",
					view_type_hint: "orthographic",
					confidence: "high",
					note: "revised main",
				},
				{
					region_id: "page0-side",
					view_type_hint: "section",
					confidence: "medium",
					note: "side section",
				},
				{
					region_id: "page1-iso",
					view_type_hint: "isometric",
					confidence: "high",
					note: "isometric",
				},
				{
					region_id: "page1-bottom",
					view_type_hint: "orthographic",
					confidence: "high",
					note: "bottom",
				},
			],
		});

		const manifest = response.details;
		expect(manifest.regions).toHaveLength(4);
		expect(manifest.regions.find(({ region_id }) => region_id === "page0-main")).toMatchObject({
			page: 0,
			roi: [25, 35, 500, 540],
			width: 102,
			height: 72,
		});
		expect(regionManifestLooksCurrent(manifest, "input-sha", 2)).toBe(true);

		const canonicalRoot = join(workspace, cacheDir, "regions");
		expect((await readdir(canonicalRoot)).sort()).toEqual(
			["page0-main", "page0-side", "page1-bottom", "page1-iso"].sort(),
		);
		for (const region of manifest.regions) {
			expect(await readdir(join(canonicalRoot, region.region_id))).toEqual(["crop.png"]);
			const bytes = await readFile(join(workspace, region.crop_path));
			expect(sha256(bytes)).toBe(region.crop_sha256);
		}
		expect(
			JSON.parse(await readFile(join(workspace, cacheDir, "region_manifest.json"), "utf8")),
		).toEqual(manifest);
		const workRegionsRoot = join(workspace, cacheDir, "work", "regions");
		expect((await readdir(workRegionsRoot)).sort()).toEqual(
			["page0-main", "page0-side", "page1-bottom", "page1-iso"].sort(),
		);
		for (const regionId of await readdir(workRegionsRoot)) {
			expect(await readdir(join(workRegionsRoot, regionId))).toEqual(["crop.png"]);
		}
		const mainBytes = await readFile(
			join(workspace, manifest.regions.find(({ region_id }) => region_id === "page0-main")!.crop_path),
		);
		expect(mainBytes).toEqual(makePng(102, 72, 2));
	});

	it("reserves call 32 for region submission after 31 segmentation work calls", async () => {
		const workspace = await makeWorkspace("pi-segmentation-budget-");
		const cropPath = "drawing_feedback/input_cache/input/work/regions/detail/crop.png";
		const cropBytes = makePng(20, 10, 9);
		await mkdir(dirname(join(workspace, cropPath)), { recursive: true });
		await writeFile(join(workspace, cropPath), cropBytes);
		const state = makeSegmentationState();
		state.toolCalls = 31;
		state.crops.set("detail", {
			regionId: "detail",
			page: 0,
			roi: [0, 0, 1000, 1000],
			cropPath,
			width: 20,
			height: 10,
			cropSha256: sha256(cropBytes),
		});
		const request = vi.fn();
		const tools = createDrawingSegmentationTools({
			worker: { request } as unknown as WorkerClient,
			workspace,
			pagePaths: ["page.png"],
			workDir: "drawing_feedback/input_cache/input/work",
			canonicalRegionsDir: "drawing_feedback/input_cache/input/regions",
			manifestPath: "drawing_feedback/input_cache/input/region_manifest.json",
			inputSha256: "input",
			controller: new RunController(),
			state,
		});
		await expect(
			invokeTool(tools[0] as TestTool, "work-call-32", {
				region_id: "detail",
				page: 0,
				roi: [0, 0, 1000, 1000],
			}),
		).rejects.toThrow("31/32 total calls; call 32 is reserved for submit_region_manifest");
		const submitted = await tools[1].execute(
			"submit-call-32",
			{
				regions: [
					{
						region_id: "detail",
						view_type_hint: "detail",
						confidence: "low",
						note: "physical detail view",
					},
				],
			},
			undefined,
			undefined,
			{} as never,
		);
		expect(state.toolCalls).toBe(32);
		expect(JSON.stringify(submitted.content)).toContain("32/32 total calls used");
		expect(request).not.toHaveBeenCalled();
	});

	it("rejects duplicate/unknown submissions and unsafe worker artifact paths", async () => {
		const workspace = await makeWorkspace("pi-segmentation-safety-");
		await writeFile(join(workspace, "page.png"), "page");
		const state = makeSegmentationState();
		const request = vi.fn(async (): Promise<WorkerCropDrawingRegionResponse> => ({
			protocol_version: PROTOCOL_VERSION,
			operation: "crop_drawing_region",
			status: "ok",
			region_id: "main",
			input_path: "page.png",
			roi: [0, 0, 500, 500],
			crop_path: "../escaped.png",
			width: 10,
			height: 10,
		}));
		const tools = createDrawingSegmentationTools({
			worker: { request } as unknown as WorkerClient,
			workspace,
			pagePaths: ["page.png"],
			workDir: "drawing_feedback/input_cache/sha/work",
			canonicalRegionsDir: "drawing_feedback/input_cache/sha/regions",
			manifestPath: "drawing_feedback/input_cache/sha/region_manifest.json",
			inputSha256: "sha",
			controller: new RunController(),
			state,
		});
		await expect(
			invokeTool(tools[0] as TestTool, "crop", {
				region_id: "main",
				page: 0,
				roi: [0, 0, 500, 500],
			}),
		).rejects.toThrow("crop identity did not match the request");
		await expect(
			invokeTool(tools[1] as TestTool, "unknown", {
				regions: [
					{
						region_id: "unknown",
						view_type_hint: "unknown",
						confidence: "low",
						note: "not cropped",
					},
				],
			}),
		).rejects.toThrow("has no controller-saved crop");
	});

	it("lets search change only the view classification, not locked crop identity", async () => {
		const regionManifest: DrawingRegionManifest = {
			version: 1,
			input_sha256: "a".repeat(64),
			regions: [
				{
					region_id: "page0-main",
					page: 0,
					roi: [10, 20, 900, 950],
					crop_path: "drawing_feedback/input_cache/input/regions/page0-main/crop.png",
					view_type_hint: "unknown",
					confidence: "low",
					note: "uncertain",
					width: 640,
					height: 480,
					crop_sha256: "b".repeat(64),
				},
			],
		};
		const region = regionManifest.regions[0]!;
		const state = makeSearchState(region);
		const rankings = Array.from({ length: 2 }, (_, index) => {
			const candidateId = `candidate-00${index + 1}`;
			const score = 0.9 - index * 0.1;
			return {
				candidate_id: candidateId,
				score,
				raw_score: score,
				eligible: true,
				failed_gates: [],
				scoring_mode: "capacity_transport" as const,
				capacity_transport: { ...transportDiagnostics, score, normalized_cost: 1 - score },
				view_spec: {
					name: candidateId,
					direction: [0, -1, 0] as DrawingViewSpec["direction"],
					up: [0, 0, 1] as DrawingViewSpec["up"],
				},
				render_path: `renders/${candidateId}.png`,
				comparison_path: `comparisons/${candidateId}.png`,
				...alignmentFields(`aligned/${candidateId}.png`),
			};
		});
		state.ranking = {
			rankings,
			margin: 0.1,
			comparisonPath: "comparisons/page0-main.png",
		};
		const tools = createViewSearchTools({
			worker: { request: vi.fn() } as unknown as WorkerClient,
			workspace: "/tmp",
			region,
			outputDir: "drawing_feedback/run",
			expectedStepSha256: "step-sha",
			controller: new RunController(),
			state,
		});
		expect(tools.map(({ name }) => name)).toEqual([...VIEW_SEARCH_AGENT_TOOLS]);
		expect(
			Value.Check(SUBMIT_VIEW_RESULT_PARAMETERS_SCHEMA, {
				view_type: "detail",
				status: "unsupported",
				confidence: "medium",
				note: "corrected classification",
			}),
		).toBe(false);
		expect(
			Value.Check(SUBMIT_VIEW_RESULT_PARAMETERS_SCHEMA, {
				view_type: "detail",
				status: "ambiguous",
				confidence: "medium",
				note: "candidate is required",
			}),
		).toBe(false);

		const submitted = await invokeTool<Record<string, unknown>>(
			tools[2] as TestTool,
			"submit-view",
			{
				view_type: "detail",
				status: "ambiguous",
				selected_candidate_id: "candidate-001",
				confidence: "medium",
				note: "corrected classification with the best available Top-2 view",
			},
		);
		expect(submitted.details).toMatchObject({
			region_id: "page0-main",
			page: regionManifest.regions[0]!.page,
			roi: regionManifest.regions[0]!.roi,
			crop_path: regionManifest.regions[0]!.crop_path,
			width: regionManifest.regions[0]!.width,
			height: regionManifest.regions[0]!.height,
			crop_sha256: regionManifest.regions[0]!.crop_sha256,
			view_type: "detail",
			status: "ambiguous",
			selected: { candidate_id: "candidate-001" },
		});
	});

	it("lets search correct wrong hints to supported orthographic, isometric, and section types", async () => {
		const scenarios: Array<{
			regionId: string;
			hint: DrawingRegionManifest["regions"][number]["view_type_hint"];
			viewType: "orthographic" | "isometric" | "section";
			spec: Omit<DrawingViewSpec, "name">;
		}> = [
			{
				regionId: "corrected-ortho",
				hint: "isometric",
				viewType: "orthographic",
				spec: { direction: [1, 0, 0], up: [0, 0, 1] },
			},
			{
				regionId: "corrected-iso",
				hint: "orthographic",
				viewType: "isometric",
				spec: { direction: [1, -1, 1], up: [1, 1, 0] },
			},
			{
				regionId: "corrected-section",
				hint: "unknown",
				viewType: "section",
				spec: {
					direction: [0, -1, 0],
					up: [0, 0, 1],
					section_offset_fraction: 0,
					section_keep: "front",
				},
			},
		];
		const regionManifest: DrawingRegionManifest = {
			version: 1,
			input_sha256: "a".repeat(64),
			regions: scenarios.map((scenario, index) => ({
				region_id: scenario.regionId,
				page: 0,
				roi: [index * 100, 0, index * 100 + 90, 900],
				crop_path: `drawing_feedback/input_cache/input/regions/${scenario.regionId}/crop.png`,
				view_type_hint: scenario.hint,
				confidence: "low",
				note: "fallible hint",
				width: 100,
				height: 100,
				crop_sha256: String(index + 1).repeat(64),
			})),
		};
		const submitted = [];
		for (const [scenarioIndex, scenario] of scenarios.entries()) {
			const region = regionManifest.regions[scenarioIndex]!;
			const state = makeSearchState(region);
			const rankings = Array.from({ length: 2 }, (_, rank) => {
				const candidateId = `candidate-${scenarioIndex}-${rank}`;
				const score = 0.9 - rank * 0.1;
				return {
					candidate_id: candidateId,
					score,
					raw_score: score,
					eligible: true,
					failed_gates: [],
					scoring_mode: "capacity_transport" as const,
					capacity_transport: { ...transportDiagnostics, score, normalized_cost: 1 - score },
					view_spec: { name: candidateId, ...scenario.spec },
					render_path: `renders/${candidateId}.png`,
					comparison_path: `comparisons/${candidateId}.png`,
					...alignmentFields(`aligned/${candidateId}.png`),
				};
			});
			state.ranking = {
				rankings,
				margin: 0.1,
				comparisonPath: `comparisons/${scenario.regionId}.png`,
			};
			const tools = createViewSearchTools({
				worker: { request: vi.fn() } as unknown as WorkerClient,
				workspace: "/tmp",
				region,
				outputDir: `drawing_feedback/run/view-search/${scenario.regionId}`,
				expectedStepSha256: "step-sha",
				controller: new RunController(),
				state,
			});
			const result = await invokeTool<{ region_id: string; view_type: string }>(
				tools[2] as TestTool,
				`submit-${scenario.regionId}`,
				{
					view_type: scenario.viewType,
					status: "matched",
					selected_candidate_id: `candidate-${scenarioIndex}-1`,
					confidence: "high",
					note: "corrected from segmentation hint",
				},
			);
			submitted.push(result.details);
		}
		expect(submitted.map(({ region_id, view_type }) => ({ region_id, view_type }))).toEqual(
			scenarios.map(({ regionId, viewType }) => ({ region_id: regionId, view_type: viewType })),
		);
	});
});

describe("standalone drawing segmentation session", () => {
	it("caches a verified multi-page RegionManifest and skips the second LLM session", async () => {
		const workspace = await makeWorkspace("pi-segmentation-cache-");
		await writeFile(join(workspace, "page-0.png"), "merged page zero");
		await writeFile(join(workspace, "page-1.png"), "merged page one");
		const regions = [
			{
				region_id: "page0-main",
				page: 0,
				roi: [10, 20, 480, 600] as [number, number, number, number],
				view_type_hint: "orthographic" as const,
			},
			{
				region_id: "page0-section",
				page: 0,
				roi: [500, 30, 980, 620] as [number, number, number, number],
				view_type_hint: "section" as const,
			},
			{
				region_id: "page1-iso",
				page: 1,
				roi: [30, 400, 500, 980] as [number, number, number, number],
				view_type_hint: "isometric" as const,
			},
		];
		const promptImages: Array<Array<{ type: string; data: string; mimeType: string }>> = [];
		installSuccessfulSession({
			regions,
			onPromptImages: (images) => promptImages.push(images),
		});
		const worker = makeCropWorker(workspace);
		const usages: SubagentRunUsage[] = [];
		const options = {
			job: job(),
			worker: worker as unknown as WorkerClient,
			workspace,
			resolvedModel: fakeResolvedModel(),
			controller: new RunController(),
			onUsage: async (usage: SubagentRunUsage) => {
				usages.push(usage);
			},
		};

		const first = await runDrawingSegmentation(options);
		expect(first.cached).toBe(false);
		expect(first.manifest.regions).toHaveLength(3);
		expect(first.usage).toMatchObject({ role: "segmentation", status: "success" });
		expect(createAgentSessionMock).toHaveBeenCalledOnce();
		expect(worker.request).toHaveBeenCalledTimes(4);
		const prepareRequest = worker.request.mock.calls[0]?.[1] as WorkerRequest;
		expect(prepareRequest).toMatchObject({
			operation: "prepare_images",
			paths: ["page-0.png", "page-1.png"],
			max_edge: 2000,
		});
		expect(promptImages).toHaveLength(1);
		expect(promptImages[0]).toEqual(
			worker.prepared.map((bytes) => ({
				type: "image",
				data: bytes.toString("base64"),
				mimeType: "image/png",
			})),
		);

		const second = await runDrawingSegmentation({ ...options, controller: new RunController() });
		expect(second.cached).toBe(true);
		expect(second.usage).toBeUndefined();
		expect(second.manifest).toEqual(first.manifest);
		expect(createAgentSessionMock).toHaveBeenCalledOnce();
		expect(worker.request).toHaveBeenCalledTimes(4);
		expect(promptImages).toHaveLength(1);
		expect(usages).toHaveLength(1);
	});

	it("invalidates a cached crop whose bytes no longer match its locked hash", async () => {
		const workspace = await makeWorkspace("pi-segmentation-hash-");
		await writeFile(join(workspace, "page-0.png"), "merged page zero");
		await writeFile(join(workspace, "page-1.png"), "merged page one");
		const regions = [
			{
				region_id: "page0-main",
				page: 0,
				roi: [10, 20, 900, 950] as [number, number, number, number],
				view_type_hint: "orthographic" as const,
			},
		];
		installSuccessfulSession({ regions });
		const worker = makeCropWorker(workspace);
		const baseOptions = {
			job: job(),
			worker: worker as unknown as WorkerClient,
			workspace,
			resolvedModel: fakeResolvedModel(),
			onUsage: async (_usage: SubagentRunUsage) => undefined,
		};
		const first = await runDrawingSegmentation({ ...baseOptions, controller: new RunController() });
		const crop = first.manifest.regions[0]!;
		await writeFile(join(workspace, crop.crop_path), makePng(crop.width, crop.height, 255));

		const second = await runDrawingSegmentation({ ...baseOptions, controller: new RunController() });
		expect(second.cached).toBe(false);
		expect(createAgentSessionMock).toHaveBeenCalledTimes(2);
		expect(second.manifest.regions[0]!.crop_sha256).not.toBe(crop.crop_sha256);
		expect(sha256(await readFile(join(workspace, second.manifest.regions[0]!.crop_path)))).toBe(
			second.manifest.regions[0]!.crop_sha256,
		);
	});

	it.each([
		{
			name: "a truncated PNG",
			corrupt: (png: Buffer) => png.subarray(0, Math.max(0, png.length - 8)),
			width: 101,
			height: 71,
		},
		{
			name: "a signature-plus-IHDR impostor",
			corrupt: (_png: Buffer) => fakeIhdrOnly(101, 71),
			width: 101,
			height: 71,
		},
		{
			name: "a PNG with a bad IHDR CRC",
			corrupt: (png: Buffer) => corruptIhdrCrc(png),
			width: 101,
			height: 71,
		},
		{
			name: "a PNG claiming oversized dimensions",
			corrupt: (png: Buffer) => rewritePngDimensions(png, 20_000, 2_000),
			width: 20_000,
			height: 2_000,
		},
	])("rejects $name even when the cached hash and metadata are rewritten to trust it", async (scenario) => {
		const workspace = await makeWorkspace("pi-segmentation-malformed-png-");
		await writeFile(join(workspace, "page-0.png"), "merged page zero");
		await writeFile(join(workspace, "page-1.png"), "merged page one");
		const regions = [
			{
				region_id: "page0-main",
				page: 0,
				roi: [10, 20, 900, 950] as [number, number, number, number],
				view_type_hint: "orthographic" as const,
			},
		];
		installSuccessfulSession({ regions });
		const worker = makeCropWorker(workspace);
		const options = {
			job: job(),
			worker: worker as unknown as WorkerClient,
			workspace,
			resolvedModel: fakeResolvedModel(),
			onUsage: async (_usage: SubagentRunUsage) => undefined,
		};
		const first = await runDrawingSegmentation({ ...options, controller: new RunController() });
		const region = first.manifest.regions[0]!;
		const validBytes = await readFile(join(workspace, region.crop_path));
		const corruptBytes = scenario.corrupt(validBytes);
		await writeFile(join(workspace, region.crop_path), corruptBytes);
		const trustedCorruption = structuredClone(first.manifest);
		trustedCorruption.regions[0]!.crop_sha256 = sha256(corruptBytes);
		trustedCorruption.regions[0]!.width = scenario.width;
		trustedCorruption.regions[0]!.height = scenario.height;
		await writeFile(
			join(workspace, first.manifestPath),
			`${JSON.stringify(trustedCorruption, null, 2)}\n`,
		);

		const second = await runDrawingSegmentation({ ...options, controller: new RunController() });
		expect(second.cached).toBe(false);
		expect(createAgentSessionMock).toHaveBeenCalledTimes(2);
		expect(second.manifest.regions[0]!.crop_sha256).not.toBe(sha256(corruptBytes));
	});

	it("propagates SIGINT to segmentation and records its usage as interrupted", async () => {
		const workspace = await makeWorkspace("pi-segmentation-abort-");
		await writeFile(join(workspace, "page-0.png"), "merged page zero");
		await writeFile(join(workspace, "page-1.png"), "merged page one");
		let promptStartedResolve: (() => void) | undefined;
		const promptStarted = new Promise<void>((resolve) => {
			promptStartedResolve = resolve;
		});
		let rejectPrompt: ((error: Error) => void) | undefined;
		const abort = vi.fn(async () => {
			rejectPrompt?.(new Error("segmentation session aborted"));
		});
		const unsubscribe = vi.fn();
		createAgentSessionMock.mockImplementation(async (creation: SessionCreationOptions) => ({
			session: {
				sessionId: "segmentation-aborted-session",
				getActiveToolNames: () => [...creation.tools],
				getSessionStats: () => ({
					sessionFile: undefined,
					tokens: { input: 21, output: 2, cacheRead: 3, cacheWrite: 1, total: 27 },
					cost: 0.04,
					toolCalls: 1,
				}),
				subscribe: vi.fn(() => unsubscribe),
				prompt: vi.fn(async () => {
					promptStartedResolve?.();
					await new Promise<never>((_resolve, reject) => {
						rejectPrompt = reject;
					});
				}),
				abort,
				dispose: vi.fn(),
			},
		}));
		const usages: SubagentRunUsage[] = [];
		const controller = new RunController();
		const worker = makeCropWorker(workspace);
		const execution = runDrawingSegmentation({
			job: job(),
			worker: worker as unknown as WorkerClient,
			workspace,
			resolvedModel: fakeResolvedModel(),
			controller,
			onUsage: async (usage) => {
				usages.push(usage);
			},
		});

		await promptStarted;
		await controller.interrupt("SIGINT");
		await expect(execution).rejects.toThrow("segmentation session aborted");
		expect(abort).toHaveBeenCalledOnce();
		expect(unsubscribe).toHaveBeenCalledOnce();
		expect(usages).toHaveLength(1);
		expect(usages[0]).toMatchObject({
			role: "segmentation",
			status: "interrupted",
			accounting_id: "segmentation-aborted-session",
			tokens: { input: 21, output: 2, cache_read: 3, cache_write: 1, total: 27 },
			cost: 0.04,
			tool_calls: 1,
		});
	});
});
