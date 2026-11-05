import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

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
	CURRENT_CAD_REFERENCE_VIEW_SPEC,
	createFixedGridReflectionTools as createReflectionTools,
	createDrawingReflectionTool,
	restoreReflectionInspectionManifest,
	runDrawingViewSearchOnly,
} from "../src/drawing-reflection.js";
import type { WorkerClient, WorkerRequest } from "../src/local-worker.js";
import type { ResolvedModel } from "../src/runner.js";
import { RunController } from "../src/runner.js";
import {
	DRAWING_FEEDBACK_VERSION,
	PROTOCOL_VERSION,
	REFLECTION_AGENT_TOOLS,
	SEGMENTATION_AGENT_TOOLS,
	VIEW_SEARCH_AGENT_TOOLS,
	type AgentJob,
	type AgentResult,
	type ArtifactPayload,
	type DrawingFeedbackSummary,
	type DrawingRegionManifest,
	type DrawingReflectionReport,
	type DrawingViewManifest,
	type DrawingViewSpec,
	type SubagentRunUsage,
	type WorkerResponse,
} from "../src/types.js";
import { makePng } from "./png-fixture.js";

interface TestTool {
	name: string;
	execute: (...arguments_: unknown[]) => Promise<unknown>;
}

interface SessionCreationOptions {
	tools: string[];
	customTools: TestTool[];
}

interface TestToolExecutionEvent {
	type: "tool_execution_start" | "tool_execution_end";
	toolName: string;
	toolCallId: string;
}

interface ToolResult<T> {
	details: T;
}

type LlmDrawingViewSpec = Omit<DrawingViewSpec, "name">;

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

function capacityFields(score: number) {
	return {
		raw_score: score,
		eligible: true,
		failed_gates: [] as string[],
		scoring_mode: "capacity_transport" as const,
		capacity_transport: { ...transportDiagnostics, score, normalized_cost: 1 - score },
	};
}

function alignmentFields(alignedRenderPath: string) {
	return {
		aligned_render_path: alignedRenderPath,
		transform: {
			scale_px_per_unit: 1,
			rotation_deg: 0,
			center_x_px: 200,
			center_y_px: 150,
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

const workspaces: string[] = [];

beforeEach(() => {
	createAgentSessionMock.mockReset();
});

afterEach(async () => {
	vi.useRealTimers();
	for (const workspace of workspaces.splice(0)) {
		await rm(workspace, { recursive: true, force: true });
	}
});

async function makeWorkspace(prefix: string): Promise<string> {
	const workspace = await mkdtemp(join(tmpdir(), prefix));
	workspaces.push(workspace);
	return workspace;
}

function sha256(value: string | Uint8Array): string {
	return createHash("sha256").update(value).digest("hex");
}

function drawingInputSha(path: string, contents: string): string {
	return createHash("sha256").update(path).update("\0").update(contents).update("\0").digest("hex");
}

function feedbackKey(inputSha256: string, stepSha256: string, provider: string, model: string): string {
	return createHash("sha256")
		.update([DRAWING_FEEDBACK_VERSION, inputSha256, stepSha256, provider, model].join("\0"))
		.digest("hex");
}

function job(stagedPaths = ["drawing.png"]): AgentJob {
	return {
		protocol_version: PROTOCOL_VERSION,
		operation: "run_fixture",
		run_id: "pipeline-test",
		fixture: {
			id: "145",
			description: "A merged engineering drawing",
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

function validArtifact(stepSha256: string): ArtifactPayload {
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
			volume: 100,
			bbox: { size_x: 10, size_y: 20, size_z: 5 },
		},
		validation_error: null,
		render_path: null,
		render_error: null,
	};
}

function agentResult(sourceSha256: string, stepSha256: string): AgentResult {
	return {
		fixture_id: "145",
		status: "running",
		provider: "test-provider",
		model: "test-model",
		thinking_level: "low",
		pi_version: "0.84.1",
		container_image: "cadgenbench-pi-agent:0.5.0",
		container_image_digest: "sha256:test",
		started_at: "2026-08-08T00:00:00.000Z",
		updated_at: "2026-08-08T00:00:00.000Z",
		duration_s: 0,
		last_turn: 1,
		artifact: validArtifact(stepSha256),
		last_executed_source_sha256: sourceSha256,
		last_executed_step_sha256: stepSha256,
		tokens: { input: 0, output: 0, cache_read: 0, cache_write: 0, total: 0 },
		cost: 0,
		tool_calls: 0,
	};
}

function inspectResponse(stepSha256: string): WorkerResponse {
	return {
		protocol_version: PROTOCOL_VERSION,
		operation: "inspect",
		artifact: validArtifact(stepSha256),
	};
}

function fakeResolvedModel(): ResolvedModel {
	return { runtime: {}, model: {} } as unknown as ResolvedModel;
}

async function invokeTool<T>(tool: TestTool, callId: string, parameters: object): Promise<ToolResult<T>> {
	return (await tool.execute(callId, parameters, undefined, undefined, {})) as ToolResult<T>;
}

function makePublicTool(options: {
	workspace: string;
	worker: WorkerClient;
	result: AgentResult;
	stagedPaths?: string[];
	lockedViewManifestPath?: string;
	controller?: RunController;
	onUsage?: (usage: SubagentRunUsage) => Promise<void>;
	onComplete?: (summary: DrawingFeedbackSummary) => Promise<void>;
}) {
	return createDrawingReflectionTool({
		job: job(options.stagedPaths),
		worker: options.worker,
		workspace: options.workspace,
		resolvedModel: fakeResolvedModel(),
		controller: options.controller ?? new RunController(),
		...(options.lockedViewManifestPath
			? { lockedViewManifestPath: options.lockedViewManifestPath }
			: {}),
		getResult: () => options.result,
		onUsage: options.onUsage ?? (async () => undefined),
		onComplete: options.onComplete ?? (async () => undefined),
	});
}

async function writeJson(path: string, value: unknown): Promise<void> {
	await mkdir(dirname(path), { recursive: true });
	await writeFile(path, `${JSON.stringify(value, null, 2)}\n`);
}

async function writeArtifact(
	workspace: string,
	relativePath: string,
	width = 400,
	height = 300,
	marker = 1,
): Promise<void> {
	const absolutePath = join(workspace, relativePath);
	await mkdir(dirname(absolutePath), { recursive: true });
	await writeFile(absolutePath, makePng(width, height, marker));
}

async function prepareResponse(workspace: string, request: WorkerRequest): Promise<WorkerResponse> {
	const paths = request.paths as string[];
	const images = [];
	for (const [index, sourcePath] of paths.entries()) {
		const path = join("prepared_inputs", `${String(index).padStart(2, "0")}.png`);
		await writeArtifact(workspace, path, 320, 240, 70 + index);
		images.push({
			source_path: sourcePath,
			path,
			mime_type: "image/png" as const,
			original_width: 640,
			original_height: 480,
			width: 320,
			height: 240,
		});
	}
	return { protocol_version: PROTOCOL_VERSION, operation: "prepare_images", images };
}

async function reflectionInspectionResponse(
	workspace: string,
	request: WorkerRequest,
): Promise<WorkerResponse> {
	const outputDir = String(request.output_dir);
	const mode = request.mode as "grid_overview" | "grid_comparison";
	if (mode === "grid_overview") {
		const contextPath = join(outputDir, "grid-overview.png");
		await writeArtifact(workspace, contextPath, 800, 500, 94);
		return {
			protocol_version: PROTOCOL_VERSION,
			operation: "compose_reflection_inspection",
			status: "ok",
			region_id: String(request.region_id),
			mode,
			context_path: contextPath,
			width: 400,
			height: 300,
			sha256: { context: sha256(await readFile(join(workspace, contextPath))) },
		};
	}
	const referenceCropPath = join(outputDir, "reference-crop.png");
	const cadCropPath = join(outputDir, "cad-crop.png");
	const contextPath = join(outputDir, "context.png");
	const comparisonPath = join(outputDir, "comparison.png");
	await Promise.all([
		writeArtifact(workspace, referenceCropPath, 100, 100, 92),
		writeArtifact(workspace, cadCropPath, 100, 100, 93),
		writeArtifact(workspace, contextPath, 800, 500, 94),
		writeArtifact(workspace, comparisonPath, 800, 900, 95),
	]);
	const cellRoi = (cell: string): [number, number, number, number] => {
		const column = cell.charCodeAt(0) - "A".charCodeAt(0);
		const row = Number.parseInt(cell.slice(1), 10) - 1;
		return [column * 250, row * 250, (column + 1) * 250, (row + 1) * 250];
	};
	const referenceCell = String(request.reference_cell);
	const cadCell = String(request.cad_cell);
	const referenceRoi = cellRoi(referenceCell);
	const cadRoi = cellRoi(cadCell);
	return {
		protocol_version: PROTOCOL_VERSION,
		operation: "compose_reflection_inspection",
		status: "ok",
		region_id: String(request.region_id),
		mode,
		reference_cell: referenceCell,
		cad_cell: cadCell,
		comparison_path: comparisonPath,
		width: 400,
		height: 300,
		reference_roi: referenceRoi,
		cad_roi: cadRoi,
		reference_pixel_roi: [40, 30, 160, 120],
		cad_pixel_roi: [80, 45, 200, 135],
		reference_crop_path: referenceCropPath,
		cad_crop_path: cadCropPath,
		context_path: contextPath,
		sha256: {
			reference_crop: sha256(await readFile(join(workspace, referenceCropPath))),
			cad_crop: sha256(await readFile(join(workspace, cadCropPath))),
			context: sha256(await readFile(join(workspace, contextPath))),
			comparison: sha256(await readFile(join(workspace, comparisonPath))),
		},
	};
}

async function inputPageCropResponse(
	workspace: string,
	request: WorkerRequest,
): Promise<WorkerResponse> {
	const cropPath = join(String(request.output_dir), "crop.png");
	await writeArtifact(workspace, cropPath, 180, 120, 96);
	return {
		protocol_version: PROTOCOL_VERSION,
		operation: "crop_drawing_region",
		status: "ok",
		region_id: String(request.region_id),
		input_path: String(request.input_path),
		roi: request.roi as [number, number, number, number],
		crop_path: cropPath,
		width: 180,
		height: 120,
	};
}

async function currentCadReferenceResponse(
	workspace: string,
	request: WorkerRequest,
	stepSha256: string,
): Promise<WorkerResponse> {
	const renderPath = join(String(request.output_dir), "render.png");
	await writeArtifact(workspace, renderPath, 400, 300, 99);
	return {
		protocol_version: PROTOCOL_VERSION,
		operation: "render_drawing_view",
		status: "ok",
		region_id: String(request.region_id),
		view_spec: request.view_spec as DrawingViewSpec,
		render_path: renderPath,
		width: 400,
		height: 300,
		step_sha256: stepSha256,
	};
}

function zeroUsage(role: SubagentRunUsage["role"]): SubagentRunUsage {
	return {
		role,
		accounting_id: `${role}-cached`,
		started_at: "2026-08-08T00:00:00.000Z",
		finished_at: "2026-08-08T00:00:01.000Z",
		duration_s: 1,
		status: "success",
		tokens: { input: 0, output: 0, cache_read: 0, cache_write: 0, total: 0 },
		cost: 0,
		tool_calls: 0,
	};
}

async function installRegionCache(options: {
	workspace: string;
	drawingContents: string;
	regionId?: string;
}): Promise<DrawingRegionManifest> {
	const inputSha256 = drawingInputSha("drawing.png", options.drawingContents);
	const regionId = options.regionId ?? "page0-main";
	const cacheDir = join("drawing_feedback", "input_cache", inputSha256);
	const cropPath = join(cacheDir, "regions", regionId, "crop.png");
	await writeArtifact(options.workspace, cropPath);
	const cropSha256 = sha256(await readFile(join(options.workspace, cropPath)));
	const manifest: DrawingRegionManifest = {
		version: 1,
		input_sha256: inputSha256,
		regions: [
			{
				region_id: regionId,
				page: 0,
				roi: [0, 0, 1000, 1000],
				crop_path: cropPath,
				view_type_hint: "orthographic",
				confidence: "high",
				note: "Cached complete drawing view",
				width: 400,
				height: 300,
				crop_sha256: cropSha256,
			},
		],
	};
	await writeJson(join(options.workspace, cacheDir, "region_manifest.json"), manifest);
	return manifest;
}

async function installExactCache(options: {
	workspace: string;
	sourceSha256: string;
	stepSha256: string;
	drawingContents: string;
}): Promise<DrawingFeedbackSummary> {
	const inputSha256 = drawingInputSha("drawing.png", options.drawingContents);
	const key = feedbackKey(inputSha256, options.stepSha256, "test-provider", "test-model");
	const runDir = join("drawing_feedback", "runs", key);
	const manifestPath = join(runDir, "view_manifest.json");
	const reportPath = join(runDir, "reflection.json");
	const sheetPath = join(runDir, "sheet", "comparison.png");
	const inputCacheDir = join("drawing_feedback", "input_cache", inputSha256);
	const regionManifestPath = join(inputCacheDir, "region_manifest.json");
	const cropPath = join(inputCacheDir, "regions", "page0-detail", "crop.png");
	await writeArtifact(options.workspace, cropPath);
	const cropBytes = await readFile(join(options.workspace, cropPath));
	const regionManifest: DrawingRegionManifest = {
		version: 1,
		input_sha256: inputSha256,
		regions: [
			{
				region_id: "page0-detail",
				page: 0,
				roi: [0, 0, 1000, 1000],
				crop_path: cropPath,
				view_type_hint: "detail",
				confidence: "low",
				note: "Unsupported local detail",
				width: 400,
				height: 300,
				crop_sha256: sha256(cropBytes),
			},
		],
	};
	const cachedCandidate = (candidateId: string, score: number, direction: DrawingViewSpec["direction"]) => ({
		candidate_id: candidateId,
		score,
		...capacityFields(score),
		view_spec: { name: candidateId, direction, up: [0, 0, 1] as DrawingViewSpec["up"] },
		render_path: join(runDir, "view-search", "page0-detail", "renders", `${candidateId}.png`),
		comparison_path: join(runDir, "view-search", "page0-detail", "comparisons", `${candidateId}.png`),
		...alignmentFields(join(runDir, "view-search", "page0-detail", "aligned", `${candidateId}.png`)),
	});
	const selected = cachedCandidate("candidate-001", 0.9, [0, -1, 0]);
	const alternative = cachedCandidate("candidate-002", 0.8, [0, 1, 0]);
	const manifest: DrawingViewManifest = {
		version: 3,
		scorer_profile: "capacity_transport_v1",
		input_sha256: inputSha256,
		source_sha256: options.sourceSha256,
		step_sha256: options.stepSha256,
		regions: [
			{
				region_id: "page0-detail",
				page: 0,
				roi: [0, 0, 1000, 1000],
				crop_path: cropPath,
				view_type: "detail",
				status: "ambiguous",
				selected,
				alternatives: [alternative],
				margin: 0.1,
				confidence: "low",
				note: "Best available Top-2 view for the local detail",
				width: 400,
				height: 300,
				crop_sha256: sha256(cropBytes),
			},
		],
	};
	const report: DrawingReflectionReport = {
		should_revise: false,
		overall_summary: "Cached diagnosis",
		findings: [],
	};
	const reflectionUsage = zeroUsage("reflection");
	const summary: DrawingFeedbackSummary = {
		cache_key: key,
		created_at: "2026-08-08T00:00:00.000Z",
		input_sha256: inputSha256,
		source_sha256: options.sourceSha256,
		step_sha256: options.stepSha256,
		segmentation_mode: "segmented",
		search_mode: "searched",
		region_manifest_path: regionManifestPath,
		manifest_path: manifestPath,
		report_path: reportPath,
		sheet_path: sheetPath,
		region_count: 1,
		matched_count: 0,
		ambiguous_count: 1,
		error_count: 0,
		should_revise: false,
		usage: {
			segmentation: zeroUsage("segmentation"),
			reflection: reflectionUsage,
			total: reflectionUsage,
		},
	};
	await writeJson(join(options.workspace, regionManifestPath), regionManifest);
	await writeJson(join(options.workspace, manifestPath), manifest);
	await writeJson(join(options.workspace, reportPath), report);
	for (const candidate of [selected, alternative]) {
		await writeArtifact(options.workspace, candidate.render_path);
		await writeArtifact(options.workspace, candidate.comparison_path);
		await writeArtifact(options.workspace, candidate.aligned_render_path);
	}
	await writeArtifact(options.workspace, sheetPath);
	await writeJson(join(options.workspace, runDir, "run.json"), {
		version: DRAWING_FEEDBACK_VERSION,
		summary,
		manifest,
		report,
	});
	return summary;
}

async function installPartialManifest(options: {
	workspace: string;
	sourceSha256: string;
	stepSha256: string;
	drawingContents: string;
	omitLastComparison?: boolean;
}): Promise<{
	manifest: DrawingViewManifest;
	manifestPath: string;
	reportPath: string;
	runPath: string;
	searchUsage: SubagentRunUsage;
	omittedPath?: string;
}> {
	const inputSha256 = drawingInputSha("drawing.png", options.drawingContents);
	const key = feedbackKey(inputSha256, options.stepSha256, "test-provider", "test-model");
	const runDir = join("drawing_feedback", "runs", key);
	const manifestPath = join(runDir, "view_manifest.json");
	const reportPath = join(runDir, "reflection.json");
	const runPath = join(runDir, "run.json");
	const candidate = (
		id: string,
		score: number,
		direction: DrawingViewSpec["direction"],
	) => ({
		candidate_id: id,
		score,
		...capacityFields(score),
		view_spec: { name: id, direction, up: [0, 0, 1] as DrawingViewSpec["up"] },
		render_path: join(runDir, "rankings", "page0-main", "renders", `${id}.png`),
		comparison_path: join(runDir, "rankings", "page0-main", "comparisons", `${id}.png`),
		...alignmentFields(join(runDir, "rankings", "page0-main", "aligned", `${id}.png`)),
	});
	const selected = candidate("candidate-001", 0.9, [0, -1, 0]);
	const alternatives = [candidate("candidate-002", 0.8, [0, 1, 0])];
	const cropPath = join(runDir, "regions", "page0-main", "crop.png");
	const manifest: DrawingViewManifest = {
		version: 3,
		scorer_profile: "capacity_transport_v1",
		input_sha256: inputSha256,
		source_sha256: options.sourceSha256,
		step_sha256: options.stepSha256,
		regions: [
			{
				region_id: "page0-main",
				page: 0,
				roi: [0, 0, 1000, 1000],
				crop_path: cropPath,
				view_type: "orthographic",
				status: "matched",
				selected,
				alternatives,
				margin: 0.1,
				confidence: "high",
				note: "Locked main view from the completed search session",
			},
		],
	};
	const candidates = [selected, ...alternatives];
	const omittedPath = options.omitLastComparison ? alternatives[0]!.comparison_path : undefined;
	await writeArtifact(options.workspace, cropPath);
	for (const item of candidates) {
		await writeArtifact(options.workspace, item.render_path);
		if (item.comparison_path !== omittedPath) await writeArtifact(options.workspace, item.comparison_path);
		await writeArtifact(options.workspace, item.aligned_render_path);
	}
	await writeJson(join(options.workspace, manifestPath), manifest);
	const searchUsage: SubagentRunUsage = {
		role: "view_search",
		region_id: "page0-main",
		accounting_id: "completed-view-search-session",
		session_file: join(runDir, "sessions", "view_search", "session.jsonl"),
		result_path: manifestPath,
		started_at: "2026-08-08T00:00:00.000Z",
		finished_at: "2026-08-08T00:00:30.000Z",
		duration_s: 30,
		status: "success",
		tokens: { input: 100, output: 20, cache_read: 5, cache_write: 2, total: 127 },
		cost: 0.25,
		tool_calls: 7,
	};
	return {
		manifest,
		manifestPath,
		reportPath,
		runPath,
		searchUsage,
		...(omittedPath ? { omittedPath } : {}),
	};
}

describe("2D drawing reflection controller pipeline", () => {
	it("enforces a global draft before fixed-grid finding updates and restores evidence", async () => {
		const workspace = await makeWorkspace("pi-reflection-fixed-grid-");
		await writeArtifact(workspace, "drawing.png", 640, 480, 89);
		const cropPath = "drawing_feedback/crop.png";
		const alignedPath = "drawing_feedback/aligned.png";
		await Promise.all([
			writeArtifact(workspace, cropPath),
			writeArtifact(workspace, alignedPath),
			writeArtifact(workspace, "drawing_feedback/comparison.png"),
		]);
		const candidate = {
			candidate_id: "candidate-001",
			score: 0.9,
			...capacityFields(0.9),
			view_spec: { name: "front", direction: [0, -1, 0], up: [0, 0, 1] } as DrawingViewSpec,
			render_path: "drawing_feedback/render.png",
			comparison_path: "drawing_feedback/comparison.png",
			...alignmentFields(alignedPath),
		};
		const manifest: DrawingViewManifest = {
			version: 3,
			scorer_profile: "capacity_transport_v1",
			input_sha256: "input-sha",
			source_sha256: "source-sha",
			step_sha256: "step-sha",
			regions: [{
				region_id: "main",
				page: 0,
				roi: [0, 0, 1000, 1000],
				crop_path: cropPath,
				view_type: "orthographic",
				status: "matched",
				selected: candidate,
				alternatives: [{ ...candidate, candidate_id: "candidate-002" }],
				margin: 0.1,
				confidence: "high",
				note: "locked front",
			}],
		};
		const state = { manifest, report: null };
		const request = vi.fn(async (_workspace: string, workerRequest: WorkerRequest) =>
			workerRequest.operation === "crop_drawing_region"
				? await inputPageCropResponse(workspace, workerRequest)
				: await reflectionInspectionResponse(workspace, workerRequest));
		const tools = createReflectionTools({
			workspace,
			state,
			worker: { request } as unknown as WorkerClient,
			controller: new RunController(),
			outputDir: "drawing_feedback/run",
		});
		expect(tools.map(({ name }) => name)).toEqual([
			"inspect_region",
			"submit_initial_findings",
			"compare_region_grid_cells",
			"update_findings",
			"submit_reflection",
		]);
		await expect(invokeTool(tools[1] as TestTool, "draft-too-early", { findings: [] }))
			.rejects.toThrow("Inspect every comparable region");
		await expect(invokeTool(tools[2] as TestTool, "compare-too-early", {
			region_id: "main",
			finding_ids: ["finding-001"],
			reference_cell: "A1",
			cad_cell: "A1",
		})).rejects.toThrow("Submit the complete initial findings draft");
		await invokeTool(tools[0] as TestTool, "inspect-main", { region_id: "main" });
		const draft = await invokeTool<{ findings: Array<{ finding_id: string }> }>(
			tools[1] as TestTool,
			"submit-draft",
			{
				findings: [{
					region_id: "main",
					obs: "The feature is too small; the note also says too small.",
					suggested_change: "Increase the feature and keep its position.",
				}],
			},
		);
		const findingId = draft.details.findings[0]!.finding_id;
		expect(findingId).toBe("finding-001");
		await expect(invokeTool(tools[1] as TestTool, "submit-draft-twice", { findings: [] }))
			.rejects.toThrow("already submitted");
		await expect(invokeTool(tools[4] as TestTool, "submit-before-review", {
			should_revise: true,
			overall_summary: "Pending.",
		})).rejects.toThrow("Verify every initial finding");

		const compared = await invokeTool<{
			crop_id: string;
			reference_roi: [number, number, number, number];
			cad_roi: [number, number, number, number];
		}>(tools[2] as TestTool, "compare", {
			region_id: "main",
			finding_ids: [findingId],
			reference_cell: "B2",
			cad_cell: "C3",
		});
		expect(compared.details.reference_roi).toEqual([250, 250, 500, 500]);
		expect(compared.details.cad_roi).toEqual([500, 500, 750, 750]);
		await expect(invokeTool(tools[3] as TestTool, "ambiguous-update", {
			region_id: "main",
			crop_id: compared.details.crop_id,
			reviewed_finding_ids: [findingId],
			changes: [{
				action: "update",
				finding_id: findingId,
				field: "obs",
				old_text: "too small",
				new_text: "different",
			}],
		})).rejects.toThrow("old_text occurs more than once");
		const updated = await invokeTool<{ findings: Array<{ finding_id: string }> }>(tools[3] as TestTool, "update", {
			region_id: "main",
			crop_id: compared.details.crop_id,
			reviewed_finding_ids: [findingId],
			changes: [
				{
					action: "update",
					finding_id: findingId,
					field: "obs",
					old_text: "The feature is too small",
					new_text: "The corresponding feature differs",
				},
				{
					action: "add",
					obs: "A second supported detail differs.",
					suggested_change: "Correct the second detail.",
				},
			],
		});
		expect(updated.details.findings.map(({ finding_id }) => finding_id)).toEqual(["finding-001", "finding-002"]);

		const restoredState = {
			manifest,
			report: null,
			inspectionManifestPath: "drawing_feedback/run/reflection/inspection_manifest.json",
		};
		await restoreReflectionInspectionManifest(workspace, restoredState);
		const restoredTools = createReflectionTools({
			workspace,
			state: restoredState,
			worker: { request } as unknown as WorkerClient,
			controller: new RunController(),
			outputDir: "drawing_feedback/run",
		});
		await invokeTool(restoredTools[0] as TestTool, "inspect-restored", { region_id: "main" });
		const secondCrop = await invokeTool<{ crop_id: string }>(restoredTools[2] as TestTool, "compare-second", {
			region_id: "main",
			finding_ids: ["finding-002"],
			reference_cell: "D4",
			cad_cell: "D4",
		});
		await invokeTool(restoredTools[3] as TestTool, "delete-second", {
			region_id: "main",
			crop_id: secondCrop.details.crop_id,
			reviewed_finding_ids: ["finding-002"],
			changes: [{ action: "delete", finding_id: "finding-002" }],
		});
		const submitted = await invokeTool<DrawingReflectionReport>(restoredTools[4] as TestTool, "submit", {
			should_revise: true,
			overall_summary: "One verified issue.",
		});
		expect(submitted.details.findings).toEqual([{
			region_id: "main",
			obs: "The corresponding feature differs; the note also says too small.",
			suggested_change: "Increase the feature and keep its position.",
		}]);
		expect(JSON.stringify(submitted.details)).not.toContain("finding_id");
		const persisted = JSON.parse(await readFile(
			join(workspace, "drawing_feedback/run/reflection/inspection_manifest.json"),
			"utf8",
		)) as { version: number; phase: string; grid_overviews: unknown[]; comparisons: unknown[] };
		expect(persisted).toMatchObject({ version: 7, phase: "grid_verification" });
		expect(persisted.grid_overviews).toHaveLength(1);
		expect(persisted.comparisons).toHaveLength(2);
	});

	it.skip("binds input-page crops to the current region and persists editable findings", async () => {
		const workspace = await makeWorkspace("pi-reflection-dual-roi-");
		await writeArtifact(workspace, "drawing.png", 640, 480, 89);
		const cropPath = "drawing_feedback/crop.png";
		const alignedPath = "drawing_feedback/aligned.png";
		await writeArtifact(workspace, cropPath);
		await writeArtifact(workspace, alignedPath);
		await writeArtifact(workspace, "drawing_feedback/comparison.png");
		const candidate = {
			candidate_id: "candidate-001",
			score: 0.9,
			...capacityFields(0.9),
			view_spec: { name: "front", direction: [0, -1, 0], up: [0, 0, 1] } as DrawingViewSpec,
			render_path: "drawing_feedback/render.png",
			comparison_path: "drawing_feedback/comparison.png",
			...alignmentFields(alignedPath),
		};
		const manifest: DrawingViewManifest = {
			version: 3,
			scorer_profile: "capacity_transport_v1",
			input_sha256: "input-sha",
			source_sha256: "source-sha",
			step_sha256: "step-sha",
			regions: [{
				region_id: "main",
				page: 0,
				roi: [0, 0, 1000, 1000],
				crop_path: cropPath,
				view_type: "orthographic",
				status: "matched",
				selected: candidate,
				alternatives: [{ ...candidate, candidate_id: "candidate-002", view_spec: { ...candidate.view_spec, name: "rear" } }],
				margin: 0.1,
				confidence: "high",
				note: "locked front",
			}],
		};
		const state = { manifest, report: null };
		const request = vi.fn(async (_workspace: string, workerRequest: WorkerRequest) =>
			workerRequest.operation === "crop_drawing_region"
				? await inputPageCropResponse(workspace, workerRequest)
				: await reflectionInspectionResponse(workspace, workerRequest));
		const tools = createReflectionTools({
			workspace,
			state,
			worker: { request } as unknown as WorkerClient,
			controller: new RunController(),
			outputDir: "drawing_feedback/run",
		});
		expect(tools.map(({ name }) => name)).toEqual([
			"inspect_region",
			"submit_initial_findings",
			"compare_region_grid_cells",
			"update_findings",
			"submit_reflection",
		]);
		await expect(invokeTool(tools[4] as TestTool, "submit-too-early", {
			should_revise: false,
			overall_summary: "No issue.",
		})).rejects.toThrow("Review findings for every comparable region");

		await expect(invokeTool(tools[0] as TestTool, "crop-before-inspect", {
			page: 0,
			roi: { x_min: 100, y_min: 100, x_max: 400, y_max: 400 },
		})).rejects.toThrow("Call inspect_region");
		const inspected = await invokeTool<{ region_id: string; selected: { comparison_path: string } }>(
			tools[1] as TestTool,
			"inspect-main",
			{ region_id: "main" },
		);
		expect(inspected.details.region_id).toBe("main");
		expect(inspected.details.selected.comparison_path).toBe("drawing_feedback/comparison.png");
		await expect(invokeTool(tools[0] as TestTool, "cross-page", {
			page: 1,
			roi: { x_min: 100, y_min: 100, x_max: 400, y_max: 400 },
		})).rejects.toThrow("bound to input page 0; page 1 is not allowed");
		const pageCropParameters = {
			page: 0,
			roi: { x_min: 100, y_min: 100, x_max: 400, y_max: 400 },
		};
		const pageCrop = await invokeTool<{ crop_id: string }>(tools[0] as TestTool, "page-crop", pageCropParameters);
		const repeatedPageCrop = await invokeTool<{ crop_id: string }>(
			tools[0] as TestTool,
			"page-crop-repeat",
			pageCropParameters,
		);
		expect(repeatedPageCrop.details.crop_id).toBe(pageCrop.details.crop_id);
		expect(request.mock.calls.filter(([, workerRequest]) =>
			(workerRequest as WorkerRequest).operation === "crop_drawing_region")).toHaveLength(1);

		await expect(invokeTool(tools[3] as TestTool, "edit-without-local-evidence", {
			region_id: "main",
			changes: [{ action: "add", obs: "Mismatch.", suggested_change: "Revise it." }],
		})).rejects.toThrow("Call compare_region_crops");
		await expect(invokeTool(tools[2] as TestTool, "too-small", {
			region_id: "main",
			reference_roi: { x_min: 0, y_min: 0, x_max: 49, y_max: 100 },
			cad_roi: { x_min: 0, y_min: 0, x_max: 100, y_max: 100 },
		})).rejects.toThrow("at least 50");
		const parameters = {
			region_id: "main",
			reference_roi: { x_min: 100, y_min: 100, x_max: 400, y_max: 400 },
			cad_roi: { x_min: 200, y_min: 150, x_max: 500, y_max: 450 },
		};
		const first = await invokeTool<{ crop_id: string }>(tools[2] as TestTool, "crop-first", parameters);
		const repeated = await invokeTool<{ crop_id: string }>(tools[2] as TestTool, "crop-repeat", parameters);
		expect(repeated.details.crop_id).toBe(first.details.crop_id);
		expect(request.mock.calls.filter(([, workerRequest]) =>
			(workerRequest as WorkerRequest).operation === "compose_reflection_inspection")).toHaveLength(1);
		const initialized = await invokeTool<{
			findings: Array<{ finding_id: string }>;
		}>(tools[3] as TestTool, "add-first", {
			region_id: "main",
			changes: [{
				action: "add",
				obs: "The feature is too small; the repeated note says too small.",
				suggested_change: "Increase the feature and keep its position.",
			}],
		});
		const initialFindingId = initialized.details.findings[0]!.finding_id;
		expect(initialFindingId).toBe("finding-001");
		const unchanged = await invokeTool<{ findings: Array<{ finding_id: string }> }>(
			tools[3] as TestTool,
			"empty-edit",
			{
				region_id: "main",
				changes: [],
			},
		);
		expect(unchanged.details.findings).toHaveLength(1);
		await expect(invokeTool(tools[3] as TestTool, "ambiguous-replacement", {
			region_id: "main",
			changes: [{
				action: "update",
				finding_id: initialFindingId,
				field: "obs",
				old_text: "too small",
				new_text: "different",
			}],
		})).rejects.toThrow("old_text occurs more than once");
		await expect(invokeTool(tools[3] as TestTool, "atomic-rollback", {
			region_id: "main",
			changes: [
				{
					action: "add",
					obs: "This addition must roll back.",
					suggested_change: "Do not persist this.",
				},
				{
					action: "update",
					finding_id: "finding-999",
					field: "obs",
					old_text: "missing",
					new_text: "replacement",
				},
			],
		})).rejects.toThrow("Unknown finding finding-999");
		const added = await invokeTool<{ findings: Array<{ finding_id: string }> }>(tools[3] as TestTool, "add-second", {
			region_id: "main",
			changes: [{
				action: "add",
				obs: "Temporary second observation.",
				suggested_change: "Temporary second suggestion.",
			}],
		});
		const addedFindingId = added.details.findings[1]!.finding_id;
		expect(addedFindingId).toBe("finding-002");
		await invokeTool(tools[3] as TestTool, "update-first", {
			region_id: "main",
			changes: [{
				action: "update",
				finding_id: initialFindingId,
				field: "obs",
				old_text: "The feature is too small",
				new_text: "The corresponding feature differs",
			}],
		});
		await invokeTool(tools[3] as TestTool, "delete-second", {
			region_id: "main",
			changes: [{ action: "delete", finding_id: addedFindingId }],
		});

		const restoredState = {
			manifest,
			report: null,
			inspectionManifestPath: "drawing_feedback/run/reflection/inspection_manifest.json",
		};
		await restoreReflectionInspectionManifest(workspace, restoredState);
		const restoredTools = createReflectionTools({
			workspace,
			state: restoredState,
			worker: { request } as unknown as WorkerClient,
			controller: new RunController(),
			outputDir: "drawing_feedback/run",
		});
		await invokeTool(restoredTools[1] as TestTool, "inspect-restored-main", { region_id: "main" });
		const addedAfterRestore = await invokeTool<{ findings: Array<{ finding_id: string }> }>(
			restoredTools[3] as TestTool,
			"add-after-restore",
			{
				region_id: "main",
				changes: [{
					action: "add",
					obs: "Temporary restored observation.",
					suggested_change: "Temporary restored suggestion.",
				}],
			},
		);
		const restoredFindingId = addedAfterRestore.details.findings[1]!.finding_id;
		expect(restoredFindingId).toBe("finding-003");
		await invokeTool(restoredTools[3] as TestTool, "delete-after-restore", {
			region_id: "main",
			changes: [{ action: "delete", finding_id: restoredFindingId }],
		});
		await invokeTool(restoredTools[3] as TestTool, "update-suggestion-after-restore", {
			region_id: "main",
			changes: [{
				action: "update",
				finding_id: initialFindingId,
				field: "suggested_change",
				old_text: "Increase the feature",
				new_text: "Revise the feature geometry",
			}],
		});
		const submitted = await invokeTool<DrawingReflectionReport>(restoredTools[4] as TestTool, "submit", {
			should_revise: true,
			overall_summary: "Local mismatch.",
		});
		expect(submitted.details.findings).toEqual([{
			region_id: "main",
			obs: "The corresponding feature differs; the repeated note says too small.",
			suggested_change: "Revise the feature geometry and keep its position.",
		}]);
		expect(JSON.stringify(submitted.details)).not.toContain("finding_id");
		const persisted = JSON.parse(await readFile(
			join(workspace, "drawing_feedback/run/reflection/inspection_manifest.json"),
			"utf8",
		)) as {
			version: number;
			inspected_regions: string[];
			input_page_crops: unknown[];
			comparisons: unknown[];
			region_findings: Array<{
				region_id: string;
				next_finding_sequence: number;
				findings: Array<{ finding_id: string; obs: string }>;
			}>;
		};
		expect(persisted.version).toBe(5);
		expect(persisted.inspected_regions).toEqual(["main"]);
		expect(persisted.input_page_crops).toHaveLength(1);
		expect(persisted.comparisons).toHaveLength(1);
		expect(persisted.region_findings).toHaveLength(1);
		expect(persisted.region_findings[0]?.findings[0]?.finding_id).toBe(initialFindingId);
		expect(persisted.region_findings[0]?.next_finding_sequence).toBe(4);
		expect(persisted.region_findings[0]?.findings[0]?.obs).toBe(
			"The corresponding feature differs; the repeated note says too small.",
		);
	});

	it("allows an inspected region to submit an empty global draft without a local crop", async () => {
		const workspace = await makeWorkspace("pi-reflection-global-finding-");
		const cropPath = "drawing_feedback/crop.png";
		const comparisonPath = "drawing_feedback/comparison.png";
		await writeArtifact(workspace, cropPath);
		await writeArtifact(workspace, comparisonPath);
		const candidate = {
			candidate_id: "candidate-001",
			score: 0.9,
			...capacityFields(0.9),
			view_spec: { name: "front", direction: [0, -1, 0], up: [0, 0, 1] } as DrawingViewSpec,
			render_path: "drawing_feedback/render.png",
			comparison_path: comparisonPath,
			...alignmentFields("drawing_feedback/aligned.png"),
		};
		const manifest: DrawingViewManifest = {
			version: 3,
			scorer_profile: "capacity_transport_v1",
			input_sha256: "input-sha",
			source_sha256: "source-sha",
			step_sha256: "step-sha",
			regions: [{
				region_id: "main",
				page: 0,
				roi: [0, 0, 1000, 1000],
				crop_path: cropPath,
				view_type: "orthographic",
				status: "matched",
				selected: candidate,
				alternatives: [{ ...candidate, candidate_id: "candidate-002", view_spec: { ...candidate.view_spec, name: "rear" } }],
				margin: 0.1,
				confidence: "high",
				note: "locked front",
			}],
		};
		const request = vi.fn(async (_workspace: string, workerRequest: WorkerRequest) =>
			await reflectionInspectionResponse(workspace, workerRequest));
		const tools = createReflectionTools({
			workspace,
			state: { manifest, report: null },
			worker: { request } as unknown as WorkerClient,
			controller: new RunController(),
			outputDir: "drawing_feedback/run",
		});
		await invokeTool(tools[0] as TestTool, "inspect-main", { region_id: "main" });
		await invokeTool(tools[1] as TestTool, "submit-empty-draft", { findings: [] });
		const submitted = await invokeTool<DrawingReflectionReport>(tools[4] as TestTool, "submit-empty", {
			should_revise: false,
			overall_summary: "No verified issue.",
		});
		expect(submitted.details.findings).toEqual([]);
		expect(request).toHaveBeenCalledOnce();
	});

	it("aggregates persisted findings in locked manifest order rather than draft order", async () => {
		const workspace = await makeWorkspace("pi-reflection-finding-order-");
		const makeCandidate = (regionId: string) => ({
			candidate_id: `${regionId}-candidate-001`,
			score: 0.9,
			...capacityFields(0.9),
			view_spec: { name: `${regionId}-front`, direction: [0, -1, 0], up: [0, 0, 1] } as DrawingViewSpec,
			render_path: `drawing_feedback/${regionId}-render.png`,
			comparison_path: `drawing_feedback/${regionId}-comparison.png`,
			...alignmentFields(`drawing_feedback/${regionId}-aligned.png`),
		});
		const regions = ["first", "second"].map((regionId) => {
			const selected = makeCandidate(regionId);
			return {
				region_id: regionId,
				page: 0,
				roi: [0, 0, 1000, 1000] as [number, number, number, number],
				crop_path: `drawing_feedback/${regionId}-crop.png`,
				view_type: "orthographic" as const,
				status: "matched" as const,
				selected,
				alternatives: [{ ...selected, candidate_id: `${regionId}-candidate-002` }],
				margin: 0.1,
				confidence: "high" as const,
				note: "locked front",
			};
		});
		for (const region of regions) {
			await Promise.all([
				writeArtifact(workspace, region.crop_path),
				writeArtifact(workspace, region.selected.comparison_path),
				writeArtifact(workspace, region.selected.aligned_render_path),
			]);
		}
		const manifest: DrawingViewManifest = {
			version: 3,
			scorer_profile: "capacity_transport_v1",
			input_sha256: "input-sha",
			source_sha256: "source-sha",
			step_sha256: "step-sha",
			regions,
		};
		const tools = createReflectionTools({
			workspace,
			state: { manifest, report: null },
			worker: {
				request: async (_workspace: string, workerRequest: WorkerRequest) =>
					await reflectionInspectionResponse(workspace, workerRequest),
			} as unknown as WorkerClient,
			controller: new RunController(),
			outputDir: "drawing_feedback/run",
		});
		for (const regionId of ["second", "first"]) {
			await invokeTool(tools[0] as TestTool, `inspect-${regionId}`, { region_id: regionId });
		}
		const draft = await invokeTool<{
			findings: Array<{ region_id: string; finding_id: string }>;
		}>(tools[1] as TestTool, "submit-initial", {
			findings: ["second", "first"].map((regionId) => ({
				region_id: regionId,
				obs: `${regionId === "first" ? "First" : "Second"} issue.`,
				suggested_change: `Fix ${regionId}.`,
			})),
		});
		expect(draft.details.findings.map(({ region_id }) => region_id)).toEqual(["first", "second"]);
		for (const finding of draft.details.findings) {
			await invokeTool(tools[0] as TestTool, `inspect-grid-${finding.region_id}`, {
				region_id: finding.region_id,
			});
			const compared = await invokeTool<{ crop_id: string }>(tools[2] as TestTool, `compare-${finding.region_id}`, {
				region_id: finding.region_id,
				finding_ids: [finding.finding_id],
				reference_cell: "A1",
				cad_cell: "A1",
			});
			await invokeTool(tools[3] as TestTool, `retain-${finding.region_id}`, {
				region_id: finding.region_id,
				crop_id: compared.details.crop_id,
				reviewed_finding_ids: [finding.finding_id],
				changes: [],
			});
		}
		const submitted = await invokeTool<DrawingReflectionReport>(tools[4] as TestTool, "submit", {
			should_revise: true,
			overall_summary: "Two verified issues.",
		});
		expect(submitted.details.findings.map(({ region_id }) => region_id)).toEqual(["first", "second"]);
	});

	it("rejects stale solution.py before creating either subagent", async () => {
		const workspace = await makeWorkspace("pi-reflection-stale-");
		const source = "print('current source')\n";
		await writeFile(join(workspace, "solution.py"), source);
		await writeFile(join(workspace, "drawing.png"), "drawing");
		const result = agentResult(sha256("different source"), "step-sha");
		const request = vi.fn(async () => inspectResponse("step-sha"));
		const tool = makePublicTool({
			workspace,
			worker: { request } as unknown as WorkerClient,
			result,
		});

		await expect(invokeTool(tool as TestTool, "reflect-1", {})).rejects.toThrow(
			"solution.py changed after the latest execute_build123d call",
		);

		expect(request).toHaveBeenCalledOnce();
		expect(createAgentSessionMock).not.toHaveBeenCalled();
	});

	it("rejects a stale output.step before creating either subagent", async () => {
		const workspace = await makeWorkspace("pi-reflection-stale-step-");
		const source = "print('current source')\n";
		const sourceSha256 = sha256(source);
		await writeFile(join(workspace, "solution.py"), source);
		await writeFile(join(workspace, "drawing.png"), "drawing");
		const result = agentResult(sourceSha256, "recorded-step-sha");
		const request = vi.fn(async () => inspectResponse("replaced-step-sha"));
		const tool = makePublicTool({
			workspace,
			worker: { request } as unknown as WorkerClient,
			result,
		});

		await expect(invokeTool(tool as TestTool, "reflect-stale-step", {})).rejects.toThrow(
			"output.step is missing or changed after the latest execute_build123d call",
		);

		expect(request).toHaveBeenCalledOnce();
		expect(createAgentSessionMock).not.toHaveBeenCalled();
	});

	it("uses an exact cache without LLM/search work and performs a second formal SHA inspection", async () => {
		const workspace = await makeWorkspace("pi-reflection-cache-");
		const source = "print('stable source')\n";
		const drawing = "merged drawing bytes";
		const sourceSha256 = sha256(source);
		const stepSha256 = "stable-step-sha";
		await writeFile(join(workspace, "solution.py"), source);
		await writeFile(join(workspace, "drawing.png"), drawing);
		const summary = await installExactCache({ workspace, sourceSha256, stepSha256, drawingContents: drawing });
		const result = agentResult(sourceSha256, stepSha256);
		const request = vi.fn(async (_workspace: string, _request: WorkerRequest) => inspectResponse(stepSha256));
		const onComplete = vi.fn(async () => undefined);
		const tool = makePublicTool({
			workspace,
			worker: { request } as unknown as WorkerClient,
			result,
			onComplete,
		});

		const response = await invokeTool<{ cached: boolean }>(tool as TestTool, "reflect-2", {});

		expect(response.details.cached).toBe(true);
		expect(request).toHaveBeenCalledTimes(2);
		expect(request.mock.calls.map((call) => (call[1] as WorkerRequest).operation)).toEqual([
			"inspect",
			"inspect",
		]);
		expect(createAgentSessionMock).not.toHaveBeenCalled();
		expect(onComplete).toHaveBeenCalledWith(summary);
	});

	it("rejects an exact cache when the second inspection observes a changed STEP", async () => {
		const workspace = await makeWorkspace("pi-reflection-cache-race-");
		const source = "print('stable source')\n";
		const drawing = "merged drawing bytes";
		const sourceSha256 = sha256(source);
		const stepSha256 = "first-step-sha";
		await writeFile(join(workspace, "solution.py"), source);
		await writeFile(join(workspace, "drawing.png"), drawing);
		await installExactCache({ workspace, sourceSha256, stepSha256, drawingContents: drawing });
		const result = agentResult(sourceSha256, stepSha256);
		const request = vi
			.fn()
			.mockResolvedValueOnce(inspectResponse(stepSha256))
			.mockResolvedValueOnce(inspectResponse("replaced-step-sha"));
		const onComplete = vi.fn(async () => undefined);
		const tool = makePublicTool({
			workspace,
			worker: { request } as unknown as WorkerClient,
			result,
			onComplete,
		});

		await expect(invokeTool(tool as TestTool, "reflect-3", {})).rejects.toThrow(
			"Formal source or STEP changed while loading cached 2D feedback",
		);

		expect(request).toHaveBeenCalledTimes(2);
		expect(createAgentSessionMock).not.toHaveBeenCalled();
		expect(onComplete).not.toHaveBeenCalled();
	});

	it("uses the controller-bound View Search manifest without starting a new search", async () => {
		const workspace = await makeWorkspace("pi-reflection-locked-view-");
		const source = "print('stable source')\n";
		const drawing = "locked drawing bytes";
		const sourceSha256 = sha256(source);
		const stepSha256 = "locked-step-sha";
		await writeFile(join(workspace, "solution.py"), source);
		await writeFile(join(workspace, "drawing.png"), drawing);
		const regionManifest = await installRegionCache({ workspace, drawingContents: drawing });
		const lockedRegion = regionManifest.regions[0]!;
		const lockedRoot = join("drawing_feedback", "locked-view-search");
		const candidate = (id: string, score: number, direction: DrawingViewSpec["direction"]) => ({
			candidate_id: id,
			score,
			...capacityFields(score),
			view_spec: { name: id, direction, up: [0, 0, 1] as DrawingViewSpec["up"] },
			render_path: join(lockedRoot, "renders", `${id}.png`),
			comparison_path: join(lockedRoot, "comparisons", `${id}.png`),
			...alignmentFields(join(lockedRoot, "aligned", `${id}.png`)),
		});
		const selected = candidate("candidate-001", 0.9, [0, -1, 0]);
		const alternative = candidate("candidate-002", 0.8, [0, 1, 0]);
		for (const item of [selected, alternative]) {
			await writeArtifact(workspace, item.render_path);
			await writeArtifact(workspace, item.comparison_path);
			await writeArtifact(workspace, item.aligned_render_path);
		}
		const lockedManifestPath = join(lockedRoot, "view_manifest.json");
		const lockedManifest: DrawingViewManifest = {
			version: 3,
			scorer_profile: "capacity_transport_v1",
			input_sha256: regionManifest.input_sha256,
			source_sha256: sourceSha256,
			step_sha256: stepSha256,
			regions: [{
				region_id: lockedRegion.region_id,
				page: lockedRegion.page,
				roi: lockedRegion.roi,
				crop_path: lockedRegion.crop_path,
				view_type: "orthographic",
				status: "matched",
				selected,
				alternatives: [alternative],
				margin: 0.1,
				confidence: "high",
				note: "Controller-bound locked view",
				width: lockedRegion.width,
				height: lockedRegion.height,
				crop_sha256: lockedRegion.crop_sha256,
			}],
		};
		await writeJson(join(workspace, lockedManifestPath), lockedManifest);
		const operations: string[] = [];
		const request = vi.fn(async (_workspace: string, workerRequest: WorkerRequest): Promise<WorkerResponse> => {
			operations.push(workerRequest.operation);
			if (workerRequest.operation === "inspect") return inspectResponse(stepSha256);
			throw new Error("stop after locked manifest was accepted");
		});
		const tool = makePublicTool({
			workspace,
			worker: { request } as unknown as WorkerClient,
			result: agentResult(sourceSha256, stepSha256),
			lockedViewManifestPath: lockedManifestPath,
		});

		await expect(invokeTool(tool as TestTool, "reflect-locked", {})).rejects.toThrow(
			"stop after locked manifest was accepted",
		);
		expect(operations).toEqual(["inspect", "compare_drawing_sheet"]);
		expect(createAgentSessionMock).not.toHaveBeenCalled();
	});

	it("passes the locked multi-region manifest unchanged through the sheet and reflection session", async () => {
		vi.useFakeTimers();
		const workspace = await makeWorkspace("pi-reflection-pipeline-");
		const source = "# source snapshot\n";
		const drawingPage0 = "drawing page zero bytes";
		const drawingPage1 = "drawing page one bytes";
		const stagedPaths = ["drawing-page-0.png", "drawing-page-1.png"];
		const sourceSha256 = sha256(source);
		const stepSha256 = "pipeline-step-sha";
		await writeFile(join(workspace, "solution.py"), source);
		await writeFile(join(workspace, stagedPaths[0]!), drawingPage0);
		await writeFile(join(workspace, stagedPaths[1]!), drawingPage1);
		const result = agentResult(sourceSha256, stepSha256);
		const workerRequests: WorkerRequest[] = [];
		let delayedFirstRank = false;
		const request = vi.fn(async (_workspace: string, workerRequest: WorkerRequest): Promise<WorkerResponse> => {
			workerRequests.push(workerRequest);
			if (workerRequest.operation === "inspect") return inspectResponse(stepSha256);
			if (workerRequest.operation === "prepare_images") {
				const paths = workerRequest.paths as string[];
				const images = [];
				for (const [index, sourcePath] of paths.entries()) {
					const path = join("prepared_inputs", `${String(index).padStart(2, "0")}.png`);
					await writeArtifact(workspace, path, 320, 240, 70 + index);
					images.push({
						source_path: sourcePath,
						path,
						mime_type: "image/png" as const,
						original_width: 640,
						original_height: 480,
						width: 320,
						height: 240,
					});
				}
				return { protocol_version: PROTOCOL_VERSION, operation: "prepare_images", images };
			}
			if (workerRequest.operation === "render_drawing_view") {
				return await currentCadReferenceResponse(workspace, workerRequest, stepSha256);
			}
			if (workerRequest.operation === "crop_drawing_region") {
				const cropPath = join(String(workerRequest.output_dir), "crop.png");
				await writeArtifact(workspace, cropPath);
				return {
					protocol_version: PROTOCOL_VERSION,
					operation: "crop_drawing_region",
					status: "ok",
					region_id: String(workerRequest.region_id),
					input_path: String(workerRequest.input_path),
					roi: workerRequest.roi as [number, number, number, number],
					crop_path: cropPath,
					width: 400,
					height: 300,
				};
			}
			if (workerRequest.operation === "rank_drawing_views") {
				if (!delayedFirstRank) {
					delayedFirstRank = true;
					await new Promise<void>((resolve) => setTimeout(resolve, 650_000));
				}
				const outputDir = String(workerRequest.output_dir);
				const candidates = workerRequest.candidates as Array<{
					candidate_id: string;
					view_spec: DrawingViewSpec;
				}>;
				const rankings = await Promise.all(
					candidates.map(async (candidate, index) => {
						const renderPath = join(outputDir, "renders", `${candidate.candidate_id}.png`);
						const comparisonPath = join(outputDir, "comparisons", `${candidate.candidate_id}.png`);
						const alignedRenderPath = join(outputDir, "aligned", `${candidate.candidate_id}.png`);
						await writeArtifact(workspace, renderPath);
						await writeArtifact(workspace, comparisonPath);
						await writeArtifact(workspace, alignedRenderPath);
							return {
								candidate_id: candidate.candidate_id,
								score: 0.9 - index * 0.1,
								...capacityFields(0.9 - index * 0.1),
							view_spec: candidate.view_spec,
							render_path: renderPath,
							comparison_path: comparisonPath,
							...alignmentFields(alignedRenderPath),
						};
					}),
				);
				const comparisonPath = join(outputDir, "overview.png");
				await writeArtifact(workspace, comparisonPath);
				return {
					protocol_version: PROTOCOL_VERSION,
					operation: "rank_drawing_views",
					status: "ok",
					region_id: String(workerRequest.region_id),
					step_path: "output.step",
					step_sha256: stepSha256,
					rankings,
						margin: 0.1,
						comparison_path: comparisonPath,
						scoring_mode: "capacity_transport",
						scorer_profile: "capacity_transport_v1",
				};
			}
			if (workerRequest.operation === "compare_drawing_sheet") {
				const sheetPath = join(String(workerRequest.output_dir), "comparison-sheet.png");
				await writeArtifact(workspace, sheetPath);
				return {
					protocol_version: PROTOCOL_VERSION,
					operation: "compare_drawing_sheet",
					status: "ok",
					sheet_path: sheetPath,
					regions: workerRequest.regions as never,
				};
			}
			if (workerRequest.operation === "compose_reflection_inspection") {
				return await reflectionInspectionResponse(workspace, workerRequest);
			}
			throw new Error(`Unexpected worker operation ${workerRequest.operation}`);
		});

		const sessionToolNames: string[][] = [];
		const sessionPrompts: string[] = [];
		const sessionImageCounts: Array<{ role: string; count: number }> = [];
		const sessionInputs: Array<{
			role: string;
			prompt: string;
			images: Array<{ data?: string }>;
		}> = [];
		const segmentationToolTrace: string[] = [];
		const searchToolTrace: string[] = [];
			const llmBatchSpecs: LlmDrawingViewSpec[][] = [];
			const subscriptionCleanups: Array<ReturnType<typeof vi.fn>> = [];
			const sessionAborts: Array<{ role: string; abort: ReturnType<typeof vi.fn> }> = [];
			let sessionIndex = 0;
			let activeSearchPrompts = 0;
			let maxActiveSearchPrompts = 0;
		const regions = [
			{
				id: "page0-main",
				page: 0,
				roi: [0, 0, 500, 500] as [number, number, number, number],
				viewTypeHint: "orthographic" as const,
			},
			{
				id: "page1-side",
				page: 1,
				roi: [500, 0, 1000, 500] as [number, number, number, number],
				viewTypeHint: "section" as const,
			},
			{
				id: "page0-top",
				page: 0,
				roi: [0, 500, 1000, 1000] as [number, number, number, number],
				viewTypeHint: "orthographic" as const,
			},
		];
			createAgentSessionMock.mockImplementation(async (creation: SessionCreationOptions) => {
				const role = creation.tools.includes("submit_region_manifest")
					? "segmentation"
					: creation.tools.includes("submit_view_result")
						? "view_search"
						: "reflection";
				sessionIndex += 1;
				const sessionId = `${role}-session-${sessionIndex}`;
			let promptCount = 0;
			let eventListener: ((event: TestToolExecutionEvent) => void) | undefined;
			const unsubscribe = vi.fn(() => {
				eventListener = undefined;
			});
			const abort = vi.fn(async () => undefined);
			subscriptionCleanups.push(unsubscribe);
			sessionAborts.push({ role, abort });
			const session = {
					sessionId,
				getActiveToolNames: () => [...creation.tools],
				getSessionStats: () => ({
					sessionFile: undefined,
					tokens: { input: 10, output: 5, cacheRead: 0, cacheWrite: 0, total: 15 },
					cost: 0.01,
					toolCalls: role === "segmentation" ? 4 : role === "view_search" ? 4 : 3,
				}),
				subscribe: vi.fn((listener: (event: TestToolExecutionEvent) => void) => {
					eventListener = listener;
					return unsubscribe;
				}),
				prompt: vi.fn(async (prompt: string, promptOptions?: { images?: unknown[] }) => {
					promptCount += 1;
					if (promptCount > 1) throw new Error(`${role} unexpectedly needed a corrective prompt`);
					sessionToolNames.push([...creation.tools]);
					sessionPrompts.push(prompt);
					sessionImageCounts.push({ role, count: promptOptions?.images?.length ?? 0 });
					sessionInputs.push({
						role,
						prompt,
						images: (promptOptions?.images ?? []) as Array<{ data?: string }>,
					});
					const tools = new Map(creation.customTools.map((tool) => [tool.name, tool]));
					if (role === "segmentation") {
						for (const region of regions) {
							await invokeTool(tools.get("crop_region")!, `crop-${region.id}`, {
								region_id: region.id,
								page: region.page,
								roi: region.roi,
							});
							segmentationToolTrace.push("crop_region");
						}
						await invokeTool(tools.get("submit_region_manifest")!, "submit-regions", {
							regions: regions.map((region) => ({
								region_id: region.id,
								view_type_hint: region.viewTypeHint,
								confidence: "high",
								note: `${region.id} locked crop`,
							})),
						});
						segmentationToolTrace.push("submit_region_manifest");
					} else if (role === "view_search") {
						activeSearchPrompts += 1;
						maxActiveSearchPrompts = Math.max(maxActiveSearchPrompts, activeSearchPrompts);
						try {
							const region = regions.find((candidate) => prompt.includes(`"region_id": "${candidate.id}"`));
							if (!region) throw new Error("View-search prompt omitted its locked region");
							const regionIndex = regions.indexOf(region);
							const specs: LlmDrawingViewSpec[] = [
								{ direction: [0, -1, 0], up: [0, 0, 1] },
								{ direction: [0, 1, 0], up: [0, 0, 1] },
								{ direction: [-1, 0, 0], up: [0, 0, 1] },
							];
							llmBatchSpecs.push(specs);
							const rankCallId = `rank-${region.id}`;
							eventListener?.({
								type: "tool_execution_start",
								toolName: "rank_view_candidates",
								toolCallId: rankCallId,
							});
							const rankingPromise = invokeTool<{
								rankings: Array<{ candidate_id: string; eligible: boolean }>;
							}>(tools.get("rank_view_candidates")!, rankCallId, { candidates: specs });
							await vi.advanceTimersByTimeAsync(650_000);
							const ranked = await rankingPromise;
							eventListener?.({
								type: "tool_execution_end",
								toolName: "rank_view_candidates",
								toolCallId: rankCallId,
							});
							searchToolTrace.push("rank_view_candidates");
							const ids = ranked.details.rankings
								.filter(({ eligible }) => eligible)
								.map(({ candidate_id }) => candidate_id);
							await invokeTool(tools.get("submit_view_result")!, `submit-${region.id}`, {
								view_type: "orthographic",
								status: "matched",
								selected_candidate_id: ids[regionIndex === 1 ? 1 : 0],
								confidence: regionIndex === 0 ? "high" : "medium",
								note: `${region.id} locked view`,
							});
							searchToolTrace.push("submit_view_result");
						} finally {
							activeSearchPrompts -= 1;
						}
					} else {
						await invokeTool(tools.get("inspect_region")!, "inspect-main", {
							region_id: "page0-main",
						});
						await invokeTool(tools.get("inspect_region")!, "inspect-side", {
							region_id: "page1-side",
						});
						await invokeTool(tools.get("inspect_region")!, "inspect-top", {
							region_id: "page0-top",
						});
						await invokeTool(tools.get("submit_reflection")!, "submit-reflection", {
							should_revise: true,
							overall_summary: "The side view needs revision.",
							findings: [{
								region_id: "page1-side",
								obs: "The outline differs.",
								suggested_change: "Adjust the side feature.",
							}],
						});
					}
				}),
				abort,
				dispose: vi.fn(),
			};
			return { session };
		});

		const usages: SubagentRunUsage[] = [];
		const onComplete = vi.fn(async () => undefined);
		const tool = makePublicTool({
			workspace,
			worker: { request } as unknown as WorkerClient,
			result,
			stagedPaths,
			onUsage: async (usage) => {
				usages.push(usage);
			},
			onComplete,
		});

		const response = await invokeTool<{
			manifest: DrawingViewManifest;
			cached: boolean;
		}>(tool as TestTool, "reflect-pipeline", {});

		expect(response.details.cached).toBe(false);
		const prepareRequest = workerRequests.find(({ operation }) => operation === "prepare_images");
		expect(prepareRequest).toMatchObject({
			paths: stagedPaths,
			max_edge: 2000,
		});
		const referenceRequests = workerRequests.filter(
			({ operation }) => operation === "render_drawing_view",
		);
		expect(referenceRequests).toHaveLength(1);
		const referenceRequest = referenceRequests[0]!;
		expect(referenceRequest).toMatchObject({
			region_id: "current-cad-reference",
			view_spec: CURRENT_CAD_REFERENCE_VIEW_SPEC,
			show_hidden_lines: true,
		});
		expect(String(referenceRequest.output_dir)).toContain("/current-cad-reference");
		const referenceMetadata = JSON.parse(
			await readFile(join(workspace, String(referenceRequest.output_dir), "metadata.json"), "utf8"),
		) as Record<string, unknown>;
		expect(referenceMetadata).toMatchObject({
			purpose: "shape_reference_only",
			not_a_ranked_candidate: true,
			step_sha256: stepSha256,
			view_spec: CURRENT_CAD_REFERENCE_VIEW_SPEC,
		});
			expect(sessionToolNames.filter((tools) => tools.includes("submit_view_result"))).toHaveLength(3);
			expect(sessionToolNames).toContainEqual([...SEGMENTATION_AGENT_TOOLS]);
			expect(sessionToolNames).toContainEqual([...REFLECTION_AGENT_TOOLS]);
			expect(createAgentSessionMock).toHaveBeenCalledTimes(5);
			expect(maxActiveSearchPrompts).toBe(3);
		const manifest = response.details.manifest;
		expect(manifest.regions.map(({ region_id }) => region_id)).toEqual([
			"page0-main",
			"page1-side",
			"page0-top",
		]);
		expect(segmentationToolTrace).toEqual([
			"crop_region",
			"crop_region",
			"crop_region",
			"submit_region_manifest",
		]);
			expect(searchToolTrace.filter((name) => name === "rank_view_candidates")).toHaveLength(3);
			expect(searchToolTrace.filter((name) => name === "submit_view_result")).toHaveLength(3);
		expect(llmBatchSpecs).toHaveLength(3);
		expect(llmBatchSpecs.flat().every((spec) => !("name" in spec))).toBe(true);
		const rankRequests = workerRequests.filter(({ operation }) => operation === "rank_drawing_views");
		expect(rankRequests).toHaveLength(3);
		const workerCandidates = rankRequests.flatMap(
			(workerRequest) => workerRequest.candidates as Array<{
				candidate_id: string;
				view_spec: DrawingViewSpec;
			}>,
		);
			expect(workerCandidates.map(({ candidate_id }) => candidate_id).sort()).toEqual(
				Array.from({ length: 3 }, (_, index) => `candidate-${String(index + 1).padStart(3, "0")}`).flatMap(
					(id) => [id, id, id],
				).sort(),
			);
		for (const candidate of workerCandidates) {
			expect(candidate.candidate_id).toMatch(/^candidate-[0-9]{3}$/);
			expect(candidate.view_spec.name).toBe(candidate.candidate_id);
		}
			const allCandidates = manifest.regions.flatMap((region) => [region.selected!, ...region.alternatives]);
			expect(new Set(allCandidates.map(({ render_path }) => render_path)).size).toBe(6);
			expect(new Set(allCandidates.map(({ comparison_path }) => comparison_path)).size).toBe(6);
		for (const region of manifest.regions) {
			for (const candidate of [region.selected!, ...region.alternatives]) {
					expect(candidate.render_path).toContain(`/view-search/${region.region_id}/rankings/`);
					expect(candidate.comparison_path).toContain(`/view-search/${region.region_id}/rankings/`);
			}
		}
		const sheetRequest = workerRequests.find(({ operation }) => operation === "compare_drawing_sheet")!;
		const sheetRegions = sheetRequest.regions as Array<Record<string, unknown>>;
		expect(sheetRegions).toEqual(
			manifest.regions.map((region) => ({
				region_id: region.region_id,
				crop_path: region.crop_path,
				render_path: region.selected!.render_path,
				comparison_path: region.selected!.comparison_path,
				view_spec: region.selected!.view_spec,
				score: region.selected!.score,
				margin: region.margin,
				status: region.status,
			})),
		);
			const viewInputs = sessionInputs.filter(({ role }) => role === "view_search");
			expect(viewInputs).toHaveLength(3);
			const cadReferenceData = makePng(400, 300, 99).toString("base64");
			for (const region of regions) {
				const input = viewInputs.find(({ prompt }) =>
					prompt.includes(`"region_id": "${region.id}"`),
				);
				expect(input).toBeDefined();
				expect(input!.images).toHaveLength(3);
				expect(input!.images[1]?.data).toBe(
					makePng(320, 240, 70 + region.page).toString("base64"),
				);
				expect(input!.images[1]?.data).not.toBe(
					makePng(320, 240, 70 + (region.page === 0 ? 1 : 0)).toString("base64"),
				);
				expect(input!.images[2]?.data).toBe(cadReferenceData);
				expect(input!.prompt).toContain(
					`1=the only source page for this region, page ${region.page} (${stagedPaths[region.page]})`,
				);
				expect(input!.prompt).not.toContain(stagedPaths[region.page === 0 ? 1 : 0]);
				expect(input!.prompt).toContain('"not_a_ranked_candidate":true');
				expect(input!.prompt).toContain('"direction":[1,-1,1]');
				expect(input!.prompt).toContain('"up":[1,1,0]');
			}
			expect(sessionPrompts.some((prompt) => prompt.includes(JSON.stringify(manifest, null, 2)))).toBe(true);
			expect(sessionImageCounts.filter(({ role }) => role === "view_search")).toEqual([
				{ role: "view_search", count: 3 },
				{ role: "view_search", count: 3 },
				{ role: "view_search", count: 3 },
			]);
			const reflectionInput = sessionInputs.find(({ role }) => role === "reflection");
			expect(reflectionInput?.images).toHaveLength(1);
			expect(reflectionInput?.prompt).not.toContain(stagedPaths[0]);
			expect(reflectionInput?.prompt).not.toContain(stagedPaths[1]);
			expect(reflectionInput?.prompt).toContain("The attached sheet and manifest were generated from locked, ranked view candidates");
			expect(reflectionInput?.prompt).toContain("Inspect individual regions when the sheet is not enough");
			expect(reflectionInput?.prompt).not.toContain("compare_region_grid_cells");
			expect(usages.filter(({ role, status }) => role === "view_search" && status === "success")).toHaveLength(3);
			expect(usages.some(({ role, status }) => role === "segmentation" && status === "success")).toBe(true);
			expect(usages.some(({ role, status }) => role === "reflection" && status === "success")).toBe(true);
		expect(usages[0]?.tool_calls).toBe(4);
			expect(usages.filter(({ role }) => role === "view_search").every(({ tool_calls }) => tool_calls === 4)).toBe(true);
			expect(sessionAborts.filter(({ role }) => role === "view_search").every(({ abort }) => abort.mock.calls.length === 0)).toBe(true);
			expect(subscriptionCleanups).toHaveLength(5);
		for (const unsubscribe of subscriptionCleanups) expect(unsubscribe).toHaveBeenCalledOnce();
		expect(vi.getTimerCount()).toBe(0);
		expect(onComplete).toHaveBeenCalledOnce();
	});

	it("runs segmentation and View Search only without creating a Reflection Agent", async () => {
		const workspace = await makeWorkspace("pi-view-search-only-");
		const source = "# stable source\n";
		const drawing = "single drawing page";
		const sourceSha256 = sha256(source);
		const stepSha256 = "search-only-step-sha";
		await writeFile(join(workspace, "solution.py"), source);
		await writeFile(join(workspace, "drawing.png"), drawing);
		const result = agentResult(sourceSha256, stepSha256);
		const requests: WorkerRequest[] = [];
		const request = vi.fn(async (_workspace: string, workerRequest: WorkerRequest): Promise<WorkerResponse> => {
			requests.push(workerRequest);
			if (workerRequest.operation === "inspect") return inspectResponse(stepSha256);
			if (workerRequest.operation === "prepare_images") return await prepareResponse(workspace, workerRequest);
			if (workerRequest.operation === "crop_drawing_region") {
				const cropPath = join(String(workerRequest.output_dir), "crop.png");
				await writeArtifact(workspace, cropPath);
				return {
					protocol_version: PROTOCOL_VERSION,
					operation: "crop_drawing_region",
					status: "ok",
					region_id: String(workerRequest.region_id),
					input_path: String(workerRequest.input_path),
					roi: workerRequest.roi as [number, number, number, number],
					crop_path: cropPath,
					width: 400,
					height: 300,
				};
			}
			if (workerRequest.operation === "render_drawing_view") {
				return await currentCadReferenceResponse(workspace, workerRequest, stepSha256);
			}
			if (workerRequest.operation === "rank_drawing_views") {
				const outputDir = String(workerRequest.output_dir);
				const candidates = workerRequest.candidates as Array<{
					candidate_id: string;
					view_spec: DrawingViewSpec;
				}>;
				const rankings = [];
				for (const [index, candidate] of candidates.entries()) {
					const renderPath = join(outputDir, "renders", `${candidate.candidate_id}.png`);
					const comparisonPath = join(outputDir, "comparisons", `${candidate.candidate_id}.png`);
					const alignedRenderPath = join(outputDir, "aligned", `${candidate.candidate_id}.png`);
					await writeArtifact(workspace, renderPath);
					await writeArtifact(workspace, comparisonPath);
					await writeArtifact(workspace, alignedRenderPath);
					const score = 0.9 - index * 0.1;
					rankings.push({
						candidate_id: candidate.candidate_id,
						score,
						...capacityFields(score),
						view_spec: candidate.view_spec,
						render_path: renderPath,
						comparison_path: comparisonPath,
						...alignmentFields(alignedRenderPath),
					});
				}
				const overviewPath = join(outputDir, "overview.png");
				await writeArtifact(workspace, overviewPath);
				return {
					protocol_version: PROTOCOL_VERSION,
					operation: "rank_drawing_views",
					status: "ok",
					region_id: String(workerRequest.region_id),
					step_path: "output.step",
					step_sha256: stepSha256,
					rankings,
					margin: 0.1,
					comparison_path: overviewPath,
					scoring_mode: "capacity_transport",
					scorer_profile: "capacity_transport_v1",
				};
			}
			throw new Error(`Unexpected worker operation ${workerRequest.operation}`);
		});
		const roles: string[] = [];
		createAgentSessionMock.mockImplementation(async (creation: SessionCreationOptions) => {
			const role = creation.tools.includes("submit_region_manifest") ? "segmentation" : "view_search";
			roles.push(role);
			return {
				session: {
					sessionId: `${role}-only-session`,
					getActiveToolNames: () => [...creation.tools],
					getSessionStats: () => ({
						sessionFile: undefined,
						tokens: { input: 2, output: 1, cacheRead: 0, cacheWrite: 0, total: 3 },
						cost: 0.01,
						toolCalls: 2,
					}),
					subscribe: vi.fn(() => vi.fn()),
					prompt: vi.fn(async () => {
						const tools = new Map(creation.customTools.map((tool) => [tool.name, tool]));
						if (role === "segmentation") {
							await invokeTool(tools.get("crop_region")!, "crop-only", {
								region_id: "page0-main",
								page: 0,
								roi: [0, 0, 1000, 1000],
							});
							await invokeTool(tools.get("submit_region_manifest")!, "submit-regions-only", {
								regions: [{
									region_id: "page0-main",
									view_type_hint: "orthographic",
									confidence: "high",
									note: "complete main view",
								}],
							});
							return;
						}
						const ranked = await invokeTool<{
							rankings: Array<{ candidate_id: string; eligible: boolean }>;
						}>(tools.get("rank_view_candidates")!, "rank-only", {
							candidates: [
								{ direction: [0, -1, 0], up: [0, 0, 1] },
								{ direction: [0, 1, 0], up: [0, 0, 1] },
							],
						});
						await invokeTool(tools.get("submit_view_result")!, "submit-view-only", {
							view_type: "orthographic",
							status: "matched",
							selected_candidate_id: ranked.details.rankings[0]!.candidate_id,
							confidence: "high",
							note: "front projection matches",
						});
					}),
					abort: vi.fn(async () => undefined),
					dispose: vi.fn(),
				},
			};
		});
		const usages: SubagentRunUsage[] = [];
		const output = await runDrawingViewSearchOnly({
			job: job(),
			worker: { request } as unknown as WorkerClient,
			workspace,
			resolvedModel: fakeResolvedModel(),
			controller: new RunController(),
			getResult: () => result,
			onUsage: async (usage) => {
				usages.push(usage);
			},
		});

		expect(roles).toEqual(["segmentation", "view_search"]);
		expect(createAgentSessionMock).toHaveBeenCalledTimes(2);
		expect(output.manifest.regions).toHaveLength(1);
		expect(output.manifest.regions[0]).toMatchObject({ status: "matched" });
		expect(usages.map(({ role }) => role)).toEqual(["segmentation", "view_search"]);
		expect(requests.some(({ operation }) => operation === "compare_drawing_sheet")).toBe(false);
		expect(await readFile(join(workspace, output.manifest_path), "utf8")).toContain('"status": "matched"');
	});

	it("keeps a controller-accepted submission when the active deadline aborts trailing model output", async () => {
		vi.useFakeTimers();
		const workspace = await makeWorkspace("pi-submission-timeout-race-");
		const source = "# stable source\n";
		const drawing = "single drawing page";
		const sourceSha256 = sha256(source);
		const stepSha256 = "submission-timeout-step-sha";
		await writeFile(join(workspace, "solution.py"), source);
		await writeFile(join(workspace, "drawing.png"), drawing);
		await installRegionCache({ workspace, drawingContents: drawing });
		const result = agentResult(sourceSha256, stepSha256);
		const request = vi.fn(async (_workspace: string, workerRequest: WorkerRequest): Promise<WorkerResponse> => {
			if (workerRequest.operation === "inspect") return inspectResponse(stepSha256);
			if (workerRequest.operation === "prepare_images") return await prepareResponse(workspace, workerRequest);
			if (workerRequest.operation === "render_drawing_view") {
				return await currentCadReferenceResponse(workspace, workerRequest, stepSha256);
			}
			if (workerRequest.operation === "rank_drawing_views") {
				const outputDir = String(workerRequest.output_dir);
				const candidates = workerRequest.candidates as Array<{
					candidate_id: string;
					view_spec: DrawingViewSpec;
				}>;
				const rankings = [];
				for (const [index, candidate] of candidates.entries()) {
					const renderPath = join(outputDir, "renders", `${candidate.candidate_id}.png`);
					const comparisonPath = join(outputDir, "comparisons", `${candidate.candidate_id}.png`);
					const alignedRenderPath = join(outputDir, "aligned", `${candidate.candidate_id}.png`);
					await writeArtifact(workspace, renderPath);
					await writeArtifact(workspace, comparisonPath);
					await writeArtifact(workspace, alignedRenderPath);
					const score = 0.9 - index * 0.1;
					rankings.push({
						candidate_id: candidate.candidate_id,
						score,
						...capacityFields(score),
						view_spec: candidate.view_spec,
						render_path: renderPath,
						comparison_path: comparisonPath,
						...alignmentFields(alignedRenderPath),
					});
				}
				const overviewPath = join(outputDir, "overview.png");
				await writeArtifact(workspace, overviewPath);
				return {
					protocol_version: PROTOCOL_VERSION,
					operation: "rank_drawing_views",
					status: "ok",
					region_id: String(workerRequest.region_id),
					step_path: "output.step",
					step_sha256: stepSha256,
					rankings,
					margin: 0.1,
					comparison_path: overviewPath,
					scoring_mode: "capacity_transport",
					scorer_profile: "capacity_transport_v1",
				};
			}
			throw new Error(`Unexpected worker operation ${workerRequest.operation}`);
		});
		let submittedResolve: (() => void) | undefined;
		const submitted = new Promise<void>((resolve) => {
			submittedResolve = resolve;
		});
		let rejectPrompt: ((reason: Error) => void) | undefined;
		const abort = vi.fn(async () => {
			rejectPrompt?.(new Error("deadline aborted trailing response"));
		});
		createAgentSessionMock.mockImplementationOnce(async (creation: SessionCreationOptions) => ({
			session: {
				sessionId: "submitted-before-timeout-session",
				getActiveToolNames: () => [...creation.tools],
				getSessionStats: () => ({
					sessionFile: undefined,
					tokens: { input: 2, output: 1, cacheRead: 0, cacheWrite: 0, total: 3 },
					cost: 0.01,
					toolCalls: 2,
				}),
				subscribe: vi.fn(() => vi.fn()),
				prompt: vi.fn(async () => {
					const tools = new Map(creation.customTools.map((tool) => [tool.name, tool]));
					const ranked = await invokeTool<{
						rankings: Array<{ candidate_id: string; eligible: boolean }>;
					}>(tools.get("rank_view_candidates")!, "rank-race", {
						candidates: [
							{ direction: [0, -1, 0], up: [0, 0, 1] },
							{ direction: [0, 1, 0], up: [0, 0, 1] },
						],
					});
					await invokeTool(tools.get("submit_view_result")!, "submit-race", {
						view_type: "orthographic",
						status: "matched",
						selected_candidate_id: ranked.details.rankings[0]!.candidate_id,
						confidence: "high",
						note: "submission completed before trailing output timed out",
					});
					submittedResolve?.();
					await new Promise<never>((_resolve, reject) => {
						rejectPrompt = reject;
					});
				}),
				abort,
				dispose: vi.fn(),
			},
		}));
		const usages: SubagentRunUsage[] = [];
		const execution = runDrawingViewSearchOnly({
			job: job(),
			worker: { request } as unknown as WorkerClient,
			workspace,
			resolvedModel: fakeResolvedModel(),
			controller: new RunController(),
			getResult: () => result,
			onUsage: async (usage) => {
				usages.push(usage);
			},
		});

		await submitted;
		await vi.advanceTimersByTimeAsync(600_000);
		const output = await execution;

		expect(abort).toHaveBeenCalledOnce();
		expect(output.manifest.regions[0]).toMatchObject({ status: "matched" });
		expect(usages).toHaveLength(1);
		expect(usages[0]).toMatchObject({ role: "view_search", status: "success" });
		expect(vi.getTimerCount()).toBe(0);
	});

	it("propagates RunController abort into the active search subagent and records interruption", async () => {
		const workspace = await makeWorkspace("pi-reflection-abort-");
		const source = "# source snapshot\n";
		const sourceSha256 = sha256(source);
		const stepSha256 = "abort-step-sha";
		await writeFile(join(workspace, "solution.py"), source);
		const drawing = "merged drawing bytes";
		await writeFile(join(workspace, "drawing.png"), drawing);
		await installRegionCache({ workspace, drawingContents: drawing });
		const result = agentResult(sourceSha256, stepSha256);
		const request = vi.fn(async (_workspace: string, workerRequest: WorkerRequest) => {
			if (workerRequest.operation === "prepare_images") {
				return await prepareResponse(workspace, workerRequest);
			}
			if (workerRequest.operation === "render_drawing_view") {
				return await currentCadReferenceResponse(workspace, workerRequest, stepSha256);
			}
			return inspectResponse(stepSha256);
		});
		let promptStartedResolve: (() => void) | undefined;
		const promptStarted = new Promise<void>((resolve) => {
			promptStartedResolve = resolve;
		});
		let rejectPrompt: ((reason: Error) => void) | undefined;
		const abort = vi.fn(async () => {
			rejectPrompt?.(new Error("fake session aborted"));
		});
		const unsubscribe = vi.fn();
		createAgentSessionMock.mockImplementation(async (creation: SessionCreationOptions) => ({
			session: {
				sessionId: "aborted-search-session",
				getActiveToolNames: () => [...creation.tools],
				getSessionStats: () => ({
					sessionFile: undefined,
					tokens: { input: 1, output: 0, cacheRead: 0, cacheWrite: 0, total: 1 },
					cost: 0,
					toolCalls: 0,
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
		const tool = makePublicTool({
			workspace,
			worker: { request } as unknown as WorkerClient,
			result,
			controller,
			onUsage: async (usage) => {
				usages.push(usage);
			},
		});

		const execution = invokeTool(tool as TestTool, "reflect-abort", {});
		await promptStarted;
		await controller.interrupt("SIGINT");

		await expect(execution).rejects.toThrow("interrupted");
		expect(controller.signal.aborted).toBe(true);
		expect(abort).toHaveBeenCalledOnce();
		expect(unsubscribe).toHaveBeenCalledOnce();
		expect(usages).toHaveLength(1);
		expect(usages[0]).toMatchObject({
			role: "view_search",
			status: "interrupted",
			accounting_id: "aborted-search-session",
		});
	});

	it("isolates active-Agent and total-wall timeouts to the affected region", async () => {
		vi.useFakeTimers();
		const scenarios = [
			{
				name: "active",
				advanceMs: 600_000,
				startControlledTool: false,
				expectedError: "view_search subagent exhausted 10 minutes of active Agent time",
			},
			{
				name: "wall",
				advanceMs: 7_200_000,
				startControlledTool: true,
				expectedError: "view_search subagent exceeded 120 minutes total wall time",
			},
		] as const;

		for (const scenario of scenarios) {
			const workspace = await makeWorkspace(`pi-reflection-${scenario.name}-timeout-`);
			const source = "# source snapshot\n";
			const sourceSha256 = sha256(source);
			const stepSha256 = `${scenario.name}-timeout-step-sha`;
			await writeFile(join(workspace, "solution.py"), source);
			const drawing = "merged drawing bytes";
			await writeFile(join(workspace, "drawing.png"), drawing);
			await installRegionCache({ workspace, drawingContents: drawing });
			const result = agentResult(sourceSha256, stepSha256);
			const request = vi.fn(async (_workspace: string, workerRequest: WorkerRequest) => {
				if (workerRequest.operation === "prepare_images") return await prepareResponse(workspace, workerRequest);
				if (workerRequest.operation === "render_drawing_view") {
					return await currentCadReferenceResponse(workspace, workerRequest, stepSha256);
				}
				if (workerRequest.operation === "compare_drawing_sheet") {
					const sheetPath = join(String(workerRequest.output_dir), "comparison-sheet.png");
					await writeArtifact(workspace, sheetPath);
					return {
						protocol_version: PROTOCOL_VERSION,
						operation: "compare_drawing_sheet",
						status: "ok",
						sheet_path: sheetPath,
						regions: workerRequest.regions as never,
					} as WorkerResponse;
				}
				return inspectResponse(stepSha256);
			});
			let promptStartedResolve: (() => void) | undefined;
			const promptStarted = new Promise<void>((resolve) => {
				promptStartedResolve = resolve;
			});
			let rejectPrompt: ((reason: Error) => void) | undefined;
			let eventListener: ((event: TestToolExecutionEvent) => void) | undefined;
			const abort = vi.fn(async () => {
				rejectPrompt?.(new Error("fake timeout abort"));
			});
			const unsubscribe = vi.fn(() => {
				eventListener = undefined;
			});
			const dispose = vi.fn();
			createAgentSessionMock.mockImplementationOnce(async (creation: SessionCreationOptions) => ({
				session: {
					sessionId: `${scenario.name}-timeout-search-session`,
					getActiveToolNames: () => [...creation.tools],
					getSessionStats: () => ({
						sessionFile: undefined,
						tokens: { input: 1, output: 0, cacheRead: 0, cacheWrite: 0, total: 1 },
						cost: 0,
						toolCalls: scenario.startControlledTool ? 1 : 0,
					}),
					subscribe: vi.fn((listener: (event: TestToolExecutionEvent) => void) => {
						eventListener = listener;
						return unsubscribe;
					}),
					prompt: vi.fn(async () => {
						if (scenario.startControlledTool) {
							eventListener?.({
								type: "tool_execution_start",
								toolName: "rank_view_candidates",
								toolCallId: "rank-timeout",
							});
						}
						promptStartedResolve?.();
						await new Promise<never>((_resolve, reject) => {
							rejectPrompt = reject;
						});
					}),
					abort,
					dispose,
				},
			})).mockImplementationOnce(async (creation: SessionCreationOptions) => ({
				session: {
					sessionId: `${scenario.name}-timeout-reflection-session`,
					getActiveToolNames: () => [...creation.tools],
					getSessionStats: () => ({
						sessionFile: undefined,
						tokens: { input: 1, output: 1, cacheRead: 0, cacheWrite: 0, total: 2 },
						cost: 0,
						toolCalls: 1,
					}),
					subscribe: vi.fn(() => vi.fn()),
					prompt: vi.fn(async () => {
						const tools = new Map(creation.customTools.map((tool) => [tool.name, tool]));
						await invokeTool(tools.get("submit_reflection")!, "submit-reflection", {
							should_revise: false,
							overall_summary: "View search timed out for the locked region.",
							findings: [],
						});
					}),
					abort: vi.fn(async () => undefined),
					dispose: vi.fn(),
				},
			}));
			const usages: SubagentRunUsage[] = [];
			const controller = new RunController();
			const tool = makePublicTool({
				workspace,
				worker: { request } as unknown as WorkerClient,
				result,
				controller,
				onUsage: async (usage) => {
					usages.push(usage);
				},
			});

			const execution = invokeTool<{ manifest: DrawingViewManifest }>(
				tool as TestTool,
				`reflect-${scenario.name}-timeout`,
				{},
			);
			await promptStarted;
			await vi.advanceTimersByTimeAsync(scenario.advanceMs);

			const response = await execution;
			expect(response.details.manifest.regions[0]).toMatchObject({
				status: "error",
				alternatives: [],
			});
			expect(response.details.manifest.regions[0]!.note).toContain(scenario.expectedError);
			expect(controller.interrupted).toBe(false);
			expect(abort).toHaveBeenCalledOnce();
			expect(unsubscribe).toHaveBeenCalledOnce();
			expect(dispose).toHaveBeenCalledOnce();
			expect(usages).toHaveLength(2);
			expect(usages.find(({ role }) => role === "view_search")).toMatchObject({
				role: "view_search",
				status: "timeout",
				error: scenario.expectedError,
			});
			expect(vi.getTimerCount()).toBe(0);
		}
	});
});
