import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";

import type { SessionEntry } from "@earendil-works/pi-coding-agent";
import { afterEach, describe, expect, it, vi } from "vitest";

import * as files from "../src/files.js";
import type { WorkerClient, WorkerRequest } from "../src/local-worker.js";
import {
	cameraSha256,
	CALIBRATE_DRAWING_SCALE_PARAMETERS_SCHEMA,
	createMainViewReflectTools,
	DUAL_ROI_CROP_PARAMETERS_SCHEMA,
	filterLockedMainViewManifest,
	INSPECT_COMPARISON_PARAMETERS_SCHEMA,
	loadLockedMainViewManifest,
	mainViewReflectContinuationPrompt,
	mainViewReflectSummary,
	reconcileMainViewInspectionEvidence,
	restoreMainViewReflectState,
	summaryHasInspectedFormalState,
} from "../src/main-view-reflect.js";
import { loadSystemPrompt } from "../src/prompts.js";
import { continuationAccepted, RunController, unchangedContinuationPrompt } from "../src/runner.js";
import { restoreTargetAnnotationState } from "../src/target-annotations.js";
import {
	MAIN_VIEW_REFLECT_AGENT_TOOLS,
	PROTOCOL_VERSION,
	TARGET_ANNOTATION_PROFILE,
	type AgentJob,
	type AgentResult,
	type ArtifactPayload,
	type CandidateInspection,
	type DrawingManifestRegion,
	type DrawingSimilarityTransform,
	type DrawingViewSpec,
	type DrawingViewManifest,
	type WorkerResponse,
} from "../src/types.js";
import { makePng } from "./png-fixture.js";

interface TestTool {
	name: string;
	description: string;
	execute: (...arguments_: unknown[]) => Promise<unknown>;
}

const workspaces: string[] = [];

afterEach(async () => {
	vi.restoreAllMocks();
	for (const workspace of workspaces.splice(0)) await rm(workspace, { recursive: true, force: true });
});

async function makeWorkspace(): Promise<string> {
	const workspace = await mkdtemp(join(tmpdir(), "main-view-reflect-"));
	workspaces.push(workspace);
	return workspace;
}

function digest(bytes: Buffer | string): string {
	return createHash("sha256").update(bytes).digest("hex");
}

async function writePng(workspace: string, path: string, marker: number): Promise<string> {
	const bytes = makePng(320, 240, marker);
	await mkdir(dirname(join(workspace, path)), { recursive: true });
	await writeFile(join(workspace, path), bytes);
	return digest(bytes);
}

function candidate(id: string, direction: [number, number, number], suffix: string) {
	return {
		candidate_id: id,
		render_path: `locked/${suffix}-render.png`,
		comparison_path: `locked/${suffix}-comparison.png`,
		aligned_render_path: `locked/${suffix}-aligned.png`,
		view_spec: { name: id, direction, up: [0, 1, 0] as [number, number, number] },
		score: 0.6,
		raw_score: 0.6,
		eligible: true,
		failed_gates: [],
		scoring_mode: "capacity_transport" as const,
		capacity_transport: {},
		transform: {
			scale_px_per_unit: 1,
			rotation_deg: 0,
			center_x_px: 100,
			center_y_px: 100,
			model_center_x: 0,
			model_center_y: 0,
			reflected: false,
		},
		alignment: {
			objective: 0.4,
			mode: "capacity_transport" as const,
			scale_at_bound: false,
			rotation_at_limit: false,
			translation_at_limit: false,
		},
	};
}

function manifest(sourceSha: string, stepSha: string): DrawingViewManifest {
	return {
		version: 3,
		scorer_profile: "capacity_transport_v1",
		input_sha256: "a".repeat(64),
		source_sha256: sourceSha,
		step_sha256: stepSha,
		regions: [{
			region_id: "front",
			page: 0,
			roi: [0, 0, 500, 500],
			crop_path: "locked/front-crop.png",
			view_type: "orthographic",
			status: "matched",
			selected: candidate("candidate-001", [0, 0, 1], "front"),
			alternatives: [candidate("candidate-002", [0, 0, -1], "alternative")],
			margin: 0.1,
			confidence: "high",
			note: "locked front view",
			width: 320,
			height: 240,
			crop_sha256: "b".repeat(64),
		}],
	} as unknown as DrawingViewManifest;
}

it("restricts a locked manifest to one selected comparison region", () => {
	const original = manifest("1".repeat(64), "2".repeat(64));
	const second = {
		...original.regions[0]!,
		region_id: "iso",
		view_type: "isometric" as const,
		crop_path: "locked/iso-crop.png",
		selected: candidate("candidate-iso", [-1, -1, 1], "iso"),
	} as unknown as DrawingManifestRegion;
	original.regions.push(second);

	const filtered = filterLockedMainViewManifest(original, "iso");

	expect(filtered.regions.map(({ region_id }) => region_id)).toEqual(["iso"]);
	expect(original.regions.map(({ region_id }) => region_id)).toEqual(["front", "iso"]);
	expect(() => filterLockedMainViewManifest(original, "missing")).toThrow("has no region missing");
});

function validArtifact(stepSha: string): ArtifactPayload {
	return {
		status: "valid",
		path: "output.step",
		sha256: stepSha,
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
}

function job(sourceSha: string, stepSha: string, manifestSha = "d".repeat(64)): AgentJob {
	return {
		protocol_version: PROTOCOL_VERSION,
		operation: "run_fixture",
		run_id: "test",
		agent_mode: "base-only",
		fixture: { id: "102", description: "test", inputFiles: ["input.png"], inputType: "drawing", taskType: "generation", sourceDir: "." },
		staged_paths: ["inputs/00.png"],
		provider: "ucloud",
		model: "gpt-5.6-sol",
		thinking_level: "high",
		pi_version: "0.84.1",
		container_image: "test",
		container_image_digest: "sha256:test",
		continuation: {
			kind: "main_view_reflect",
			id: "main-view-reflect-102",
			prompt: "inspect",
			baseline_source_sha256: sourceSha,
			baseline_step_sha256: stepSha,
			locked_view_manifest_path: "locked/view_manifest.json",
			locked_view_manifest_sha256: manifestSha,
		},
	};
}

async function setup(options: { fixedFrame?: boolean; section?: boolean } = {}) {
	const workspace = await makeWorkspace();
	await writeFile(join(workspace, "solution.py"), "baseline source\n");
	await writeFile(join(workspace, "output.step"), "baseline step\n");
	const sourceSha = digest("baseline source\n");
	const stepSha = digest("baseline step\n");
	const lockedManifest = manifest(sourceSha, stepSha);
	if (options.fixedFrame) {
		await mkdir(join(workspace, "locked"), { recursive: true });
		await writeFile(join(workspace, "locked/anchor.step"), "baseline step\n");
		lockedManifest.projection_frame = {
			policy: "fixed", anchor_step_path: "locked/anchor.step", anchor_step_sha256: stepSha,
		};
		Object.assign(lockedManifest.regions[0]!.selected!.transform, {
			rotation_deg: 27.253068131156706,
			center_x_px: 264.1600177525925,
			model_center_x: -2.0802078155877055,
		});
	}
	if (options.section) {
		Object.assign(lockedManifest.regions[0]!.selected!.view_spec, {
			section_offset_fraction: 0.8, section_keep: "back",
		});
	}
	for (const [index, path] of [
		"locked/front-crop.png",
		"locked/front-render.png",
		"locked/front-aligned.png",
		"locked/front-comparison.png",
	].entries()) await writePng(workspace, path, 10 + index);
	await writeFile(join(workspace, "locked/view_manifest.json"), `${JSON.stringify(lockedManifest, null, 2)}\n`);
	const manifestSha = digest(await readFile(join(workspace, "locked/view_manifest.json")));
	const result = {
		last_executed_source_sha256: sourceSha,
		last_executed_step_sha256: stepSha,
	} as AgentResult;
	const modes: string[] = [];
	const fixedScales: number[] = [];
	const fixedTransforms: DrawingSimilarityTransform[] = [];
	const fixedRequests: WorkerRequest[] = [];
	const reflectionRequests: WorkerRequest[] = [];
	const worker: WorkerClient = {
		async request<T extends WorkerResponse>(_workspace: string, request: WorkerRequest): Promise<T> {
			if (request.operation === "inspect") {
				const currentStep = await readFile(join(workspace, "output.step"));
				return {
					protocol_version: PROTOCOL_VERSION,
					operation: "inspect",
					artifact: validArtifact(digest(currentStep)),
				} as T;
			}
			if (request.operation === "rank_drawing_views") {
				modes.push("rank");
				const output = request.output_dir as string;
				const paths = {
					render: join(output, "render.png"),
					aligned: join(output, "aligned.png"),
					comparison: join(output, "comparison.png"),
				};
				await writePng(workspace, paths.render, 30);
				await writePng(workspace, paths.aligned, 31);
				await writePng(workspace, paths.comparison, 32);
				const requested = (request.candidates as Array<{ candidate_id: string; view_spec: DrawingViewSpec }>)[0]!;
				const currentStep = await readFile(join(workspace, "output.step"));
				const rankedTransform: DrawingSimilarityTransform = digest(currentStep) === stepSha
					? candidate(requested.candidate_id, [0, 0, 1], "ignored").transform
					: {
						scale_px_per_unit: 2,
						rotation_deg: 4,
						center_x_px: 130,
						center_y_px: 140,
						model_center_x: 2,
						model_center_y: 3,
						reflected: false,
					};
				return {
					protocol_version: PROTOCOL_VERSION,
					operation: "rank_drawing_views",
					status: "ok",
					region_id: request.region_id,
					step_path: "output.step",
					step_sha256: digest(await readFile(join(workspace, "output.step"))),
					rankings: [{
						...candidate(requested.candidate_id, [0, 0, 1], "ignored"),
						candidate_id: requested.candidate_id,
						view_spec: requested.view_spec,
						render_path: paths.render,
						aligned_render_path: paths.aligned,
						comparison_path: paths.comparison,
						transform: rankedTransform,
					}],
					margin: null,
					comparison_path: paths.comparison,
					scoring_mode: "capacity_transport",
					scorer_profile: "capacity_transport_v1",
				} as unknown as T;
			}
			if (request.operation === "render_fixed_projection") {
				fixedRequests.push(request);
				modes.push("render_fixed_projection");
				const output = request.output_dir as string;
				const renderPath = join(output, "aligned-render.png");
				const hash = await writePng(workspace, renderPath, 35 + fixedScales.length);
				const transform = request.transform as DrawingSimilarityTransform;
				fixedScales.push(transform.scale_px_per_unit);
				fixedTransforms.push({ ...transform });
				return {
					protocol_version: PROTOCOL_VERSION,
					operation: "render_fixed_projection",
					status: "ok",
					region_id: request.region_id,
					step_path: "output.step",
					step_sha256: digest(await readFile(join(workspace, "output.step"))),
					view_spec: request.view_spec,
					transform: request.transform,
					render_path: renderPath,
					width: 320,
					height: 240,
					mime_type: "image/png",
					sha256: hash,
					projection_bounds: [-50, -25, 50, 25],
					projection_width_units: 100,
					projection_height_units: 50,
					...(request.projection_anchor_step_path ? {
						projection_origin: [1, 2, 3],
						projection_anchor_step_sha256: digest(await readFile(join(workspace, request.projection_anchor_step_path as string))),
						...((request.view_spec as DrawingViewSpec).section_offset_fraction !== undefined ? {
							section_origin: [0, 0, 40],
						} : {}),
					} : {}),
				} as T;
			}
			if (request.operation === "compose_reflection_inspection") {
				reflectionRequests.push(request);
				modes.push(request.mode as string);
				const output = request.output_dir as string;
				if (request.mode === "coordinate_overview") {
					const context = join(output, "overview.png");
					const hash = await writePng(workspace, context, 40);
					return {
						protocol_version: PROTOCOL_VERSION,
						operation: "compose_reflection_inspection",
						status: "ok",
						region_id: request.region_id,
						mode: "coordinate_overview",
						context_path: context,
						width: 320,
						height: 240,
						sha256: { context: hash },
					} as T;
				}
				const paths = {
					reference: join(output, "reference.png"),
					cad: join(output, "cad.png"),
					context: join(output, "context.png"),
					comparison: join(output, "comparison.png"),
				};
				const hashes = {
					reference_crop: await writePng(workspace, paths.reference, 41),
					cad_crop: await writePng(workspace, paths.cad, 42),
					context: await writePng(workspace, paths.context, 43),
					comparison: await writePng(workspace, paths.comparison, 44),
				};
				return {
					protocol_version: PROTOCOL_VERSION,
					operation: "compose_reflection_inspection",
					status: "ok",
					region_id: request.region_id,
					mode: "dual_roi",
					width: 320,
					height: 480,
					reference_roi: request.reference_roi,
					cad_roi: request.cad_roi,
					reference_pixel_roi: [0, 0, 100, 100],
					cad_pixel_roi: [10, 10, 110, 110],
					reference_crop_path: paths.reference,
					cad_crop_path: paths.cad,
					context_path: paths.context,
					comparison_path: paths.comparison,
					sha256: hashes,
				} as T;
			}
			throw new Error(`Unexpected worker operation ${request.operation}`);
		},
	};
	const state = await restoreMainViewReflectState({
		workspace,
		fixtureId: "102",
		lockedManifest,
		lockedManifestPath: "locked/view_manifest.json",
		lockedManifestSha256: manifestSha,
		baselineSourceSha256: sourceSha,
		baselineStepSha256: stepSha,
	});
	const summaries: ReturnType<typeof mainViewReflectSummary>[] = [];
	const tools = createMainViewReflectTools({
		workspace,
		state,
		worker,
		controller: new RunController(),
		getResult: () => result,
		onUpdate: async (summary) => {
			summaries.push(summary);
		},
	}) as unknown as TestTool[];
	return {
		workspace,
		sourceSha,
		stepSha,
		lockedManifest,
		manifestSha,
		result,
		state,
		worker,
		tools,
		modes,
		reflectionRequests,
		fixedScales,
		fixedTransforms,
		fixedRequests,
		summaries,
	};
}

async function invoke(tool: TestTool, parameters: object): Promise<unknown> {
	return await tool.execute("call", parameters, undefined, undefined, {});
}

function inspectionSessionEntry(response: unknown, overrides: Record<string, unknown> = {}): SessionEntry {
	const returned = response as { content: unknown; details: unknown };
	return {
		type: "message",
		id: "inspection-result",
		parentId: null,
		timestamp: new Date().toISOString(),
		message: {
			role: "toolResult",
			toolCallId: "call",
			toolName: "inspect_comparison",
			isError: false,
			content: returned.content,
			details: returned.details,
			timestamp: Date.now(),
			...overrides,
		},
	} as SessionEntry;
}

describe("Main Agent fixed-view self-reflection contract", () => {
	it("appends dimension evidence after the normal aligned comparison image", async () => {
		const fixture = await setup({ fixedFrame: true });
		const dimensionPath = "drawing_feedback/dimensions/comparison.png";
		await writePng(fixture.workspace, dimensionPath, 77);
		const dimensionImage = { type: "image" as const, mimeType: "image/png", data: (await readFile(join(fixture.workspace, dimensionPath))).toString("base64") };
		const dimensionText = { type: "text" as const, text: "Current measured bore diameter: 4 mm" };
		const callback = vi.fn(async (regionId: string) => {
			expect(regionId).toBe("front");
			return { content: [dimensionText, dimensionImage] };
		});
		const tools = createMainViewReflectTools({
			workspace: fixture.workspace, state: fixture.state, worker: fixture.worker,
			controller: new RunController(), getResult: () => fixture.result,
			onUpdate: async summary => { fixture.summaries.push(summary); },
			dimensionComparison: callback,
		}) as unknown as TestTool[];
		const response = await invoke(tools[0]!, { region_id: "front" }) as {
			content: Array<{ type: string; text?: string; data?: string }>;
			details: { step_sha256: string };
		};
		expect(response.content[0]!.text).toContain("LEFT is the Original drawing and RIGHT is the current CAD");
		expect(response.content[1]!.type).toBe("image");
		expect(response.content[1]!.data).not.toBe(dimensionImage.data);
		expect(response.content.slice(2)).toEqual([dimensionText, dimensionImage]);
		expect(response.details.step_sha256).toBe(fixture.stepSha);
		expect(callback).toHaveBeenCalledOnce();
		expect(fixture.state.inspections.size).toBe(1);
		expect(fixture.state.inspectionAttempts).toHaveLength(1);
		expect(fixture.summaries.at(-1)!.inspection_count).toBe(1);
	});

	it("does not record a successful inspection when its dimension callback fails", async () => {
		const fixture = await setup({ fixedFrame: true });
		const failure = new Error("Dimension review failed");
		let fail = true;
		const tools = createMainViewReflectTools({
			workspace: fixture.workspace, state: fixture.state, worker: fixture.worker,
			controller: new RunController(), getResult: () => fixture.result,
			onUpdate: async summary => { fixture.summaries.push(summary); },
			dimensionComparison: async () => {
				if (fail) throw failure;
				return { content: [{ type: "text" as const, text: "No reliably corresponding dimension in this view." }] };
			},
		}) as unknown as TestTool[];
		await expect(invoke(tools[0]!, { region_id: "front" })).rejects.toBe(failure);
		expect(fixture.state.inspections.size).toBe(0);
		expect(fixture.state.inspectionAttempts).toHaveLength(0);
		expect(fixture.summaries).toHaveLength(0);
		fail = false;
		await tools[0]!.execute("successful-dimension-review", { region_id: "front" }, undefined, undefined, {});
		const successfulRecords = [...fixture.state.inspections.values()];
		const attempts = [...fixture.state.inspectionAttempts];
		fail = true;
		await expect(invoke(tools[0]!, { region_id: "front" })).rejects.toBe(failure);
		expect([...fixture.state.inspections.values()]).toEqual(successfulRecords);
		expect(fixture.state.inspectionAttempts).toEqual(attempts);
		expect(fixture.summaries).toHaveLength(1);
		const restored = await restoreMainViewReflectState({
			workspace: fixture.workspace, fixtureId: "102", lockedManifest: fixture.lockedManifest,
			lockedManifestPath: "locked/view_manifest.json", lockedManifestSha256: fixture.manifestSha,
			baselineSourceSha256: fixture.sourceSha, baselineStepSha256: fixture.stepSha,
		});
		expect([...restored.inspections.values()]).toEqual(successfulRecords);
	});

	it.each(["solution.py", "output.step"] as const)("rejects %s changes during the dimension callback", async path => {
		const fixture = await setup({ fixedFrame: true });
		const tools = createMainViewReflectTools({
			workspace: fixture.workspace, state: fixture.state, worker: fixture.worker,
			controller: new RunController(), getResult: () => fixture.result,
			onUpdate: async summary => { fixture.summaries.push(summary); },
			dimensionComparison: async () => {
				await writeFile(join(fixture.workspace, path), "changed during dimension callback\n");
				return { content: [{ type: "text" as const, text: "Stale dimensions must not count as inspection evidence." }] };
			},
		}) as unknown as TestTool[];
		const rejected = await invoke(tools[0]!, { region_id: "front" }).then(() => false, () => true);
		expect(rejected).toBe(true);
		expect(fixture.state.inspections.size).toBe(0);
		expect(fixture.state.inspectionAttempts).toHaveLength(0);
		expect(fixture.summaries).toHaveLength(0);
	});

	it.each(["audit", "summary"] as const)("rolls back inspection evidence when %s persistence fails", async (failureStage) => {
		const setupResult = await setup();
		const originalError = new Error(`${failureStage} EIO`);
		let failSummary = false;
		const writeAudit = vi.spyOn(files, "writeJsonAtomic");
		const tools = createMainViewReflectTools({
			workspace: setupResult.workspace,
			state: setupResult.state,
			worker: setupResult.worker,
			controller: new RunController(),
			getResult: () => setupResult.result,
			onUpdate: async (summary) => {
				setupResult.result.main_view_reflection = summary;
				if (failSummary) {
					failSummary = false;
					throw originalError;
				}
			},
		}) as unknown as TestTool[];
		const failNextCommit = () => {
			if (failureStage === "audit") writeAudit.mockRejectedValueOnce(originalError);
			else failSummary = true;
		};
		const restore = () => restoreMainViewReflectState({
			workspace: setupResult.workspace,
			fixtureId: "102",
			lockedManifest: setupResult.lockedManifest,
			lockedManifestPath: "locked/view_manifest.json",
			lockedManifestSha256: setupResult.manifestSha,
			baselineSourceSha256: setupResult.sourceSha,
			baselineStepSha256: setupResult.stepSha,
		});

		failNextCommit();
		await expect(invoke(tools[0]!, { region_id: "front" })).rejects.toBe(originalError);
		expect(setupResult.state.inspections.size).toBe(0);
		expect(setupResult.state.inspectionAttempts).toHaveLength(0);
		expect(setupResult.result.main_view_reflection?.inspected_formal_states).toEqual([]);
		expect((await restore()).inspections.size).toBe(0);

		await tools[0]!.execute("successful-retry", { region_id: "front" }, undefined, undefined, {});
		const successfulRecord = [...setupResult.state.inspections.values()][0]!;
		expect(successfulRecord.tool_call_id).toBe("successful-retry");
		failNextCommit();
		await expect(invoke(tools[0]!, { region_id: "front" })).rejects.toBe(originalError);
		expect([...setupResult.state.inspections.values()]).toEqual([successfulRecord]);
		expect(setupResult.state.inspectionAttempts).toHaveLength(1);
		expect(setupResult.result.main_view_reflection?.inspection_count).toBe(1);
		expect([...(await restore()).inspections.values()]).toEqual([successfulRecord]);
	});

	it("preserves the original inspection error when summary rollback also fails", async () => {
		const setupResult = await setup();
		const originalError = new Error("first summary failure");
		let updates = 0;
		const tools = createMainViewReflectTools({
			workspace: setupResult.workspace,
			state: setupResult.state,
			worker: setupResult.worker,
			controller: new RunController(),
			getResult: () => setupResult.result,
			onUpdate: async (summary) => {
				setupResult.result.main_view_reflection = summary;
				throw ++updates === 1 ? originalError : new Error("rollback summary failure");
			},
		}) as unknown as TestTool[];
		await expect(invoke(tools[0]!, { region_id: "front" })).rejects.toBe(originalError);
		expect(updates).toBe(2);
		expect(setupResult.state.inspections.size).toBe(0);
		expect(setupResult.result.main_view_reflection?.inspected_formal_states).toEqual([]);
	});

	it.each([
		["failed tool result", { isError: true }],
		["text-only tool result", { content: [{ type: "text", text: "comparison unavailable" }] }],
		["different tool call", { toolCallId: "another-call" }],
		["different tool", { toolName: "dual_roi_crop" }],
		["different inspection", { details: { inspection_id: "other", step_sha256: "wrong" } }],
		["different STEP", {}],
	])("rejects inspection evidence backed by a %s", async (_label, overrides) => {
		const setupResult = await setup();
		const response = await invoke(setupResult.tools[0]!, { region_id: "front" });
		const details = (response as { details: Record<string, unknown> }).details;
		const mismatchedReturn = _label === "different STEP"
			? { details: { ...details, step_sha256: "f".repeat(64) } }
			: _label === "different inspection"
				? { details: { ...details, inspection_id: "other" } }
				: overrides as Record<string, unknown>;
		await reconcileMainViewInspectionEvidence(setupResult.workspace, setupResult.state, [
			inspectionSessionEntry(response, mismatchedReturn),
		]);
		expect(setupResult.state.inspections.size).toBe(0);
		expect(mainViewReflectSummary(setupResult.state)).toMatchObject({
			inspection_count: 0,
			inspected_formal_states: [],
		});
		const persisted = await files.readJson<{ inspections: unknown[]; inspection_attempts: unknown[] }>(
			join(setupResult.workspace, setupResult.state.auditPath),
		);
		expect(persisted.inspections).toEqual([]);
		expect(persisted.inspection_attempts).toEqual([]);
	});

	it("retains complete successful image returns and removes unsupported attempts from full session history", async () => {
		const setupResult = await setup();
		const response = await invoke(setupResult.tools[0]!, { region_id: "front" });
		const attempt = setupResult.state.inspectionAttempts[0]!;
		setupResult.state.inspectionAttempts.unshift({ ...attempt, tool_call_id: "unfinished-call" });
		const entries: SessionEntry[] = [
			inspectionSessionEntry(response),
			{
				type: "compaction",
				id: "compacted",
				parentId: "inspection-result",
				timestamp: new Date().toISOString(),
				summary: "The comparison was inspected.",
				firstKeptEntryId: "compacted",
				tokensBefore: 1000,
			},
		];
		await reconcileMainViewInspectionEvidence(setupResult.workspace, setupResult.state, entries);
		expect(setupResult.state.inspections.size).toBe(1);
		expect(setupResult.state.inspectionAttempts).toEqual([{ ...attempt, sequence: 1 }]);
		expect(summaryHasInspectedFormalState(mainViewReflectSummary(setupResult.state), setupResult.sourceSha, setupResult.stepSha)).toBe(true);
	});

	it.each(["missing return", "legacy record"])("rejects unconfirmed persisted inspection evidence: %s", async (scenario) => {
		const setupResult = await setup();
		const response = await invoke(setupResult.tools[0]!, { region_id: "front" });
		if (scenario === "legacy record") {
			for (const record of setupResult.state.inspections.values()) delete record.tool_call_id;
			for (const attempt of setupResult.state.inspectionAttempts) delete attempt.tool_call_id;
		}
		await reconcileMainViewInspectionEvidence(
			setupResult.workspace,
			setupResult.state,
			scenario === "missing return" ? [] : [inspectionSessionEntry(response)],
		);
		expect(setupResult.state.inspections.size).toBe(0);
		expect(setupResult.state.inspectionAttempts).toHaveLength(0);
	});

	it("does not record inspection evidence when the returned annotated image cannot be rendered", async () => {
		const setupResult = await setup();
		const targetAnnotationState = await restoreTargetAnnotationState({
			workspace: setupResult.workspace,
			fixtureId: "102",
			profile: TARGET_ANNOTATION_PROFILE,
			lockedViewManifestSha256: setupResult.manifestSha,
		});
		const now = new Date().toISOString();
		targetAnnotationState.annotations.push({
			mark: 1,
			name: "mounting hole",
			description: "The circular mounting hole on the target drawing.",
			region_id: "front",
			geometry: { kind: "point", point: [500, 500] },
			created_at: now,
			updated_at: now,
		});
		let failAnnotatedImage = true;
		const worker: WorkerClient = {
			async request<T extends WorkerResponse>(workspace: string, request: WorkerRequest): Promise<T> {
				if (request.operation === "compose_reflection_inspection" && request.target_annotations && failAnnotatedImage) {
					throw new Error("Annotated comparison image unavailable");
				}
				return setupResult.worker.request<T>(workspace, request);
			},
		};
		const tools = createMainViewReflectTools({
			workspace: setupResult.workspace,
			state: setupResult.state,
			worker,
			controller: new RunController(),
			getResult: () => setupResult.result,
			onUpdate: async (summary) => { setupResult.summaries.push(summary); },
			targetAnnotationState,
		}) as unknown as TestTool[];

		await expect(invoke(tools[0]!, { region_id: "front" })).rejects.toThrow("Annotated comparison image unavailable");
		expect(setupResult.state.inspections.size).toBe(0);
		expect(setupResult.state.inspectionAttempts).toHaveLength(0);
		expect(setupResult.summaries).toHaveLength(0);
		expect(summaryHasInspectedFormalState(mainViewReflectSummary(setupResult.state), setupResult.sourceSha, setupResult.stepSha)).toBe(false);
		const restored = await restoreMainViewReflectState({
			workspace: setupResult.workspace,
			fixtureId: "102",
			lockedManifest: setupResult.lockedManifest,
			lockedManifestPath: "locked/view_manifest.json",
			lockedManifestSha256: setupResult.manifestSha,
			baselineSourceSha256: setupResult.sourceSha,
			baselineStepSha256: setupResult.stepSha,
		});
		expect(restored.inspections.size).toBe(0);

		failAnnotatedImage = false;
		const response = await invoke(tools[0]!, { region_id: "front" }) as { content: Array<{ type: string }> };
		expect(response.content[1]?.type).toBe("image");
		expect(setupResult.state.inspectionAttempts).toHaveLength(1);
		expect(summaryHasInspectedFormalState(setupResult.summaries[0], setupResult.sourceSha, setupResult.stepSha)).toBe(true);
	});

	it("persists Point/Box marks and redraws only marks belonging to the inspected region", async () => {
		const setupResult = await setup();
		const targetAnnotationState = await restoreTargetAnnotationState({
			workspace: setupResult.workspace,
			fixtureId: "102",
			profile: TARGET_ANNOTATION_PROFILE,
			lockedViewManifestSha256: setupResult.manifestSha,
		});
		const tools = createMainViewReflectTools({
			workspace: setupResult.workspace,
			state: setupResult.state,
			worker: setupResult.worker,
			controller: new RunController(),
			getResult: () => setupResult.result,
			onUpdate: async () => {},
			targetAnnotationState,
		}) as unknown as TestTool[];
		expect(tools.map(({ name }) => name)).toEqual([
			"inspect_comparison",
			"dual_roi_crop",
			"calibrate_drawing_scale",
			"annotate_target",
		]);

		await invoke(tools[0]!, { region_id: "front" });
		const response = await invoke(tools[3]!, {
			operation: "add",
			name: "six-slot rounded grille",
			description: "A rounded frame contains six vertical rounded slots arranged in two rows of three.",
			box: [370, 250, 655, 735],
		}) as { content: Array<{ type: string; text?: string }>; details: { affected: object; registry: object[] } };
		expect(response.content[0]?.text).toContain("Added Target mark [1] six-slot rounded grille · Box.");
		expect(response.content[0]?.text).toContain("A rounded frame contains six vertical rounded slots arranged in two rows of three.");
		expect(response.content[0]?.text).toContain("Marks shown in this image:");
		expect(response.content[0]?.text).toContain("[1] six-slot rounded grille · Box");
		expect(response.content[1]?.type).toBe("image");
		expect(response.details.affected).toMatchObject({
			mark: 1,
			name: "six-slot rounded grille",
			region_id: "front",
			mode: "Box",
		});
		expect(response.details.registry).toHaveLength(1);
		expect(targetAnnotationState.annotations).toMatchObject([{
			mark: 1,
			name: "six-slot rounded grille",
			region_id: "front",
			geometry: { kind: "box", box: [370, 250, 655, 735] },
		}]);
		expect(setupResult.reflectionRequests.at(-1)?.target_annotations).toEqual([{
			mark: 1,
			name: "six-slot rounded grille",
			box: [370, 250, 655, 735],
		}]);
		await invoke(tools[3]!, {
			operation: "add",
			name: "upper mounting hole",
			description: "The circular through-hole at the rounded free end of the upper mounting arm.",
			point: [52, 290],
		});
		expect(setupResult.reflectionRequests.at(-1)?.target_annotations).toEqual([{
			mark: 1,
			name: "six-slot rounded grille",
			box: [370, 250, 655, 735],
		}, {
			mark: 2,
			name: "upper mounting hole",
			point: [52, 290],
		}]);

		const now = new Date().toISOString();
		targetAnnotationState.annotations.push({
			mark: 3,
			name: "other-view feature",
			description: "A feature belonging to another Target view.",
			region_id: "other",
			geometry: { kind: "point", point: [500, 500] },
			created_at: now,
			updated_at: now,
		});
		targetAnnotationState.nextMark = 4;

		await invoke(tools[0]!, { region_id: "front" });
		expect(setupResult.reflectionRequests.at(-1)?.target_annotations).toEqual([{
			mark: 1,
			name: "six-slot rounded grille",
			box: [370, 250, 655, 735],
		}, {
			mark: 2,
			name: "upper mounting hole",
			point: [52, 290],
		}]);
	});

	it("exposes comparison, crop, and drawing-scale calibration tools", async () => {
		expect(MAIN_VIEW_REFLECT_AGENT_TOOLS).toEqual([
			"inspect_comparison",
			"dual_roi_crop",
			"calibrate_drawing_scale",
		]);
		expect(MAIN_VIEW_REFLECT_AGENT_TOOLS).not.toContain("self_reflect_2d_drawing");
		expect(MAIN_VIEW_REFLECT_AGENT_TOOLS.join(" ")).not.toMatch(/search|report|submit_reflection/u);
		expect(INSPECT_COMPARISON_PARAMETERS_SCHEMA.properties).toHaveProperty("region_id");
		expect(Object.keys(INSPECT_COMPARISON_PARAMETERS_SCHEMA.properties)).toEqual(["region_id"]);
		expect(Object.keys(DUAL_ROI_CROP_PARAMETERS_SCHEMA.properties)).toEqual(["region_id", "reference_roi", "cad_roi"]);
		expect("anyOf" in CALIBRATE_DRAWING_SCALE_PARAMETERS_SCHEMA).toBe(true);
		const repositoryRoot = new URL("../../..", import.meta.url).pathname;
		const integrationRoot = new URL("..", import.meta.url).pathname;
		const prompt = await loadSystemPrompt(repositoryRoot, integrationRoot, "main-view-reflect");
		expect(prompt).toContain("inspect_comparison");
		expect(prompt).toContain("dual_roi_crop");
		expect(prompt).toContain("calibrate_drawing_scale");
		expect(prompt).toContain("global and local visual evidence");
		expect(prompt).toContain("repeat this loop for as many rounds as needed");
	});

	it("hash-locks the manifest and rejects path escape and symlinks", async () => {
		const setupResult = await setup();
		await expect(loadLockedMainViewManifest({
			workspace: setupResult.workspace,
			manifestPath: "locked/view_manifest.json",
			manifestSha256: setupResult.manifestSha,
			baselineSourceSha256: setupResult.sourceSha,
			baselineStepSha256: setupResult.stepSha,
		})).resolves.toEqual(setupResult.lockedManifest);
		await expect(loadLockedMainViewManifest({
			workspace: setupResult.workspace,
			manifestPath: "../view_manifest.json",
			manifestSha256: setupResult.manifestSha,
			baselineSourceSha256: setupResult.sourceSha,
			baselineStepSha256: setupResult.stepSha,
		})).rejects.toThrow("escaped");
		await symlink(join(setupResult.workspace, "locked/view_manifest.json"), join(setupResult.workspace, "locked/link.json"));
		await expect(loadLockedMainViewManifest({
			workspace: setupResult.workspace,
			manifestPath: "locked/link.json",
			manifestSha256: setupResult.manifestSha,
			baselineSourceSha256: setupResult.sourceSha,
			baselineStepSha256: setupResult.stepSha,
		})).rejects.toThrow("unsafe");
	});

	it("reuses the initial comparison, refreshes the same camera after STEP changes, and invalidates stale Crop", async () => {
		const setupResult = await setup();
		const inspect = setupResult.tools[0]!;
		const crop = setupResult.tools[1]!;
		expect(setupResult.tools.map(({ name }) => name)).toEqual(MAIN_VIEW_REFLECT_AGENT_TOOLS);
		expect(crop.description).toBe("查看原始工程图与当前 CAD 二维投影中相互对应的局部细节。分别指定两侧需要放大的区域，以检查局部结构的形状、位置、方向和连接关系。两个 ROI 使用各自面板独立的 0–1000 坐标。可用的 `region_id`：`front`");
		await invoke(inspect, { region_id: "front" });
		expect(setupResult.modes).toEqual(["coordinate_overview"]);
		expect([...setupResult.state.snapshots.values()][0]?.reused_locked_artifact).toBe(true);
		await expect(invoke(crop, {
			region_id: "front",
			reference_roi: { x_min: 0, y_min: 0, x_max: 49, y_max: 100 },
			cad_roi: { x_min: 0, y_min: 0, x_max: 100, y_max: 100 },
		})).rejects.toThrow("at least 50");
		const cropResult = await invoke(crop, {
			region_id: "front",
			reference_roi: { x_min: 100, y_min: 100, x_max: 300, y_max: 300 },
			cad_roi: { x_min: 120, y_min: 100, x_max: 320, y_max: 300 },
		});
		const cropText = (cropResult as { content: Array<{ type: string; text?: string }> }).content[0]?.text;
		expect(cropText).toBe("以下图像对比了 `front` 中原始工程图与当前 CAD 的对应局部。上排红框标出了两个 ROI 的位置，下排展示放大结果。在使用这些图像判断缺陷之前，请先确认：（1）两侧 ROI 是否对应同一几何特征；（2）当前视觉证据是否足以确认当前 CAD 相比原始工程图存在明确的形状错误、位置偏差、方向错误或连接关系不一致。若两侧不是同一几何特征，请调整 ROI 后重新调用 `dual_roi_crop`；若两侧对应同一特征但视觉证据仍不充分，也请调整或扩大 ROI 后重新检查。不要依据不充分的局部证据推断缺陷。涉及具体尺寸修改时，仍需仔细查看原始输入工程图中的尺寸标注。");
		expect(cropText).not.toMatch(/source=|STEP=|crop_id/u);
		expect(setupResult.modes.at(-1)).toBe("dual_roi");

		await writeFile(join(setupResult.workspace, "solution.py"), "revised source\n");
		await writeFile(join(setupResult.workspace, "output.step"), "revised step\n");
		setupResult.result.last_executed_source_sha256 = digest("revised source\n");
		setupResult.result.last_executed_step_sha256 = digest("revised step\n");
		await expect(invoke(crop, {
			region_id: "front",
			reference_roi: { x_min: 100, y_min: 100, x_max: 300, y_max: 300 },
			cad_roi: { x_min: 120, y_min: 100, x_max: 320, y_max: 300 },
		})).rejects.toThrow("Establish the current complete-view context");
		expect(setupResult.modes.filter((mode) => mode === "rank")).toHaveLength(0);
		const revisedInspection = await invoke(inspect, { region_id: "front" }) as {
			details: { snapshot: { transform: DrawingSimilarityTransform; free_fit_scale_px_per_unit?: number } };
		};
		expect(setupResult.modes.slice(-4)).toEqual([
			"rank",
			"render_fixed_projection",
			"coordinate_overview",
			"coordinate_overview",
		]);
		expect(revisedInspection.details.snapshot).toMatchObject({
			free_fit_scale_px_per_unit: 2,
			transform: {
				scale_px_per_unit: 1,
				rotation_deg: 4,
				center_x_px: 130,
				center_y_px: 140,
			},
		});
		expect(setupResult.fixedTransforms.at(-1)).toMatchObject({
			scale_px_per_unit: 1,
			rotation_deg: 4,
			center_x_px: 130,
			center_y_px: 140,
		});
		const snapshots = [...setupResult.state.snapshots.values()];
		expect(snapshots).toHaveLength(2);
		expect(snapshots[1]?.reused_locked_artifact).toBe(false);
		expect(snapshots[1]?.camera_sha256).toBe(cameraSha256(setupResult.lockedManifest.regions[0]!.selected!.view_spec));
		expect(snapshots[1]?.sha256.comparison).toMatch(/^[a-f0-9]{64}$/u);
	});

	it("refreshes an old locked manifest whose aligned render belongs to another candidate", async () => {
		const setupResult = await setup();
		const selected = setupResult.state.lockedManifest.regions[0]!.selected!;
		selected.aligned_render_path = "locked/candidates/candidate-999/aligned-render.png";
		await writePng(setupResult.workspace, selected.aligned_render_path, 99);

		await invoke(setupResult.tools[0]!, { region_id: "front" });

		expect(setupResult.modes).toEqual([
			"rank",
			"render_fixed_projection",
			"coordinate_overview",
			"coordinate_overview",
		]);
		const snapshot = [...setupResult.state.snapshots.values()][0];
		expect(snapshot?.reused_locked_artifact).toBe(false);
		expect(snapshot?.locked_candidate_id).toBe("candidate-001");
		expect(snapshot?.camera_sha256).toBe(cameraSha256(selected.view_spec));
	});

	it.each(["STEP header only", "formal geometry"])("preserves the complete calibrated frame after %s changes without ranking", async (change) => {
		const fixture = await setup({ fixedFrame: true });
		const selected = fixture.lockedManifest.regions[0]!.selected!;
		const originalTransform = { ...selected.transform };
		await invoke(fixture.tools[0]!, { region_id: "front" });
		await invoke(fixture.tools[0]!, { region_id: "front" });
		expect(fixture.fixedRequests).toHaveLength(1);
		if (change === "formal geometry") {
			await writeFile(join(fixture.workspace, "solution.py"), "revised source\n");
			fixture.result.last_executed_source_sha256 = digest("revised source\n");
		}
		await writeFile(join(fixture.workspace, "output.step"), `${change}\n`);
		fixture.result.last_executed_step_sha256 = digest(`${change}\n`);
		await invoke(fixture.tools[0]!, { region_id: "front" });
		expect(fixture.modes).not.toContain("rank");
		expect(fixture.fixedRequests).toHaveLength(2);
		for (const request of fixture.fixedRequests) {
			expect(request).toMatchObject({
				projection_anchor_step_path: "locked/anchor.step",
				view_spec: selected.view_spec,
				transform: originalTransform,
			});
		}
		const snapshots = [...fixture.state.snapshots.values()];
		expect(snapshots).toHaveLength(2);
		expect(snapshots[0]!.fixed_frame_sha256).toMatch(/^[a-f0-9]{64}$/u);
		expect(snapshots[1]).toMatchObject({
			fixed_frame_sha256: snapshots[0]!.fixed_frame_sha256,
			projection_origin: [1, 2, 3],
			projection_anchor_step_sha256: fixture.stepSha,
			transform: originalTransform,
		});
		expect(await readFile(join(fixture.workspace, "locked/anchor.step"), "utf8")).toBe("baseline step\n");
	});

	it("keeps an anchor-relative section plane and origin after the formal STEP changes", async () => {
		const fixture = await setup({ fixedFrame: true, section: true });
		await invoke(fixture.tools[0]!, { region_id: "front" });
		await writeFile(join(fixture.workspace, "output.step"), "changed bounds and centroid\n");
		fixture.result.last_executed_step_sha256 = digest("changed bounds and centroid\n");
		await invoke(fixture.tools[0]!, { region_id: "front" });
		for (const snapshot of fixture.state.snapshots.values()) {
			expect(snapshot).toMatchObject({ projection_origin: [1, 2, 3], section_origin: [0, 0, 40] });
		}
		expect(fixture.fixedRequests.map((request) => request.view_spec)).toEqual([
			fixture.lockedManifest.regions[0]!.selected!.view_spec,
			fixture.lockedManifest.regions[0]!.selected!.view_spec,
		]);
		expect(fixture.modes).not.toContain("rank");
	});

	it.each([
		["anchor hash", { projection_anchor_step_sha256: "f".repeat(64) }],
		["current STEP hash", { step_sha256: "f".repeat(64) }],
		["origin", { projection_origin: [0, Number.NaN, 0] }],
		["section plane", { section_origin: undefined }],
		["camera", { view_spec: { name: "drift", direction: [1, 0, 0], up: [0, 1, 0] } }],
		["transform", { transform: { ...candidate("a", [0, 0, 1], "a").transform, rotation_deg: 4 } }],
	])("rejects a fixed-frame worker response with mismatched %s", async (_label, override) => {
		const fixture = await setup({ fixedFrame: true, section: true });
		const originalRequest = fixture.worker.request.bind(fixture.worker);
		vi.spyOn(fixture.worker, "request").mockImplementation(async (...args) => {
			const response = await originalRequest(...args);
			return args[1].operation === "render_fixed_projection" ? { ...response, ...override } : response;
		});
		await expect(invoke(fixture.tools[0]!, { region_id: "front" })).rejects.toThrow("Locked-frame projection returned a mismatched");
		expect(fixture.state.inspections.size).toBe(0);
		expect(fixture.state.snapshots.size).toBe(0);
	});

	it.each(["projection_origin", "section_origin"] as const)("rejects %s drift across formal STEP changes in the same fixed frame", async (field) => {
		const fixture = await setup({ fixedFrame: true, section: true });
		await invoke(fixture.tools[0]!, { region_id: "front" });
		await writeFile(join(fixture.workspace, "output.step"), "edited STEP\n");
		fixture.result.last_executed_step_sha256 = digest("edited STEP\n");
		const originalRequest = fixture.worker.request.bind(fixture.worker);
		vi.spyOn(fixture.worker, "request").mockImplementation(async (...args) => {
			const response = await originalRequest(...args);
			return args[1].operation === "render_fixed_projection" ? { ...response, [field]: [3, 4, 5] } : response;
		});
		await expect(invoke(fixture.tools[0]!, { region_id: "front" })).rejects.toThrow("changed its physical origin or section plane");
		expect(fixture.state.snapshots.size).toBe(1);
	});

	it("does not reuse legacy snapshots or a different fixed 2D transform", async () => {
		const fixture = await setup();
		await invoke(fixture.tools[0]!, { region_id: "front" });
		const legacyId = [...fixture.state.snapshots.keys()][0];
		await writeFile(join(fixture.workspace, "locked/anchor.step"), "baseline step\n");
		fixture.lockedManifest.projection_frame = {
			policy: "fixed", anchor_step_path: "locked/anchor.step", anchor_step_sha256: fixture.stepSha,
		};
		await invoke(fixture.tools[0]!, { region_id: "front" });
		const firstFixed = [...fixture.state.snapshots.values()].at(-1)!;
		expect(firstFixed.snapshot_id).not.toBe(legacyId);
		expect(firstFixed.fixed_frame_sha256).toBeDefined();
		fixture.lockedManifest.regions[0]!.selected!.transform.center_x_px += 20;
		await invoke(fixture.tools[0]!, { region_id: "front" });
		const secondFixed = [...fixture.state.snapshots.values()].at(-1)!;
		expect(secondFixed.snapshot_id).not.toBe(firstFixed.snapshot_id);
		expect(secondFixed.fixed_frame_sha256).not.toBe(firstFixed.fixed_frame_sha256);
		expect(fixture.fixedRequests).toHaveLength(2);
		expect(fixture.state.inspections.size).toBe(3);
	});

	it("validates the fixed anchor on manifest loading and before cache reuse", async () => {
		const fixture = await setup({ fixedFrame: true });
		const load = () => loadLockedMainViewManifest({
			workspace: fixture.workspace, manifestPath: "locked/view_manifest.json", manifestSha256: fixture.manifestSha,
			baselineSourceSha256: fixture.sourceSha, baselineStepSha256: fixture.stepSha,
		});
		await expect(load()).resolves.toEqual(fixture.lockedManifest);
		await invoke(fixture.tools[0]!, { region_id: "front" });
		await writeFile(join(fixture.workspace, "locked/anchor.step"), "mutated anchor\n");
		await expect(load()).rejects.toThrow("anchor STEP SHA-256 mismatch");
		await expect(invoke(fixture.tools[0]!, { region_id: "front" })).rejects.toThrow("anchor STEP SHA-256 mismatch");
		expect(fixture.fixedRequests).toHaveLength(1);
	});

	it("regenerates an unbound cached snapshot and its derived inspection instead of retaining a refitted image", async () => {
		const fixture = await setup({ fixedFrame: true });
		await invoke(fixture.tools[0]!, { region_id: "front" });
		const cached = [...fixture.state.snapshots.values()][0]!;
		delete cached.fixed_frame_sha256;
		delete cached.projection_origin;
		const firstRenderHash = cached.sha256.aligned_render;
		await invoke(fixture.tools[0]!, { region_id: "front" });
		expect(fixture.fixedRequests).toHaveLength(2);
		expect(fixture.reflectionRequests).toHaveLength(4);
		expect(fixture.state.inspections.size).toBe(1);
		const regenerated = [...fixture.state.snapshots.values()][0]!;
		expect(regenerated.fixed_frame_sha256).toBeDefined();
		expect(regenerated.sha256.aligned_render).not.toBe(firstRenderHash);
	});

	it.each(["../anchor.step", "locked/anchor-link.step"])("rejects unsafe fixed anchor path %s", async (anchorPath) => {
		const fixture = await setup({ fixedFrame: true });
		await symlink(join(fixture.workspace, "locked/anchor.step"), join(fixture.workspace, "locked/anchor-link.step"));
		fixture.lockedManifest.projection_frame!.anchor_step_path = anchorPath;
		const bytes = JSON.stringify(fixture.lockedManifest);
		await writeFile(join(fixture.workspace, "locked/view_manifest.json"), bytes);
		await expect(loadLockedMainViewManifest({
			workspace: fixture.workspace, manifestPath: "locked/view_manifest.json", manifestSha256: digest(bytes),
			baselineSourceSha256: fixture.sourceSha, baselineStepSha256: fixture.stepSha,
		})).rejects.toThrow(/escaped|unsafe/u);
	});

	it("uses the same immutable anchor for explicit scale calibration and retains its complete confirmed transform", async () => {
		const fixture = await setup({ fixedFrame: true, section: true });
		const scale = fixture.tools[2]!;
		await invoke(scale, { operation: "propose", region_id: "front", boundary: "overall_width", target_dimension_mm: 80 });
		await invoke(scale, { operation: "confirm", calibration_ref: "S001" });
		await writeFile(join(fixture.workspace, "output.step"), "revised STEP\n");
		fixture.result.last_executed_step_sha256 = digest("revised STEP\n");
		await invoke(fixture.tools[0]!, { region_id: "front" });
		expect(fixture.modes).not.toContain("rank");
		expect(fixture.fixedRequests).toHaveLength(4);
		for (const request of fixture.fixedRequests) expect(request.projection_anchor_step_path).toBe("locked/anchor.step");
		const snapshot = [...fixture.state.snapshots.values()][0]!;
		expect(snapshot.transform).toEqual(fixture.state.scaleCalibrations[0]!.locked_transform);
		expect(snapshot.transform).toMatchObject({ scale_px_per_unit: 1.25, rotation_deg: 27.253068131156706 });
		expect(snapshot).toMatchObject({ projection_origin: [1, 2, 3], section_origin: [0, 0, 40] });
	});

	it.each(["free-measurement", "locked-render"])("rejects physical frame drift in scale calibration %s", async (outputName) => {
		const fixture = await setup({ fixedFrame: true });
		const originalRequest = fixture.worker.request.bind(fixture.worker);
		vi.spyOn(fixture.worker, "request").mockImplementation(async (...args) => {
			const response = await originalRequest(...args);
			return args[1].operation === "render_fixed_projection" && String(args[1].output_dir).endsWith(outputName)
				? { ...response, projection_origin: [9, 9, 9] } : response;
		});
		await expect(invoke(fixture.tools[2]!, {
			operation: "propose", region_id: "front", boundary: "overall_width", target_dimension_mm: 80,
		})).rejects.toThrow("Drawing-scale projection drifted from the locked physical frame");
		expect(fixture.state.scaleCalibrations).toHaveLength(0);
	});

	it("proposes, confirms, persists, applies, and deletes a fixed drawing scale without target pixels", async () => {
		const setupResult = await setup();
		const inspect = setupResult.tools[0]!;
		const scale = setupResult.tools[2]!;
		expect(scale.description).toContain("without selecting target pixels");
		expect(scale.description).toContain("complete projected boundary");

		const proposal = await invoke(scale, {
			operation: "propose",
			region_id: "front",
			boundary: "overall_width",
			target_dimension_mm: 80,
		}) as {
			content: Array<{ type: string; text?: string }>;
			details: { calibration_ref: string; locked_scale_px_per_unit: number; status: string };
		};
		expect(proposal.details).toMatchObject({
			calibration_ref: "S001",
			locked_scale_px_per_unit: 1.25,
			status: "proposed",
		});
		expect(proposal.content[0]?.text).toContain("current CAD overall_width is 100.000 model units");
		expect(setupResult.fixedScales).toEqual([1, 1.25]);
		expect(setupResult.modes).toEqual([
			"rank",
			"render_fixed_projection",
			"render_fixed_projection",
			"coordinate_overview",
		]);

		await invoke(scale, { operation: "confirm", calibration_ref: "S001" });
		expect(setupResult.state.scaleCalibrations[0]?.status).toBe("confirmed");
		const inspection = await invoke(inspect, { region_id: "front" }) as {
			content: Array<{ type: string; text?: string }>;
		};
		expect(inspection.content[0]?.text).toBe("front: LEFT is the Original drawing and RIGHT is the current CAD. Each panel has its own 0-1000 coordinates.");
		expect(setupResult.fixedScales.at(-1)).toBe(1.25);
		expect([...setupResult.state.snapshots.values()][0]).toMatchObject({
			scale_calibration_ref: "S001",
			free_fit_scale_px_per_unit: 1,
			transform: { scale_px_per_unit: 1.25 },
		});

		const restored = await restoreMainViewReflectState({
			workspace: setupResult.workspace,
			fixtureId: "102",
			lockedManifest: setupResult.lockedManifest,
			lockedManifestPath: "locked/view_manifest.json",
			lockedManifestSha256: setupResult.manifestSha,
			baselineSourceSha256: setupResult.sourceSha,
			baselineStepSha256: setupResult.stepSha,
		});
		expect(restored.scaleCalibrations).toHaveLength(1);
		expect(restored.scaleCalibrations[0]?.status).toBe("confirmed");

		await invoke(scale, { operation: "delete", calibration_ref: "S001" });
		expect(setupResult.state.scaleCalibrations[0]?.status).toBe("deleted");
		expect(setupResult.state.snapshots).toHaveLength(0);
	});

	it("restores its cache and permits a no-op only after final global and local comparison evidence", async () => {
		const setupResult = await setup();
		setupResult.state.auditPath = "drawing_feedback/main-self-reflect/continuation-a.json";
		await invoke(setupResult.tools[0]!, { region_id: "front" });
		await invoke(setupResult.tools[1]!, {
			region_id: "front",
			reference_roi: { x_min: 100, y_min: 100, x_max: 300, y_max: 300 },
			cad_roi: { x_min: 100, y_min: 100, x_max: 300, y_max: 300 },
		});
		const restored = await restoreMainViewReflectState({
			workspace: setupResult.workspace,
			fixtureId: "102",
			lockedManifest: setupResult.lockedManifest,
			lockedManifestPath: "locked/view_manifest.json",
			lockedManifestSha256: setupResult.manifestSha,
			baselineSourceSha256: setupResult.sourceSha,
			baselineStepSha256: setupResult.stepSha,
			auditPath: setupResult.state.auditPath,
		});
		const summary = mainViewReflectSummary(restored);
		expect(summary.inspection_count).toBe(1);
		expect(summary.crop_count).toBe(1);
		expect(summaryHasInspectedFormalState(summary, setupResult.sourceSha, setupResult.stepSha)).toBe(true);
		const differentContinuation = await restoreMainViewReflectState({
			workspace: setupResult.workspace,
			fixtureId: "102",
			lockedManifest: setupResult.lockedManifest,
			lockedManifestPath: "locked/view_manifest.json",
			lockedManifestSha256: setupResult.manifestSha,
			baselineSourceSha256: setupResult.sourceSha,
			baselineStepSha256: setupResult.stepSha,
			auditPath: "drawing_feedback/main-self-reflect/continuation-b.json",
		});
		const differentSummary = mainViewReflectSummary(differentContinuation);
		expect(differentSummary.inspection_count).toBe(0);
		expect(differentSummary.inspected_formal_states).toEqual([]);
		expect(summaryHasInspectedFormalState(differentSummary, setupResult.sourceSha, setupResult.stepSha)).toBe(false);
		const continuationJob = job(setupResult.sourceSha, setupResult.stepSha, setupResult.manifestSha);
		const result = {
			applied_continuation: {
				id: continuationJob.continuation!.id,
				prompt_sha256: digest(continuationJob.continuation!.prompt),
				applied_at: new Date().toISOString(),
			},
			main_view_reflection: summary,
		} as AgentResult;
		const validInspection: CandidateInspection = {
			artifact: validArtifact(setupResult.stepSha),
			formal_execution_current: true,
			reason: "current",
			source_sha256: setupResult.sourceSha,
		};
		expect(continuationAccepted(continuationJob, result, validInspection)).toBe(true);
		const withoutInspection = { ...result };
		delete withoutInspection.main_view_reflection;
		expect(continuationAccepted(continuationJob, withoutInspection, validInspection)).toBe(false);
		expect(continuationAccepted(continuationJob, result, { ...validInspection, formal_execution_current: false })).toBe(false);
		expect(continuationAccepted(continuationJob, result, {
			...validInspection,
			artifact: { ...validInspection.artifact, status: "invalid" },
		})).toBe(false);
		expect(unchangedContinuationPrompt(continuationJob)).toContain("preserve the Direct output");
		expect(mainViewReflectContinuationPrompt("102", setupResult.lockedManifest)).not.toContain("submit_reflection");
	});
});
