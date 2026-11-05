import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import {
	causalRepairSummary,
	createCausalRepairTool,
	recordCausalRepairFormalExecution,
	restoreCausalRepairState,
} from "../src/causal-repair.js";
import type { WorkerClient, WorkerRequest } from "../src/local-worker.js";
import { RunController } from "../src/runner.js";
import {
	CAUSAL_REPAIR_PROFILE,
	PROTOCOL_VERSION,
	type AgentResult,
	type ArtifactPayload,
	type CausalRepairSummary,
	type DrawingViewManifest,
	type WorkerResponse,
} from "../src/types.js";
import { makePng } from "./png-fixture.js";

interface TestTool {
	description: string;
	parameters: { properties: Record<string, unknown> };
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

function validArtifact(stepSha: string, path = "output.step"): ArtifactPayload {
	return {
		status: "valid",
		path,
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

async function invoke(tool: TestTool, parameters: object): Promise<unknown> {
	return await tool.execute("call", parameters, undefined, undefined, {});
}

describe("counterfactual causal repair verification", () => {
	it("evaluates isolated candidates in one frozen frame and preserves formal files", async () => {
		const workspace = await mkdtemp(join(tmpdir(), "causal-repair-"));
		workspaces.push(workspace);
		const source = "radius = 10\nheight = 20\n";
		const step = "formal step\n";
		await writeFile(join(workspace, "solution.py"), source);
		await writeFile(join(workspace, "output.step"), step);
		await Promise.all([
			writePng(workspace, "locked/front-crop.png", 1),
			writePng(workspace, "locked/front-render.png", 2),
			writePng(workspace, "locked/front-aligned.png", 3),
			writePng(workspace, "locked/front-comparison.png", 4),
		]);
		const sourceSha = digest(source);
		const stepSha = digest(step);
		const transform = {
			scale_px_per_unit: 1,
			rotation_deg: 0,
			center_x_px: 100,
			center_y_px: 100,
			model_center_x: 0,
			model_center_y: 0,
			reflected: false,
		};
		const manifest: DrawingViewManifest = {
			version: 3,
			scorer_profile: "capacity_transport_v1",
			input_sha256: "a".repeat(64),
			source_sha256: sourceSha,
			step_sha256: stepSha,
			regions: [{
				region_id: "front",
				page: 0,
				roi: [0, 0, 1000, 1000],
				crop_path: "locked/front-crop.png",
				view_type: "orthographic",
				status: "matched",
				selected: {
					candidate_id: "front-camera",
					render_path: "locked/front-render.png",
					comparison_path: "locked/front-comparison.png",
					aligned_render_path: "locked/front-aligned.png",
					view_spec: { name: "front-camera", direction: [0, -1, 0], up: [0, 0, 1] },
					score: 0.5,
					raw_score: 0.5,
					eligible: true,
					failed_gates: [],
					scoring_mode: "capacity_transport",
					capacity_transport: {} as never,
					transform,
					alignment: {
						objective: 0.5,
						mode: "capacity_transport",
						scale_at_bound: false,
						rotation_at_limit: false,
						translation_at_limit: false,
					},
				},
				alternatives: [],
				margin: null,
				confidence: "high",
				note: "front",
			}],
		};
		const result = {
			last_executed_source_sha256: sourceSha,
			last_executed_step_sha256: stepSha,
		} as AgentResult;
		const operations: string[] = [];
		const worker: WorkerClient = {
			async request<T extends WorkerResponse>(_workspace: string, request: WorkerRequest): Promise<T> {
				operations.push(request.operation);
				if (request.operation === "inspect") {
					return {
						protocol_version: PROTOCOL_VERSION,
						operation: "inspect",
						artifact: validArtifact(stepSha),
					} as unknown as T;
				}
				if (request.operation === "execute_repair_candidate") {
					const candidateSource = await readFile(join(workspace, request.source_path as string));
					if (request.candidate_id === "broken") {
						return {
							protocol_version: PROTOCOL_VERSION,
							operation: "execute_repair_candidate",
							candidate_id: request.candidate_id,
							source_path: request.source_path,
							source_sha256: digest(candidateSource),
							formal_source_sha256: sourceSha,
							formal_step_sha256: stepSha,
							formal_unchanged: true,
							execution: {
								success: false,
								exit_code: 1,
								timed_out: false,
								stdout: "",
								stderr: "NameError: broken candidate",
								files_produced: {},
								duration_s: 0.1,
							},
							artifact: {
								status: "missing",
								path: null,
								sha256: null,
								validation: null,
								validation_error: null,
								render_path: null,
								render_error: null,
							},
						} as unknown as T;
					}
					const outputPath = join(request.output_dir as string, "output.step");
					await mkdir(dirname(join(workspace, outputPath)), { recursive: true });
					await writeFile(join(workspace, outputPath), `step-${String(request.candidate_id)}\n`);
					const candidateStepSha = digest(await readFile(join(workspace, outputPath)));
					return {
						protocol_version: PROTOCOL_VERSION,
						operation: "execute_repair_candidate",
						candidate_id: request.candidate_id,
						source_path: request.source_path,
						source_sha256: digest(candidateSource),
						formal_source_sha256: sourceSha,
						formal_step_sha256: stepSha,
						formal_unchanged: true,
						execution: {
							success: true,
							exit_code: 0,
							timed_out: false,
							stdout: "",
							stderr: "",
							files_produced: { "output.step": 10 },
							duration_s: 0.1,
						},
						artifact: validArtifact(candidateStepSha, outputPath),
					} as unknown as T;
				}
				if (request.operation === "render_fixed_projection") {
					const renderPath = join(request.output_dir as string, "aligned-render.png");
					const marker = String(request.step_path).includes("/no_effect/") ? 3 : operations.length;
					const renderSha = await writePng(workspace, renderPath, marker);
					return {
						protocol_version: PROTOCOL_VERSION,
						operation: "render_fixed_projection",
						status: "ok",
						region_id: request.region_id,
						step_path: request.step_path,
						step_sha256: digest(await readFile(join(workspace, request.step_path as string))),
						view_spec: request.view_spec,
						transform: request.transform,
						render_path: renderPath,
						width: 320,
						height: 240,
						mime_type: "image/png",
						sha256: renderSha,
					} as T;
				}
				if (request.operation === "compose_repair_comparison") {
					const sheetPath = join(request.output_dir as string, "comparison.png");
					const sheetSha = await writePng(workspace, sheetPath, 9);
					return {
						protocol_version: PROTOCOL_VERSION,
						operation: "compose_repair_comparison",
						status: "ok",
						sheet_path: sheetPath,
						width: 320,
						height: 240,
						mime_type: "image/png",
						sha256: sheetSha,
					} as T;
				}
				throw new Error(`Unexpected worker operation ${request.operation}`);
			},
		};
		const state = await restoreCausalRepairState({
			workspace,
			fixtureId: "102",
			profile: CAUSAL_REPAIR_PROFILE,
			lockedViewManifestSha256: "b".repeat(64),
		});
		const summaries: CausalRepairSummary[] = [];
		const tool = createCausalRepairTool({
			workspace,
			state,
			lockedManifest: manifest,
			worker,
			controller: new RunController(),
			getResult: () => result,
			onUpdate: async (summary) => {
				summaries.push(summary);
			},
		}) as unknown as TestTool;

		expect(tool.description).toContain("当你已经发现当前 CAD 存在视觉缺陷");
		expect(tool.description).toContain("可用的 `region_id`：`front`");
		expect(tool.parameters.properties).toHaveProperty("region_id");
		expect(tool.parameters.properties).not.toHaveProperty("region_ids");
		const response = await invoke(tool, {
			region_id: "front",
			hypotheses: [
				{
					id: "wider",
					diagnosis: "radius is too small",
					expected_visual_effect: "outer silhouette becomes wider",
					edits: [{ old_text: "radius = 10", new_text: "radius = 12" }],
				},
				{
					id: "shorter",
					diagnosis: "height is too large",
					expected_visual_effect: "front silhouette becomes shorter",
					edits: [{ old_text: "height = 20", new_text: "height = 18" }],
				},
			],
		}) as { content: Array<{ type: string; text?: string }> };
		expect(response.content[0]?.text).toContain(
			"以下对比图从左到右分别展示了 `front` 区域的原始工程图、当前 CAD 二维投影",
		);
		expect(response.content[0]?.text).toContain("此次比较没有修改正式 solution.py 和 output.step");
		expect(response.content[0]?.text).toContain("候选执行状态：");

		expect(await readFile(join(workspace, "solution.py"), "utf8")).toBe(source);
		expect(await readFile(join(workspace, "output.step"), "utf8")).toBe(step);
		expect(operations).toEqual([
			"inspect",
			"execute_repair_candidate",
			"execute_repair_candidate",
			"render_fixed_projection",
			"render_fixed_projection",
			"compose_repair_comparison",
		]);
		expect(causalRepairSummary(state)).toMatchObject({ evaluation_count: 1, candidate_count: 2 });
		expect(summaries).toHaveLength(1);
		const selected = state.evaluations[0]!.candidates[0]!;
		await recordCausalRepairFormalExecution({
			workspace,
			state,
			turn: 8,
			sourceSha256: selected.source_sha256,
			stepSha256: "c".repeat(64),
		});
		expect(state.formalizations[0]).toMatchObject({
			turn: 8,
			selection: "candidate",
			candidate_id: "wider",
		});

		const beforeNoEffect = operations.length;
		const noEffectResponse = await invoke(tool, {
			region_id: "front",
			hypotheses: [
				{
					id: "no_effect",
					diagnosis: "candidate has no visible effect",
					expected_visual_effect: "projection should change",
					edits: [{ old_text: "radius = 10", new_text: "radius = 11" }],
				},
				{
					id: "visible",
					diagnosis: "candidate changes the projection",
					expected_visual_effect: "projection changes",
					edits: [{ old_text: "height = 20", new_text: "height = 19" }],
				},
			],
		}) as { content: Array<{ type: string; text?: string }> };
		expect(noEffectResponse.content).toHaveLength(1);
		expect(noEffectResponse.content[0]?.type).toBe("text");
		expect(noEffectResponse.content[0]?.text).toContain("no_effect: Patch 执行成功");
		expect(noEffectResponse.content[0]?.text).toContain("没有产生可见的二维投影变化");
		expect(operations.slice(beforeNoEffect)).toEqual([
			"inspect",
			"execute_repair_candidate",
			"execute_repair_candidate",
			"render_fixed_projection",
			"render_fixed_projection",
		]);
		expect(state.evaluations[1]).toMatchObject({ feedback: "error_only" });
		expect(state.evaluations[1]).not.toHaveProperty("sheet_path");

		const beforeFailure = operations.length;
		const failureResponse = await invoke(tool, {
			region_id: "front",
			hypotheses: [
				{
					id: "broken",
					diagnosis: "candidate execution fails",
					expected_visual_effect: "no projection is available",
					edits: [{ old_text: "radius = 10", new_text: "radius = 13" }],
				},
				{
					id: "still_valid",
					diagnosis: "candidate remains executable",
					expected_visual_effect: "projection changes",
					edits: [{ old_text: "height = 20", new_text: "height = 17" }],
				},
			],
		}) as { content: Array<{ type: string; text?: string }> };
		expect(failureResponse.content).toHaveLength(1);
		expect(failureResponse.content[0]?.type).toBe("text");
		expect(failureResponse.content[0]?.text).toContain("broken: Patch 执行失败：NameError: broken candidate");
		expect(operations.slice(beforeFailure)).toEqual([
			"inspect",
			"execute_repair_candidate",
			"execute_repair_candidate",
		]);
		expect(state.evaluations[2]).toMatchObject({ feedback: "error_only" });
		expect(state.evaluations[2]).not.toHaveProperty("sheet_path");
	});
});
