import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import type { CodeConstructionState } from "../src/code-construction.js";
import type { WorkerClient, WorkerRequest } from "../src/local-worker.js";
import { RunController } from "../src/runner.js";
import {
	createVisualDiffTool,
	injectPersistentVisualDiffs,
	restoreVisualDiffState,
	VISUAL_DIFF_CONTEXT_TYPE,
	VISUAL_DIFF_PARAMETERS_SCHEMA,
} from "../src/visual-diffs.js";
import {
	CODE_CONSTRUCTION_PROFILE,
	VISUAL_DIFF_PROFILE,
	type AgentResult,
	type WorkerResponse,
} from "../src/types.js";
import { makePng } from "./png-fixture.js";

interface TestTool {
	name: string;
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

function result(sourceSha256: string, stepSha256: string): AgentResult {
	return {
		fixture_id: "103",
		status: "valid",
		agent_mode: "base-only",
		provider: "test",
		model: "test",
		thinking_level: "high",
		pi_version: "0.84.1",
		container_image: "test",
		container_image_digest: "sha256:test",
		started_at: "2026-08-31T00:00:00.000Z",
		updated_at: "2026-08-31T00:00:00.000Z",
		duration_s: 0,
		last_turn: 1,
		artifact: null,
		last_executed_source_sha256: sourceSha256,
		last_executed_step_sha256: stepSha256,
		tokens: { input: 0, output: 0, cache_read: 0, cache_write: 0, total: 0 },
		cost: 0,
		tool_calls: 0,
	};
}

async function fixture() {
	const workspace = await mkdtemp(join(tmpdir(), "visual-diffs-"));
	roots.push(workspace);
	const sourceSha256 = "a".repeat(64);
	const stepSha256 = "b".repeat(64);
	const cameraSha256 = "c".repeat(64);
	const image = makePng(200, 160, 7);
	for (const path of ["reference.png", "current.png", "ablated.png"]) {
		await writeFile(join(workspace, path), image);
	}
	const codeState: CodeConstructionState = {
		fixtureId: "103",
		profile: CODE_CONSTRUCTION_PROFILE,
		lockedViewManifestSha256: "e".repeat(64),
		auditPath: "drawing_feedback/code-ablation/audit.json",
		startedAt: "2026-08-31T00:00:00.000Z",
		constructions: [{
			construction_ref: "C001",
			created_at: "2026-08-31T00:00:00.000Z",
			formal_source_sha256: sourceSha256,
			formal_step_sha256: stepSha256,
			code_block: "part = part + feature",
			code_block_sha256: "f".repeat(64),
			line_start: 10,
			line_end: 10,
			binding_status: "current",
		}],
		activeConstructionRefs: ["C001"],
		inspections: [{
			sequence: 1,
			created_at: "2026-08-31T00:00:00.000Z",
			construction_ref: "C001",
			region_id: "iso",
			status: "highlighted",
			formal_source_sha256: sourceSha256,
			formal_step_sha256: stepSha256,
			code_block: "part = part + feature",
			code_block_sha256: "f".repeat(64),
			line_start: 10,
			line_end: 10,
			ablated_source_path: "submitted.py",
			ablated_source_sha256: "1".repeat(64),
			ablation_replacement: "pass",
			execution: {
				success: true,
				exit_code: 0,
				timed_out: false,
				stdout: "",
				stderr: "",
				files_produced: {},
				duration_s: 1,
			},
			artifact: {
				status: "valid",
				path: "ablated.step",
				sha256: "2".repeat(64),
				validation: null,
				validation_error: null,
				render_path: null,
				render_error: null,
			},
			camera_sha256: cameraSha256,
			reference_path: "reference.png",
			current_render_path: "current.png",
			ablated_render_path: "ablated.png",
			comparison_path: "current.png",
		}],
	};
	const state = await restoreVisualDiffState({
		workspace,
		fixtureId: "103",
		profile: VISUAL_DIFF_PROFILE,
		lockedViewManifestSha256: "e".repeat(64),
	});
	const worker: WorkerClient = {
		async request<T extends WorkerResponse>(_workspace: string, request: WorkerRequest): Promise<T> {
			if (request.operation !== "compose_visual_diff") throw new Error("unexpected operation");
			const outputDir = String(request.output_dir);
			const comparisonPath = join(outputDir, "visual-diff.png");
			const targetMaskPath = join(outputDir, "target-mask.png");
			const currentMaskPath = join(outputDir, "current-mask.png");
			await mkdir(dirname(join(workspace, comparisonPath)), { recursive: true });
			const comparison = makePng(300, 220, 9);
			const targetMask = makePng(200, 160, 10);
			const currentMask = makePng(200, 160, 11);
			await Promise.all([
				writeFile(join(workspace, comparisonPath), comparison),
				writeFile(join(workspace, targetMaskPath), targetMask),
				writeFile(join(workspace, currentMaskPath), currentMask),
			]);
			return {
				protocol_version: 5,
				operation: "compose_visual_diff",
				status: "ok",
				region_id: String(request.region_id),
				diff_id: String(request.diff_id),
				construction_ref: String(request.construction_ref),
				target_roi: request.target_roi as [number, number, number, number],
				target_pixel_roi: [20, 30, 80, 100],
				target_geometry_pixel_bbox: [22, 34, 78, 96],
				current_geometry_pixel_bbox: [120, 40, 180, 120],
				comparison_path: comparisonPath,
				target_mask_path: targetMaskPath,
				current_mask_path: currentMaskPath,
				metrics: {
					overlap_pixel_count: 40,
					target_only_pixel_count: 20,
					current_only_pixel_count: 30,
					iou: 40 / 90,
					centroid_delta_px: [100, 15],
				},
				width: 300,
				height: 220,
				mime_type: "image/png",
				sha256: {
					comparison: sha(comparison),
					target_mask: sha(targetMask),
					current_mask: sha(currentMask),
				},
			} as T;
		},
	};
	return { workspace, sourceSha256, stepSha256, codeState, state, worker };
}

async function invoke(tool: TestTool, params: unknown): Promise<unknown> {
	return await tool.execute("call", params, undefined, undefined, {});
}

describe("one-step persistent visual diffs", () => {
	it("accepts only the two selections and immediately returns one diff image", async () => {
		expect(Object.keys(VISUAL_DIFF_PARAMETERS_SCHEMA.properties)).toEqual([
			"region_id",
			"target_roi",
			"current_construction_ref",
		]);
		const item = await fixture();
		const agentResult = result(item.sourceSha256, item.stepSha256);
		const tool = createVisualDiffTool({
			workspace: item.workspace,
			state: item.state,
			codeState: item.codeState,
			worker: item.worker,
			controller: new RunController(),
			getResult: () => agentResult,
		}) as unknown as TestTool;
		expect(tool.name).toBe("visual_diff");
		expect(tool.description).toContain("No prose description, relation, tags, proposal, or confirmation");

		const first = await invoke(tool, {
			region_id: "iso",
			target_roi: { x_min: 100, y_min: 200, x_max: 450, y_max: 700 },
			current_construction_ref: "C001",
		}) as { content: Array<{ type: string }>; details: { diff: { diff_id: string } } };
		expect(first.content.map(({ type }) => type)).toEqual(["image"]);
		expect(first.details.diff.diff_id).toBe("D001");
		expect(item.state.diffs).toHaveLength(1);

		await invoke(tool, {
			region_id: "iso",
			target_roi: { x_min: 120, y_min: 220, x_max: 430, y_max: 680 },
			current_construction_ref: "C001",
		});
		expect(item.state.diffs).toHaveLength(1);
		expect(item.state.diffs[0]?.diff_id).toBe("D001");
		expect(item.state.diffs[0]?.target_roi).toEqual([120, 220, 430, 680]);
		expect(item.state.revision).toBe(2);
	});

	it("injects only fresh verified images and hides stale ones after construction edits", async () => {
		const item = await fixture();
		const agentResult = result(item.sourceSha256, item.stepSha256);
		const tool = createVisualDiffTool({
			workspace: item.workspace,
			state: item.state,
			codeState: item.codeState,
			worker: item.worker,
			controller: new RunController(),
			getResult: () => agentResult,
		}) as unknown as TestTool;
		await invoke(tool, {
			region_id: "iso",
			target_roi: { x_min: 100, y_min: 200, x_max: 450, y_max: 700 },
			current_construction_ref: "C001",
		});
		const fresh = await injectPersistentVisualDiffs({
			messages: [],
			workspace: item.workspace,
			state: item.state,
			codeState: item.codeState,
			result: agentResult,
		}) as unknown as Array<{ customType?: string; content?: Array<{ type: string; text?: string }> }>;
		const freshContext = fresh.find(({ customType }) => customType === VISUAL_DIFF_CONTEXT_TYPE);
		expect(freshContext?.content?.map(({ type }) => type)).toEqual(["text", "image"]);
		expect(freshContext?.content?.[0]?.text).toContain("freshness=fresh");
		expect(freshContext?.content?.[0]?.text).not.toContain("relation=");
		expect(freshContext?.content?.[0]?.text).not.toContain("delta=");

		item.codeState.constructions[0]!.binding_status = "modified";
		const stale = await injectPersistentVisualDiffs({
			messages: [],
			workspace: item.workspace,
			state: item.state,
			codeState: item.codeState,
			result: agentResult,
		}) as unknown as Array<{ customType?: string; content?: Array<{ type: string; text?: string }> }>;
		const staleContext = stale.find(({ customType }) => customType === VISUAL_DIFF_CONTEXT_TYPE);
		expect(staleContext?.content?.map(({ type }) => type)).toEqual(["text"]);
		expect(staleContext?.content?.[0]?.text).toContain("freshness=stale");
	});

	it("rejects a persisted comparison whose image hash changed", async () => {
		const item = await fixture();
		const agentResult = result(item.sourceSha256, item.stepSha256);
		const tool = createVisualDiffTool({
			workspace: item.workspace,
			state: item.state,
			codeState: item.codeState,
			worker: item.worker,
			controller: new RunController(),
			getResult: () => agentResult,
		}) as unknown as TestTool;
		await invoke(tool, {
			region_id: "iso",
			target_roi: { x_min: 100, y_min: 200, x_max: 450, y_max: 700 },
			current_construction_ref: "C001",
		});
		const path = item.state.diffs[0]!.comparison_path;
		await writeFile(join(item.workspace, path), makePng(300, 220, 12));

		await expect(injectPersistentVisualDiffs({
			messages: [],
			workspace: item.workspace,
			state: item.state,
			codeState: item.codeState,
			result: agentResult,
		})).rejects.toThrow("no longer matches its stored hashes");
		const stored = JSON.parse(await readFile(join(item.workspace, item.state.statePath), "utf8")) as {
			revision: number;
		};
		expect(stored.revision).toBe(1);
	});
});
