import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import type { DockerAgentRuntime } from "../src/docker-worker.js";
import {
	prepareMainViewReflectCase,
	treeSha256,
} from "../src/main-view-reflect-continuation-cli.js";
import { JOB_FILE, type AgentJob, type AgentResult } from "../src/types.js";
import { makePng } from "./png-fixture.js";

const roots: string[] = [];

afterEach(async () => {
	for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true });
});

function sha(bytes: Buffer | string): string {
	return createHash("sha256").update(bytes).digest("hex");
}

async function png(path: string, marker: number): Promise<void> {
	await mkdir(dirname(path), { recursive: true });
	await writeFile(path, makePng(200, 160, marker));
}

function candidate(id: string, suffix: string) {
	return {
		candidate_id: id,
		render_path: `drawing_feedback/locked/${suffix}-render.png`,
		comparison_path: `drawing_feedback/locked/${suffix}-comparison.png`,
		aligned_render_path: `drawing_feedback/locked/${suffix}-aligned.png`,
		view_spec: { name: id, direction: suffix === "front" ? [0, 0, 1] : [0, 0, -1], up: [0, 1, 0] },
		score: 0.5,
		raw_score: 0.5,
		eligible: true,
		failed_gates: [],
		scoring_mode: "capacity_transport",
		capacity_transport: { version: 1 },
		transform: {
			scale_px_per_unit: 1,
			rotation_deg: 0,
			center_x_px: 100,
			center_y_px: 80,
			model_center_x: 0,
			model_center_y: 0,
			reflected: false,
		},
		alignment: {
			objective: 0.5,
			mode: "capacity_transport",
			scale_at_bound: false,
			rotation_at_limit: false,
			translation_at_limit: false,
		},
	};
}

async function fixture() {
	const root = await mkdtemp(join(tmpdir(), "main-view-reflect-cli-"));
	roots.push(root);
	const source = join(root, "direct", "102");
	const viewRoot = join(root, "view");
	const view = join(viewRoot, "102");
	const runDir = join(root, "output", "run");
	await Promise.all([
		mkdir(join(source, "sessions"), { recursive: true }),
		mkdir(join(source, "turns", "0001"), { recursive: true }),
		mkdir(join(source, "inputs"), { recursive: true }),
		mkdir(view, { recursive: true }),
		mkdir(runDir, { recursive: true }),
	]);
	const sourceText = "from build123d import Box\n";
	const stepText = "ISO-10303-21;\n";
	const sourceSha = sha(sourceText);
	const stepSha = sha(stepText);
	await Promise.all([
		writeFile(join(source, "solution.py"), sourceText),
		writeFile(join(source, "output.step"), stepText),
		writeFile(join(source, "sessions", "session.jsonl"), "{}\n"),
		png(join(source, "turns", "0001", "iso.png"), 1),
		png(join(source, "inputs", "00.png"), 2),
		writeFile(join(view, "solution.py"), sourceText),
		writeFile(join(view, "output.step"), stepText),
	]);
	const job: AgentJob = {
		protocol_version: 5,
		operation: "run_fixture",
		run_id: "direct",
		agent_mode: "base-only",
		fixture: { id: "102", description: "fixture", inputFiles: ["input.png"], inputType: "drawing", taskType: "generation", sourceDir: "." },
		staged_paths: ["inputs/00.png"],
		provider: "ucloud",
		model: "gpt-5.6-sol",
		thinking_level: "high",
		pi_version: "0.84.1",
		container_image: "old",
		container_image_digest: "sha256:old",
	};
	const result = {
		fixture_id: "102",
		status: "valid",
		agent_mode: "base-only",
		provider: "ucloud",
		model: "gpt-5.6-sol",
		thinking_level: "high",
		pi_version: "0.84.1",
		container_image: "old",
		container_image_digest: "sha256:old",
		started_at: "2026-08-21T00:00:00.000Z",
		updated_at: "2026-08-21T00:00:01.000Z",
		duration_s: 1,
		session_file: "sessions/session.jsonl",
		last_turn: 1,
		artifact: null,
		last_executed_source_sha256: sourceSha,
		last_executed_step_sha256: stepSha,
		tokens: { input: 1, output: 1, cache_read: 0, cache_write: 0, total: 2 },
		cost: 0.01,
		tool_calls: 1,
	} as AgentResult;
	await Promise.all([
		writeFile(join(source, JOB_FILE), `${JSON.stringify(job)}\n`),
		writeFile(join(source, "agent_result.json"), `${JSON.stringify(result)}\n`),
	]);
	const manifestPath = "drawing_feedback/runs/locked/view_manifest.json";
	const manifest = {
		version: 3,
		scorer_profile: "capacity_transport_v1",
		input_sha256: "a".repeat(64),
		source_sha256: sourceSha,
		step_sha256: stepSha,
		regions: [{
			region_id: "front",
			page: 0,
			roi: [0, 0, 500, 500],
			crop_path: "drawing_feedback/locked/front-crop.png",
			view_type: "orthographic",
			status: "matched",
			selected: candidate("candidate-001", "front"),
			alternatives: [candidate("candidate-002", "alternate")],
			margin: 0.1,
			confidence: "high",
			note: "front",
		}],
	};
	await mkdir(dirname(join(view, manifestPath)), { recursive: true });
	await writeFile(join(view, manifestPath), `${JSON.stringify(manifest, null, 2)}\n`);
	for (const [index, path] of [
		manifest.regions[0]!.crop_path,
		manifest.regions[0]!.selected.render_path,
		manifest.regions[0]!.selected.aligned_render_path,
		manifest.regions[0]!.selected.comparison_path,
	].entries()) await png(join(view, path), 10 + index);
	await writeFile(join(view, "view_search_only_result.json"), `${JSON.stringify({
		status: "completed",
		fixture_id: "102",
		source_workspace: source,
		reflection_agent_started: false,
		agent_roles: ["segmentation", "view_search"],
		manifest_path: manifestPath,
		source_sha256: sourceSha,
		step_sha256: stepSha,
	}, null, 2)}\n`);
	return { root, source, viewRoot, view, runDir, sourceSha, stepSha, manifestPath };
}

describe("Main View Reflect continuation preparation", () => {
	it("resumes the Direct session, copies only locked View Search evidence, and leaves both baseline trees unchanged", async () => {
		const item = await fixture();
		const sourceBefore = await treeSha256(item.source);
		const viewBefore = await treeSha256(item.view);
		const prepared = await prepareMainViewReflectCase({
			sourceWorkspace: item.source,
			viewSearchRoot: item.viewRoot,
			runDir: item.runDir,
			runId: "main-view-test",
			image: "cadgenbench-pi-agent:0.5.0-main-view-reflect-v1",
			imageId: "sha256:new",
			runtime: {} as DockerAgentRuntime,
			modelSelection: { profile: "gpt-5.6-sol", provider: "ucloud", model: "gpt-5.6-sol" },
		});
		expect(prepared.job.continuation?.kind).toBe("main_view_reflect");
		expect(prepared.job.continuation?.locked_view_manifest_path).toBe(item.manifestPath);
		expect(prepared.job.continuation?.baseline_source_sha256).toBe(item.sourceSha);
		expect(prepared.job.continuation?.baseline_step_sha256).toBe(item.stepSha);
		expect(await treeSha256(item.source)).toBe(sourceBefore);
		expect(await treeSha256(item.view)).toBe(viewBefore);
		expect(await readFile(join(prepared.workspace, "sessions", "session.jsonl"), "utf8")).toBe("{}\n");
		expect(await readFile(join(prepared.workspace, item.manifestPath), "utf8")).toBe(
			await readFile(join(item.view, item.manifestPath), "utf8"),
		);
		const copiedFiles = (await readdir(join(prepared.workspace, "drawing_feedback", "runs", "locked"))).sort();
		expect(copiedFiles).toEqual(["view_manifest.json"]);
		await expect(readFile(join(prepared.workspace, "reflection_only_result.json"))).rejects.toThrow();
	});

	it("fails before copying when the View Search result claims a different STEP", async () => {
		const item = await fixture();
		const resultPath = join(item.view, "view_search_only_result.json");
		const result = JSON.parse(await readFile(resultPath, "utf8")) as { step_sha256: string };
		result.step_sha256 = "f".repeat(64);
		await writeFile(resultPath, `${JSON.stringify(result)}\n`);
		await expect(prepareMainViewReflectCase({
			sourceWorkspace: item.source,
			viewSearchRoot: item.viewRoot,
			runDir: item.runDir,
			runId: "main-view-test",
			image: "test",
			imageId: "sha256:test",
			runtime: {} as DockerAgentRuntime,
			modelSelection: { profile: "gpt-5.6-sol", provider: "ucloud", model: "gpt-5.6-sol" },
		})).rejects.toThrow("does not exactly match");
	});
});
