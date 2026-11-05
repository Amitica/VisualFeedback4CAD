import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import {
	createReflectionRefinementState,
	createReflectionRefinementTools,
	persistReflectionRefinementState,
	promptFindings,
	restoreReflectionRefinementState,
} from "../src/drawing-reflection-refinement.js";
import { alignmentNotice } from "../src/drawing-reflection.js";
import type { WorkerClient, WorkerRequest } from "../src/local-worker.js";
import {
	REFLECTION_ACCEPTANCE_SYSTEM_PROMPT,
	reflectionAcceptancePrompt,
	REFLECTION_REFINEMENT_SYSTEM_PROMPT,
	reflectionContinuationPrompt,
	reflectionRefinementPrompt,
} from "../src/prompts.js";
import {
	PROTOCOL_VERSION,
	REFLECTION_AGENT_TOOLS,
	REFLECTION_REFINEMENT_AGENT_TOOLS,
	type DrawingReflectionReport,
	type DrawingViewManifest,
	type WorkerResponse,
} from "../src/types.js";
import { makePng } from "./png-fixture.js";

interface TestTool {
	name: string;
	execute: (...arguments_: unknown[]) => Promise<unknown>;
}

interface ToolResult<T> {
	details: T;
}

const workspaces: string[] = [];

afterEach(async () => {
	for (const workspace of workspaces.splice(0)) await rm(workspace, { recursive: true, force: true });
});

async function makeWorkspace(): Promise<string> {
	const workspace = await mkdtemp(join(tmpdir(), "reflection-refinement-"));
	workspaces.push(workspace);
	return workspace;
}

function sha256(bytes: Buffer): string {
	return createHash("sha256").update(bytes).digest("hex");
}

async function writePng(workspace: string, path: string, width = 400, height = 300, marker = 1): Promise<string> {
	const bytes = makePng(width, height, marker);
	await mkdir(dirname(join(workspace, path)), { recursive: true });
	await writeFile(join(workspace, path), bytes);
	return sha256(bytes);
}

function report(): DrawingReflectionReport {
	return {
		should_revise: true,
		overall_summary: "Initial report",
		findings: [
			{ region_id: "front", obs: "CAD hole is too small.", suggested_change: "Increase the hole diameter." },
			{ region_id: "side", obs: "CAD slot is absent.", suggested_change: "Add the side slot." },
		],
	};
}

function candidate(id: string, comparisonPath: string, alignedPath: string) {
	return {
		rank: 1,
		candidate_id: id,
		render_path: `${id}.png`,
		comparison_path: comparisonPath,
		aligned_render_path: alignedPath,
		view_spec: { name: id, direction: [0, 0, 1], up: [0, 1, 0] },
		score: 0.5,
		raw_score: 0.5,
		eligible: true,
		failed_gates: [],
		scoring_mode: "capacity_transport",
		capacity_transport: {},
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
			objective: 0.5,
			mode: "capacity_transport",
			scale_at_bound: false,
			rotation_at_limit: false,
			translation_at_limit: false,
		},
	};
}

function manifest(): DrawingViewManifest {
	return {
		version: 3,
		input_sha256: "a".repeat(64),
		source_sha256: "b".repeat(64),
		step_sha256: "c".repeat(64),
		regions: [
			{
				region_id: "front",
				page: 0,
				roi: [0, 0, 500, 500],
				crop_path: "images/front.png",
				view_type: "orthographic",
				status: "matched",
				selected: candidate("candidate-001", "images/front-comparison.png", "images/front-aligned.png"),
				alternatives: [],
				margin: 0.1,
				confidence: "high",
				note: "front",
				width: 400,
				height: 300,
				crop_sha256: "d".repeat(64),
			},
			{
				region_id: "side",
				page: 0,
				roi: [500, 0, 1000, 500],
				crop_path: "images/side.png",
				view_type: "orthographic",
				status: "matched",
				selected: candidate("candidate-002", "images/side-comparison.png", "images/side-aligned.png"),
				alternatives: [],
				margin: 0.2,
				confidence: "high",
				note: "side",
				width: 400,
				height: 300,
				crop_sha256: "e".repeat(64),
			},
		],
	} as unknown as DrawingViewManifest;
}

async function invoke<T>(tool: TestTool, parameters: object): Promise<ToolResult<T>> {
	return (await tool.execute("call", parameters, undefined, undefined, {})) as ToolResult<T>;
}

async function setup(phase: "refinement" | "acceptance" = "refinement") {
	const workspace = await makeWorkspace();
	for (const [index, path] of [
		"images/front.png",
		"images/front-aligned.png",
		"images/front-comparison.png",
		"images/side.png",
		"images/side-aligned.png",
		"images/side-comparison.png",
	].entries()) await writePng(workspace, path, 400, 300, 20 + index);
	const workerModes: string[] = [];
	const worker: WorkerClient = {
		async request<T extends WorkerResponse>(_workspace: string, request: WorkerRequest): Promise<T> {
			workerModes.push(request.mode as string);
			const output = request.output_dir as string;
			expect(output.startsWith("drawing_feedback/")).toBe(true);
			if (request.mode === "coordinate_overview") {
				const contextPath = join(output, "coordinate-overview.png");
				const contextHash = await writePng(workspace, contextPath, 300, 200, 70);
				return {
					protocol_version: PROTOCOL_VERSION,
					operation: "compose_reflection_inspection",
					status: "ok",
					region_id: request.region_id as string,
					mode: "coordinate_overview",
					width: 400,
					height: 300,
					context_path: contextPath,
					sha256: { context: contextHash },
				} as T;
			}
			expect(request.mode).toBe("dual_roi");
			const files = {
				reference: join(output, "reference-crop.png"),
				cad: join(output, "cad-crop.png"),
				context: join(output, "context.png"),
				comparison: join(output, "comparison.png"),
			};
			const hashes = {
				reference_crop: await writePng(workspace, files.reference, 100, 90, 71),
				cad_crop: await writePng(workspace, files.cad, 120, 80, 72),
				context: await writePng(workspace, files.context, 300, 200, 73),
				comparison: await writePng(workspace, files.comparison, 300, 400, 74),
			};
			return {
				protocol_version: PROTOCOL_VERSION,
				operation: "compose_reflection_inspection",
				status: "ok",
				region_id: request.region_id as string,
				mode: "dual_roi",
				width: 400,
				height: 300,
				reference_roi: request.reference_roi,
				cad_roi: request.cad_roi,
				reference_pixel_roi: [0, 0, 100, 90],
				cad_pixel_roi: [10, 10, 130, 90],
				reference_crop_path: files.reference,
				cad_crop_path: files.cad,
				context_path: files.context,
				comparison_path: files.comparison,
				sha256: hashes,
			} as T;
		},
	};
	const initial = report();
	const state = createReflectionRefinementState({
		fixtureId: "102",
		viewManifest: manifest(),
		initialReport: initial,
		initialReportSha256: createHash("sha256").update(JSON.stringify(initial)).digest("hex"),
		phase,
	});
	const tools = new Map(
		createReflectionRefinementTools({ workspace, state, worker }).map((tool) => [tool.name, tool as TestTool]),
	);
	return { workspace, state, tools, workerModes };
}

describe("post-submit Reflection refinement", () => {
	it("keeps the 0.0165 tools separate and exposes only the refinement tools", async () => {
		expect(REFLECTION_AGENT_TOOLS).toEqual(["inspect_region", "submit_reflection"]);
			expect(REFLECTION_REFINEMENT_AGENT_TOOLS).toEqual([
				"inspect_region_for_crop",
				"compare_region_crops",
				"manage_reflection_findings",
				"submit_refined_reflection",
		]);
		const { tools } = await setup();
		expect([...tools.keys()]).toEqual(REFLECTION_REFINEMENT_AGENT_TOOLS);
	});

	it("surfaces historical locked-camera replay gate failures to the acceptance Agent", () => {
		const region = manifest().regions[0]!;
		region.selected!.locked_replay_original_eligible = false;
		region.selected!.locked_replay_original_failed_gates = ["projection_outside_crop"];
		expect(alignmentNotice(region)).toContain("projection_outside_crop");
		expect(alignmentNotice(region)).toContain("Treat this view and all local geometry claims from it as uncertain");
	});

	it("requires one coordinate inspection, then allows repeated and multiple crops without reinspecting", async () => {
		const { state, tools, workerModes } = await setup();
		const compare = tools.get("compare_region_crops")!;
		const validParameters = {
			region_id: "front",
			reference_roi: { x_min: 100, y_min: 120, x_max: 400, y_max: 500 },
			cad_roi: { x_min: 300, y_min: 200, x_max: 700, y_max: 600 },
		};
		await expect(invoke(compare, validParameters)).rejects.toThrow("Call inspect_region_for_crop");
		const inspect = tools.get("inspect_region_for_crop")!;
		await invoke(inspect, { region_id: "front" });
		await invoke(inspect, { region_id: "front" });
		await expect(invoke(compare, {
			region_id: "front",
			reference_roi: { x_min: 0, y_min: 0, x_max: 49, y_max: 100 },
			cad_roi: { x_min: 0, y_min: 0, x_max: 100, y_max: 100 },
		})).rejects.toThrow("at least 50");
		const first = await invoke<{ crop_id: string }>(compare, validParameters);
		const second = await invoke<{ crop_id: string }>(compare, validParameters);
		await invoke(compare, {
			...validParameters,
			cad_roi: { x_min: 350, y_min: 220, x_max: 750, y_max: 620 },
		});
		expect(first.details.crop_id).toBe(second.details.crop_id);
		expect(workerModes.filter((mode) => mode === "coordinate_overview")).toHaveLength(1);
		expect(workerModes.filter((mode) => mode === "dual_roi")).toHaveLength(2);
		expect(state.cropAttempts.map(({ sequence, cache_hit }) => [sequence, cache_hit])).toEqual([
			[1, false],
			[2, true],
			[3, false],
		]);
	});

	it("adds, locally updates, deletes, and persists stable finding IDs", async () => {
		const { workspace, state, tools } = await setup();
		await invoke(tools.get("inspect_region_for_crop")!, { region_id: "front" });
		const comparison = await invoke<{ crop_id: string }>(tools.get("compare_region_crops")!, {
			region_id: "front",
			reference_roi: { x_min: 100, y_min: 100, x_max: 500, y_max: 500 },
			cad_roi: { x_min: 200, y_min: 100, x_max: 600, y_max: 500 },
		});
		const manage = tools.get("manage_reflection_findings")!;
		await expect(invoke(manage, {
			operation: "apply_changes",
			crop_id: comparison.details.crop_id,
			changes: [{ action: "add", obs: "CAD has an extra rim.", suggested_change: "Remove the extra rim." }],
		})).rejects.toThrow("operation=read");
		await invoke(manage, { operation: "read" });
		await invoke(manage, {
			operation: "apply_changes",
			crop_id: comparison.details.crop_id,
			changes: [
				{ action: "update", finding_id: "finding-001", field: "obs", old_text: "too small", new_text: "slightly too small" },
				{ action: "add", obs: "CAD has an extra rim.", suggested_change: "Remove the extra rim." },
			],
		});
		expect(promptFindings(state).map(({ finding_id }) => finding_id)).toEqual(["finding-001", "finding-002", "finding-003"]);
		await expect(invoke(manage, {
			operation: "apply_changes",
			crop_id: comparison.details.crop_id,
			changes: [{ action: "delete", finding_id: "finding-002" }],
		})).rejects.toThrow("belongs to side");
		await invoke(tools.get("inspect_region_for_crop")!, { region_id: "side" });
		const sideComparison = await invoke<{ crop_id: string }>(tools.get("compare_region_crops")!, {
			region_id: "side",
			reference_roi: { x_min: 100, y_min: 100, x_max: 500, y_max: 500 },
			cad_roi: { x_min: 100, y_min: 100, x_max: 500, y_max: 500 },
		});
		await invoke(manage, {
			operation: "apply_changes",
			crop_id: sideComparison.details.crop_id,
			changes: [{ action: "delete", finding_id: "finding-002" }],
		});
		await persistReflectionRefinementState(workspace, state);
		const restored = await restoreReflectionRefinementState({
			workspace,
			fixtureId: "102",
			viewManifest: state.viewManifest,
			initialReport: state.initialReport,
			initialReportSha256: state.initialReportSha256,
		});
		expect(restored.nextFindingSequence).toBe(4);
		expect(promptFindings(restored).map(({ finding_id }) => finding_id)).toEqual(["finding-001", "finding-003"]);
		expect(restored.findings.find(({ finding_id }) => finding_id === "finding-001")?.obs).toContain("slightly too small");
		expect(restored.coordinateInspections.has("front")).toBe(true);
		expect(restored.coordinateInspections.has("side")).toBe(true);
		expect(restored.cropAttempts).toHaveLength(2);
	});

	it("uses explicit read/apply/confirm operations and rejects exact duplicate additions", async () => {
		const { state, tools } = await setup();
		const manage = tools.get("manage_reflection_findings")!;
		const queried = await invoke<{ crop_id: null; findings: unknown[] }>(manage, { operation: "read" });
		expect(queried.details.crop_id).toBeNull();
		expect(queried.details.findings).toHaveLength(2);
		expect(state.changeHistory).toHaveLength(0);
		await expect(invoke(manage, {
			operation: "apply_changes",
			changes: [{ action: "delete", finding_id: "finding-001" }],
		})).rejects.toThrow("crop_id is required");
		await invoke(tools.get("inspect_region_for_crop")!, { region_id: "front" });
		const comparison = await invoke<{ crop_id: string }>(tools.get("compare_region_crops")!, {
			region_id: "front",
			reference_roi: { x_min: 100, y_min: 100, x_max: 500, y_max: 500 },
			cad_roi: { x_min: 100, y_min: 100, x_max: 500, y_max: 500 },
		});
		await invoke(manage, { operation: "read" });
		await expect(invoke(manage, {
			operation: "apply_changes",
			crop_id: comparison.details.crop_id,
			changes: [{
				action: "add",
				obs: " CAD HOLE   IS TOO SMALL. ",
				suggested_change: " increase the hole diameter. ",
			}],
		})).rejects.toThrow("duplicates existing finding-001");
		await invoke(manage, { operation: "confirm_no_change", crop_id: comparison.details.crop_id });
		expect(state.changeHistory.at(-1)?.operation).toBe("confirm_no_change");
	});

	it("requires one crop and submits a report without internal IDs", async () => {
		const { state, tools } = await setup();
		await expect(invoke(tools.get("submit_refined_reflection")!, {
			should_revise: true,
			overall_summary: "Still needs revision.",
		})).rejects.toThrow("at least once");
		await invoke(tools.get("inspect_region_for_crop")!, { region_id: "front" });
		const comparison = await invoke<{ crop_id: string }>(tools.get("compare_region_crops")!, {
			region_id: "front",
			reference_roi: { x_min: 100, y_min: 100, x_max: 500, y_max: 500 },
			cad_roi: { x_min: 100, y_min: 100, x_max: 500, y_max: 500 },
		});
		const manage = tools.get("manage_reflection_findings")!;
		await invoke(manage, { operation: "confirm_no_change", crop_id: comparison.details.crop_id });
		await invoke(manage, { operation: "read" });
		const submitted = await invoke<DrawingReflectionReport>(tools.get("submit_refined_reflection")!, {
			should_revise: true,
			overall_summary: "Crop confirmed the initial report.",
		});
		expect(submitted.details.findings).toEqual(state.initialReport.findings);
		expect(JSON.stringify(submitted.details)).not.toContain("finding_id");
	});

	it("requires a fresh read of the final findings after the last edit", async () => {
		const { tools } = await setup();
		await invoke(tools.get("inspect_region_for_crop")!, { region_id: "front" });
		const comparison = await invoke<{ crop_id: string }>(tools.get("compare_region_crops")!, {
			region_id: "front",
			reference_roi: { x_min: 100, y_min: 100, x_max: 500, y_max: 500 },
			cad_roi: { x_min: 100, y_min: 100, x_max: 500, y_max: 500 },
		});
		const manage = tools.get("manage_reflection_findings")!;
		await invoke(manage, {
			operation: "apply_changes",
			crop_id: comparison.details.crop_id,
			changes: [{
				action: "update",
				finding_id: "finding-001",
				field: "obs",
				old_text: "too small",
				new_text: "slightly too small",
			}],
		});
		const submit = tools.get("submit_refined_reflection")!;
		await expect(invoke(submit, {
			should_revise: true,
			overall_summary: "Final state.",
		})).rejects.toThrow("after the final edit");
		await invoke(manage, { operation: "read" });
		const submitted = await invoke<DrawingReflectionReport>(submit, {
			should_revise: true,
			overall_summary: "Final state.",
		});
		expect(submitted.details.findings[0]?.obs).toContain("slightly too small");
	});

	it("requires acceptance to inspect every current region and review every carried finding", async () => {
		const { state, tools } = await setup("acceptance");
		const inspect = tools.get("inspect_region_for_crop")!;
		const compare = tools.get("compare_region_crops")!;
		const manage = tools.get("manage_reflection_findings")!;
		const submit = tools.get("submit_refined_reflection")!;
		await invoke(inspect, { region_id: "front" });
		const frontCrop = await invoke<{ crop_id: string }>(compare, {
			region_id: "front",
			reference_roi: { x_min: 100, y_min: 100, x_max: 500, y_max: 500 },
			cad_roi: { x_min: 100, y_min: 100, x_max: 500, y_max: 500 },
		});
		await expect(invoke(manage, {
			operation: "confirm_no_change",
			crop_id: frontCrop.details.crop_id,
		})).rejects.toThrow("requires reviewed_finding_ids");
		await invoke(manage, {
			operation: "confirm_no_change",
			crop_id: frontCrop.details.crop_id,
			reviewed_finding_ids: ["finding-001"],
		});
		await invoke(manage, { operation: "read" });
		await expect(invoke(submit, {
			should_revise: true,
			overall_summary: "One finding remains.",
		})).rejects.toThrow("missing: side");

		await invoke(inspect, { region_id: "side" });
		const sideCrop = await invoke<{ crop_id: string }>(compare, {
			region_id: "side",
			reference_roi: { x_min: 100, y_min: 100, x_max: 500, y_max: 500 },
			cad_roi: { x_min: 100, y_min: 100, x_max: 500, y_max: 500 },
		});
		await invoke(manage, {
			operation: "apply_changes",
			crop_id: sideCrop.details.crop_id,
			changes: [{ action: "delete", finding_id: "finding-002" }],
		});
		await invoke(manage, { operation: "read" });
		await expect(invoke(submit, {
			should_revise: false,
			overall_summary: "Incorrect empty decision.",
		})).rejects.toThrow("exactly when one or more findings remain");
		const accepted = await invoke<DrawingReflectionReport>(submit, {
			should_revise: true,
			overall_summary: "The front finding remains; the side finding was resolved.",
		});
		expect(accepted.details.findings).toHaveLength(1);
		expect([...state.reviewedFindingIds].sort()).toEqual(["finding-001", "finding-002"]);
	});

	it("persists acceptance review coverage without changing carried finding IDs", async () => {
		const { workspace, state, tools } = await setup("acceptance");
		await invoke(tools.get("inspect_region_for_crop")!, { region_id: "front" });
		const crop = await invoke<{ crop_id: string }>(tools.get("compare_region_crops")!, {
			region_id: "front",
			reference_roi: { x_min: 100, y_min: 100, x_max: 500, y_max: 500 },
			cad_roi: { x_min: 100, y_min: 100, x_max: 500, y_max: 500 },
		});
		await invoke(tools.get("manage_reflection_findings")!, {
			operation: "apply_changes",
			crop_id: crop.details.crop_id,
			changes: [{
				action: "update",
				finding_id: "finding-001",
				field: "obs",
				old_text: "too small",
				new_text: "still too small",
			}],
		});
		await persistReflectionRefinementState(workspace, state);
		const restored = await restoreReflectionRefinementState({
			workspace,
			fixtureId: "102",
			viewManifest: state.viewManifest,
			initialReport: state.initialReport,
			initialReportSha256: state.initialReportSha256,
			phase: "acceptance",
		});
		expect(restored.requiredFindingIds).toEqual(new Set(["finding-001", "finding-002"]));
		expect(restored.reviewedFindingIds).toEqual(new Set(["finding-001"]));
		expect(promptFindings(restored)[0]?.obs).toContain("still too small");
	});

	it("does not alter the existing Continuation prompt contract", () => {
		const initial = report();
		const before = reflectionContinuationPrompt("102", initial);
		reflectionRefinementPrompt({
			fixtureId: "102",
			findings: initial.findings.map((finding, index) => ({ finding_id: `finding-00${index + 1}`, ...finding })),
		});
		expect(reflectionContinuationPrompt("102", initial)).toBe(before);
	});

	it("teaches the complete finding-to-crop-to-edit workflow without rewriting the report", () => {
		const prompt = reflectionRefinementPrompt({
			fixtureId: "102",
			findings: report().findings.map((finding, index) => ({ finding_id: `finding-00${index + 1}`, ...finding })),
		});
		expect(prompt).toContain("inspect_region_for_crop");
		expect(prompt).toContain("finding_id or finding_ids being tested");
		expect(prompt).toContain("Each panel independently uses 0-1000");
		expect(prompt).toContain("as many times as needed");
		expect(prompt).toContain("Do not write a replacement report");
		expect(prompt).toContain("First inspect the upper red boxes");
		expect(prompt).toContain("confirm, weaken, correct, or disprove");
		expect(prompt).toContain("Original-versus-CAD difference");
		expect(prompt).toContain("manage_reflection_findings({ operation: \"read\" })");
		expect(prompt).toContain("Before adding any finding");
		expect(prompt).toContain("no current finding already covers it");
		expect(prompt).toContain("Do not infer hidden 3D geometry");
		expect(prompt).toContain("Never use an empty changes array");
		expect(REFLECTION_REFINEMENT_SYSTEM_PROMPT).toContain("claim to test, not as a fact to defend");
		expect(REFLECTION_REFINEMENT_SYSTEM_PROMPT).toContain("visual similarity alone is insufficient");
		expect(REFLECTION_REFINEMENT_SYSTEM_PROMPT).toContain("Do not use a separately centered local crop alone");
		expect(REFLECTION_REFINEMENT_SYSTEM_PROMPT).toContain("the Original drawing shows X, while the CAD shows Y");
	});

	it("makes the third-stage prompt validate the updated CAD and edit every carried finding", () => {
		const prompt = reflectionAcceptancePrompt({
			fixtureId: "102",
			findings: report().findings.map((finding, index) => ({ finding_id: `finding-00${index + 1}`, ...finding })),
		});
		expect(prompt).toContain("new valid STEP");
		expect(prompt).toContain("once for every comparable region");
		expect(prompt).toContain("Review every carried finding");
		expect(prompt).toContain("Delete it if resolved");
		expect(prompt).toContain("confirm_no_change");
		expect(prompt).toContain("newly introduced problem");
		expect(REFLECTION_ACCEPTANCE_SYSTEM_PROMPT).toContain("CURRENT CAD after the Main Agent's changes");
		expect(REFLECTION_ACCEPTANCE_SYSTEM_PROMPT).toContain("A carried finding cannot remain unreviewed");
	});
});
