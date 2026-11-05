import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { appendFile, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
	createSession: vi.fn(),
	openSession: vi.fn(),
	loadManifest: vi.fn(),
	createComparisonTools: vi.fn(),
	createRepairTool: vi.fn(),
}));

vi.mock("@earendil-works/pi-coding-agent", async () => {
	const actual = await vi.importActual<typeof import("@earendil-works/pi-coding-agent")>(
		"@earendil-works/pi-coding-agent",
	);
	return {
		...actual,
		createAgentSession: mocks.createSession,
		DefaultResourceLoader: class {
			async reload(): Promise<void> {}
		},
		getAgentDir: () => "/tmp/cadgenbench-optional-reflect-test-agent",
		SessionManager: { open: mocks.openSession },
		SettingsManager: { inMemory: () => ({}) },
	};
});

vi.mock("../src/main-view-reflect.js", async () => {
	const actual = await vi.importActual<typeof import("../src/main-view-reflect.js")>("../src/main-view-reflect.js");
	return {
		...actual,
		loadLockedMainViewManifest: mocks.loadManifest,
		createMainViewReflectTools: mocks.createComparisonTools,
	};
});

vi.mock("../src/repair-geometry.js", () => ({ createRepairGeometryTool: mocks.createRepairTool }));

import type { SessionEntry } from "@earendil-works/pi-coding-agent";
import type { WorkerClient, WorkerRequest } from "../src/local-worker.js";
import {
	mainViewReflectSummary,
	persistMainViewReflectState,
	type MainViewReflectState,
} from "../src/main-view-reflect.js";
import { runFixtureInContainer, RunController, type ResolvedModel } from "../src/runner.js";
import {
	PROTOCOL_VERSION,
	GEOMETRY_REPAIR_PROFILE,
	REPAIR_PREVIEW_PROFILE,
	SHAPE_FEEDBACK_PROFILE,
	VISUAL_REVIEW_PROFILE,
	type AgentJob,
	type AgentResult,
	type ArtifactPayload,
	type MainViewReflectSummary,
	type WorkerResponse,
	type SubagentRunUsage,
} from "../src/types.js";

interface TestTool {
	name: string;
	execute: (id: string, parameters: object, signal?: AbortSignal) => Promise<unknown>;
}

interface TestSessionManager {
	path: string;
	getEntries: () => SessionEntry[];
	append: (message: object) => Promise<void>;
}

interface SessionCreation {
	cwd: string;
	tools: string[];
	customTools: TestTool[];
	sessionManager: TestSessionManager;
}

interface PromptContext {
	workspace: string;
	callTool: (name: string, parameters?: object) => Promise<void>;
}

type ModelTurn = (context: PromptContext) => Promise<"stop" | "aborted" | "error">;

const workspaces: string[] = [];
const sha256 = (bytes: string | Buffer): string => createHash("sha256").update(bytes).digest("hex");
const now = "2026-09-06T00:00:00Z";

beforeEach(() => {
	vi.clearAllMocks();
	mocks.loadManifest.mockResolvedValue({ regions: [] });
	mocks.createRepairTool.mockImplementation(() => ({ name: "repair_geometry", execute: async () => ({ content: [] }) }));
	mocks.openSession.mockImplementation((path: string): TestSessionManager => {
		const entries = readFileSync(path, "utf8").trim().split("\n").filter(Boolean)
			.map((line) => JSON.parse(line) as SessionEntry);
		return {
			path,
			getEntries: () => entries,
			append: async (message) => {
				const entry = {
					type: "message", id: `entry-${entries.length}`, parentId: null, timestamp: now, message,
				} as SessionEntry;
				entries.push(entry);
				await appendFile(path, `${JSON.stringify(entry)}\n`);
			},
		};
	});
	mocks.createComparisonTools.mockImplementation((options: {
		workspace: string;
		state: MainViewReflectState;
		getResult: () => AgentResult;
		onUpdate: (summary: MainViewReflectSummary) => Promise<void>;
	}): TestTool[] => [{
		name: "inspect_comparison",
		execute: async (toolCallId) => {
			const result = options.getResult();
			const stepSha = sha256(await readFile(join(options.workspace, "output.step")));
			const sourceSha = sha256(await readFile(join(options.workspace, "solution.py")));
			expect(sourceSha).toBe(result.last_executed_source_sha256);
			expect(stepSha).toBe(result.last_executed_step_sha256);
			const inspectionId = `inspection-${options.state.inspectionAttempts.length + 1}`;
			options.state.inspections.set(inspectionId, {
				inspection_id: inspectionId, tool_call_id: toolCallId, snapshot_id: "snapshot-1", region_id: "front",
				source_sha256: sourceSha, step_sha256: stepSha, camera_sha256: "c".repeat(64),
				context_path: "comparison.png", context_sha256: "d".repeat(64), created_at: now,
			});
			options.state.inspectionAttempts.push({
				sequence: options.state.inspectionAttempts.length + 1, inspection_id: inspectionId,
				tool_call_id: toolCallId, region_id: "front", source_sha256: sourceSha, step_sha256: stepSha,
				cache_hit: false, created_at: now,
			});
			await persistMainViewReflectState(options.workspace, options.state);
			await options.onUpdate(mainViewReflectSummary(options.state));
			return {
				content: [{ type: "image", data: "cG5n", mimeType: "image/png" }],
				details: { inspection_id: inspectionId, step_sha256: stepSha },
			};
		},
	}]);
});

afterEach(async () => {
	await Promise.all(workspaces.splice(0).map((workspace) => rm(workspace, { recursive: true, force: true })));
});

async function setup(modelTurns: ModelTurn[], toolResults: Readonly<Record<string, unknown>> = {}) {
	const workspace = await mkdtemp(join(tmpdir(), "cadgenbench-optional-reflect-resume-"));
	workspaces.push(workspace);
	await mkdir(join(workspace, "sessions"));
	await mkdir(join(workspace, "turns", "0001"), { recursive: true });
	await writeFile(join(workspace, "sessions", "test.jsonl"), "");
	const baselineSource = "# Original formal source\n";
	const baselineStep = `STEP:${baselineSource}`;
	await writeFile(join(workspace, "solution.py"), baselineSource);
	await writeFile(join(workspace, "output.step"), baselineStep);
	const job: AgentJob = {
		protocol_version: PROTOCOL_VERSION, operation: "run_fixture", run_id: "resume-test", agent_mode: "base-only",
		fixture: { id: "146", description: "resume test", inputFiles: [], inputType: "png", taskType: "generation", sourceDir: "." },
		staged_paths: [], provider: "test", model: "test", thinking_level: "high", pi_version: "0.84.1",
		container_image: "test", container_image_digest: "sha256:test",
		continuation: {
			kind: "main_self_reflect_inspect", id: "optional-reflect-resume", prompt: "Inspect the final formal CAD and finish.",
			image_paths: [], baseline_source_sha256: sha256(baselineSource), baseline_step_sha256: sha256(baselineStep),
			locked_view_manifest_path: "drawing_feedback/locked.json", locked_view_manifest_sha256: "c".repeat(64),
		},
	};
	const artifact = async (): Promise<ArtifactPayload> => ({
		status: "valid", path: "output.step", sha256: sha256(await readFile(join(workspace, "output.step"))),
		validation: { is_valid: true, is_watertight: true, topology_errors: [], solid_count: 1, shell_count: 1, face_count: 6, volume: 1, bbox: {} },
		validation_error: null, render_path: null, render_error: null,
	});
	const baseline: AgentResult = {
		fixture_id: "146", provider: "test", model: "test", thinking_level: "high", pi_version: "0.84.1",
		container_image: "test", container_image_digest: "sha256:test", started_at: now, updated_at: now,
		duration_s: 0, last_turn: 1, status: "valid", agent_mode: "base-only", session_file: "sessions/test.jsonl",
		tokens: { input: 0, output: 0, cache_read: 0, cache_write: 0, total: 0 }, cost: 0, tool_calls: 0,
		last_executed_source_sha256: sha256(baselineSource), last_executed_step_sha256: sha256(baselineStep),
		artifact: await artifact(),
	};
	await writeFile(join(workspace, "agent_result.json"), JSON.stringify(baseline));
	const requests: WorkerRequest[] = [];
	const worker: WorkerClient = {
		async request<T extends WorkerResponse>(_workspace: string, request: WorkerRequest): Promise<T> {
			requests.push(request);
			const base = { protocol_version: PROTOCOL_VERSION, operation: request.operation };
			if (request.operation === "prepare_images") return { ...base, images: [] } as unknown as T;
			if (request.operation === "inspect") return { ...base, artifact: await artifact() } as unknown as T;
			if (request.operation === "finalize") return { ...base, finalized: true } as unknown as T;
			if (request.operation === "execute") {
				const source = await readFile(join(workspace, "solution.py"), "utf8");
				await writeFile(join(workspace, "output.step"), `STEP:${source}`);
				await mkdir(join(workspace, "turns", String(request.turn).padStart(4, "0")), { recursive: true });
				return {
					...base, turn: request.turn, source_sha256: sha256(source), artifact: await artifact(),
					execution: { success: true, exit_code: 0, timed_out: false, stdout: "", stderr: "", files_produced: {}, duration_s: 0 },
				} as unknown as T;
			}
			throw new Error(`Unexpected worker operation: ${request.operation}`);
		},
	};
	const promptCalls: string[] = [];
	mocks.createSession.mockImplementation(async (creation: SessionCreation) => {
		const manager = creation.sessionManager;
		return {
			session: {
				get messages() {
					return manager.getEntries().flatMap((entry) => entry.type === "message" ? [entry.message] : []);
				},
				getSessionStats: () => ({
					sessionFile: manager.path, cost: 0, toolCalls: 0,
					tokens: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
				}),
				prompt: async (prompt: string) => {
					promptCalls.push(prompt);
					const turn = modelTurns[promptCalls.length - 1];
					if (!turn) throw new Error("Unexpected extra model turn");
					const stopReason = await turn({
						workspace,
						callTool: async (name, parameters = {}) => {
							const tool = creation.customTools.find((candidate) => candidate.name === name);
							if (!tool) throw new Error(`Missing tool ${name}`);
							const callId = `call-${promptCalls.length}-${manager.getEntries().length}`;
							const override = toolResults[name];
							const response = name in toolResults
								? typeof override === "function" ? await override(parameters) : override
								: await tool.execute(callId, parameters);
							await manager.append({
								role: "toolResult", toolCallId: callId, toolName: name, isError: false, timestamp: Date.now(),
								...(response as object),
							});
						},
					});
					await manager.append({ role: "assistant", content: [{ type: "text", text: "Review finished." }], stopReason, timestamp: Date.now() });
				},
				abort: vi.fn(async () => undefined),
				dispose: vi.fn(),
			},
		};
	});
	return {
		workspace, job, requests, promptCalls,
		readResult: async () => JSON.parse(await readFile(join(workspace, "agent_result.json"), "utf8")) as AgentResult,
		run: () => runFixtureInContainer({
			workspace, job, worker, controller: new RunController(),
			resolvedModel: { runtime: {}, model: {} } as ResolvedModel,
			repositoryRoot: new URL("../../..", import.meta.url).pathname,
			integrationRoot: new URL("..", import.meta.url).pathname,
		}),
	};
}

describe("Optional Reflect runner interruption and resume", () => {
	it.each(["aborted", "error"] as const)("requires another model turn after %s resolves despite a successful final inspection", async (stopReason) => {
		const fixture = await setup([
			async ({ callTool }) => { await callTool("inspect_comparison"); return stopReason; },
			async () => "stop",
		]);
		await expect(fixture.run()).rejects.toThrow();
		const interrupted = await fixture.readResult();
		expect(interrupted.status).not.toBe("valid");
		expect(interrupted.main_view_reflection?.inspection_count).toBe(1);
		expect(interrupted.optional_reflect_turn?.status).toBe("running");
		expect(fixture.requests.filter(({ operation }) => operation === "finalize")).toHaveLength(0);

		await fixture.run();
		expect(fixture.promptCalls).toHaveLength(2);
		expect(mocks.createSession).toHaveBeenCalledTimes(2);
		const completed = await fixture.readResult();
		expect(completed.status).toBe("valid");
		expect(completed.optional_reflect_turn?.status).toBe("completed");
		expect(completed.main_view_reflection?.inspection_count).toBe(1);
		expect(fixture.requests.filter(({ operation }) => operation === "finalize")).toHaveLength(1);
	});

	it("resumes formal edits made before the first inspection instead of rechecking the original baseline", async () => {
		const fixture = await setup([
			async ({ workspace, callTool }) => {
				await writeFile(join(workspace, "solution.py"), "# Revised formal source\n");
				await callTool("execute_build123d");
				throw new Error("Connection lost before the first comparison");
			},
			async ({ callTool }) => { await callTool("inspect_comparison"); return "stop"; },
		]);
		await expect(fixture.run()).rejects.toThrow("Connection lost before the first comparison");
		const interrupted = await fixture.readResult();
		expect(interrupted.status).not.toBe("valid");
		expect(interrupted.main_view_reflection?.inspection_count).toBe(0);
		expect(interrupted.last_executed_source_sha256).not.toBe(fixture.job.continuation!.baseline_source_sha256);
		expect(interrupted.last_executed_step_sha256).not.toBe(fixture.job.continuation!.baseline_step_sha256);

		await fixture.run();
		expect(fixture.promptCalls).toHaveLength(2);
		const completed = await fixture.readResult();
		expect(completed.status).toBe("valid");
		expect(completed.optional_reflect_turn?.status).toBe("completed");
		expect(completed.main_view_reflection?.inspected_formal_states).toEqual([{
			source_sha256: completed.last_executed_source_sha256, step_sha256: completed.last_executed_step_sha256,
		}]);
		expect(fixture.requests.filter(({ operation }) => operation === "execute")).toHaveLength(1);
		expect(fixture.requests.filter(({ operation }) => operation === "finalize")).toHaveLength(1);
	});
});

describe("Optional Reflect shape feedback registration", () => {
	it("registers repair and adoption for Main, hides Preview, and records each child usage once across resume", async () => {
		mocks.createRepairTool.mockImplementation((options: { onUsage: (usage: SubagentRunUsage) => Promise<void> }) => ({
			name: "repair_geometry",
			execute: async () => {
				const usage: SubagentRunUsage = {
					role: "geometry_repair", accounting_id: "child-103-1", status: "success",
					started_at: now, finished_at: now, duration_s: 1,
					tokens: { input: 100, output: 20, cache_read: 0, cache_write: 0, total: 120 }, cost: 0.5, tool_calls: 3,
				};
				await options.onUsage(usage);
				await options.onUsage(usage);
				return { content: [{ type: "text", text: "No supported repair was found." }] };
			},
		}));
		const fixture = await setup([
			async ({ callTool }) => { await callTool("repair_geometry"); await callTool("inspect_comparison"); return "aborted"; },
			async () => "stop",
		]);
		fixture.job.continuation!.visual_review_profile = GEOMETRY_REPAIR_PROFILE;
		fixture.job.fixture.inputFiles = ["original-drawing.png"];
		fixture.job.staged_paths = ["inputs/00-original-drawing.png"];
		await expect(fixture.run()).rejects.toThrow();
		const first = await fixture.readResult();
		expect(first.subagent_runs).toHaveLength(1);
		expect(first.recorded_subagent_usage_ids).toEqual(["child-103-1"]);
		expect(first.subagent_usage?.tokens.total).toBe(120);
		expect(first.subagent_usage?.cost).toBe(0.5);
		await fixture.run();
		const completed = await fixture.readResult();
		expect(completed.status).toBe("valid");
		expect(completed.subagent_runs).toHaveLength(1);
		expect(completed.tokens.total).toBe(120);
		expect(completed.cost).toBe(0.5);
		expect(completed.tool_calls).toBe(3);
		const creation = mocks.createSession.mock.calls[0]![0] as SessionCreation;
		for (const names of [creation.tools, creation.customTools.map(({ name }) => name)]) {
			expect(names).toContain("repair_geometry");
			expect(names).toContain("adopt_shape_candidate");
			expect(names).toContain("inspect_comparison");
			expect(names).not.toContain("preview_shape_feedback");
		}
		expect(mocks.createRepairTool.mock.calls[0]![0]).toMatchObject({
			workspace: fixture.workspace, fixture: fixture.job.fixture, thinkingLevel: fixture.job.thinking_level,
			stagedPaths: ["inputs/00-original-drawing.png"],
			resolvedModel: { runtime: {}, model: {} },
		});
	});

	it("rejects repair state when resuming the historical shape-feedback profile", async () => {
		const fixture = await setup([]);
		fixture.job.continuation!.visual_review_profile = SHAPE_FEEDBACK_PROFILE;
		const result = await fixture.readResult();
		result.subagent_runs = [];
		await writeFile(join(fixture.workspace, "agent_result.json"), JSON.stringify(result));
		await expect(fixture.run()).rejects.toThrow("state is forbidden");
		expect(mocks.createSession).not.toHaveBeenCalled();
	});

	it("rejects non-repair child records before restoring a repair-profile session", async () => {
		const fixture = await setup([]);
		fixture.job.continuation!.visual_review_profile = GEOMETRY_REPAIR_PROFILE;
		const result = await fixture.readResult();
		result.subagent_runs = [{ role: "reflection", accounting_id: "foreign" } as SubagentRunUsage];
		await writeFile(join(fixture.workspace, "agent_result.json"), JSON.stringify(result));
		await expect(fixture.run()).rejects.toThrow("Only geometry_repair");
		expect(mocks.createSession).not.toHaveBeenCalled();
	});

	it("registers preview and adoption in both tool lists and still requires a formal comparison", async () => {
		const previewResult = {
			content: [{ type: "image", data: "cHJldmlldy1pbWFnZQ==", mimeType: "image/png" }],
			details: { preview_id: "preview-1", status: "ok" },
		};
		const fixture = await setup([
			async ({ callTool }) => { await callTool("preview_shape_feedback"); return "stop"; },
			async ({ callTool }) => {
				const pending = await fixture.readResult();
				expect(pending.status).toBe("running");
				expect(pending.main_view_reflection?.inspected_formal_states).toEqual([]);
				expect(fixture.requests.filter(({ operation }) => operation === "finalize")).toHaveLength(0);
				await callTool("inspect_comparison");
				return "stop";
			},
		], { preview_shape_feedback: previewResult });
		fixture.job.continuation!.visual_review_profile = SHAPE_FEEDBACK_PROFILE;

		await fixture.run();

		const creation = mocks.createSession.mock.calls[0]![0] as SessionCreation;
		for (const registered of [creation.tools, creation.customTools.map(({ name }) => name)]) {
			expect(registered).toContain("preview_shape_feedback");
			expect(registered).toContain("adopt_shape_candidate");
			expect(registered).not.toContain("visualize_code_hypothesis");
			expect(registered).not.toContain("compare_repair_hypotheses");
			expect(registered).not.toContain("preview_repair_candidates");
		}
		expect(fixture.promptCalls).toHaveLength(2);
		expect(fixture.promptCalls[1]).toContain("inspect_comparison");
		const completed = await fixture.readResult();
		expect(completed.status).toBe("valid");
		expect(completed.optional_reflect_turn?.status).toBe("completed");
		expect(completed.main_view_reflection?.inspection_count).toBe(1);
		expect(fixture.requests.filter(({ operation }) => operation === "finalize")).toHaveLength(1);
	});

	it.each([undefined, REPAIR_PREVIEW_PROFILE, VISUAL_REVIEW_PROFILE])("does not register shape feedback or adoption for profile %s", async (profile) => {
		const fixture = await setup([
			async ({ callTool }) => { await callTool("inspect_comparison"); return "stop"; },
		]);
		if (profile) fixture.job.continuation!.visual_review_profile = profile;

		await fixture.run();

		const creation = mocks.createSession.mock.calls[0]![0] as SessionCreation;
		expect(creation.tools).not.toContain("preview_shape_feedback");
		expect(creation.customTools.map(({ name }) => name)).not.toContain("preview_shape_feedback");
		expect(creation.tools).not.toContain("adopt_shape_candidate");
		expect(creation.customTools.map(({ name }) => name)).not.toContain("adopt_shape_candidate");
		expect((await fixture.readResult()).status).toBe("valid");
		expect(fixture.promptCalls).toHaveLength(1);
	});

	it("keeps adopted source pending until the model formally executes and inspects it", async () => {
		const adoptedSource = "# Source from the selected candidate\n";
		const adopt = vi.fn(async (parameters: object) => {
			expect(parameters).toEqual({ candidate_id: "wider" });
			await writeFile(join(fixture.workspace, "solution.py"), adoptedSource);
			return { content: [{ type: "text", text: "Candidate wider applied to solution.py." }], details: { candidate_id: "wider", adopted: true } };
		});
		const fixture = await setup([
			async ({ callTool }) => {
				await callTool("inspect_comparison");
				await callTool("adopt_shape_candidate", { candidate_id: "wider" });
				return "stop";
			},
			async ({ callTool }) => {
				const pending = await fixture.readResult();
				expect(pending.status).toBe("running");
				expect(pending.last_executed_source_sha256).toBe(fixture.job.continuation!.baseline_source_sha256);
				expect(sha256(await readFile(join(fixture.workspace, "output.step")))).toBe(fixture.job.continuation!.baseline_step_sha256);
				expect(fixture.requests.filter(({ operation }) => operation === "execute" || operation === "finalize")).toHaveLength(0);
				await callTool("execute_build123d");
				await callTool("inspect_comparison");
				return "stop";
			},
		], { adopt_shape_candidate: adopt });
		fixture.job.continuation!.visual_review_profile = SHAPE_FEEDBACK_PROFILE;

		await fixture.run();

		expect(adopt).toHaveBeenCalledOnce();
		expect(fixture.promptCalls).toHaveLength(2);
		const completed = await fixture.readResult();
		expect(completed.status).toBe("valid");
		expect(completed.last_executed_source_sha256).toBe(sha256(adoptedSource));
		expect(completed.main_view_reflection?.inspected_formal_states).toContainEqual({
			source_sha256: completed.last_executed_source_sha256, step_sha256: completed.last_executed_step_sha256,
		});
		expect(fixture.requests.filter(({ operation }) => operation === "execute")).toHaveLength(1);
		expect(fixture.requests.filter(({ operation }) => operation === "finalize")).toHaveLength(1);
	});
});
