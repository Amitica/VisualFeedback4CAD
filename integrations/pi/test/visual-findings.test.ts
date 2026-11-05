import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { resolveActiveCodeConstructionGroundings, type CodeConstructionState } from "../src/code-construction.js";
import {
	createUpdateVisualFindingsTool,
	injectPersistentVisualFindings,
	renderVisualFindings,
	restoreVisualFindingState,
	UPDATE_VISUAL_FINDINGS_PARAMETERS_SCHEMA,
	VISUAL_FINDINGS_CONTEXT_TYPE,
} from "../src/visual-findings.js";
import { CODE_CONSTRUCTION_PROFILE, VISUAL_FINDING_PROFILE } from "../src/types.js";

interface TestTool {
	name: string;
	label: string;
	description: string;
	execute: (...arguments_: unknown[]) => Promise<unknown>;
}

const workspaces: string[] = [];

afterEach(async () => {
	for (const workspace of workspaces.splice(0)) await rm(workspace, { recursive: true, force: true });
});

async function makeState() {
	const workspace = await mkdtemp(join(tmpdir(), "visual-findings-"));
	workspaces.push(workspace);
	const state = await restoreVisualFindingState({
		workspace,
		fixtureId: "103",
		profile: VISUAL_FINDING_PROFILE,
		lockedViewManifestSha256: "a".repeat(64),
	});
	return { workspace, state };
}

async function invoke(tool: TestTool, operations: unknown[]): Promise<unknown> {
	return await tool.execute("call", { operations }, undefined, undefined, {});
}

function addOperation(index: number, references: string[] = []) {
	return {
		operation: "add",
		target_geometry: `Target geometry ${index}`,
		current_geometry: `Current geometry ${index}`,
		difference: `Unresolved shape difference ${index}`,
		current_construction_refs: references,
	};
}

describe("persistent visual findings collection", () => {
	it("exposes explicit add, update, and delete operations", () => {
		expect(Object.keys(UPDATE_VISUAL_FINDINGS_PARAMETERS_SCHEMA.properties)).toEqual(["operations"]);
		const operationUnion = UPDATE_VISUAL_FINDINGS_PARAMETERS_SCHEMA.properties.operations.items;
		expect("anyOf" in operationUnion).toBe(true);
	});

	it("starts empty, has no five-finding template, and supports add, update, delete, and restore", async () => {
		const { workspace, state } = await makeState();
		const tool = createUpdateVisualFindingsTool({ workspace, state }) as unknown as TestTool;
		expect(tool.name).toBe("update_visual_findings");
		expect(tool.label).toBe("Manage visual findings");
		expect(tool.description).toContain("may initially be empty and has no fixed finding count");
		expect(tool.description).toContain("never recreate or clear it");
		expect(tool.description).toContain("References are not cleared automatically when their code changes");
		expect(tool.description).toContain("may add or remove references");
		expect(tool.description).toContain("explicit add, update, and delete operations");
		expect(tool.description).toContain("`target_geometry`");
		expect(tool.description).toContain("`current_geometry`");
		expect(tool.description).toContain("`difference`");
		expect(tool.description).toContain("Do not record similarities");
		expect(tool.description).toContain("Delete a finding when its entire difference is resolved");
		expect(state.findings).toEqual([]);

		const initialMessages = injectPersistentVisualFindings([
			{ role: "user", content: "original" },
		], state) as Array<{ customType?: string }>;
		expect(initialMessages.some(({ customType }) => customType === VISUAL_FINDINGS_CONTEXT_TYPE)).toBe(false);

		const firstResponse = await invoke(tool, Array.from({ length: 6 }, (_, index) => addOperation(index + 1))) as {
			content: Array<{ type: string; text: string }>;
			details: { added_finding_ids: string[]; active_finding_count: number };
		};
		expect(firstResponse.content[0]?.text).toContain("Visual findings revision 1 has been saved");
		expect(firstResponse.details.added_finding_ids).toEqual([
			"F001", "F002", "F003", "F004", "F005", "F006",
		]);
		expect(firstResponse.details.active_finding_count).toBe(6);
		expect(state.findings).toHaveLength(6);

		await invoke(tool, [
			{ operation: "update", finding_id: "F002", difference: "Revised unresolved position difference" },
			{ operation: "delete", finding_id: "F004" },
		]);
		expect(state.revision).toBe(2);
		expect(state.findings).toHaveLength(5);
		expect(state.findings.find(({ finding_id }) => finding_id === "F002")?.difference)
			.toBe("Revised unresolved position difference");
		expect(state.findings.some(({ finding_id }) => finding_id === "F004")).toBe(false);

		const stored = JSON.parse(await readFile(join(workspace, state.statePath), "utf8")) as {
			version: number;
			findings: unknown[];
			next_finding_index: number;
			history: unknown[];
		};
		expect(stored.version).toBe(2);
		expect(stored.findings).toHaveLength(5);
		expect(stored.next_finding_index).toBe(7);
		expect(stored.history).toHaveLength(2);

		const restored = await restoreVisualFindingState({
			workspace,
			fixtureId: "103",
			profile: VISUAL_FINDING_PROFILE,
			lockedViewManifestSha256: "a".repeat(64),
		});
		expect(restored.findings).toEqual(state.findings);
		expect(restored.nextFindingIndex).toBe(7);
		expect(restored.revision).toBe(2);
	});

	it("renders target, Current, and difference in order and injects only the latest collection", async () => {
		const { workspace, state } = await makeState();
		const tool = createUpdateVisualFindingsTool({
			workspace,
			state,
			getActiveConstructionRefs: () => ["C007"],
		}) as unknown as TestTool;
		await invoke(tool, [addOperation(1, ["C007"])]);
		const rendered = renderVisualFindings(state.findings);
		expect(rendered).toBe([
			"[F001]",
			"Target: Target geometry 1",
			"Current [C007]: Current geometry 1",
			"Difference: Unresolved shape difference 1",
		].join("\n"));

		const constructionGroundings = [{
			construction_ref: "C007",
			current_source_sha256: "b".repeat(64),
			current_line_start: 12,
			current_line_end: 14,
			code_block: "far_strap = make_strap()\npart = part + far_strap",
			code_block_sha256: "c".repeat(64),
			binding_status: "current" as const,
		}];
		const injected = injectPersistentVisualFindings([
			{ role: "user", content: "original" },
			{ role: "custom", customType: VISUAL_FINDINGS_CONTEXT_TYPE, content: "stale" },
		], state, constructionGroundings) as Array<{ customType?: string; content?: string }>;
		const visualMessages = injected.filter(({ customType }) => customType === VISUAL_FINDINGS_CONTEXT_TYPE);
		expect(visualMessages).toHaveLength(1);
		expect(visualMessages[0]?.content).toContain("revision 1");
		expect(visualMessages[0]?.content).toContain("Current [C007 current]: Current geometry 1");
		expect(visualMessages[0]?.content).toContain("[Automatically resolved Current-code grounding]");
		expect(visualMessages[0]?.content).toContain("[C007 code grounding]");
		expect(visualMessages[0]?.content).toContain("Current source: solution.py:12-14");
		expect(visualMessages[0]?.content).toContain("far_strap = make_strap()");
		expect(visualMessages[0]?.content).not.toContain("stale");
	});

	it("resolves active references to exact blocks at their current source lines", async () => {
		const { workspace } = await makeState();
		const codeBlock = "far_strap = make_strap()\npart = part + far_strap";
		await writeFile(join(workspace, "solution.py"), [
			"from build123d import Box",
			"part = Box(1, 1, 1)",
			codeBlock,
			"export(part)",
		].join("\n"), "utf8");
		const codeState: CodeConstructionState = {
			fixtureId: "103",
			profile: CODE_CONSTRUCTION_PROFILE,
			lockedViewManifestSha256: "a".repeat(64),
			auditPath: "drawing_feedback/code-ablation/audit.json",
			startedAt: "2026-08-31T00:00:00.000Z",
			constructions: [{
				construction_ref: "C001",
				created_at: "2026-08-31T00:00:00.000Z",
				formal_source_sha256: "d".repeat(64),
				formal_step_sha256: "e".repeat(64),
				code_block: codeBlock,
				code_block_sha256: "f".repeat(64),
				line_start: 40,
				line_end: 41,
			}],
			activeConstructionRefs: ["C001"],
			inspections: [],
		};
		const resolved = await resolveActiveCodeConstructionGroundings({
			workspace,
			state: codeState,
			constructionRefs: ["C001", "C001"],
		});
		expect(resolved).toHaveLength(1);
		expect(resolved[0]).toMatchObject({
			construction_ref: "C001",
			current_line_start: 3,
			current_line_end: 4,
			code_block: codeBlock,
			binding_status: "current",
		});
		expect(resolved[0]?.current_source_sha256).toMatch(/^[a-f0-9]{64}$/u);
		await expect(resolveActiveCodeConstructionGroundings({
			workspace,
			state: codeState,
			constructionRefs: ["C002"],
		})).rejects.toThrow("inactive construction C002");
	});

	it("keeps construction references structured and accepts only active new registrations", async () => {
		const { workspace, state } = await makeState();
		const tool = createUpdateVisualFindingsTool({
			workspace,
			state,
			getActiveConstructionRefs: () => ["C001"],
		}) as unknown as TestTool;
		await expect(invoke(tool, [{
			...addOperation(1),
			current_geometry: "Current construction C001 is displaced",
		}])).rejects.toThrow("must not embed construction references");
		await expect(invoke(tool, [addOperation(1, ["C002"])]))
			.rejects.toThrow("Construction reference C002 is not active");
		await invoke(tool, [addOperation(1, ["C001"])]);
		expect(state.findings[0]?.current_construction_refs).toEqual(["C001"]);
	});

	it("rejects unknown, empty, and no-op updates", async () => {
		const { workspace, state } = await makeState();
		const tool = createUpdateVisualFindingsTool({ workspace, state }) as unknown as TestTool;
		await expect(invoke(tool, [{ operation: "update", finding_id: "F001", difference: "x" }]))
			.rejects.toThrow("unknown finding_id F001");
		await invoke(tool, [addOperation(1)]);
		await expect(invoke(tool, [{ operation: "update", finding_id: "F001" }]))
			.rejects.toThrow("must change at least one field");
		await expect(invoke(tool, [{
			operation: "update",
			finding_id: "F001",
			difference: "Unresolved shape difference 1",
		}])).rejects.toThrow("does not change finding F001");
	});

	it("lets the Agent add and remove registered references while rejecting unknown ones", async () => {
		const { workspace, state } = await makeState();
		const tool = createUpdateVisualFindingsTool({
			workspace,
			state,
			getActiveConstructionRefs: () => ["C001", "C002"],
		}) as unknown as TestTool;
		await invoke(tool, [addOperation(1, ["C001"])]);
		await invoke(tool, [{
			operation: "update",
			finding_id: "F001",
			difference: "Revised unresolved shape difference",
		}]);
		expect(state.findings[0]?.current_construction_refs).toEqual(["C001"]);
		await invoke(tool, [{
			operation: "update",
			finding_id: "F001",
			current_construction_refs: ["C002"],
		}]);
		expect(state.findings[0]?.current_construction_refs).toEqual(["C002"]);
		await expect(invoke(tool, [{
			operation: "update",
			finding_id: "F001",
			current_construction_refs: ["C002", "C003"],
		}])).rejects.toThrow("Construction reference C003 is not active");
	});

	it("keeps a modified stable reference and tells the Agent to re-highlight it by ID", async () => {
		const { workspace, state } = await makeState();
		const tool = createUpdateVisualFindingsTool({
			workspace,
			state,
			getActiveConstructionRefs: () => ["C001"],
		}) as unknown as TestTool;
		await invoke(tool, [addOperation(1, ["C001"])]);
		const injected = injectPersistentVisualFindings([], state, [{
			construction_ref: "C001",
			current_source_sha256: "b".repeat(64),
			current_line_start: 10,
			current_line_end: 10,
			code_block: "part = part + revised_support",
			code_block_sha256: "d".repeat(64),
			binding_status: "modified",
		}]) as Array<{
			customType?: string;
			content?: string;
		}>;
		const visualMessages = injected.filter(({ customType }) => customType === VISUAL_FINDINGS_CONTEXT_TYPE);
		expect(visualMessages).toHaveLength(1);
		expect(visualMessages[0]?.content).toContain("Current [C001 modified]: Current geometry 1");
		expect(visualMessages[0]?.content).toContain("Pass its construction_ref directly to inspect_code_ablation");
		expect(visualMessages[0]?.content).toContain("part = part + revised_support");
		expect(state.findings[0]?.current_construction_refs).toEqual(["C001"]);
		expect(state.revision).toBe(1);
	});
});
