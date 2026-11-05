import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import {
	CODE_CONSTRUCTION_CONTEXT_TYPE,
	createCodeConstructionTool,
	injectPersistentCodeConstructionRegistry,
	rebindCodeConstructionsAfterEdit,
	reconcileCodeConstructionBindings,
	type CodeConstructionReferenceRecord,
	type CodeConstructionState,
} from "../src/code-construction.js";
import { CODE_CONSTRUCTION_PROFILE } from "../src/types.js";

const workspaces: string[] = [];

afterEach(async () => {
	for (const workspace of workspaces.splice(0)) await rm(workspace, { recursive: true, force: true });
});

function construction(reference: string, codeBlock: string, createdAt: string): CodeConstructionReferenceRecord {
	return {
		construction_ref: reference,
		created_at: createdAt,
		formal_source_sha256: reference === "C001" ? "a".repeat(64) : "b".repeat(64),
		formal_step_sha256: "c".repeat(64),
		code_block: codeBlock,
		code_block_sha256: reference === "C001" ? "d".repeat(64) : "e".repeat(64),
		line_start: reference === "C001" ? 2 : 3,
		line_end: reference === "C001" ? 2 : 3,
	};
}

describe("code-construction registration lifecycle", () => {
	it("gives concise mandatory guidance for selecting and interpreting an ablation", () => {
		const tool = createCodeConstructionTool({
			workspace: "/tmp/not-used",
			state: {
				fixtureId: "103",
				profile: CODE_CONSTRUCTION_PROFILE,
				lockedViewManifestSha256: "f".repeat(64),
				auditPath: "drawing_feedback/code-ablation/audit.json",
				startedAt: "2026-08-30T00:00:00.000Z",
				constructions: [],
				activeConstructionRefs: [],
				inspections: [],
			},
			lockedManifest: {
				regions: [{ region_id: "front", selected: {} }],
			} as never,
			worker: {} as never,
			controller: {} as never,
			getResult: () => ({}) as never,
		});

		expect(tool.description).toContain("for every important unresolved 2D difference");
		expect(tool.description).toContain("To register a construction for the first time");
		expect(tool.description).toContain("submit its `construction_ref` instead");
		expect(tool.description).toContain("do not copy the code again");
		expect(tool.description).toContain("exactly one of `code_block` or `construction_ref`");
		expect(tool.description).toContain("export a valid STEP");
		expect(tool.description).toContain("not target segmentation or proof of correspondence");
		expect(tool.description).toContain("surrounding geometry and connectivity");
		expect(tool.description).toContain("try another code block");
		expect(tool.description).toContain("never infer them from pixel ratios or distances");
		expect(tool.description).toContain("Available `region_id` values: `front`");
		expect(tool.description).not.toContain("reference remains active while");
	});

	it("rebinds an edited block to the same stable reference while retaining unrelated references", async () => {
		const workspace = await mkdtemp(join(tmpdir(), "code-construction-lifecycle-"));
		workspaces.push(workspace);
		await writeFile(join(workspace, "solution.py"), [
			"unrelated_setting = 2",
			"part = part + unchanged_support",
			"part = part + revised_foot",
			"",
		].join("\n"));
		const state: CodeConstructionState = {
			fixtureId: "103",
			profile: CODE_CONSTRUCTION_PROFILE,
			lockedViewManifestSha256: "f".repeat(64),
			auditPath: "drawing_feedback/code-ablation/audit.json",
			startedAt: "2026-08-30T00:00:00.000Z",
			constructions: [
				construction("C001", "part = part + unchanged_support", "2026-08-30T00:00:01.000Z"),
				construction("C002", "part = part + original_foot", "2026-08-30T00:00:02.000Z"),
			],
			activeConstructionRefs: ["C001", "C002"],
			inspections: [],
		};

		const rebound = await rebindCodeConstructionsAfterEdit({
			workspace,
			state,
			path: "solution.py",
			edits: [{
				oldText: "part = part + original_foot",
				newText: "part = part + revised_foot",
			}],
		});
		expect(rebound).toEqual(["C002"]);
		expect(state.activeConstructionRefs).toEqual(["C001", "C002"]);
		expect(state.constructions[0]).toMatchObject({
			construction_ref: "C001",
			code_block: "part = part + unchanged_support",
		});
		expect(state.constructions[1]).toMatchObject({
			construction_ref: "C002",
			code_block: "part = part + revised_foot",
			line_start: 3,
			line_end: 3,
			binding_status: "modified",
		});
		expect(await reconcileCodeConstructionBindings({ workspace, state })).toEqual([]);
		const audit = JSON.parse(await readFile(join(workspace, state.auditPath), "utf8")) as {
			constructions: Array<{ construction_ref: string; code_block: string; binding_status?: string }>;
			active_construction_refs: string[];
		};
		expect(audit.constructions).toHaveLength(2);
		expect(audit.constructions[1]).toMatchObject({
			construction_ref: "C002",
			code_block: "part = part + revised_foot",
			binding_status: "modified",
		});
		expect(audit.active_construction_refs).toEqual(["C001", "C002"]);
	});

	it("extracts only the modified registered block when an edit includes unchanged context", async () => {
		const workspace = await mkdtemp(join(tmpdir(), "code-construction-context-edit-"));
		workspaces.push(workspace);
		await writeFile(join(workspace, "solution.py"), [
			"before = keep()",
			"part = part + revised_rib",
			"after = keep()",
		].join("\n"));
		const state: CodeConstructionState = {
			fixtureId: "103",
			profile: CODE_CONSTRUCTION_PROFILE,
			lockedViewManifestSha256: "f".repeat(64),
			auditPath: "drawing_feedback/code-ablation/audit.json",
			startedAt: "2026-08-30T00:00:00.000Z",
			constructions: [
				construction("C001", "part = part + original_rib", "2026-08-30T00:00:01.000Z"),
			],
			activeConstructionRefs: ["C001"],
			inspections: [],
		};

		await rebindCodeConstructionsAfterEdit({
			workspace,
			state,
			path: "solution.py",
			edits: [{
				oldText: "before = keep()\npart = part + original_rib\nafter = keep()",
				newText: "before = keep()\npart = part + revised_rib\nafter = keep()",
			}],
		});

		expect(state.constructions[0]).toMatchObject({
			construction_ref: "C001",
			code_block: "part = part + revised_rib",
			line_start: 2,
			line_end: 2,
			binding_status: "modified",
		});
	});

	it("removes a retired registration and its paired historical tool messages from model context", () => {
		const state: CodeConstructionState = {
			fixtureId: "103",
			profile: CODE_CONSTRUCTION_PROFILE,
			lockedViewManifestSha256: "f".repeat(64),
			auditPath: "drawing_feedback/code-ablation/audit.json",
			startedAt: "2026-08-30T00:00:00.000Z",
			constructions: [
				construction("C001", "part = part + unchanged_support", "2026-08-30T00:00:01.000Z"),
				construction("C002", "part = part + original_foot", "2026-08-30T00:00:02.000Z"),
			],
			activeConstructionRefs: ["C001"],
			inspections: [],
		};
		const messages = [
			{ role: "user", content: "Continue the fixture." },
			{
				role: "assistant",
				content: [{
					type: "toolCall",
					id: "stale-call",
					name: "inspect_code_ablation",
					arguments: { code_block: "part = part + original_foot" },
				}],
			},
			{
				role: "toolResult",
				toolCallId: "stale-call",
				toolName: "inspect_code_ablation",
				content: [{ type: "text", text: "Construction reference C002 has been registered." }],
				details: { construction_ref: "C002" },
			},
			{
				role: "toolResult",
				toolCallId: "active-call",
				toolName: "inspect_code_ablation",
				content: [{ type: "text", text: "Construction reference C001 remains active." }],
				details: { construction_ref: "C001" },
			},
			{
				role: "custom",
				customType: CODE_CONSTRUCTION_CONTEXT_TYPE,
				content: "old registry containing C001 and C002",
			},
		];

		const injected = injectPersistentCodeConstructionRegistry(messages, state);
		const serialized = JSON.stringify(injected);
		expect(serialized).toContain("C001");
		expect(serialized).not.toContain("C002");
		expect(serialized).not.toContain("original_foot");
		expect(serialized).not.toContain("stale-call");
		expect(injected.filter((message) => (
			typeof message === "object" &&
			message !== null &&
			(message as { customType?: string }).customType === CODE_CONSTRUCTION_CONTEXT_TYPE
		))).toHaveLength(1);
	});
});
