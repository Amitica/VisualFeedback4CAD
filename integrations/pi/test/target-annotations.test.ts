import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { ImageContent } from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it } from "vitest";

import {
	ANNOTATE_TARGET_PARAMETERS_SCHEMA,
	createTargetAnnotationTool,
	injectPersistentTargetAnnotations,
	renderTargetAnnotationContext,
	restoreTargetAnnotationState,
	TARGET_ANNOTATION_STATE_PATH,
	TARGET_ANNOTATIONS_CONTEXT_TYPE,
} from "../src/target-annotations.js";
import { TARGET_ANNOTATION_PROFILE } from "../src/types.js";

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

async function setup() {
	const workspace = await mkdtemp(join(tmpdir(), "target-annotations-"));
	workspaces.push(workspace);
	const state = await restoreTargetAnnotationState({
		workspace,
		fixtureId: "103",
		profile: TARGET_ANNOTATION_PROFILE,
		lockedViewManifestSha256: "a".repeat(64),
	});
	let latestRegionId: string | undefined = "p0_top_right_isometric";
	const renders: Array<{ regionId: string; names: string[] }> = [];
	const tool = createTargetAnnotationTool({
		workspace,
		state,
		getLatestInspectedRegionId: () => latestRegionId,
		renderComparison: async (regionId, annotations): Promise<ImageContent> => {
			renders.push({ regionId, names: annotations.map(({ name }) => name) });
			return { type: "image", data: "iVBORw0KGgo=", mimeType: "image/png" };
		},
	}) as unknown as TestTool;
	return {
		workspace,
		state,
		tool,
		renders,
		setLatestRegionId: (value: string | undefined) => {
			latestRegionId = value;
		},
	};
}

async function invoke(tool: TestTool, parameters: object): Promise<unknown> {
	return await tool.execute("call", parameters, undefined, undefined, {});
}

describe("persistent numeric Target annotations", () => {
	it("offers mutually exclusive Point and Box variants", () => {
		expect("anyOf" in ANNOTATE_TARGET_PARAMETERS_SCHEMA).toBe(true);
		const schema = JSON.stringify(ANNOTATE_TARGET_PARAMETERS_SCHEMA);
		expect(schema).toContain("actual [x, y] anchor");
		expect(schema).toContain("Do not move the anchor into blank space");
		expect(schema).toContain("complete but tight");
		expect(schema).toContain("short, unique, specific semantic name");
		expect(schema).toContain("position, adjacency, or connectivity");
		expect(schema).toContain("Use its name, never its numeric visual mark");
	});

	it("keeps independent semantics and stable marks across rename, movement, mode switch, delete, and restore", async () => {
		const { workspace, state, tool, renders, setLatestRegionId } = await setup();
		expect(tool.name).toBe("annotate_target");
		expect(tool.label).toBe("Annotate Target drawing");
		expect(tool.description).toContain("Add numbered marks to the Target engineering drawing on the left side");
		expect(tool.description).toContain("Use point for a hole, endpoint, local bend");
		expect(tool.description).toContain("the point must land on the actual feature");
		expect(tool.description).toContain("draws a short leader back to that position");
		expect(tool.description).toContain("Use box for a grille, bracket, frame");
		expect(tool.description).toContain("enclose its complete range");
		expect(tool.description).toContain("unique short name");
		expect(tool.description).toContain("description of only the Target structure");
		expect(tool.description).toContain("the affected mark's description");
		expect(tool.description).toContain("the number and name of every mark shown in that image");
		expect(tool.description).not.toContain("construction_ref");

		const boxResponse = await invoke(tool, {
			operation: "add",
			name: "rear tower lower bend",
			description: "The tall rear support turns through a rounded lower transition before meeting its short horizontal floor leg.",
			box: [120, 260, 360, 720],
		}) as {
			content: Array<{ type: string; text?: string }>;
			details: { mark: number; name: string; revision: number; affected: object; registry: object[] };
		};
		expect(boxResponse.content[0]?.text).toContain("Added Target mark [1] rear tower lower bend · Box.");
		expect(boxResponse.content[0]?.text).toContain("The tall rear support turns through a rounded lower transition");
		expect(boxResponse.content[0]?.text).toContain("Marks shown in this image:");
		expect(boxResponse.content[0]?.text).toContain("[1] rear tower lower bend · Box");
		expect(boxResponse.content[0]?.text).toContain("Inspect the returned image. Update an inaccurate mark by its semantic name.");
		expect(boxResponse.content[1]?.type).toBe("image");
		expect(boxResponse.details).toMatchObject({ mark: 1, name: "rear tower lower bend", revision: 1 });
		expect(boxResponse.details.affected).toMatchObject({
			mark: 1,
			name: "rear tower lower bend",
			mode: "Box",
			geometry: { kind: "box", box: [120, 260, 360, 720] },
		});
		expect(boxResponse.details.registry).toHaveLength(1);

		const pointResponse = await invoke(tool, {
			operation: "add",
			name: "upper mounting hole",
			description: "The circular through-hole at the rounded free end of the upper mounting arm.",
			point: [52, 290],
		}) as { content: Array<{ text?: string }>; details: { mark: number; revision: number } };
		expect(pointResponse.details).toMatchObject({ mark: 2, revision: 2 });
		expect(pointResponse.content[0]?.text).toContain("Added Target mark [2] upper mounting hole · Point.");
		expect(pointResponse.content[0]?.text).toContain("The circular through-hole at the rounded free end of the upper mounting arm.");
		expect(pointResponse.content[0]?.text).toContain("[1] rear tower lower bend · Box");
		expect(pointResponse.content[0]?.text).toContain("[2] upper mounting hole · Point");
		expect(pointResponse.content[0]?.text).not.toContain("The tall rear support turns through a rounded lower transition");
		expect(renders).toEqual([{
			regionId: "p0_top_right_isometric",
			names: ["rear tower lower bend"],
		}, {
			regionId: "p0_top_right_isometric",
			names: ["rear tower lower bend", "upper mounting hole"],
		}]);

		setLatestRegionId("side");
		const sideResponse = await invoke(tool, {
			operation: "add",
			name: "side-view datum hole",
			description: "The circular datum hole at the free end of the side-view mounting tab.",
			point: [850, 500],
		}) as { content: Array<{ text?: string }>; details: { mark: number; registry: object[] } };
		expect(sideResponse.details.mark).toBe(3);
		expect(sideResponse.details.registry).toHaveLength(3);
		expect(sideResponse.content[0]?.text).not.toContain("p0_top_right_isometric");
		expect(sideResponse.content[0]?.text).not.toContain("rear tower lower bend");
		expect(sideResponse.content[0]?.text).toContain("[3] side-view datum hole · Point");

		const context = renderTargetAnnotationContext(state);
		expect(context).toContain("Current Target mark registry — revision 3");
		expect(context).toContain("p0_top_right_isometric");
		expect(context).toContain("[1] rear tower lower bend · Box");
		expect(context).toContain("[2] upper mounting hole · Point");
		expect(context).toContain("side");
		expect(context).toContain("[3] side-view datum hole · Point");
		expect(context).toContain("rounded lower transition");
		expect(context).not.toContain("shown in the returned image");
		const injected = injectPersistentTargetAnnotations([
			{ role: "user", content: "original" },
			{ role: "custom", customType: TARGET_ANNOTATIONS_CONTEXT_TYPE, content: "stale" },
		], state) as Array<{ customType?: string; content?: string }>;
		const memories = injected.filter(({ customType }) => customType === TARGET_ANNOTATIONS_CONTEXT_TYPE);
		expect(memories).toHaveLength(1);
		expect(memories[0]?.content).toBe(context);
		expect(memories[0]?.content).not.toContain("stale");

		setLatestRegionId("p0_top_right_isometric");
		const updateResponse = await invoke(tool, {
			operation: "update",
			name: "rear tower lower bend",
			new_name: "rear tower lower transition",
			point: [165, 700],
		}) as { content: Array<{ text?: string }>; details: { mark: number; affected: object; registry: object[] } };
		expect(updateResponse.content[0]?.text).toContain("Updated Target mark [1] rear tower lower transition · Point. Its number is unchanged.");
		expect(updateResponse.content[0]?.text).toContain("The tall rear support turns through a rounded lower transition");
		expect(updateResponse.content[0]?.text).not.toContain("side-view datum hole");
		expect(updateResponse.details).toMatchObject({
			mark: 1,
			affected: { mark: 1, name: "rear tower lower transition", mode: "Point" },
		});
		expect(updateResponse.details.registry).toHaveLength(3);
		expect(state.annotations[0]).toMatchObject({
			mark: 1,
			name: "rear tower lower transition",
			geometry: { kind: "point", point: [165, 700] },
		});
		expect(state.annotations[1]).toMatchObject({ mark: 2, name: "upper mounting hole" });

		const stored = JSON.parse(await readFile(join(workspace, state.statePath), "utf8")) as {
			version: number;
			annotations: unknown[];
			next_mark: number;
			revision: number;
		};
		expect(stored).toMatchObject({ version: 2, next_mark: 4, revision: 4 });
		expect(stored.annotations).toHaveLength(3);
		const restored = await restoreTargetAnnotationState({
			workspace,
			fixtureId: "103",
			profile: TARGET_ANNOTATION_PROFILE,
			lockedViewManifestSha256: "a".repeat(64),
		});
		expect(restored.annotations).toEqual(state.annotations);
		expect(restored.nextMark).toBe(4);

		const deleteResponse = await invoke(tool, { operation: "delete", name: "upper mounting hole" }) as {
			content: Array<{ text?: string }>;
			details: { affected: object; registry: object[] };
		};
		expect(deleteResponse.content[0]?.text).toContain("Deleted Target mark [2] upper mounting hole.");
		expect(deleteResponse.content[0]?.text).not.toContain("[2] upper mounting hole · Point");
		expect(deleteResponse.details.affected).toMatchObject({ mark: 2, name: "upper mounting hole", mode: "Point" });
		expect(deleteResponse.details.registry).toHaveLength(2);
		const replacement = await invoke(tool, {
			operation: "add",
			name: "lower mounting hole",
			description: "The circular through-hole at the lower mounting arm.",
			point: [60, 760],
		}) as { details: { mark: number } };
		expect(replacement.details.mark).toBe(4);
		expect(state.annotations.map(({ mark }) => mark)).toEqual([1, 3, 4]);
		expect(state.nextMark).toBe(5);
	});

	it("reports when no Target marks remain in the returned image", async () => {
		const { tool } = await setup();
		await invoke(tool, {
			operation: "add",
			name: "single hole",
			description: "The isolated circular hole at the free end of the upper mounting arm.",
			point: [100, 100],
		});
		const response = await invoke(tool, { operation: "delete", name: "single hole" }) as {
			content: Array<{ text?: string }>;
			details: { annotation_count: number; registry: object[] };
		};
		expect(response.content[0]?.text).toBe("Deleted Target mark [1] single hole.\n\nNo Target marks remain in this image.");
		expect(response.details).toMatchObject({ annotation_count: 0, registry: [] });
	});

	it("migrates legacy Box-only state in original order", async () => {
		const workspace = await mkdtemp(join(tmpdir(), "target-annotations-legacy-"));
		workspaces.push(workspace);
		const statePath = join(workspace, TARGET_ANNOTATION_STATE_PATH);
		await mkdir(join(statePath, ".."), { recursive: true });
		await writeFile(statePath, JSON.stringify({
			version: 1,
			profile: TARGET_ANNOTATION_PROFILE,
			fixture_id: "103",
			locked_view_manifest_sha256: "a".repeat(64),
			annotations: [{
				name: "first legacy box",
				description: "First legacy Target structure.",
				region_id: "front",
				box: [100, 100, 300, 300],
				created_at: "2026-01-01T00:00:00.000Z",
				updated_at: "2026-01-01T00:00:00.000Z",
			}, {
				name: "second legacy box",
				description: "Second legacy Target structure.",
				region_id: "front",
				box: [400, 400, 700, 700],
				created_at: "2026-01-01T00:00:00.000Z",
				updated_at: "2026-01-01T00:00:00.000Z",
			}],
			revision: 4,
			started_at: "2026-01-01T00:00:00.000Z",
			updated_at: "2026-01-01T00:00:00.000Z",
		}), "utf8");
		const restored = await restoreTargetAnnotationState({
			workspace,
			fixtureId: "103",
			profile: TARGET_ANNOTATION_PROFILE,
			lockedViewManifestSha256: "a".repeat(64),
		});
		expect(restored.annotations).toMatchObject([
			{ mark: 1, name: "first legacy box", geometry: { kind: "box", box: [100, 100, 300, 300] } },
			{ mark: 2, name: "second legacy box", geometry: { kind: "box", box: [400, 400, 700, 700] } },
		]);
		expect(restored.nextMark).toBe(3);
		const migrated = JSON.parse(await readFile(statePath, "utf8")) as { version: number; next_mark: number };
		expect(migrated).toMatchObject({ version: 2, next_mark: 3 });
	});

	it("requires a current inspected comparison and rejects duplicate, ambiguous, missing, or no-op geometry", async () => {
		const { tool, setLatestRegionId } = await setup();
		setLatestRegionId(undefined);
		await expect(invoke(tool, {
			operation: "add",
			name: "tower",
			description: "Target tower.",
			box: [100, 100, 200, 300],
		})).rejects.toThrow("Call inspect_comparison");
		setLatestRegionId("p0_top_right_isometric");
		await invoke(tool, {
			operation: "add",
			name: "Tower",
			description: "Target tower.",
			box: [100, 100, 200, 300],
		});
		await expect(invoke(tool, {
			operation: "add",
			name: "tower",
			description: "Duplicate target tower.",
			box: [300, 100, 400, 300],
		})).rejects.toThrow("already exists");
		await expect(invoke(tool, {
			operation: "add",
			name: "ambiguous geometry",
			description: "Invalid Target geometry.",
			point: [100, 100],
			box: [100, 100, 200, 200],
		})).rejects.toThrow("point or box, never both");
		await expect(invoke(tool, {
			operation: "add",
			name: "missing geometry",
			description: "Invalid Target geometry.",
		})).rejects.toThrow("requires point or box");
		await expect(invoke(tool, {
			operation: "update",
			name: "Tower",
			box: [100, 100, 200, 300],
		})).rejects.toThrow("does not change");
	});
});
