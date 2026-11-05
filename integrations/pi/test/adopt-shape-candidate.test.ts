import { createHash } from "node:crypto";
import * as fs from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";

import { generateUnifiedPatch } from "@earendil-works/pi-coding-agent";
import { Value } from "typebox/value";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ADOPT_SHAPE_CANDIDATE_PARAMETERS_SCHEMA, createAdoptShapeCandidateTool, verifyShapeCandidateEvidence } from "../src/adopt-shape-candidate.js";
import { RunController } from "../src/runner.js";
import { SHAPE_FEEDBACK_LATEST_PATH, type ShapeFeedbackAudit } from "../src/shape-feedback.js";
import type { DrawingViewManifest } from "../src/types.js";
import { applyPiStyleEdits } from "../src/visual-review.js";
import { makePng } from "./png-fixture.js";

vi.mock("node:fs/promises", async (importOriginal) => {
	const original = await importOriginal<typeof import("node:fs/promises")>();
	return { ...original, open: vi.fn(original.open), rename: vi.fn(original.rename) };
});

const roots: string[] = [];
afterEach(async () => {
	const original = await vi.importActual<typeof import("node:fs/promises")>("node:fs/promises");
	vi.mocked(fs.open).mockReset().mockImplementation(original.open);
	vi.mocked(fs.rename).mockReset().mockImplementation(original.rename);
	for (const root of roots.splice(0)) await fs.rm(root, { recursive: true, force: true });
});

function sha(value: string | Buffer): string { return createHash("sha256").update(value).digest("hex"); }
interface AdoptionResult {
	content: Array<{ type: string; text: string }>;
	details: { status: "applied" | "already_applied"; candidate_id: string; evaluation_id: string; source_sha256: string; patch: string };
}

async function fixture() {
	const workspace = await fs.mkdtemp(join(tmpdir(), "adopt-shape-"));
	roots.push(workspace);
	const source = "width = 10\nheight = 20\nexport_step(part, 'output.step')\n";
	const step = "formal STEP";
	await fs.writeFile(join(workspace, "solution.py"), source);
	await fs.writeFile(join(workspace, "output.step"), step);
	await fs.writeFile(join(workspace, "agent_result.json"), '{"formal_execution_current":true}');
	const manifest = { version: 4, regions: [] } as unknown as DrawingViewManifest;
	const controller = new RunController();
	const save = async (relativePath: string, data: string | Buffer) => {
		await fs.mkdir(dirname(join(workspace, relativePath)), { recursive: true });
		await fs.writeFile(join(workspace, relativePath), data);
	};
	const persist = async (audit: ShapeFeedbackAudit) => {
		await save(`drawing_feedback/shape-feedback/${audit.evaluation_id}/audit.json`, JSON.stringify(audit));
	};
	const createPreview = async (evaluationId = "0123456789abcdef-1234abcd", width = 12) => {
		const outputRoot = `drawing_feedback/shape-feedback/${evaluationId}`;
		const png = makePng(120, 100, 1);
		const card = `${outputRoot}/views/front/feedback/shape-feedback.png`;
		await save(card, png);
		const definitions = [
			{ id: "wide", edits: [{ oldText: "width = 10", newText: `width = ${width}` }, { oldText: "height = 20", newText: "height = 24" }] },
			{ id: "tall", edits: [{ oldText: "height = 20", newText: "height = 30" }] },
		];
		const audit = {
			version: 1, kind: "shape_feedback_preview", evaluation_id: evaluationId,
			created_at: "2026-09-07T00:00:00Z", updated_at: "2026-09-07T00:00:00Z", status: "completed",
			formal_source_sha256: sha(source), formal_step_sha256: sha(step), locked_view_manifest_sha256: sha(JSON.stringify(manifest)),
			requested_views: [{ region_id: "front", target_roi: [0, 0, 1000, 1000] }],
			candidates: [], frames: [], projections: [], feedback: [], topology: null, errors: [],
		} as ShapeFeedbackAudit;
		for (const definition of definitions) {
			const proposed = applyPiStyleEdits(source, definition.edits);
			const sourcePath = `${outputRoot}/candidates/${definition.id}/submitted-solution.py`;
			const stepPath = `${outputRoot}/candidates/${definition.id}/execution/output.step`;
			const renderPath = `${outputRoot}/views/front/candidates/${definition.id}/projection/aligned-render.png`;
			await save(sourcePath, proposed);
			await save(stepPath, `STEP ${proposed}`);
			await save(renderPath, png);
			audit.candidates.push({
				...definition, patch: generateUnifiedPatch("solution.py", source, proposed), source_path: sourcePath,
				source_sha256: sha(proposed), status: "valid",
				execution: {
					protocol_version: 5, operation: "execute_repair_candidate", candidate_id: `shape-${evaluationId.slice(0, 16)}-${definition.id}`,
					source_path: sourcePath, source_sha256: sha(proposed), formal_source_sha256: sha(source), formal_step_sha256: sha(step), formal_unchanged: true,
					execution: { success: true, timed_out: false, exit_code: 0, stdout: "", stderr: "", files_produced: {}, duration_s: 0 },
					artifact: { status: "valid", path: stepPath, sha256: sha(`STEP ${proposed}`), validation: { is_valid: true, is_watertight: true, topology_errors: [], solid_count: 1, shell_count: 1, face_count: 6, volume: 1, bbox: {} }, validation_error: null, render_path: null, render_error: null },
				},
			});
			audit.projections.push({ region_id: "front", candidate_id: definition.id, step_sha256: sha(`STEP ${proposed}`), render_path: renderPath, render_sha256: sha(png), camera_sha256: "0".repeat(64), transform_sha256: "1".repeat(64), projection_origin: [0, 0, 0], projection_anchor_step_sha256: sha(step) });
		}
		audit.feedback.push({
			protocol_version: 5, operation: "compose_shape_feedback", status: "ok", region_id: "front", card_path: card, sha256: sha(png), width: 120, height: 100,
			card_layout: "three_panel_row", card_count: definitions.length,
			cards: await Promise.all(definitions.map(async (candidate, index) => {
				const cardPath = index === 0 ? card : `${outputRoot}/views/front/feedback/candidates/${candidate.id}/shape-feedback.png`;
				const cardBytes = index === 0 ? png : makePng(120, 100, 2);
				await save(cardPath, cardBytes);
				return { candidate_id: candidate.id, card_path: cardPath, sha256: sha(cardBytes), width: 120, height: 100, mime_type: "image/png" as const,
					card_layout: "three_panel_row" as const, panel_count: 3,
					panel_titles: ["Target", "Current", "Preview"] };
			})),
			reference_sha256: sha(png), current_render_sha256: sha(png), target_roi: [0, 0, 1000, 1000],
			candidates: audit.candidates.map((candidate) => ({
				candidate_id: candidate.id, render_sha256: sha(png),
			})), files: {}, diagnostics: {},
		});
		await persist(audit);
		await save(SHAPE_FEEDBACK_LATEST_PATH, JSON.stringify({ version: 1, evaluation_id: evaluationId }));
		return audit;
	};
	const audit = await createPreview();
	const createTool = () => createAdoptShapeCandidateTool({ workspace, lockedManifest: manifest, controller });
	const invoke = (params: object = { candidate_id: "wide" }): Promise<AdoptionResult> => (createTool().execute as (...args: unknown[]) => Promise<AdoptionResult>)("adopt-call", params, undefined, undefined, {});
	return { workspace, source, step, manifest, controller, audit, save, persist, createPreview, invoke, createTool };
}

describe("adopt_shape_candidate", () => {
	it("accepts a safe candidate_id and optional exact evaluation_id in its schema", () => {
		expect(Value.Check(ADOPT_SHAPE_CANDIDATE_PARAMETERS_SCHEMA, { candidate_id: "candidate-1" })).toBe(true);
		expect(Value.Check(ADOPT_SHAPE_CANDIDATE_PARAMETERS_SCHEMA, { candidate_id: "candidate-1", evaluation_id: "0123456789abcdef-1234abcd" })).toBe(true);
		for (const params of [{}, { candidate_id: "../x" }, { candidate_id: "/x" }, { candidate_id: "a".repeat(65) }, { candidate_id: "x", edits: [] }]) expect(Value.Check(ADOPT_SHAPE_CANDIDATE_PARAMETERS_SCHEMA, params)).toBe(false);
	});

	it.each(["../0123456789abcdef-1234abcd", "0123456789abcdef-1234abcd/extra", "", "0123456789ABCDEF-1234abcd", null, 7])("rejects unsafe or invalid evaluation ID %s before adopting", async (evaluation_id) => {
		const f = await fixture();
		const params = { candidate_id: "wide", evaluation_id };
		expect(Value.Check(ADOPT_SHAPE_CANDIDATE_PARAMETERS_SCHEMA, params)).toBe(false);
		await expect(f.invoke(params)).rejects.toThrow("optionally evaluation_id");
		expect(await fs.readFile(join(f.workspace, "solution.py"), "utf8")).toBe(f.source);
	});

	it("adopts an earlier evaluation despite a newer preview using the same candidate ID", async () => {
		const f = await fixture();
		const newer = await f.createPreview("fedcba9876543210-ffffaaaa", 19);
		const result = await f.invoke({ candidate_id: "wide", evaluation_id: f.audit.evaluation_id });
		expect(result.details).toMatchObject({ status: "applied", evaluation_id: f.audit.evaluation_id });
		expect(await fs.readFile(join(f.workspace, "solution.py"), "utf8")).toContain("width = 12");
		expect(JSON.parse(await fs.readFile(join(f.workspace, SHAPE_FEEDBACK_LATEST_PATH), "utf8")).evaluation_id).toBe(newer.evaluation_id);
	});

	it("does not fall back to latest when an explicitly selected evaluation is missing", async () => {
		const f = await fixture();
		await expect(f.invoke({ candidate_id: "wide", evaluation_id: "fedcba9876543210-ffffaaaa" })).rejects.toThrow("Missing or unsafe file");
		expect(await fs.readFile(join(f.workspace, "solution.py"), "utf8")).toBe(f.source);
	});

	it("keeps candidate-only selection tied to the newest evaluation", async () => {
		const f = await fixture();
		const newer = await f.createPreview("fedcba9876543210-ffffaaaa", 19);
		const result = await f.invoke({ candidate_id: "wide" });
		expect(result.details.evaluation_id).toBe(newer.evaluation_id);
		expect(await fs.readFile(join(f.workspace, "solution.py"), "utf8")).toContain("width = 19");
	});

	it.each(["missing", "corrupt", "symlink"])("explicit selection does not require the latest pointer when it is %s", async (kind) => {
		const f = await fixture();
		const pointer = join(f.workspace, SHAPE_FEEDBACK_LATEST_PATH);
		await fs.unlink(pointer);
		if (kind === "corrupt") await fs.writeFile(pointer, "invalid pointer");
		if (kind === "symlink") await fs.symlink(join(f.workspace, "solution.py"), pointer);
		await expect(f.invoke({ candidate_id: "wide", evaluation_id: f.audit.evaluation_id })).resolves.toMatchObject({ details: { status: "applied" } });
	});

	it.each(["solution.py", "output.step"])("explicit selection rejects changed formal %s", async (path) => {
		const f = await fixture();
		await fs.writeFile(join(f.workspace, path), "changed externally");
		await expect(f.invoke({ candidate_id: "wide", evaluation_id: f.audit.evaluation_id })).rejects.toThrow("changed after the preview");
	});

	it.each([false, true])("rejects recalibrated views for explicit=%s", async (explicit) => {
		const f = await fixture();
		f.manifest.version = 5 as unknown as DrawingViewManifest["version"];
		await expect(f.invoke({ candidate_id: "wide", ...(explicit ? { evaluation_id: f.audit.evaluation_id } : {}) })).rejects.toThrow("locked drawing views changed");
	});

	it("explicit already-applied selection is idempotent only while the formal STEP is unchanged", async () => {
		const f = await fixture();
		const params = { candidate_id: "wide", evaluation_id: f.audit.evaluation_id };
		await f.invoke(params);
		await fs.unlink(join(f.workspace, SHAPE_FEEDBACK_LATEST_PATH));
		await expect(f.invoke(params)).resolves.toMatchObject({ details: { status: "already_applied" } });
		await fs.writeFile(join(f.workspace, "output.step"), "unrelated rebuilt STEP");
		await expect(f.invoke(params)).rejects.toThrow("output.step changed after the preview");
	});

	it.each([false, true])("a concurrent latest-pointer change only invalidates implicit selection (explicit=%s)", async (explicit) => {
		const f = await fixture();
		const original = await vi.importActual<typeof import("node:fs/promises")>("node:fs/promises");
		vi.mocked(fs.open).mockImplementation(async (path, flags, mode) => {
			const handle = await original.open(path, flags, mode);
			if (String(path).includes(".solution.py.adopt-")) {
				const sync = handle.sync.bind(handle);
				handle.sync = async () => {
					await sync();
					await original.writeFile(join(f.workspace, SHAPE_FEEDBACK_LATEST_PATH), JSON.stringify({ version: 1, evaluation_id: "fedcba9876543210-ffffaaaa" }));
				};
			}
			return handle;
		});
		const promise = f.invoke({ candidate_id: "wide", ...(explicit ? { evaluation_id: f.audit.evaluation_id } : {}) });
		if (explicit) await expect(promise).resolves.toMatchObject({ details: { status: "applied" } });
		else await expect(promise).rejects.toThrow("preview changed during adoption");
	});

	it("explicit adoption still rejects a concurrent edit to its own audit", async () => {
		const f = await fixture();
		const original = await vi.importActual<typeof import("node:fs/promises")>("node:fs/promises");
		vi.mocked(fs.open).mockImplementation(async (path, flags, mode) => {
			const handle = await original.open(path, flags, mode);
			if (String(path).includes(".solution.py.adopt-")) {
				const sync = handle.sync.bind(handle);
				handle.sync = async () => {
					await sync();
					await original.writeFile(join(f.workspace, `drawing_feedback/shape-feedback/${f.audit.evaluation_id}/audit.json`), "changed audit");
				};
			}
			return handle;
		});
		await expect(f.invoke({ candidate_id: "wide", evaluation_id: f.audit.evaluation_id })).rejects.toThrow("preview changed during adoption");
		expect(await fs.readFile(join(f.workspace, "solution.py"), "utf8")).toBe(f.source);
		expect((await fs.readdir(f.workspace)).some((name) => name.startsWith(".solution.py.adopt-"))).toBe(false);
	});

	it("adopts all selected edits after resume from disk, preserving siblings, STEP and result state", async () => {
		const f = await fixture();
		const sibling = await fs.readFile(join(f.workspace, f.audit.candidates[1]!.source_path));
		const resultState = await fs.readFile(join(f.workspace, "agent_result.json"));
		const selected = await fs.readFile(join(f.workspace, f.audit.candidates[0]!.source_path));
		const result = await f.invoke();
		expect(f.createTool().executionMode).toBe("sequential");
		expect(result.details).toMatchObject({ status: "applied", candidate_id: "wide", evaluation_id: f.audit.evaluation_id, source_sha256: sha(selected), patch: f.audit.candidates[0]!.patch });
		expect(result.content[0]!.text).toBe("Applied candidate wide to solution.py. output.step unchanged; CAD has not been rebuilt.");
		expect(await fs.readFile(join(f.workspace, "solution.py"))).toEqual(selected);
		expect(await fs.readFile(join(f.workspace, f.audit.candidates[1]!.source_path))).toEqual(sibling);
		expect(await fs.readFile(join(f.workspace, "output.step"), "utf8")).toBe(f.step);
		expect(await fs.readFile(join(f.workspace, "agent_result.json"))).toEqual(resultState);
		expect((await fs.readdir(f.workspace)).some((name) => name.startsWith(".solution.py.adopt-"))).toBe(false);
	});

	it.each(["solution.py", "output.step"])("rejects changed formal %s before any edit", async (path) => {
		const f = await fixture();
		await fs.writeFile(join(f.workspace, path), "changed externally");
		await expect(f.invoke()).rejects.toThrow("changed after the preview");
		expect(await fs.readFile(join(f.workspace, "solution.py"), "utf8")).toBe(path === "solution.py" ? "changed externally" : f.source);
	});

	it("repeated adoption is harmless after a rebuild, while another old sibling remains stale", async () => {
		const f = await fixture();
		await f.invoke();
		await fs.writeFile(join(f.workspace, "output.step"), "newly rebuilt STEP");
		const result = await f.invoke();
		expect(result.details.status).toBe("already_applied");
		expect(await fs.readFile(join(f.workspace, "output.step"), "utf8")).toBe("newly rebuilt STEP");
		await expect(f.invoke({ candidate_id: "tall" })).rejects.toThrow("solution.py changed");
	});

	it.each(["unknown", "../wide", "a".repeat(65)])("rejects unknown or unsafe candidate %s", async (candidate_id) => {
		const f = await fixture();
		await expect(f.invoke({ candidate_id })).rejects.toThrow();
		expect(await fs.readFile(join(f.workspace, "solution.py"), "utf8")).toBe(f.source);
	});

	it.each(["source_path", "edits"])("rejects inconsistent %s without partially applying edits", async (kind) => {
		const f = await fixture();
		const candidate = f.audit.candidates[0]!;
		if (kind === "source_path") candidate.source_path = "solution.py";
		if (kind === "edits") candidate.edits[1]!.newText = "height = 99";
		await f.persist(f.audit);
		await expect(f.invoke()).rejects.toThrow();
		expect(await fs.readFile(join(f.workspace, "solution.py"), "utf8")).toBe(f.source);
	});

	it.each(["source", "step", "card"])("rejects corrupted candidate %s files", async (kind) => {
		const f = await fixture();
		const candidate = f.audit.candidates[0]!;
		const path = kind === "source" ? candidate.source_path : kind === "step" ? candidate.execution!.artifact.path! : f.audit.feedback[0]!.card_path;
		await fs.writeFile(join(f.workspace, path), "corrupted");
		await expect(f.invoke()).rejects.toThrow();
		expect(await fs.readFile(join(f.workspace, "solution.py"), "utf8")).toBe(f.source);
	});

	it("checks the selected non-first candidate card before adopting its patch", async () => {
		const f = await fixture();
		const card = f.audit.feedback[0]!.cards[1]!;
		await fs.writeFile(join(f.workspace, card.card_path), "corrupted second card");
		await expect(f.invoke({ candidate_id: "tall" })).rejects.toThrow();
		expect(await fs.readFile(join(f.workspace, "solution.py"), "utf8")).toBe(f.source);
	});

	it("adopts a non-first candidate using its own card when a sibling card is missing", async () => {
		const f = await fixture();
		await fs.unlink(join(f.workspace, f.audit.feedback[0]!.cards[0]!.card_path));
		const result = await f.invoke({ candidate_id: "tall" });
		expect(result.details).toMatchObject({ status: "applied", candidate_id: "tall" });
		expect(await fs.readFile(join(f.workspace, "solution.py"), "utf8")).toBe(f.source.replace("height = 20", "height = 30"));
	});

	it.each(["solution.py", SHAPE_FEEDBACK_LATEST_PATH])("rejects symlink input %s even when its target stays in the workspace", async (path) => {
		const f = await fixture();
		const target = join(f.workspace, "linked-file");
		await fs.rename(join(f.workspace, path), target);
		await fs.symlink(target, join(f.workspace, path));
		await expect(f.invoke()).rejects.toThrow();
	});

	it("rejects a symlink in a candidate directory", async () => {
		const f = await fixture();
		const directory = dirname(join(f.workspace, f.audit.candidates[0]!.source_path));
		await fs.rename(directory, `${directory}-moved`);
		await fs.symlink(`${directory}-moved`, directory);
		await expect(f.invoke()).rejects.toThrow("unsafe file");
	});

	it("rechecks Current after writing the temporary source and cleans up on a concurrent edit", async () => {
		const f = await fixture();
		const original = await vi.importActual<typeof import("node:fs/promises")>("node:fs/promises");
		vi.mocked(fs.open).mockImplementation(async (path, flags, mode) => {
			const handle = await original.open(path, flags, mode);
			if (String(path).includes(".solution.py.adopt-")) {
				const sync = handle.sync.bind(handle);
				handle.sync = async () => { await sync(); await original.writeFile(join(f.workspace, "solution.py"), "concurrent edit"); };
			}
			return handle;
		});
		await expect(f.invoke()).rejects.toThrow("changed during adoption");
		expect(await fs.readFile(join(f.workspace, "solution.py"), "utf8")).toBe("concurrent edit");
		expect((await fs.readdir(f.workspace)).some((name) => name.startsWith(".solution.py.adopt-"))).toBe(false);
	});

	it("leaves the source whole if atomic rename fails", async () => {
		const f = await fixture();
		vi.mocked(fs.rename).mockRejectedValueOnce(new Error("injected rename failure"));
		await expect(f.invoke()).rejects.toThrow("injected rename failure");
		expect(await fs.readFile(join(f.workspace, "solution.py"), "utf8")).toBe(f.source);
		expect((await fs.readdir(f.workspace)).some((name) => name.startsWith(".solution.py.adopt-"))).toBe(false);
	});

	it("does not write when cancelled", async () => {
		const f = await fixture();
		await f.controller.interrupt("cancelled");
		await expect(f.invoke()).rejects.toThrow("cancelled");
		expect(await fs.readFile(join(f.workspace, "solution.py"), "utf8")).toBe(f.source);
	});
});


describe("verifyShapeCandidateEvidence", () => {
	it("returns saved candidate and card evidence without modifying Current or reading the latest pointer", async () => {
		const f = await fixture();
		await fs.unlink(join(f.workspace, SHAPE_FEEDBACK_LATEST_PATH));
		const result = await verifyShapeCandidateEvidence({ workspace: f.workspace, lockedManifest: f.manifest, evaluationId: f.audit.evaluation_id, candidateId: "wide" });
		expect(result.candidate).toEqual(f.audit.candidates[0]);
		expect(result.cards).toEqual([{ region_id: "front", card_path: f.audit.feedback[0]!.cards[0]!.card_path, sha256: f.audit.feedback[0]!.cards[0]!.sha256, width: 120, height: 100 }]);
		expect(await fs.readFile(join(f.workspace, "solution.py"), "utf8")).toBe(f.source);
		expect(await fs.readFile(join(f.workspace, "output.step"), "utf8")).toBe(f.step);
	});

	it.each(["solution.py", "output.step"])("rejects stale formal %s in read-only verification", async (path) => {
		const f = await fixture();
		await fs.writeFile(join(f.workspace, path), "external change");
		await expect(verifyShapeCandidateEvidence({ workspace: f.workspace, lockedManifest: f.manifest, evaluationId: f.audit.evaluation_id, candidateId: "wide" })).rejects.toThrow("changed after the preview");
	});

	it("checks patch content even though no patch is applied", async () => {
		const f = await fixture();
		f.audit.candidates[0]!.patch = "altered patch";
		await f.persist(f.audit);
		await expect(verifyShapeCandidateEvidence({ workspace: f.workspace, lockedManifest: f.manifest, evaluationId: f.audit.evaluation_id, candidateId: "wide" })).rejects.toThrow("does not match its edits");
	});

	it.each(["execution", "anchor", "missing_card", "duplicate_projection", "duplicate_view"])("rejects incomplete or ambiguous %s evidence", async (kind) => {
		const f = await fixture();
		if (kind === "execution") f.audit.candidates[0]!.execution!.execution.success = false;
		if (kind === "anchor") f.audit.projections[0]!.projection_anchor_step_sha256 = "2".repeat(64);
		if (kind === "missing_card") f.audit.feedback[0]!.cards = [];
		if (kind === "duplicate_projection") f.audit.projections.push({ ...f.audit.projections[0]! });
		if (kind === "duplicate_view") f.audit.feedback.push(structuredClone(f.audit.feedback[0]!));
		await f.persist(f.audit);
		await expect(verifyShapeCandidateEvidence({ workspace: f.workspace, lockedManifest: f.manifest, evaluationId: f.audit.evaluation_id, candidateId: "wide" })).rejects.toThrow();
	});

	it("returns all candidate views and validates the images after the first view", async () => {
		const f = await fixture();
		const second = structuredClone(f.audit.feedback[0]!);
		second.region_id = "side";
		second.cards[0]!.card_path = `drawing_feedback/shape-feedback/${f.audit.evaluation_id}/views/side/card.png`;
		await f.save(second.cards[0]!.card_path, makePng(120, 100, 1));
		f.audit.feedback.push(second);
		f.audit.projections.push({ ...f.audit.projections[0]!, region_id: "side" });
		await f.persist(f.audit);
		const options = { workspace: f.workspace, lockedManifest: f.manifest, evaluationId: f.audit.evaluation_id, candidateId: "wide" };
		expect((await verifyShapeCandidateEvidence(options)).cards.map(card => card.region_id)).toEqual(["front", "side"]);
		await f.save(second.cards[0]!.card_path, makePng(120, 100, 3));
		await expect(verifyShapeCandidateEvidence(options)).rejects.toThrow("saved preview image changed");
	});
});
