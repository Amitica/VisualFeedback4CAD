import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import {
  createDimensionAnnotationService, DIMENSION_SUBAGENT_TOOLS, dimensionsCurrent,
} from "../src/dimension-annotations.js";
import type { WorkerClient, WorkerRequest } from "../src/local-worker.js";
import { RunController, type ResolvedModel } from "../src/runner.js";
import {
  DIMENSION_ANNOTATION_PROFILE, type AgentJob, type AgentResult,
  type DimensionAnnotationSummary, type DrawingViewManifest,
  type SubagentRunUsage, type WorkerResponse,
} from "../src/types.js";
import type { CurrentFrame } from "../src/visual-review.js";
import { makePng } from "./png-fixture.js";

type ServiceOptions = Parameters<typeof createDimensionAnnotationService>[0];
type SessionOptions = Parameters<NonNullable<ServiceOptions["runSession"]>>[0];
interface TestTool {
  name: string;
  execute: (...args: unknown[]) => Promise<unknown>;
}
const roots: string[] = [];
afterEach(async () => {
  for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true });
});
function sha(value: string | Buffer): string {
  return createHash("sha256").update(value).digest("hex");
}
async function invoke(session: SessionOptions, name: string, args: object): Promise<unknown> {
  const tool = session.tools.find(t => t.name === name) as TestTool | undefined;
  if (!tool) throw new Error(`Missing child tool ${name}`);
  return tool.execute(`test-${name}`, args, undefined, undefined, {});
}
function choice(entity = "E001", id = "bore_diameter") {
  return { dimension_id: id, description: "Diameter of the central circular bore", quantity: "diameter", entity_ids: [entity] };
}
async function measure(session: SessionOptions, entity = "E001", id = "bore_diameter") {
  return invoke(session, "measure_dimensions", { measurements: [choice(entity, id)] });
}
async function submit(session: SessionOptions, args: object = {}) {
  return invoke(session, "submit_dimension_annotations", {
    status: "annotated", reason: "Measured the selected current bore and checked the image.", removed_dimensions: [], ...args,
  });
}

async function fixture(options: { section?: boolean } = {}) {
  const workspace = await mkdtemp(join(tmpdir(), "dimension-annotations-"));
  roots.push(workspace);
  const save = async (path: string, value: string | Buffer) => {
    await mkdir(dirname(join(workspace, path)), { recursive: true });
    await writeFile(join(workspace, path), value);
  };
  await save("solution.py", "radius = 2\n");
  await save("output.step", "STEP-v1");
  await save("reference.png", makePng(200, 160, 1));
  await save("current.png", makePng(200, 160, 2));
  const manifestHash = sha("locked manual alignment");
  const manifest = {
    version: 4, source_sha256: sha("original solution"), step_sha256: sha("anchor STEP"),
    projection_frame: { policy: "fixed", anchor_step_path: "anchor.step", anchor_step_sha256: sha("anchor STEP") },
    regions: [{
      region_id: "front", view_type: options.section ? "section" : "orthographic", crop_path: "reference.png",
      selected: {
        view_spec: { name: "front", direction: [0, -1, 0], up: [0, 0, 1] },
        transform: { scale_px_per_unit: 5, rotation_deg: 4, center_x_px: 110, center_y_px: 90, model_center_x: 1, model_center_y: -2, reflected: false },
      },
    }],
  } as unknown as DrawingViewManifest;
  const job = {
    thinking_level: "low", continuation: { id: "dimension-test", locked_view_manifest_sha256: manifestHash },
  } as AgentJob;
  const requests: WorkerRequest[] = [];
  const sessions: SessionOptions[] = [];
  const updates: DimensionAnnotationSummary[] = [];
  const usages: SubagentRunUsage[] = [];
  const catalogSessions = new Map<string, { step: string; entity: string }>();
  const control: { failOperation?: string; failMessage?: string } = {};
  const hashFile = async (path: string) => sha(await readFile(join(workspace, path)));
  const worker: WorkerClient = {
    async request<T extends WorkerResponse>(_workspace: string, request: WorkerRequest): Promise<T> {
      requests.push(structuredClone(request));
      if (request.operation === control.failOperation) {
        return { protocol_version: 5, operation: request.operation, status: "error", error: { kind: "drawing", message: control.failMessage ?? "synthetic worker failure" } } as unknown as T;
      }
      if (request.operation === "prepare_dimension_view") {
        const digest = await hashFile(String(request.step_path));
        expect(request.expected_step_sha256).toBe(digest);
        const entity = (await readFile(join(workspace, "output.step"), "utf8")) === "STEP-v1" ? "E001" : "E011";
        const out = String(request.output_dir);
        const sessionPath = join(out, "session.json");
        const objects = { [entity]: { id: entity, kind: "edge", geometry_type: "CIRCLE", visible: true, radius: entity === "E001" ? 2 : 3 } };
        await save(sessionPath, JSON.stringify({ step_sha256: digest }));
        const entitiesPath = join(out, "entities.json");
        await save(entitiesPath, JSON.stringify({ objects, pages: [{ entity_ids: [entity] }] }));
        const numbered = join(out, "numbered.png");
        const cad = join(out, "cad.png");
        await save(numbered, makePng(200, 160, entity === "E001" ? 3 : 4));
        await save(cad, makePng(200, 160, 5));
        catalogSessions.set(sessionPath, { step: digest, entity });
        return {
          protocol_version: 5, operation: request.operation, status: "ok", region_id: request.region_id,
          step_sha256: digest, session_path: sessionPath, entities_path: entitiesPath,
          cad_path: cad, reference_path: "reference.png", numbered_paths: [numbered], objects,
        } as unknown as T;
      }
      if (request.operation === "measure_dimension_view") {
        const catalog = catalogSessions.get(String(request.session_path));
        if (!catalog) throw new Error("Unknown prepared session");
        expect(request.expected_step_sha256).toBe(catalog.step);
        const measurements = request.requests as Array<{ quantity: string; entity_ids: string[] }>;
        for (const measurement of measurements) {
          expect(measurement.entity_ids).toEqual([catalog.entity]);
          expect(measurement).not.toHaveProperty("value");
          expect(measurement).not.toHaveProperty("description");
          expect(measurement).not.toHaveProperty("dimension_id");
        }
        const out = String(request.output_dir);
        const paths = {
          selection_path: join(out, "selection.png"), comparison_path: join(out, "comparison.png"),
          annotated_path: join(out, "annotated.png"), measurements_path: join(out, "measurements.json"),
        };
        const results = measurements.map(m => ({ ...m, value: catalog.entity === "E001" ? 4 : 6, unit: "mm", model_sha256: catalog.step, annotation_status: "dimension_drawn" }));
        await save(paths.selection_path, makePng(200, 160, 6));
        await save(paths.comparison_path, makePng(400, 200, catalog.entity === "E001" ? 7 : 8));
        await save(paths.annotated_path, makePng(200, 160, 9));
        await save(paths.measurements_path, JSON.stringify({ results, model_sha256: catalog.step }));
        const artifact_sha256 = Object.fromEntries(await Promise.all(Object.values(paths).map(async p => [p, await hashFile(p)])));
        return { protocol_version: 5, operation: request.operation, status: "ok", step_sha256: catalog.step, results, artifact_sha256, ...paths } as unknown as T;
      }
      throw new Error(`Unexpected worker request ${request.operation}`);
    },
  };
  const frame = (): CurrentFrame => ({
    viewSpec: manifest.regions[0]!.selected!.view_spec,
    transform: manifest.regions[0]!.selected!.transform,
    referencePath: "reference.png", currentRenderPath: "current.png", projectionOrigin: [1, 2, 3],
    projectionAnchorStepPath: "anchor.step", projectionAnchorStepSha256: sha("anchor STEP"),
    cameraSha256: sha("camera"), transformSha256: sha("transform"),
  });
  const readFormal: NonNullable<ServiceOptions["readFormal"]> = async () => {
    const source = await readFile(join(workspace, "solution.py"), "utf8");
    return { source, sourceSha256: sha(source), stepSha256: await hashFile("output.step") };
  };
  const optionsBase: Omit<ServiceOptions, "runSession"> = {
    workspace, job, lockedManifest: manifest, worker, controller: new RunController(),
    resolvedModel: { runtime: {}, model: {} } as unknown as ResolvedModel,
    getResult: () => ({}) as AgentResult,
    readFormal, makeFrame: async () => frame(),
    onUpdate: async state => { updates.push(structuredClone(state)); },
    onUsage: async usage => { usages.push(usage); },
  };
  const service = (script: (session: SessionOptions) => Promise<void>, continuationId?: string) => {
    const runSession = async (session: SessionOptions) => {
      sessions.push(session);
      await script(session);
      const usage: SubagentRunUsage = {
        role: "dimension_annotation", accounting_id: `dimension-test-${sessions.length}`, status: "success",
        started_at: "2026-09-30T00:00:00Z", finished_at: "2026-09-30T00:00:01Z", duration_s: 1,
        tokens: { input: 5, output: 5, cache_read: 0, cache_write: 0, total: 10 }, cost: 0, tool_calls: 2,
      };
      await session.onUsage(usage);
      return { value: session.getSubmitted(), usage };
    };
    return createDimensionAnnotationService({
      ...optionsBase,
      ...(continuationId ? { job: { ...job, continuation: { ...job.continuation!, id: continuationId } } } : {}),
      runSession: runSession as NonNullable<ServiceOptions["runSession"]>,
    });
  };
  const revise = async () => {
    await save("solution.py", "radius = 3\n");
    await save("output.step", "STEP-v2");
  };
  return { workspace, manifest, manifestHash, job, save, hashFile, requests, sessions, updates, usages, control, service, revise, frame, optionsBase };
}

describe("dimension annotation SubAgent lifecycle", () => {
  it("prepares a fixed view, runs restricted measurement tools, and publishes actual values", async () => {
    const f = await fixture();
    const service = f.service(async session => {
      expect(session.toolNames).toEqual(DIMENSION_SUBAGENT_TOOLS);
      expect(session.tools.map(t => t.name)).toEqual([...DIMENSION_SUBAGENT_TOOLS]);
      expect(session.images).toHaveLength(2);
      await invoke(session, "inspect_dimension_entities", { kind: "edge", geometry_type: "CIRCLE" });
      await measure(session);
      await submit(session);
    });
    const state = await service.ensureCurrent("first formal execute");
    expect(state.status).toBe("completed");
    expect(state.regions[0]).toMatchObject({ status: "annotated", dimensions: [choice()], values: [{ value: 4, unit: "mm" }] });
    expect(state.step_sha256).toBe(await f.hashFile("output.step"));
    expect(f.requests.map(r => r.operation)).toEqual(["prepare_dimension_view", "measure_dimension_view"]);
    expect(f.requests[0]).toMatchObject({ view_spec: f.frame().viewSpec, projection_origin: [1, 2, 3], transform: f.frame().transform, image_size: [200, 160] });
    expect(f.usages).toHaveLength(1);
    expect(f.updates[0]!.status).toBe("running");
    const returned = await service.comparison("front");
    expect(returned.content.some(c => c.type === "image")).toBe(true);
    expect(f.sessions).toHaveLength(1);
    expect(await readFile(join(f.workspace, "solution.py"), "utf8")).toBe("radius = 2\n");
  });

  it("publishes more than twelve reference dimensions without truncation", async () => {
    const f = await fixture();
    const choices = Array.from({ length: 30 }, (_, i) => choice("E001", `drawing_dim_${i}`));
    const service = f.service(async session => {
      await invoke(session, "measure_dimensions", { measurements: choices });
      await submit(session, { reason: "All 30 reference dimensions measured." });
    });
    const state = await service.ensureCurrent("full reference coverage");
    expect(state.regions[0]!.dimensions).toHaveLength(30);
    expect(state.regions[0]!.values).toHaveLength(30);
  });

  it("retains measured evidence alongside first-time unresolved reference dimensions", async () => {
    const f = await fixture();
    const removed = Array.from({ length: 30 }, (_, i) => ({ dimension_id: `unsupported_${i}`, reason: "Reference callout cannot be measured with current tools." }));
    const service = f.service(async session => {
      await measure(session);
      await submit(session, { status: "unresolved", reason: "1 measured; 30 unresolved reference dimensions.", removed_dimensions: removed });
    });
    const state = await service.ensureCurrent("partial measurement coverage");
    expect(state.regions[0]).toMatchObject({ status: "unresolved", dimensions: [choice()], values: [{ value: 4 }], removed_dimensions: removed });
    expect(state.regions[0]!.comparison_path).toBeTruthy();
    const visible = await service.comparison("front");
    expect(visible.content.some(item => item.type === "image")).toBe(true);
    const text = visible.content.filter(item => item.type === "text").map(item => item.text).join("\n");
    expect(text).not.toContain("unresolved");
    expect(text).not.toContain("unsupported_");
    expect(text).not.toContain("Reference callout");
    expect(text).not.toContain("30");
    expect(text).toContain('"value":4');
  });

  it("reuses verified evidence for an unchanged STEP, including source-only edits", async () => {
    const f = await fixture();
    const service = f.service(async s => { await measure(s); await submit(s); });
    const initial = await service.ensureCurrent("initial");
    await f.save("solution.py", "# equivalent code\nradius = 2\n");
    const reused = await service.ensureCurrent("execute unchanged geometry");
    expect(reused.revision_id).toBe(initial.revision_id);
    expect(reused.source_sha256).toBe(await f.hashFile("solution.py"));
    expect(reused.source_sha256).not.toBe(initial.source_sha256);
    expect(reused.regions[0]!.comparison_path).toBe(initial.regions[0]!.comparison_path);
    expect(f.sessions).toHaveLength(1);
    expect(f.requests).toHaveLength(2);
  });

  it("remeasures unchanged CAD for each new reflection round while reusing evidence within that round", async () => {
    const f = await fixture();
    const sourceHash = await f.hashFile("solution.py");
    const stepHash = await f.hashFile("output.step");
    const rounds = ["reflect-round-1", "reflect-round-2", "reflect-round-3"];
    const revisions = new Set<string>();
    const statePaths = new Set<string>();
    const comparisonPaths = new Set<string>();
    const priorRoundStates = new Map<string, string>();

    for (const [index, continuationId] of rounds.entries()) {
      const service = f.service(async session => { await measure(session); await submit(session); }, continuationId);
      const measured = await service.ensureCurrent(`before ${continuationId}`);
      expect(measured).toMatchObject({ status: "completed", source_sha256: sourceHash, step_sha256: stepHash });
      expect(measured.regions[0]).toMatchObject({ dimensions: [choice()], values: [{ value: 4, unit: "mm" }] });
      revisions.add(measured.revision_id);
      statePaths.add(measured.state_path);
      comparisonPaths.add(measured.regions[0]!.comparison_path!);
      expect(f.sessions).toHaveLength(index + 1);
      expect(f.requests.filter(request => request.operation === "measure_dimension_view")).toHaveLength(index + 1);

      const cached = await service.ensureCurrent(`repeat inspect within ${continuationId}`);
      expect(cached.revision_id).toBe(measured.revision_id);
      expect(cached.state_path).toBe(measured.state_path);
      expect(cached.regions[0]!.comparison_path).toBe(measured.regions[0]!.comparison_path);
      expect(f.sessions).toHaveLength(index + 1);
      expect(f.requests.filter(request => request.operation === "measure_dimension_view")).toHaveLength(index + 1);
      for (const [path, saved] of priorRoundStates) {
        expect(await readFile(join(f.workspace, path), "utf8")).toBe(saved);
      }
      priorRoundStates.set(cached.state_path, await readFile(join(f.workspace, cached.state_path), "utf8"));
    }

    expect(revisions.size).toBe(3);
    expect(statePaths.size).toBe(3);
    expect(comparisonPaths.size).toBe(3);
    expect(f.requests.map(request => request.operation)).toEqual(rounds.flatMap(() => ["prepare_dimension_view", "measure_dimension_view"]));
    expect(await f.hashFile("solution.py")).toBe(sourceHash);
    expect(await f.hashFile("output.step")).toBe(stepHash);
  });

  it("remeasures new STEP IDs and retains the intended dimension after an edit", async () => {
    const f = await fixture();
    const service = f.service(async s => {
      if (f.sessions.length === 1) { await measure(s); }
      else {
        expect(s.prompt).toContain("bore_diameter");
        expect(s.prompt).toContain("E011");
        await expect(measure(s, "E001")).rejects.toThrow("known current entities");
        await measure(s, "E011");
      }
      await submit(s);
    });
    const initial = await service.ensureCurrent("initial");
    await f.revise();
    const updated = await service.ensureCurrent("after edit");
    expect(updated.revision_id).not.toBe(initial.revision_id);
    expect(updated.step_sha256).not.toBe(initial.step_sha256);
    expect(updated.regions[0]).toMatchObject({ dimensions: [choice("E011")], values: [{ value: 6 }] });
    expect(f.sessions).toHaveLength(2);
    const prepares = f.requests.filter(r => r.operation === "prepare_dimension_view");
    expect(prepares[1]!.projection_origin).toEqual(prepares[0]!.projection_origin);
    expect(prepares[1]!.transform).toEqual(prepares[0]!.transform);
    expect(prepares[1]!.view_spec).toEqual(prepares[0]!.view_spec);
  });

  it("requires an explicit reason when a prior dimension is removed", async () => {
    const f = await fixture();
    const service = f.service(async s => {
      if (f.sessions.length === 1) { await measure(s); await submit(s); return; }
      await expect(submit(s, { status: "unresolved", reason: "The corresponding bore was removed." })).rejects.toThrow("Reconcile every previous dimension");
      await expect(submit(s, { status: "unresolved", removed_dimensions: [{ dimension_id: "bore_diameter", reason: " " }] })).rejects.toThrow("nonblank reasons");
      await submit(s, { status: "unresolved", reason: "The selected bore is missing in the current shape.", removed_dimensions: [{ dimension_id: "bore_diameter", reason: "No corresponding current bore can be selected." }] });
    });
    await service.ensureCurrent("initial");
    await f.revise();
    const updated = await service.ensureCurrent("removed bore");
    expect(updated.regions[0]).toMatchObject({ status: "unresolved", dimensions: [], removed_dimensions: [{ dimension_id: "bore_diameter", reason: "No corresponding current bore can be selected." }] });
  });

  it("cannot submit annotated without measuring or provide an invented numeric value", async () => {
    const f = await fixture();
    const service = f.service(async s => {
      await expect(submit(s)).rejects.toThrow("Measure the current selections");
      await expect(invoke(s, "measure_dimensions", { measurements: [{ ...choice(), value: 50 }] })).rejects.toThrow("Invalid measurement request");
      await submit(s, { status: "not_needed", reason: "No dimension has an unambiguous matching entity." });
    });
    expect((await service.ensureCurrent("no reliable dimension")).regions[0]!.status).toBe("not_needed");
    expect(f.requests.map(r => r.operation)).toEqual(["prepare_dimension_view"]);
  });

  it("marks the current update failed when the measurement worker fails", async () => {
    const f = await fixture();
    f.control.failOperation = "measure_dimension_view";
    const service = f.service(async s => { await measure(s); await submit(s); });
    await expect(service.ensureCurrent("worker error")).rejects.toThrow("synthetic worker failure");
    expect(f.updates.at(-1)!.status).toBe("failed");
    expect(JSON.parse(await readFile(join(f.workspace, f.updates.at(-1)!.state_path), "utf8")).status).toBe("failed");
  });

  it("marks the update failed when the child ends without submitting", async () => {
    const f = await fixture();
    const service = f.service(async () => {});
    await expect(service.ensureCurrent("incomplete child")).rejects.toThrow("without a submitted review");
    expect(f.updates.at(-1)!.status).toBe("failed");
  });

  it("rejects formal file changes during the child and fails the state", async () => {
    const f = await fixture();
    const service = f.service(async s => {
      await measure(s); await submit(s);
      await f.save("output.step", "STEP changed while child was finishing");
    });
    await expect(service.ensureCurrent("changed formal file")).rejects.toThrow("changed the formal source or STEP");
    expect(f.updates.at(-1)!.status).toBe("failed");
  });

  it("records section views as explicitly unsupported without invoking a child", async () => {
    const f = await fixture({ section: true });
    const service = f.service(async () => { throw new Error("No child should inspect cut topology"); });
    const state = await service.ensureCurrent("section");
    expect(state.status).toBe("completed");
    expect(state.regions[0]).toMatchObject({ status: "unsupported", dimensions: [] });
    expect(state.regions[0]!.reason).toContain("Section-generated topology");
    expect(f.sessions).toHaveLength(0);
    expect(f.requests).toHaveLength(0);
  });

  it("rebuilds annotated evidence when the cached comparison image changes", async () => {
    const f = await fixture();
    const service = f.service(async s => { await measure(s); await submit(s); });
    const original = await service.ensureCurrent("initial");
    const comparison = original.regions[0]!.comparison_path!;
    await f.save(comparison, makePng(400, 200, 99));
    const rebuilt = await service.ensureCurrent("verify cached image");
    expect(rebuilt.revision_id).not.toBe(original.revision_id);
    expect(rebuilt.regions[0]!.comparison_path).not.toBe(comparison);
    expect(f.sessions).toHaveLength(2);
    expect(f.requests).toHaveLength(4);
  });

  it("rebuilds the evidence when a cached comparison PNG is missing", async () => {
    const f = await fixture();
    const service = f.service(async s => { await measure(s); await submit(s); });
    const original = await service.ensureCurrent("initial");
    await rm(join(f.workspace, original.regions[0]!.comparison_path!));
    const rebuilt = await service.ensureCurrent("missing cached image");
    expect(rebuilt.status).toBe("completed");
    expect(rebuilt.revision_id).not.toBe(original.revision_id);
    expect(await f.hashFile(rebuilt.regions[0]!.comparison_path!)).toBe(rebuilt.regions[0]!.comparison_sha256);
    expect(f.sessions).toHaveLength(2);
  });

  it("invalidates cached evidence if the formal STEP changes during the reuse check", async () => {
    const f = await fixture();
    let mutateDuringRead = false;
    const readFormal = f.optionsBase.readFormal!;
    f.optionsBase.readFormal = async options => {
      const formal = await readFormal(options);
      if (mutateDuringRead) await f.save("output.step", "STEP changed during cache validation");
      return formal;
    };
    const service = f.service(async s => { await measure(s); await submit(s); });
    await service.ensureCurrent("initial");
    mutateDuringRead = true;
    await expect(service.ensureCurrent("reuse race")).rejects.toThrow("changed the formal source or STEP");
    expect(f.updates.at(-1)!.status).toBe("failed");
    expect(JSON.parse(await readFile(join(f.workspace, f.updates.at(-1)!.state_path), "utf8")).status).toBe("failed");
    expect(f.sessions).toHaveLength(1);
  });

  it("shares a pending update between concurrent consumers", async () => {
    const f = await fixture();
    const service = f.service(async s => { await measure(s); await submit(s); });
    const [a, b] = await Promise.all([service.ensureCurrent("execute"), service.ensureCurrent("inspect")]);
    expect(a.revision_id).toBe(b.revision_id);
    expect(f.sessions).toHaveLength(1);
  });
});

describe("dimensionsCurrent", () => {
  const summary: DimensionAnnotationSummary = {
    profile: DIMENSION_ANNOTATION_PROFILE, state_path: "drawing_feedback/state.json", revision_id: "revision-1",
    source_sha256: sha("source"), step_sha256: sha("step"), locked_manifest_sha256: sha("manifest"), status: "completed", regions: [],
  };
  it("requires matching formal source, STEP and requested locked manifest", () => {
    expect(dimensionsCurrent(summary, sha("source"), sha("step"), sha("manifest"))).toBe(true);
    expect(dimensionsCurrent(summary, sha("different source"), sha("step"), sha("manifest"))).toBe(false);
    expect(dimensionsCurrent(summary, sha("source"), sha("different STEP"), sha("manifest"))).toBe(false);
    expect(dimensionsCurrent(summary, sha("source"), sha("step"), sha("different manifest"))).toBe(false);
    expect(dimensionsCurrent(undefined, sha("source"), sha("step"), sha("manifest"))).toBe(false);
    expect(dimensionsCurrent(summary, undefined, sha("step"), sha("manifest"))).toBe(false);
    expect(dimensionsCurrent(summary, sha("source"), undefined, sha("manifest"))).toBe(false);
    expect(dimensionsCurrent({ ...summary, status: "running" }, sha("source"), sha("step"), sha("manifest"))).toBe(false);
    expect(dimensionsCurrent({ ...summary, status: "failed" }, sha("source"), sha("step"), sha("manifest"))).toBe(false);
  });
});
