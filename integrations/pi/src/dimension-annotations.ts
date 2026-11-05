/** Automatic, read-only dimension annotation between formal CAD execution and Main inspection. */
import { createHash, randomUUID } from "node:crypto";
import { mkdir } from "node:fs/promises";
import { isAbsolute, join, relative, resolve } from "node:path";

import type { ImageContent, TextContent } from "@earendil-works/pi-ai";
import { defineTool } from "@earendil-works/pi-coding-agent";
import { Type, type Static } from "typebox";
import { Value } from "typebox/value";

import { imageFromWorkspace, inspectPngFile, requireDrawingSuccess, runRestrictedSession } from "./drawing-reflection.js";
import { InfrastructureError } from "./errors.js";
import { isoNow, readJson, writeJsonAtomic } from "./files.js";
import type { WorkerClient } from "./local-worker.js";
import type { ResolvedModel, RunController } from "./runner.js";
import { sha256File } from "./solution.js";
import {
  DIMENSION_ANNOTATION_PROFILE, PROTOCOL_VERSION,
  type AgentJob, type AgentResult, type DimensionAnnotationSummary,
  type DimensionAnnotationRegionSummary, type DrawingViewManifest,
  type SubagentRunUsage, type WorkerBaseResponse,
} from "./types.js";
import { assertFormalUnchanged, currentFormalState, currentFrame, type FormalState } from "./visual-review.js";

const shortText = Type.String({ minLength: 1, maxLength: 1500 });
const entityId = Type.String({ pattern: "^[EF][0-9]{3,}$" });
const vector = Type.Tuple([Type.Number(), Type.Number(), Type.Number()]);
const measurementSchema = Type.Object({
  dimension_id: Type.String({ pattern: "^[A-Za-z][A-Za-z0-9_-]{0,63}$" }),
  description: shortText,
  quantity: Type.Union(["diameter", "radius", "edge_length", "center_distance", "directional_distance", "face_distance", "angle"].map(q => Type.Literal(q))),
  entity_ids: Type.Array(entityId, { minItems: 1, maxItems: 2, uniqueItems: true }),
  direction: Type.Optional(vector),
  offset_px: Type.Optional(Type.Number()),
  label_position_px: Type.Optional(Type.Tuple([Type.Number(), Type.Number()])),
  annotation_style: Type.Optional(Type.Literal("leader")),
}, { additionalProperties: false });
export const MEASURE_DIMENSIONS_SCHEMA = Type.Object({
  measurements: Type.Array(measurementSchema, { minItems: 1 }),
}, { additionalProperties: false });
export const SUBMIT_DIMENSIONS_SCHEMA = Type.Object({
  status: Type.Union([Type.Literal("annotated"), Type.Literal("not_needed"), Type.Literal("unresolved")]),
  reason: shortText,
  removed_dimensions: Type.Array(Type.Object({ dimension_id: shortText, reason: shortText }, { additionalProperties: false })),
}, { additionalProperties: false });
const catalogSchema = Type.Object({
  kind: Type.Union([Type.Literal("edge"), Type.Literal("face"), Type.Literal("all")]),
  geometry_type: Type.Optional(Type.String({ pattern: "^[A-Z_]+$", maxLength: 40 })),
  page: Type.Optional(Type.Integer({ minimum: 1, maximum: 100 })),
  entity_ids: Type.Optional(Type.Array(entityId, { minItems: 1, maxItems: 24, uniqueItems: true })),
}, { additionalProperties: false });
type DimensionChoice = Static<typeof measurementSchema>;
type Submission = Static<typeof SUBMIT_DIMENSIONS_SCHEMA>;
interface PreparedView extends WorkerBaseResponse {
  region_id: string; step_sha256: string; session_path: string; entities_path: string;
  cad_path: string; reference_path: string; numbered_paths: string[];
  objects: Record<string, { id: string; kind: string; geometry_type: string; visible: boolean; [key: string]: unknown }>;
}
interface MeasuredView extends WorkerBaseResponse {
  step_sha256: string; comparison_path: string; annotated_path: string; selection_path: string;
  measurements_path: string; results: Array<Record<string, unknown>>;
  artifact_sha256: Record<string, string>;
}
interface RegionRecord extends DimensionAnnotationRegionSummary {
  dimensions: DimensionChoice[];
  removed_dimensions: Submission["removed_dimensions"];
  measurements_path?: string;
  values?: Array<Record<string, unknown>>;
}
interface DimensionState extends DimensionAnnotationSummary {
  version: 1;
  regions: RegionRecord[];
  updated_at: string;
  trigger: string;
  error?: string;
}
export const DIMENSION_SUBAGENT_TOOLS = ["inspect_dimension_entities", "measure_dimensions", "submit_dimension_annotations"] as const;
export const DIMENSION_SUBAGENT_PROMPT = `You maintain actual CAD dimensions on a locked aligned view for a separate Main CAD repair agent.
Compare the original engineering drawing with the current numbered CAD view. First inventory EVERY dimension explicitly annotated in this reference engineering-drawing view, then account for every item. Cover all linear, diameter, radius, angular, spacing, depth, repeated-feature and section/detail dimensions, including dimension callouts in notes. Do not select only salient, easy or useful dimensions and do not stop at an arbitrary count. Assign each drawing dimension a stable dimension_id and describe its drawing label, location and intended feature. Measure every item that the current geometry and tools support. For each missing feature, unsupported quantity, illegible label or uncertain correspondence, list its dimension_id and a specific reason in removed_dimensions; this field also records reference dimensions that could not be measured for the first time. Never silently omit an annotated reference dimension. Drawing labels identify the intended checks, not actual CAD values. Use entity IDs from THIS current model's catalog; the geometry tool computes the actual values and places dimension lines/leaders. Never supply or copy expected drawing values as actual CAD values.
You may inspect circular edges, straight edges and faces. The catalog's E/F IDs name topology, not semantic holes. Circle arcs and cylinders may be incomplete or external. Match the drawing structure using the complete view and highlight evidence. Request directional_distance with a world-coordinate direction when a drawing dimension is directional. Inspect the returned measurement image before finishing; fix selection or label placement if necessary.
After a CAD change, earlier dimension descriptions are a checklist, but their entity IDs and values are obsolete. Reidentify current entities. Preserve a dimension_id when checking the same drawing dimension, and explicitly account for each obsolete, missing, unsupported or uncertain dimension with a reason. Re-scan the entire reference view on every revision; the previous selection may be incomplete. Never silently reuse a previous STEP's image, anchors or IDs. Even when the desired annotations remain the same, remeasure them on the current STEP.
Use not_needed only when the reference view contains no annotated dimensions at all. Use unresolved whenever any annotated reference dimension cannot be measured reliably, even if other dimensions were measured successfully; explain every uncovered item in removed_dimensions and include the measured and unresolved counts in reason. Use annotated only when EVERY inventoried reference dimension has a reliable current measurement, and state the total covered count in reason. Explain the reason rather than inventing geometry. Submit annotated only after a successful measure_dimensions call. The last successful measurement call is the complete set to publish for this view, not an incremental addition. Include all measurable reference dimensions together in that final call; there is no 12-dimension limit. Inspect all labels for readability and adjust their placement without dropping dimensions.
You cannot edit code, run arbitrary shell commands, alter cameras, or change the formal CAD. You only select objects and maintain dimensions. Main decides whether and how to repair the CAD.`;

export function dimensionsCurrent(summary: DimensionAnnotationSummary | undefined, source: string | null | undefined, step: string | null | undefined, manifest?: string): boolean {
  return !!summary && summary.profile === DIMENSION_ANNOTATION_PROFILE && summary.status === "completed" &&
    !!source && !!step && summary.source_sha256 === source && summary.step_sha256 === step &&
    (!manifest || summary.locked_manifest_sha256 === manifest);
}

export function createDimensionAnnotationService(options: {
  workspace: string; job: AgentJob; lockedManifest: DrawingViewManifest;
  worker: WorkerClient; controller: RunController; resolvedModel: ResolvedModel;
  getResult: () => AgentResult;
  onUpdate: (summary: DimensionAnnotationSummary) => Promise<void>;
  onUsage: (usage: SubagentRunUsage) => Promise<void>;
  runSession?: typeof runRestrictedSession;
  readFormal?: typeof currentFormalState;
  makeFrame?: typeof currentFrame;
}) {
  const manifestHash = options.job.continuation?.locked_view_manifest_sha256;
  if (!manifestHash || options.lockedManifest.projection_frame?.policy !== "fixed") {
    throw new InfrastructureError("Dimension annotations require a locked calibrated fixed projection frame");
  }
  const namespace = createHash("sha256").update(options.job.continuation!.id).digest("hex").slice(0, 16);
  const root = join("drawing_feedback", "dimension-annotations", namespace);
  const statePath = join(root, "state.json");
  const lastCompletedPath = join(root, "last-completed.json");
  let pending: Promise<DimensionState> | undefined;

  function safePath(path: string): string {
    const absolute = resolve(options.workspace, path);
    const rel = relative(resolve(options.workspace), absolute);
    if (isAbsolute(path) || rel.startsWith("..") || isAbsolute(rel)) throw new InfrastructureError("Dimension artifact escaped the workspace");
    return absolute;
  }
  const persist = async (state: DimensionState) => {
    state.updated_at = isoNow();
    await writeJsonAtomic(safePath(statePath), state);
    await options.onUpdate(state);
  };
  const checkUnchanged = async (formal: FormalState) => assertFormalUnchanged(options.workspace, formal);
  const image = async (path: string, expected?: string): Promise<ImageContent> => {
    safePath(path);
    const inspected = await inspectPngFile(options.workspace, path);
    if (expected && inspected.sha256 !== expected) throw new InfrastructureError("Dimension image hash changed");
    return imageFromWorkspace(options.workspace, path);
  };
  async function validateCached(state: DimensionState): Promise<boolean> {
    if (state.profile !== DIMENSION_ANNOTATION_PROFILE || state.status !== "completed") return false;
    const expected = options.lockedManifest.regions.filter(r => r.selected).map(r => r.region_id).sort();
    if (!Array.isArray(state.regions) || JSON.stringify(state.regions.map(r => r.region_id).sort()) !== JSON.stringify(expected)) return false;
    for (const region of state.regions) {
      if (region.status === "annotated") {
        if (!region.comparison_path || !region.comparison_sha256) return false;
        if (await sha256File(safePath(region.comparison_path)).catch(() => null) !== region.comparison_sha256) return false;
      }
    }
    return true;
  }

  async function update(trigger: string): Promise<DimensionState> {
    const formal = await (options.readFormal ?? currentFormalState)(options);
    const previous = await readJson<DimensionState>(safePath(lastCompletedPath)).catch(() => null);
    if (previous?.version === 1 && previous.locked_manifest_sha256 === manifestHash && previous.step_sha256 === formal.stepSha256 && await validateCached(previous)) {
      const reused: DimensionState = { ...previous, status: "running", source_sha256: formal.sourceSha256, trigger: `${trigger}: STEP unchanged; reuse verified dimensions` };
      await persist(reused);
      try {
        await checkUnchanged(formal);
        reused.status = "completed";
        await persist(reused);
        await writeJsonAtomic(safePath(lastCompletedPath), reused);
        return reused;
      } catch (error) {
        reused.status = "failed";
        reused.error = error instanceof Error ? error.message : String(error);
        await persist(reused);
        throw error;
      }
    }
    const revision = `revision-${randomUUID()}`;
    const outputRoot = join(root, revision);
    await mkdir(safePath(outputRoot), { recursive: true });
    const state: DimensionState = {
      version: 1, profile: DIMENSION_ANNOTATION_PROFILE, state_path: statePath, revision_id: revision,
      source_sha256: formal.sourceSha256, step_sha256: formal.stepSha256, locked_manifest_sha256: manifestHash!,
      status: "running", regions: [], trigger, updated_at: isoNow(),
    };
    await persist(state); // Invalidate the old evidence before any new work starts.
    try {
      for (const region of options.lockedManifest.regions.filter(r => r.selected)) {
        if (options.controller.signal.aborted) throw new Error("Dimension update interrupted");
        await checkUnchanged(formal);
        const old = previous && previous.locked_manifest_sha256 === manifestHash ? previous.regions.find(r => r.region_id === region.region_id) : undefined;
        if (region.view_type === "section" || region.selected!.view_spec.section_offset_fraction !== undefined || region.selected!.view_spec.section_keep !== undefined) {
          state.regions.push({ region_id: region.region_id, status: "unsupported", reason: "Section-generated topology is not yet mapped to dimension entities; use the original aligned comparison.", dimensions: [], removed_dimensions: (old?.dimensions ?? []).map(d => ({ dimension_id: d.dimension_id, reason: "Section topology unsupported" })) });
          await persist(state);
          continue;
        }
        const viewRoot = join(outputRoot, region.region_id);
        const frame = await (options.makeFrame ?? currentFrame)({ ...options, manifest: options.lockedManifest, formal, regionId: region.region_id, outputRoot: viewRoot });
        const crop = await inspectPngFile(options.workspace, frame.referencePath);
        let catalogSequence = 0;
        let measureSequence = 0;
        const prepare = async (kind: "edge" | "face" | "all", geometry?: string): Promise<PreparedView> => {
          await checkUnchanged(formal);
          const response = await options.worker.request<PreparedView>(options.workspace, {
            protocol_version: PROTOCOL_VERSION, operation: "prepare_dimension_view",
            step_path: "output.step", expected_step_sha256: formal.stepSha256, region_id: region.region_id,
            reference_path: frame.referencePath, view_spec: frame.viewSpec,
            projection_origin: frame.projectionOrigin, transform: frame.transform, image_size: [crop.width, crop.height],
            kind, ...(geometry ? { geometry_type: geometry } : {}),
            output_dir: join(viewRoot, `catalog-${++catalogSequence}`),
          }, { signal: options.controller.signal, timeoutMs: 240_000 });
          requireDrawingSuccess(response, "prepare_dimension_view");
          if (response.step_sha256 !== formal.stepSha256 || response.region_id !== region.region_id) throw new InfrastructureError("Dimension catalog belongs to the wrong STEP/view");
          for (const p of [response.session_path, response.entities_path, ...response.numbered_paths]) safePath(p);
          await checkUnchanged(formal);
          return response;
        };
        let catalog = await prepare("edge", "CIRCLE");
        const catalogs = new Map<string, PreparedView>([["edge:CIRCLE", catalog]]);
        let last: { choices: DimensionChoice[]; measured: MeasuredView } | undefined;
        let submitted: RegionRecord | null = null;
        const tools = [
          defineTool({
            name: "inspect_dimension_entities", label: "Inspect dimension entities",
            description: "Show numbered current CAD edges/faces and their exact geometry. Change kind/type or page to find appropriate entities. entity_ids optionally retrieves specific current entity records. Does not change CAD or camera.",
            parameters: catalogSchema, executionMode: "sequential",
            async execute(_id, args) {
              if (!Value.Check(catalogSchema, args)) throw new Error("Invalid catalog selection");
              await checkUnchanged(formal);
              const key = `${args.kind}:${args.geometry_type ?? ""}`;
              catalog = catalogs.get(key) ?? await prepare(args.kind, args.geometry_type);
              catalogs.set(key, catalog);
              const pages = (await readJson<{pages: Array<{ entity_ids: string[] }> }>(safePath(catalog.entities_path))).pages;
              const page = (args.page ?? 1)-1;
              if (!pages[page] || !catalog.numbered_paths[page]) throw new Error(`Catalog has ${pages.length} pages`);
              const ids = args.entity_ids ?? pages[page]!.entity_ids;
              if (ids.some(id => !catalog.objects[id])) throw new Error("Entity ID is not in the current STEP catalog");
              return { content: [
                { type: "text" as const, text: JSON.stringify({ step_sha256: formal.stepSha256, page:page+1, page_count:pages.length, objects:ids.map(id => catalog.objects[id]) }) },
                await image(catalog.numbered_paths[page]!),
              ], details: { step_sha256:formal.stepSha256, entity_ids:ids } };
            },
          }),
          defineTool({
            name: "measure_dimensions", label: "Measure and preview dimensions",
            description: "Replace the proposed set for this view with all measurable reference-drawing dimensions; no fixed dimension-count limit. Returns exact numeric results plus selection and annotation images. dimension_id remains stable for the same intended drawing dimension; entity_ids must be selected again after STEP changes.",
            parameters: MEASURE_DIMENSIONS_SCHEMA, executionMode: "sequential",
            async execute(_id, args) {
              if (!Value.Check(MEASURE_DIMENSIONS_SCHEMA, args)) throw new Error("Invalid measurement request");
              if (new Set(args.measurements.map(m=>m.dimension_id)).size !== args.measurements.length) throw new Error("dimension_id values must be unique");
              if (args.measurements.some(m=>!m.description.trim() || m.entity_ids.some(id=>!catalog.objects[id]))) throw new Error("Choose known current entities and a nonblank dimension description");
              await checkUnchanged(formal);
              const response = await options.worker.request<MeasuredView>(options.workspace, {
                protocol_version: PROTOCOL_VERSION, operation: "measure_dimension_view",
                session_path: catalog.session_path, expected_step_sha256: formal.stepSha256,
                requests: args.measurements.map(({dimension_id: _key, description: _description, ...request}) => request),
                output_dir: join(viewRoot, `measurement-${++measureSequence}`),
              }, { signal: options.controller.signal, timeoutMs: 240_000 });
              requireDrawingSuccess(response, "measure_dimension_view");
              if (response.step_sha256 !== formal.stepSha256 || response.results.length !== args.measurements.length) throw new InfrastructureError("Dimension measurement returned stale/incomplete results");
              const selection = await image(response.selection_path, response.artifact_sha256[response.selection_path]);
              const comparison = await image(response.comparison_path, response.artifact_sha256[response.comparison_path]);
              await checkUnchanged(formal);
              last = { choices:structuredClone(args.measurements), measured:response };
              return { content: [{ type:"text" as const,text:JSON.stringify(response.results) }, selection, comparison], details:response };
            },
          }),
          defineTool({
            name: "submit_dimension_annotations", label: "Finish dimension review",
            description: "Publish the last successful measurement set, or report that the reference has no annotated dimensions / itemize every unresolved reference dimension in removed_dimensions (including first-time unmeasurable items). Account for every prior dimension_id by retaining it in the published set or listing its removal with a reason.",
            parameters:SUBMIT_DIMENSIONS_SCHEMA, executionMode:"sequential",
            async execute(_id,args) {
              if (!Value.Check(SUBMIT_DIMENSIONS_SCHEMA,args) || !args.reason.trim() || args.removed_dimensions.some(d=>!d.reason.trim())) throw new Error("Provide a status and explicit nonblank reasons");
              if (args.status === "annotated" && !last) throw new Error("Measure the current selections before submitting annotated");
              const hasMeasurements = args.status !== "not_needed" && last !== undefined;
              const dimensions = hasMeasurements ? last!.choices : [];
              const retained = new Set(dimensions.map(d=>d.dimension_id));
              const removed = new Set(args.removed_dimensions.map(d=>d.dimension_id));
              if (removed.size !== args.removed_dimensions.length || [...removed].some(id=>retained.has(id))) throw new Error("Removed dimensions must be unique and cannot also be retained");
              if ((old?.dimensions ?? []).some(d=>!retained.has(d.dimension_id) && !removed.has(d.dimension_id))) throw new Error("Reconcile every previous dimension: remeasure it or state why it is removed");
              await checkUnchanged(formal);
              submitted = {
                region_id:region.region_id,status:args.status,reason:args.reason,dimensions,
                removed_dimensions:args.removed_dimensions,
                ...(hasMeasurements ? {
                  comparison_path:last!.measured.comparison_path,
                  comparison_sha256:(await inspectPngFile(options.workspace,last!.measured.comparison_path)).sha256,
                  measurements_path:last!.measured.measurements_path, values:last!.measured.results,
                } : {}),
              };
              await writeJsonAtomic(safePath(join(viewRoot,"result.json")),submitted);
              return {content:[{type:"text" as const,text:"Dimension review submitted for this current STEP."}],details:submitted};
            },
          }),
        ];
        const firstPage = (await readJson<{pages:Array<{entity_ids:string[]}>}>(safePath(catalog.entities_path))).pages[0];
        await (options.runSession ?? runRestrictedSession)({
          role:"dimension_annotation",regionId:region.region_id,workspace:options.workspace,outputDir:viewRoot,
          resolvedModel:options.resolvedModel,thinkingLevel:options.job.thinking_level,controller:options.controller,
          systemPrompt:DIMENSION_SUBAGENT_PROMPT,
          prompt:`View ${region.region_id}; current STEP ${formal.stepSha256}. Trigger: ${trigger}. Camera and image frame are frozen.\nCurrent first-page entities: ${JSON.stringify((firstPage?.entity_ids ?? []).map(id=>catalog.objects[id]))}\nPrevious dimensions (IDs are obsolete if STEP differs): ${JSON.stringify(old ?? null)}\nPrevious STEP: ${previous?.step_sha256 ?? "none"}. Inspect the target drawing, current numbered image and exact geometry; inventory every dimension annotated in the reference view and measure all of them. Do not sample a compact subset. Account for each unmeasurable reference dimension individually in removed_dimensions with a specific reason; report unresolved if any remain. Submit annotated only for complete measured coverage.`,
          images:[await image(catalog.reference_path),await image(catalog.numbered_paths[0]!)],
          toolNames:DIMENSION_SUBAGENT_TOOLS,tools,getSubmitted:()=>submitted,
          activeTimeoutMs:8*60_000,wallTimeoutMs:15*60_000,
          pauseActiveTimeoutForTools:["inspect_dimension_entities","measure_dimensions"],
          resultPath:join(viewRoot,"result.json"),onUsage:options.onUsage,signal:options.controller.signal,
        });
        if (!submitted) throw new InfrastructureError("Dimension SubAgent ended without a submitted review");
        await checkUnchanged(formal);
        state.regions.push(submitted);
        await persist(state);
      }
      await checkUnchanged(formal);
      state.status="completed";
      await persist(state);
      await writeJsonAtomic(safePath(lastCompletedPath),state);
      return state;
    } catch (error) {
      state.status="failed";
      state.error=error instanceof Error ? error.message : String(error);
      await persist(state);
      throw error;
    }
  }
  return {
    async ensureCurrent(trigger:string):Promise<DimensionState> {
      if (!pending) pending=update(trigger).finally(()=>{pending=undefined;});
      return pending;
    },
    async comparison(regionId:string):Promise<{content:Array<TextContent|ImageContent>;state:DimensionState}> {
      const state=await this.ensureCurrent("inspect_comparison");
      const region=state.regions.find(r=>r.region_id===regionId);
      if (!region) throw new InfrastructureError("Dimension review missing the requested view");
      // Coverage status, reasons and unresolved-item reports remain audit-only.
      const content:Array<TextContent|ImageContent>=[];
      if (region.comparison_path) {
        content.push(await image(region.comparison_path!,region.comparison_sha256));
        content.push({type:"text",text:JSON.stringify({dimensions:region.dimensions,measurements:region.values})});
      }
      return {content,state};
    },
  };
}
