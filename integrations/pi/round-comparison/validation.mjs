// Read-only acceptance of saved Initial and reflection-round evidence.
// Keep imports free of compiled Pi modules so --setup works on a fresh clone.
import { createHash } from "node:crypto";
import { lstat, readFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { digestFile, safeFile, validateFormalWorkspace } from "../portable-full-method/run.mjs";

const dimensionProfile = "aligned-dimension-subagent-v1";
const forbiddenResultFields = [
  "drawing_feedback", "main_view_reflection", "dimension_annotations", "causal_repair",
  "subagent_usage", "subagent_runs", "recorded_subagent_usage_ids",
];
const sha = value => createHash("sha256").update(value).digest("hex");
const require = (condition, message) => { if (!condition) throw Error(message); };

async function reflectionModules() {
  const [reflection, continuation] = await Promise.all([
    import("../dist/src/main-self-reflect.js"),
    import("../dist/src/main-self-reflect-continuation-cli.js"),
  ]);
  return { ...reflection, selfReflectCompletionAccepted: continuation.selfReflectCompletionAccepted };
}

async function requireNoAuxiliary(workspace, result) {
  const unexpected = forbiddenResultFields.filter(field => result[field] !== undefined);
  require(unexpected.length === 0, `Unexpected auxiliary state: ${unexpected.join(", ")}`);
  require(!(await lstat(join(workspace, "drawing_feedback")).catch(() => null)), "Unexpected drawing_feedback in Initial/Vanilla");
}

async function boundJson(workspace, path) {
  const file = await safeFile(workspace, path);
  const bytes = await readFile(file);
  return { value: JSON.parse(bytes.toString("utf8")), sha256: sha(bytes) };
}

function requireHashPair(value, expected, label) {
  require(value.source_sha256 === expected.source && value.step_sha256 === expected.step, `${label} source/STEP binding mismatch`);
}

/** Only a pristine, formally executed Main generation may be a shared Initial. */
export async function validateInitial(workspace, config) {
  const formal = await validateFormalWorkspace(workspace, config);
  require(formal.result.artifact?.status === "valid" && formal.result.artifact.validation?.is_valid === true && formal.result.artifact.validation?.is_watertight === true, "Initial lacks valid/watertight formal artifact evidence");
  const { classifySelfReflectSource } = await reflectionModules();
  require(classifySelfReflectSource(formal.job, formal.result) === "original", "Initial must be an original Main generation, not a continuation");
  require(formal.result.optional_reflect_turn === undefined, "Initial contains prior reflection-turn state");
  await requireNoAuxiliary(workspace, formal.result);
  return formal;
}

/**
 * Validate one requested outer round, not one model call or CAD execution.
 * Geometry validity is inherited from the formal worker's accepted execution;
 * this function checks its recorded hashes and reflection evidence, not GT fit.
 */
export async function validateRound(workspace, config, expected) {
  require(["vanilla", "ours"].includes(expected?.method), "Unknown reflection method");
  require(Number.isInteger(expected.round) && expected.round >= 1, "Invalid reflection round");
  require(expected.method !== "vanilla" || expected.round === 1, "Historical Vanilla supports exactly one reflection round");
  const formal = await validateFormalWorkspace(workspace, config);
  const { job, result, source, step } = formal;
  const modules = await reflectionModules();
  require(modules.classifySelfReflectSource(job, result) === "main_self_reflect", "Round lacks a verifiably applied Main continuation");
  const continuation = job.continuation;
  const before = expected.predecessor;
  const initial = expected.initial;
  require(before && initial && Number.isInteger(before.last_turn) && before.last_turn >= 0, "Missing predecessor/Initial execution binding");
  requireHashPair({ source_sha256: continuation.baseline_source_sha256, step_sha256: continuation.baseline_step_sha256 }, before, "Round predecessor");
  if (expected.round === 1) {
    require(before.source === initial.source && before.step === initial.step, "Round 1 must start from shared Initial");
    require(continuation.previous_continuation_id === undefined, "Round 1 unexpectedly imports a previous continuation");
  } else {
    require(typeof before.continuation_id === "string" && before.continuation_id.length > 0, "Chained round lacks predecessor continuation ID");
    require(continuation.previous_continuation_id === before.continuation_id, "Round links to the wrong predecessor continuation");
    require(continuation.id !== before.continuation_id, "Each requested round needs a new continuation ID");
  }
  if (before.workspace !== undefined) {
    const baseline = await boundJson(workspace, "baseline/manifest.json");
    require(typeof baseline.value.source_workspace === "string" && baseline.value.source_workspace.length > 0 && resolve(baseline.value.source_workspace) === resolve(before.workspace), "CLI baseline points to the wrong predecessor workspace");
    requireHashPair(baseline.value, before, "CLI baseline");
  }
  require(Number.isInteger(result.last_turn) && result.last_turn >= before.last_turn, "Round execution counter moved backwards");
  const additionalExecutions = result.last_turn - before.last_turn;
  const promptSha256 = sha(continuation.prompt);
  const after = {
    source_sha256: source, step_sha256: step, formal_executions_added: additionalExecutions,
    is_valid: result.artifact?.validation?.is_valid === true,
    is_watertight: result.artifact?.validation?.is_watertight === true,
  };
  require(modules.selfReflectCompletionAccepted(job, result, after), "Reflection round is incomplete or has stale final evidence");
  require(continuation.no_new_images !== true, "This comparison uses the historical visual Vanilla, not the no-new-images variant");
  require((continuation.image_paths ?? []).length === 0, "Unexpected extra continuation images");
  const profiles = Object.keys(continuation).filter(key => key.endsWith("_profile") && continuation[key] !== undefined);

  const record = {
    status: "completed", method: expected.method, round: expected.round, fixture_id: result.fixture_id,
    workspace: resolve(workspace), session_file: result.session_file,
    continuation_id: continuation.id, previous_continuation_id: continuation.previous_continuation_id ?? null,
    prompt_sha256: promptSha256,
    initial_source_sha256: initial.source, initial_step_sha256: initial.step,
    predecessor_source_sha256: before.source, predecessor_step_sha256: before.step,
    source_sha256: source, step_sha256: step,
    source_changed: source !== before.source, step_changed: step !== before.step,
    unchanged: source === before.source && step === before.step,
    last_turn: result.last_turn, formal_executions_added: additionalExecutions,
    is_valid: true, is_watertight: true, geometry_accuracy_scored: false,
    inspected_regions: [],
  };

  if (expected.method === "vanilla") {
    require(continuation.kind === "main_self_reflect", "Vanilla has an unexpected continuation kind");
    require(continuation.prompt === modules.mainSelfReflectPrompt(result.fixture_id), "Vanilla must preserve the historical Direct Reflect prompt");
    require(profiles.length === 0, "Vanilla cannot enable extra reflection profiles");
    require(!continuation.locked_view_manifest_path && !continuation.locked_view_manifest_sha256 && !continuation.reuse_fixed_view_from_source && !continuation.view_origin_source_sha256 && !continuation.view_origin_step_sha256, "Vanilla cannot import aligned-view state");
    await requireNoAuxiliary(workspace, result);
    // Historical Vanilla is not an Optional Reflect turn: retain its original
    // fresh-execution and changed-source/STEP gate instead of inventing a marker.
    return { formal, record };
  }

  require(continuation.kind === "main_self_reflect_inspect", "Ours must use Main Self Reflect plus Inspect");
  require(profiles.length === 1 && continuation.dimension_annotation_profile === dimensionProfile, "Ours must enable only the dimension-annotation profile");
  require(continuation.prompt === modules.mainSelfReflectInspectPrompt(result.fixture_id, undefined, undefined, undefined, undefined, undefined, undefined, undefined, dimensionProfile), "Ours prompt does not match the dimension-aware Main reflection protocol");
  modules.assertSelfReflectAuxiliaryState(result, undefined, dimensionProfile);
  require(result.causal_repair === undefined, "Unexpected causal-repair auxiliary state");
  require(expected.round === 1 ? !continuation.reuse_fixed_view_from_source : continuation.reuse_fixed_view_from_source === true, "Chained dimension rounds must reuse the original fixed view");
  require(Boolean(continuation.view_origin_source_sha256) === Boolean(continuation.view_origin_step_sha256), "Alignment origin hash pair is incomplete");
  const origin = {
    source_sha256: continuation.view_origin_source_sha256 ?? continuation.baseline_source_sha256,
    step_sha256: continuation.view_origin_step_sha256 ?? continuation.baseline_step_sha256,
  };
  requireHashPair(origin, initial, "Alignment origin");
  require(typeof expected.locked_manifest_sha256 === "string" && continuation.locked_view_manifest_sha256 === expected.locked_manifest_sha256, "Round changed the locked view manifest binding");
  const locked = await boundJson(workspace, continuation.locked_view_manifest_path);
  require(locked.sha256 === expected.locked_manifest_sha256, "Locked view manifest file changed");
  requireHashPair(locked.value, initial, "Locked view origin");
  require(locked.value.projection_frame?.policy === "fixed" && locked.value.projection_frame.anchor_step_sha256 === initial.step, "Alignment frame is not locked to Initial");
  require(await digestFile(await safeFile(workspace, locked.value.projection_frame.anchor_step_path)) === initial.step, "Alignment anchor STEP changed");

  const dimensions = result.dimension_annotations;
  const expectedStatePath = `drawing_feedback/dimension-annotations/${sha(continuation.id).slice(0, 16)}/state.json`;
  require(dimensions?.state_path === expectedStatePath, "Round reused another continuation's dimension namespace");
  const saved = await boundJson(workspace, dimensions.state_path);
  const state = saved.value;
  for (const candidate of [dimensions, state]) {
    require(candidate.profile === dimensionProfile && candidate.status === "completed", "Dimension review did not complete");
    requireHashPair(candidate, { source, step }, "Final dimensions");
    require(candidate.locked_manifest_sha256 === expected.locked_manifest_sha256, "Dimensions belong to another locked view manifest");
  }
  require(state.state_path === dimensions.state_path && typeof state.revision_id === "string" && state.revision_id.length > 0 && state.revision_id === dimensions.revision_id, "Saved dimension revision does not match its summary");
  require(JSON.stringify(state.regions) === JSON.stringify(dimensions.regions), "Saved dimension regions differ from their summary");
  const selectedRegions = (locked.value.regions ?? []).filter(region => region.selected).map(region => region.region_id).sort();
  require(selectedRegions.length > 0 && Array.isArray(state.regions), "Dimension review has no selected aligned regions");
  require(JSON.stringify(state.regions.map(region => region.region_id).sort()) === JSON.stringify(selectedRegions), "Dimension review does not cover the selected aligned regions");
  const counts = {};
  const regionRecords = [];
  for (const region of state.regions) {
    require(["annotated", "not_needed", "unresolved", "unsupported"].includes(region.status), "Unexpected dimension-region status");
    const measured = region.dimensions?.length ?? 0;
    const unresolved = region.removed_dimensions?.length ?? 0;
    if (region.status === "annotated") require(measured > 0 && region.comparison_path && region.comparison_sha256, "Annotated dimension region has no measured evidence");
    if (region.comparison_path) require(await digestFile(await safeFile(workspace, region.comparison_path)) === region.comparison_sha256, "Dimension comparison image changed");
    if (region.values) require(region.values.every(value => value.model_sha256 === undefined || value.model_sha256 === step), "Dimension value belongs to another STEP");
    counts[region.status] = (counts[region.status] ?? 0) + 1;
    regionRecords.push({ region_id: region.region_id, status: region.status, measured_dimension_count: measured, unresolved_dimension_count: unresolved, reason: region.reason ?? null });
  }

  const viewSummary = result.main_view_reflection;
  require(viewSummary?.locked_view_manifest_sha256 === expected.locked_manifest_sha256, "Final comparison summary belongs to another locked view");
  requireHashPair({ source_sha256: viewSummary.baseline_source_sha256, step_sha256: viewSummary.baseline_step_sha256 }, before, "Comparison baseline");
  const inspection = await boundJson(workspace, viewSummary.audit_manifest_path);
  const audit = inspection.value;
  require(audit.fixture_id === result.fixture_id && audit.locked_view_manifest_sha256 === expected.locked_manifest_sha256, "Inspection audit belongs to another fixture/view");
  requireHashPair({ source_sha256: audit.baseline_source_sha256, step_sha256: audit.baseline_step_sha256 }, before, "Inspection audit baseline");
  const seen = (audit.inspections ?? []).filter(row => row.step_sha256 === step);
  require(seen.length > 0, "Final STEP has no actual comparison inspection audit");
  for (const row of seen) {
    require(selectedRegions.includes(row.region_id) && typeof row.tool_call_id === "string" && row.tool_call_id.length > 0, "Final inspection lacks a selected region or successful tool binding");
    require(await digestFile(await safeFile(workspace, row.context_path)) === row.context_sha256, "Final inspection image changed");
  }
  record.inspected_regions = [...new Set(seen.map(row => row.region_id))].sort();
  record.current_step_inspection_count = seen.length;
  record.inspection_audit_path = viewSummary.audit_manifest_path;
  record.inspection_audit_sha256 = inspection.sha256;
  record.locked_manifest_sha256 = locked.sha256;
  record.optional_reflect_turn_status = result.optional_reflect_turn.status;
  record.dimension_annotations = {
    state_path: dimensions.state_path, state_sha256: saved.sha256, revision_id: state.revision_id,
    source_sha256: source, step_sha256: step, locked_manifest_sha256: locked.sha256,
    region_status_counts: counts, regions: regionRecords,
    measured_dimension_count: regionRecords.reduce((sum, region) => sum + region.measured_dimension_count, 0),
    unresolved_dimension_count: regionRecords.reduce((sum, region) => sum + region.unresolved_dimension_count, 0),
    coverage_complete: regionRecords.every(region => region.status === "annotated" || region.status === "not_needed"),
  };
  return { formal, record };
}
