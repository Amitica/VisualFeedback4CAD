import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { cp, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath, pathToFileURL } from "node:url";
import test from "node:test";
import { validateInitial, validateRound } from "./validation.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const sha = value => createHash("sha256").update(value).digest("hex");
const profile = "aligned-dimension-subagent-v1";
const config = { provider: "ucloud", model: "gpt-5.6-sol", thinking_level: "low" };
const writeJson = async (file, value) => { await mkdir(dirname(file), { recursive: true }); await writeFile(file, JSON.stringify(value, null, 2) + "\n"); };

async function fixture(t, { method = "ours", round = 1, original = false, unchanged = false, regionStatus = "annotated" } = {}) {
  const root = await mkdtemp(join(tmpdir(), "cad-round-validation-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const workspace = join(root, "140");
  await mkdir(join(workspace, "sessions"), { recursive: true });
  await writeFile(join(workspace, "sessions/main.jsonl"), "synthetic saved-session fixture\n");
  const initialCode = "initial source\n", initialStep = "initial STEP bytes\n";
  const beforeCode = round === 1 ? initialCode : "previous-round source\n";
  const beforeStep = round === 1 ? initialStep : "previous-round STEP bytes\n";
  const code = original || unchanged ? beforeCode : `${method} round ${round} source\n`;
  const stepBytes = original || unchanged ? beforeStep : `${method} round ${round} STEP bytes\n`;
  await writeFile(join(workspace, "solution.py"), code);
  await writeFile(join(workspace, "output.step"), stepBytes);
  const source = sha(code), step = sha(stepBytes);
  const initial = { source: sha(initialCode), step: sha(initialStep) };
  const predecessor = { source: sha(beforeCode), step: sha(beforeStep), last_turn: 3, ...(round > 1 ? { continuation_id: "ours-round-1" } : {}) };
  const job = { ...config, agent_mode: "base-only", fixture: { id: "140" } };
  const result = {
    ...config, fixture_id: "140", agent_mode: "base-only", status: "valid", session_file: "sessions/main.jsonl",
    last_turn: original || unchanged ? 3 : 4,
    last_executed_source_sha256: source, last_executed_step_sha256: step,
    artifact: { status: "valid", sha256: step, validation: { is_valid: true, is_watertight: true } },
  };
  const expected = { method, round, predecessor, initial };
  if (!original) {
    const prompts = await import("../dist/src/main-self-reflect.js");
    const prompt = method === "ours"
      ? prompts.mainSelfReflectInspectPrompt("140", undefined, undefined, undefined, undefined, undefined, undefined, undefined, profile)
      : prompts.mainSelfReflectPrompt("140");
    const id = `${method}-round-${round}`;
    job.continuation = {
      kind: method === "ours" ? "main_self_reflect_inspect" : "main_self_reflect", id, prompt, image_paths: [],
      baseline_source_sha256: predecessor.source, baseline_step_sha256: predecessor.step,
      ...(round > 1 ? { previous_continuation_id: predecessor.continuation_id } : {}),
    };
    result.applied_continuation = { id, prompt_sha256: sha(prompt) };
    if (method === "ours") {
      const manifestPath = "drawing_feedback/locked/view_manifest.json";
      const anchorPath = "drawing_feedback/locked/anchor.step";
      await mkdir(dirname(join(workspace, anchorPath)), { recursive: true });
      await writeFile(join(workspace, anchorPath), initialStep);
      const locked = {
        version: 3, source_sha256: initial.source, step_sha256: initial.step,
        projection_frame: { policy: "fixed", anchor_step_path: anchorPath, anchor_step_sha256: initial.step },
        regions: [{ region_id: "front", selected: { candidate_id: "fixed-camera" } }],
      };
      await writeJson(join(workspace, manifestPath), locked);
      const lockedSha = sha(await readFile(join(workspace, manifestPath)));
      expected.locked_manifest_sha256 = lockedSha;
      Object.assign(job.continuation, {
        dimension_annotation_profile: profile, locked_view_manifest_path: manifestPath, locked_view_manifest_sha256: lockedSha,
        ...(round > 1 ? { reuse_fixed_view_from_source: true, view_origin_source_sha256: initial.source, view_origin_step_sha256: initial.step } : {}),
      });
      result.optional_reflect_turn = { continuation_id: id, prompt_sha256: sha(prompt), status: "completed" };
      const statePath = `drawing_feedback/dimension-annotations/${sha(id).slice(0, 16)}/state.json`;
      const comparisonPath = `${dirname(statePath)}/revision-${round}/comparison.png`;
      await mkdir(dirname(join(workspace, comparisonPath)), { recursive: true });
      await writeFile(join(workspace, comparisonPath), "synthetic measured comparison bytes");
      result.dimension_annotations = {
        version: 1, profile, state_path: statePath, revision_id: `revision-${round}`, status: "completed",
        source_sha256: source, step_sha256: step, locked_manifest_sha256: lockedSha, trigger: "before Main reflection",
        regions: [{
          region_id: "front", status: regionStatus, reason: "Unit fixture for saved measurement evidence",
          dimensions: [{ dimension_id: "bore", quantity: "diameter", entity_ids: ["E001"] }],
          removed_dimensions: regionStatus === "unresolved" ? [{ dimension_id: "depth", reason: "Unsupported quantity" }] : [],
          values: [{ value: 12, model_sha256: step }],
          comparison_path: comparisonPath, comparison_sha256: sha(await readFile(join(workspace, comparisonPath))),
        }],
      };
      await writeJson(join(workspace, statePath), result.dimension_annotations);
      const auditPath = "drawing_feedback/main-view-reflect/audit.json";
      const contextPath = "drawing_feedback/main-view-reflect/current.png";
      await mkdir(dirname(join(workspace, contextPath)), { recursive: true });
      await writeFile(join(workspace, contextPath), "synthetic final STEP comparison bytes");
      const audit = {
        version: 1, kind: "main_view_reflect", fixture_id: "140", locked_view_manifest_sha256: lockedSha,
        baseline_source_sha256: predecessor.source, baseline_step_sha256: predecessor.step,
        inspections: [{ inspection_id: "inspection-1", tool_call_id: "inspect-tool-1", region_id: "front", source_sha256: source, step_sha256: step, context_path: contextPath, context_sha256: sha(await readFile(join(workspace, contextPath))) }],
      };
      await writeJson(join(workspace, auditPath), audit);
      result.main_view_reflection = {
        version: 1, kind: "main_view_reflect", locked_view_manifest_sha256: lockedSha,
        baseline_source_sha256: predecessor.source, baseline_step_sha256: predecessor.step,
        audit_manifest_path: auditPath, inspection_count: 1, inspected_formal_states: [{ source_sha256: source, step_sha256: step }],
      };
    }
  }
  const save = async () => Promise.all([
    writeJson(join(workspace, ".cadgenbench-pi-job.json"), job),
    writeJson(join(workspace, "agent_result.json"), result),
  ]);
  await save();
  return { root, workspace, job, result, expected, save };
}

test("module import works in a fresh source tree without dist or npm dependencies", async t => {
  const root = await mkdtemp(join(tmpdir(), "cad-validation-cold-import-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const target = join(root, "integrations/pi/round-comparison/validation.mjs");
  const portable = join(root, "integrations/pi/portable-full-method/run.mjs");
  await mkdir(dirname(target), { recursive: true });
  await mkdir(dirname(portable), { recursive: true });
  await cp(join(here, "validation.mjs"), target);
  await cp(join(here, "../portable-full-method/run.mjs"), portable);
  const proc = spawnSync(process.execPath, ["--input-type=module", "--eval", "await import(process.argv[1]);", pathToFileURL(target).href], { encoding: "utf8" });
  assert.equal(proc.status, 0, proc.stderr);
});

test("Initial requires a pristine original Main, not a completed continuation", async t => {
  const f = await fixture(t, { original: true });
  assert.equal((await validateInitial(f.workspace, config)).source, f.expected.initial.source);
  const reflected = await fixture(t, { method: "vanilla" });
  await assert.rejects(validateInitial(reflected.workspace, config), /original Main generation/);
});

test("Initial rejects auxiliary summaries and stray drawing-feedback directories", async t => {
  const f = await fixture(t, { original: true });
  f.result.dimension_annotations = {};
  await f.save();
  await assert.rejects(validateInitial(f.workspace, config), /auxiliary state/);
  delete f.result.dimension_annotations;
  await f.save();
  await mkdir(join(f.workspace, "drawing_feedback"));
  await assert.rejects(validateInitial(f.workspace, config), /drawing_feedback/);
});

test("Initial status alone cannot substitute for valid/watertight worker evidence", async t => {
  const f = await fixture(t, { original: true });
  f.result.artifact.validation.is_watertight = false;
  await f.save();
  await assert.rejects(validateInitial(f.workspace, config), /valid\/watertight/);
});

test("historical Vanilla keeps its actual prompt and changed-source/STEP gate without Optional Reflect metadata", async t => {
  const f = await fixture(t, { method: "vanilla" });
  const { record } = await validateRound(f.workspace, config, f.expected);
  assert.equal(record.method, "vanilla");
  assert.equal(record.formal_executions_added, 1);
  assert.equal(record.unchanged, false);
  assert.equal(f.result.optional_reflect_turn, undefined);
  const noOp = await fixture(t, { method: "vanilla", unchanged: true });
  await assert.rejects(validateRound(noOp.workspace, config, noOp.expected), /incomplete/);
});

test("Vanilla rejects no-new-images variants, extra profiles, and additional rounds", async t => {
  const f = await fixture(t, { method: "vanilla" });
  f.job.continuation.no_new_images = true;
  await f.save();
  await assert.rejects(validateRound(f.workspace, config, f.expected), /incomplete|no-new-images/);
  delete f.job.continuation.no_new_images;
  f.job.continuation.dimension_annotation_profile = profile;
  await f.save();
  await assert.rejects(validateRound(f.workspace, config, f.expected), /extra reflection profiles/);
  await assert.rejects(validateRound(f.workspace, config, { ...f.expected, round: 2 }), /exactly one/);
});

test("Ours reports measured and unresolved coverage separately from round completion", async t => {
  const f = await fixture(t, { regionStatus: "unresolved" });
  const { record } = await validateRound(f.workspace, config, f.expected);
  assert.deepEqual(record.inspected_regions, ["front"]);
  assert.equal(record.dimension_annotations.measured_dimension_count, 1);
  assert.equal(record.dimension_annotations.unresolved_dimension_count, 1);
  assert.equal(record.dimension_annotations.coverage_complete, false);
  assert.deepEqual(record.dimension_annotations.region_status_counts, { unresolved: 1 });
  assert.equal(record.geometry_accuracy_scored, false);
});

test("Ours accepts a completed no-op second round only with a new continuation and dimension namespace", async t => {
  const f = await fixture(t, { round: 2, unchanged: true });
  const { record } = await validateRound(f.workspace, config, f.expected);
  assert.equal(record.previous_continuation_id, "ours-round-1");
  assert.equal(record.formal_executions_added, 0);
  assert.equal(record.unchanged, true);
  assert.equal(record.dimension_annotations.revision_id, "revision-2");
  f.result.dimension_annotations.state_path = "drawing_feedback/dimension-annotations/old-round/state.json";
  await f.save();
  await assert.rejects(validateRound(f.workspace, config, f.expected), /namespace/);
});

test("rounds reject wrong predecessor links and wrong alignment origin", async t => {
  const f = await fixture(t, { round: 2 });
  f.job.continuation.previous_continuation_id = "other-predecessor";
  await f.save();
  await assert.rejects(validateRound(f.workspace, config, f.expected), /wrong predecessor/);
  f.job.continuation.previous_continuation_id = f.expected.predecessor.continuation_id;
  f.job.continuation.view_origin_step_sha256 = sha("wrong origin");
  await f.save();
  await assert.rejects(validateRound(f.workspace, config, f.expected), /Alignment origin/);
});

test("Ours cannot finish with an incomplete reflection turn or stale saved dimensions", async t => {
  const f = await fixture(t);
  f.result.optional_reflect_turn.status = "running";
  await f.save();
  await assert.rejects(validateRound(f.workspace, config, f.expected), /incomplete/);
  f.result.optional_reflect_turn.status = "completed";
  await f.save();
  const file = join(f.workspace, f.result.dimension_annotations.state_path);
  const state = JSON.parse(await readFile(file, "utf8"));
  state.step_sha256 = sha("old STEP");
  await writeJson(file, state);
  await assert.rejects(validateRound(f.workspace, config, f.expected), /Final dimensions/);
});

test("claimed final inspection does not replace the actual final-STEP audit", async t => {
  const f = await fixture(t);
  const file = join(f.workspace, f.result.main_view_reflection.audit_manifest_path);
  const audit = JSON.parse(await readFile(file, "utf8"));
  audit.inspections[0].step_sha256 = sha("another STEP");
  await writeJson(file, audit);
  await assert.rejects(validateRound(f.workspace, config, f.expected), /no actual comparison inspection/);
});

test("saved measurement images are verified even when region coverage is unresolved", async t => {
  const f = await fixture(t, { regionStatus: "unresolved" });
  const region = f.result.dimension_annotations.regions[0];
  await writeFile(join(f.workspace, region.comparison_path), "changed image bytes");
  await assert.rejects(validateRound(f.workspace, config, f.expected), /comparison image changed/);
});

test("Round 1 must start at Initial and cannot pass stale formal source/STEP binding", async t => {
  const f = await fixture(t);
  await assert.rejects(validateRound(f.workspace, config, { ...f.expected, initial: { ...f.expected.initial, source: sha("other Initial") } }), /shared Initial/);
  await writeFile(join(f.workspace, "solution.py"), "unexecuted source edit");
  await assert.rejects(validateRound(f.workspace, config, f.expected), /hashes are stale/);
});
