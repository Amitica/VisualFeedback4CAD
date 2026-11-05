import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";

import { loadLockedMainViewManifest } from "../dist/src/main-view-reflect.js";
import {
  buildPlan,
  casePaths,
  digestFile,
  freezeAutomaticViews,
  forEachCaseBounded,
  readJson,
  validateConfig,
  validateFormalWorkspace,
} from "./run.mjs";

const config = {
  experiment_id: "gpt56-sol-low-full-method-test",
  provider: "ucloud", model: "gpt-5.6-sol", thinking_level: "low",
  cases: ["145", "146"], image: "cadgenbench-pi-agent:full-method-test",
  data_inputs_dir: "/data/inputs", generation_timeout_minutes: 120,
  view_search_timeout_minutes: 180, repair_timeout_minutes: 120,
  search_cpus: 2, search_memory: "6g",
};
const sha = value => createHash("sha256").update(value).digest("hex");
const png = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l9sAAAAASUVORK5CYII=", "base64");
const flag = (command, name) => command[command.indexOf(name) + 1];

test("runs each case once within its configured concurrency limit", async () => {
  let active = 0;
  let peak = 0;
  const visited = [];
  const complete = await forEachCaseBounded([1, 2, 3, 4, 5, 6], 3, async id => {
    active++;
    peak = Math.max(peak, active);
    visited.push(id);
    await new Promise(resolve => setTimeout(resolve, 5));
    active--;
    return true;
  });
  assert.equal(complete, true);
  assert.equal(peak, 3);
  assert.deepEqual(visited.sort(), [1, 2, 3, 4, 5, 6]);
});

test("does not start further cases after an explicit interruption", async () => {
  const visited = [];
  assert.equal(await forEachCaseBounded([1, 2, 3], 1, async id => {
    visited.push(id);
    return false;
  }), false);
  assert.deepEqual(visited, [1]);
});

test("bounds case concurrency and subagent timeout scaling without changing old defaults", () => {
  assert.equal(validateConfig({ ...config }).thinking_level, "low");
  assert.equal(validateConfig({ ...config, case_parallelism: 3, subagent_timeout_multiplier: 3 }).case_parallelism, 3);
  assert.equal(validateConfig({ ...config, case_parallelism: 12 }).case_parallelism, 12);
  for (const value of [0, 65, 1.5]) assert.throws(() => validateConfig({ ...config, case_parallelism: value }), /case_parallelism/u);
  for (const value of [0, 13, "3"]) assert.throws(() => validateConfig({ ...config, subagent_timeout_multiplier: value }), /subagent_timeout_multiplier/u);
});

async function put(root, path, value) {
  await mkdir(dirname(join(root, path)), { recursive: true });
  await writeFile(join(root, path), typeof value === "string" || Buffer.isBuffer(value) ? value : `${JSON.stringify(value, null, 2)}\n`);
}

function candidate(id, section = false) {
  return {
    candidate_id: id,
    render_path: `drawing_feedback/search/${id}-render.png`,
    aligned_render_path: `drawing_feedback/search/${id}-aligned.png`,
    comparison_path: `drawing_feedback/search/${id}-comparison.png`,
    view_spec: {
      name: id, direction: [0, 0, id.endsWith("alternate") ? -1 : 1], up: [0, 1, 0],
      ...(section ? { section_offset_fraction: 0.15, section_keep: "positive" } : {}),
    },
    score: 0.72, raw_score: 0.72, eligible: true, failed_gates: [],
    scoring_mode: "capacity_transport", capacity_transport: { version: 1 },
    transform: {
      scale_px_per_unit: 2.3, rotation_deg: 1.5,
      center_x_px: 143, center_y_px: 217, model_center_x: 4.2, model_center_y: -3.6,
      reflected: false,
    },
    alignment: {
      objective: 0.72, mode: "capacity_transport",
      scale_at_bound: false, rotation_at_limit: false, translation_at_limit: false,
    },
  };
}

async function fixture(t) {
  const root = await mkdtemp(join(tmpdir(), "cad-portable-full-method-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const source = join(root, "source", "146");
  const view = join(root, "views", "146");
  const sourceText = "from build123d import Box\nresult = Box(10, 20, 30)\n";
  const stepText = "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n";
  const job = {
    operation: "run_fixture", fixture: { id: "146" }, agent_mode: "base-only",
    provider: config.provider, model: config.model, thinking_level: config.thinking_level,
  };
  const result = {
    fixture_id: "146", status: "valid", agent_mode: "base-only",
    provider: config.provider, model: config.model, thinking_level: config.thinking_level,
    last_executed_source_sha256: sha(sourceText), last_executed_step_sha256: sha(stepText),
    artifact: { status: "valid", sha256: sha(stepText), path: "output.step" },
    session_file: "sessions/main-session.jsonl",
  };
  const manifestPath = "drawing_feedback/search/view_manifest.json";
  const manifest = {
    version: 3, scorer_profile: "capacity_transport_v1", input_sha256: sha(png),
    source_sha256: sha(sourceText), step_sha256: sha(stepText),
    regions: [false, true].map(section => ({
      region_id: section ? "section" : "front", page: 0, roi: [0, 0, 500, 500],
      crop_path: `drawing_feedback/search/${section ? "section" : "front"}-crop.png`,
      view_type: section ? "section" : "orthographic", status: "matched",
      selected: candidate(section ? "section-selected" : "front-selected", section),
      alternatives: [candidate(section ? "section-alternate" : "front-alternate", section)],
      margin: 0.1, confidence: "high", note: "Synthetic automatic search result",
      width: 1, height: 1, crop_sha256: sha(png),
    })),
  };
  const searchResult = {
    status: "completed", fixture_id: "146", source_workspace: source,
    reflection_agent_started: false, agent_roles: ["segmentation", "view_search"],
    source_sha256: sha(sourceText), step_sha256: sha(stepText), manifest_path: manifestPath,
  };
  await Promise.all([
    put(source, ".cadgenbench-pi-job.json", job), put(source, "agent_result.json", result),
    put(source, "solution.py", sourceText), put(source, "output.step", stepText),
    put(source, "sessions/main-session.jsonl", "{\"type\":\"synthetic-session\"}\n"),
    put(view, "solution.py", sourceText), put(view, "output.step", stepText),
    put(view, manifestPath, manifest), put(view, "view_search_only_result.json", searchResult),
    ...manifest.regions.flatMap(region => [region.crop_path, ...[region.selected, ...region.alternatives].flatMap(item => [item.render_path, item.aligned_render_path, item.comparison_path])]).map(path => put(view, path, png)),
  ]);
  return { root, source, view, job, result, manifestPath, manifest, searchResult, sourceText, stepText };
}

test("plans base-only low Main generation, automatic search, then dimension-assisted continuation", () => {
  validateConfig(config);
  const output = "/tmp/cad portable/full run";
  const auth = "/tmp/pi credentials";
  const plans = buildPlan(config, output, auth, ["UCLOUD_API_KEY"]);
  assert.deepEqual(plans.map(plan => plan.fixture_id), config.cases);
  for (const plan of plans) {
    assert.deepEqual(plan.paths, casePaths(output, plan.fixture_id));
    assert.ok(plan.generate.includes("--base-agent-only"));
    assert.equal(flag(plan.generate, "--thinking-level"), "low");
    assert.equal(flag(plan.generate, "--provider"), "ucloud");
    assert.equal(flag(plan.generate, "--model"), "gpt-5.6-sol");
    assert.equal(flag(plan.generate, "--parallel"), "1");
    assert.equal(flag(plan.generate, "--output-dir"), plan.paths.generation);
    assert.equal(plan.search[0], "docker");
    assert.ok(plan.search.includes(`type=bind,src=${output},dst=${output}`));
    assert.ok(plan.search.includes(`type=bind,src=${auth},dst=/pi-agent`));
    assert.equal(flag(plan.search, "--workdir"), output);
    assert.ok(plan.search.includes("UCLOUD_API_KEY"));
    const searchCli = plan.search.indexOf("/opt/cadgenbench-pi/dist/src/drawing-search-cli.js");
    assert.equal(plan.search[searchCli + 1], plan.paths.source);
    // The search CLI inherits provider/model/thinking from this same Main job.
    assert.ok(!plan.search.includes("--provider"));
    assert.ok(!plan.search.includes("--model"));
    assert.ok(!plan.search.includes("--thinking-level"));
    assert.equal(flag(plan.search, "--output-dir"), plan.paths.search);
    assert.equal(flag(plan.repair, "--thinking-level"), "low");
    assert.ok(plan.repair.includes(plan.paths.source));
    assert.equal(flag(plan.repair, "--inspect-view-search-root"), plan.paths.search);
    assert.equal(flag(plan.repair, "--dimension-annotation-profile"), "aligned-dimension-subagent-v1");
    assert.ok(!plan.repair.includes("--visual-review-profile"));
    assert.doesNotMatch([...plan.generate, ...plan.search, ...plan.repair].join("\n"), /manual-calibrated|calibrated-view-search|shape-feedback|repair-geometry/u);
  }
});

test("rejects Docker mount paths containing delimiters", () => {
  for (const path of ["/tmp/output,invalid", "/tmp/output\ninvalid"]) {
    assert.throws(() => buildPlan(config, path, "/tmp/auth"), /Docker mount paths/u);
    assert.throws(() => buildPlan(config, "/tmp/output", path), /Docker mount paths/u);
  }
});

test("locks automatic views without changing cameras, sections, transforms, originals, or CAD", async t => {
  const f = await fixture(t);
  const originalManifest = await readFile(join(f.view, f.manifestPath));
  const originalResult = await readFile(join(f.view, "view_search_only_result.json"));
  const originalSession = await readFile(join(f.source, f.result.session_file));
  const record = await freezeAutomaticViews(f.source, f.view, config);
  const updated = await readJson(join(f.view, "view_search_only_result.json"));
  const fixed = await readJson(join(f.view, updated.manifest_path));
  assert.equal(record.origin, "automatic_view_search");
  assert.equal(record.manually_aligned, false);
  assert.equal(updated.frame_lock_origin, "automatic_view_search");
  assert.equal(updated.manually_aligned, false);
  assert.equal(updated.automatic_search_manifest_path, f.manifestPath);
  assert.equal(fixed.projection_frame.policy, "fixed");
  assert.equal(fixed.projection_frame.anchor_step_sha256, sha(f.stepText));
  assert.equal(await digestFile(join(f.view, fixed.projection_frame.anchor_step_path)), sha(f.stepText));
  assert.deepEqual(fixed.regions, f.manifest.regions);
  assert.deepEqual(await readFile(join(f.view, f.manifestPath)), originalManifest);
  assert.deepEqual(await readFile(join(f.view, "view_search_only_result.before-frame-lock.json")), originalResult);
  assert.equal(await readFile(join(f.source, "solution.py"), "utf8"), f.sourceText);
  assert.equal(await readFile(join(f.source, "output.step"), "utf8"), f.stepText);
  assert.deepEqual(await readFile(join(f.source, f.result.session_file)), originalSession);
  assert.equal(await digestFile(join(f.view, "output.step")), sha(f.stepText));
  assert.equal(record.original_manifest_sha256, sha(originalManifest));
  assert.equal(record.fixed_manifest_sha256, await digestFile(join(f.view, updated.manifest_path)));
  const validated = await loadLockedMainViewManifest({
    workspace: f.view, manifestPath: updated.manifest_path,
    manifestSha256: record.fixed_manifest_sha256,
    baselineSourceSha256: sha(f.sourceText), baselineStepSha256: sha(f.stepText),
  });
  assert.deepEqual(validated.regions[0].selected.view_spec, f.manifest.regions[0].selected.view_spec);
  assert.deepEqual(validated.regions[1].selected.view_spec, f.manifest.regions[1].selected.view_spec);
  assert.deepEqual(validated.regions[0].selected.transform, f.manifest.regions[0].selected.transform);
});

const failureCases = [
  ["stale source STEP", async f => put(f.source, "output.step", `${f.stepText}/* changed */\n`), /Formal CAD hashes are stale/u],
  ["changed search STEP", async f => put(f.view, "output.step", `${f.stepText}/* changed */\n`), /View Search changed the source CAD/u],
  ["mismatched search STEP record", async f => put(f.view, "view_search_only_result.json", { ...f.searchResult, step_sha256: "0".repeat(64) }), /does not match its generated CAD/u],
  ["mismatched manifest STEP", async f => put(f.view, f.manifestPath, { ...f.manifest, step_sha256: "0".repeat(64) }), /does not match the Direct baseline/u],
  ["partial search", async f => put(f.view, "view_search_only_result.json", { ...f.searchResult, status: "partial" }), /does not match its generated CAD/u],
  ["failed search", async f => put(f.view, "view_search_only_result.json", { ...f.searchResult, status: "error" }), /does not match its generated CAD/u],
  ["failed region", async f => {
    const manifest = structuredClone(f.manifest);
    manifest.regions[1].status = "error";
    delete manifest.regions[1].selected;
    manifest.regions[1].alternatives = [];
    await put(f.view, f.manifestPath, manifest);
  }, /does not match the Direct baseline/u],
  ["missing selected camera", async f => {
    const manifest = structuredClone(f.manifest);
    delete manifest.regions[1].selected;
    await put(f.view, f.manifestPath, manifest);
  }, /does not match the Direct baseline/u],
  ["zero projection scale", async f => {
    const manifest = structuredClone(f.manifest);
    manifest.regions[0].selected.transform.scale_px_per_unit = 0;
    await put(f.view, f.manifestPath, manifest);
  }, /Missing fixed 2D transform/u],
  ["escaping manifest path", async f => put(f.view, "view_search_only_result.json", { ...f.searchResult, manifest_path: "../outside.json" }), /Artifact escapes workspace/u],
  ["absolute manifest path", async f => put(f.view, "view_search_only_result.json", { ...f.searchResult, manifest_path: join(f.root, "outside.json") }), /Unsafe artifact path/u],
  ["escaping reference path", async f => {
    const manifest = structuredClone(f.manifest);
    manifest.regions[0].crop_path = "../outside.png";
    await put(f.view, f.manifestPath, manifest);
  }, /outside the fixture workspace|escaped the fixture workspace|escapes the fixture workspace/u],
  ["symlinked external render", async f => {
    const path = f.manifest.regions[0].selected.render_path;
    const outside = join(f.root, "outside.png");
    await put(f.root, "outside.png", png);
    await rm(join(f.view, path));
    await symlink(outside, join(f.view, path));
  }, /missing or unsafe|outside the fixture workspace/u],
];

for (const [name, mutate, error] of failureCases) {
  test(`rejects ${name} before freezing any automatic frame`, async t => {
    const f = await fixture(t);
    await mutate(f);
    const before = await readFile(join(f.view, f.manifestPath));
    await assert.rejects(freezeAutomaticViews(f.source, f.view, config), error);
    assert.deepEqual(await readFile(join(f.view, f.manifestPath)), before);
    await assert.rejects(readFile(join(f.view, "automatic_frame_lock.json")), { code: "ENOENT" });
  });
}

test("rejects job/result thinking mismatches rather than relabeling the Main history", async t => {
  const f = await fixture(t);
  const original = await readFile(join(f.source, "agent_result.json"));
  await put(f.source, ".cadgenbench-pi-job.json", { ...f.job, thinking_level: "high" });
  await assert.rejects(validateFormalWorkspace(f.source, config), /Model\/thinking mismatch/u);
  assert.deepEqual(await readFile(join(f.source, "agent_result.json")), original);
});
