import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { command, discoverInitials, exportEvaluation, findInitial, sha256, stagePaperCase, writeJson } from "./common.mjs";
import { mainSelfReflectPrompt } from "../dist/src/main-self-reflect.js";
import { main } from "./run.mjs";

const config = { provider: "test-provider", model: "test-model", thinking_level: "low" };
async function formalFixture(t) {
  const root = await mkdtemp(join(tmpdir(), "cad-baseline-binding-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const source = join(root, "cases/146/01-generate/146");
  await mkdir(join(source, "sessions"), { recursive: true });
  await mkdir(join(source, "inputs"));
  await writeFile(join(source, "solution.py"), '# original parametric source\nexport_step(part, "output.step")\n');
  await writeFile(join(source, "output.step"), "unchanged-initial-step");
  await writeFile(join(source, "inputs/page.png"), "drawing-bytes");
  await writeFile(join(source, "sessions/main.jsonl"), "original-session");
  const binding = { ...config, agent_mode: "base-only" };
  await writeJson(join(source, ".cadgenbench-pi-job.json"), { ...binding, fixture: { id: "146", description: "Drawing to CAD" }, staged_paths: ["inputs/page.png"] });
  await writeJson(join(source, "agent_result.json"), { ...binding, fixture_id: "146", status: "valid", session_file: "sessions/main.jsonl", last_executed_source_sha256: await sha256(join(source, "solution.py")), last_executed_step_sha256: await sha256(join(source, "output.step")), artifact: { sha256: await sha256(join(source, "output.step")) } });
  await writeJson(join(source, "result.json"), { status: "valid" });
  return { root, source };
}

test("Initial discovery binds the original Main workspace rather than a final STEP export", async t => {
  const { root, source } = await formalFixture(t);
  assert.equal(await findInitial(root, "146"), source);
  assert.deepEqual(await discoverInitials(root), ["146"]);
  await mkdir(join(root, "147"));
  await writeFile(join(root, "147/output.step"), "final-only");
  await assert.rejects(() => findInitial(root, "147"), /cannot resume/u);
  assert.deepEqual(await discoverInitials(root), ["146"]);
});

test("paper staging preserves Initial code, geometry and drawing and rejects stale source bindings", async t => {
  const { root, source } = await formalFixture(t);
  const destination = join(root, "new-case");
  await stagePaperCase(source, destination, "cadcodeverify", config);
  for (const file of ["solution.py", "output.step"]) {
    assert.equal(await sha256(join(source, file)), await sha256(join(destination, file)));
    assert.equal(await sha256(join(source, file)), await sha256(join(destination, "initial", file)));
  }
  await writeFile(join(destination, "solution.py"), "changed-reflection-code");
  assert.notEqual(await sha256(join(source, "solution.py")), await sha256(join(destination, "solution.py")));
  assert.equal(await readFile(join(destination, "inputs/page.png"), "utf8"), "drawing-bytes");
  await writeFile(join(source, "solution.py"), "tampered-after-execution");
  await assert.rejects(() => stagePaperCase(source, join(root, "stale-case"), "seek-cad", config), /hashes are stale/u);
});

test("evaluation export binds the exact program and STEP without modifying its source", async t => {
  const { root, source } = await formalFixture(t);
  const before = await sha256(join(source, "output.step"));
  const destination = join(root, "evaluation/146");
  const binding = await exportEvaluation(source, destination, { fixture_id: "146", method: "initial" });
  assert.equal(binding.source_workspace, source);
  assert.equal(binding.step_sha256, before);
  assert.equal(await sha256(join(source, "output.step")), before);
  assert.equal(await sha256(join(destination, "solution.py")), binding.source_sha256);
  assert.equal(JSON.parse(await readFile(join(destination, "baseline_provenance.json"))).step_sha256, before);
});

test("staging refuses a source file symlink rather than copying unrelated files", async t => {
  const { root, source } = await formalFixture(t);
  await rm(join(source, "solution.py"));
  await writeFile(join(root, "outside.py"), "outside the selected Initial workspace");
  await symlink(join(root, "outside.py"), join(source, "solution.py"));
  await assert.rejects(() => stagePaperCase(source, join(root, "unsafe-case"), "seek-cad", config));
});

test("Vanilla keeps the historical Direct Reflect prompt verbatim", async () => {
  const historical = await readFile(new URL("./vanilla/historical-direct-reflect-102.txt", import.meta.url), "utf8");
  assert.equal(mainSelfReflectPrompt("102"), historical.replace(/\n$/u, ""));
  assert.match(mainSelfReflectPrompt("146"), /Continue CADGenBench fixture 146/u);
  assert.doesNotMatch(mainSelfReflectPrompt("146"), /dimension_annotation|view_search|B-Rep/u);
});

test("Initial-only CLI exports a valid seed and retains an unusable case in the evaluation denominator", async t => {
  const { root, source } = await formalFixture(t);
  const inputs = join(root, "dataset/146");
  await mkdir(inputs, { recursive: true });
  await writeFile(join(inputs, "page.png"), "drawing-bytes");
  await writeFile(join(inputs, "description.yaml"), "description: Reconstruct the supplied drawing.\ninput_type: text+image\ntask_type: generation\ninput_files: [page.png]\n");
  const configFile = join(root, "config.json");
  await writeJson(configFile, { ...config, image: "not-needed-for-export", docker_network: "bridge", parallel: 1, timeout_minutes: 360, max_compile_attempts: 3, cases: ["146"], data_inputs_dir: join(root, "dataset") });
  const argv = output => ["--method", "initial", "--initial-root", root, "--config", configFile, "--output-dir", output];
  const good = join(root, "good-export");
  assert.equal(await main(argv(good)), 0);
  assert.equal(await sha256(join(good, "evaluation/146/output.step")), await sha256(join(source, "output.step")));
  await writeFile(join(source, "solution.py"), "changed after the formal execution");
  const failed = join(root, "failed-export");
  assert.equal(await main(argv(failed)), 2);
  const result = JSON.parse(await readFile(join(failed, "evaluation/146/result.json")));
  assert.equal(result.status, "missing");
  assert.equal(result.baseline_status, "initial_error");
  const summary = JSON.parse(await readFile(join(failed, "summary.json")));
  assert.equal(summary.cases.length, 1);
  assert.equal(summary.cases[0].status, "error");
});

test("a hanging generation command is interrupted at its configured deadline", async () => {
  const started = Date.now();
  await assert.rejects(() => command([process.execPath, "-e", "setInterval(() => {}, 1000)"], { timeoutMs: 200 }), /timed out/u);
  assert.ok(Date.now() - started < 5000, "Command must not remain alive after the deadline");
});

test("a missing Main workspace does not discard another valid Initial", async t => {
  const { root, source } = await formalFixture(t);
  const data = join(root, "inputs");
  for (const id of ["146", "147"]) {
    await mkdir(join(data, id), { recursive: true });
    await writeFile(join(data, id, "page.png"), "drawing");
    await writeFile(join(data, id, "description.yaml"), "description: Drawing to CAD.\ninput_type: text+image\ninput_files: [page.png]\n");
  }
  const configFile = join(root, "suite.json");
  await writeJson(configFile, { ...config, image: "no-container-needed", docker_network: "bridge", parallel: 2, timeout_minutes: 360, max_compile_attempts: 3, cases: ["146", "147"], data_inputs_dir: data });
  const output = join(root, "partial-export");
  assert.equal(await main(["--method", "initial", "--initial-root", root, "--config", configFile, "--cases", "146", "147", "--output-dir", output]), 2);
  assert.equal(await sha256(join(output, "evaluation/146/output.step")), await sha256(join(source, "output.step")));
  const summary = JSON.parse(await readFile(join(output, "summary.json")));
  assert.deepEqual(summary.cases.map(c => c.status), ["completed", "error"]);
  assert.equal(JSON.parse(await readFile(join(output, "evaluation/147/result.json"))).status, "missing");
});
