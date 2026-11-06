import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";
import { exportSnapshot } from "./run.mjs";

const sha = bytes => createHash("sha256").update(bytes).digest("hex");

async function fixture(t) {
  const root = await mkdtemp(join(tmpdir(), "cad-round-snapshot-test-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const workspace = join(root, "raw/run/140");
  const destination = join(root, "evaluation/ours/round-01/140");
  await mkdir(workspace, { recursive: true });
  await mkdir(destination, { recursive: true });
  const source = "# UNIT TEST: fake source, not executable model output\n";
  const step = "UNIT TEST: fake STEP bytes; not a CAD validity result\n";
  const placeholder = '{"status":"missing","method":"ours","round":1}\n';
  await writeFile(join(workspace, "solution.py"), source);
  await writeFile(join(workspace, "output.step"), step);
  await writeFile(join(workspace, "agent_result.json"), '{"test_fixture":true}\n');
  await writeFile(join(destination, "result.json"), placeholder);
  return {
    workspace, destination, source, step, placeholder,
    formal: { source: sha(source), step: sha(step) },
    metadata: { fixture_id: "140", method: "ours", round: 1, status: "completed" },
  };
}

async function exportedBytes(directory) {
  return Object.fromEntries(await Promise.all((await readdir(directory)).sort().map(async name => [name, sha(await readFile(join(directory, name)))])));
}

async function assertNoTemporaryExports(destination) {
  const siblings = await readdir(dirname(destination));
  assert.deepEqual(siblings.filter(name => name.startsWith(".export-")), []);
}

test("a formal-hash mismatch preserves the missing placeholder and publishes no STEP", async t => {
  const f = await fixture(t);
  await assert.rejects(
    () => exportSnapshot(f.workspace, f.destination, { ...f.formal, step: sha("different accepted CAD") }, f.metadata),
    /does not match the accepted formal CAD/u,
  );
  assert.equal(await readFile(join(f.destination, "result.json"), "utf8"), f.placeholder);
  assert.deepEqual(await readdir(f.destination), ["result.json"]);
  assert.equal(await readFile(join(f.workspace, "solution.py"), "utf8"), f.source);
  assert.equal(await readFile(join(f.workspace, "output.step"), "utf8"), f.step);
  await assertNoTemporaryExports(f.destination);
});

test("a successful real export cannot be overwritten by another candidate", async t => {
  const f = await fixture(t);
  const binding = await exportSnapshot(f.workspace, f.destination, f.formal, f.metadata);
  assert.equal(binding.source_sha256, f.formal.source);
  assert.equal(binding.step_sha256, f.formal.step);
  assert.equal(await readFile(join(f.destination, "solution.py"), "utf8"), f.source);
  assert.equal(await readFile(join(f.destination, "output.step"), "utf8"), f.step);
  assert.deepEqual(JSON.parse(await readFile(join(f.destination, "round.json"), "utf8")), f.metadata);
  const original = await exportedBytes(f.destination);

  const changedSource = "# UNIT TEST: second candidate source\n";
  const changedStep = "UNIT TEST: second candidate STEP\n";
  await writeFile(join(f.workspace, "solution.py"), changedSource);
  await writeFile(join(f.workspace, "output.step"), changedStep);
  await assert.rejects(
    () => exportSnapshot(f.workspace, f.destination, { source: sha(changedSource), step: sha(changedStep) }, { ...f.metadata, round: 2 }),
    /Refusing to overwrite an exported result/u,
  );
  assert.deepEqual(await exportedBytes(f.destination), original);
  await assertNoTemporaryExports(f.destination);
});
