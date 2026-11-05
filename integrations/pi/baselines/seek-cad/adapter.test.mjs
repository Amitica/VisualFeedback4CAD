import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { buildCritiquePrompt, buildInstrumentationPrompt, parseInstrumentation, parseCritique, validateStepManifest, assertGeometryEquivalent } from './adapter.mjs';

test('requires genuine step evidence and verified unchanged Initial geometry', () => {
  assert.throws(() => buildCritiquePrompt({ steps: [] }), /final-image-only/u);
  assert.throws(() => buildCritiquePrompt({ steps: [{ step_id: 'S001' }] }), /independently verified/u);
  const prompt = buildInstrumentationPrompt({ fixtureId: '146', source: 'part = Box(10, 20, 30)', sourceSha256: 'a'.repeat(64), initialStepSha256: 'b'.repeat(64) });
  assert.match(prompt, /NOT a repair/u);
  assert.match(prompt, /Do not provide private chain-of-thought/u);
  assert.match(prompt, /prior_step_path/u);
  assert.match(prompt, /current working directory set to an isolated instrumentation folder/u);
  assert.match(prompt, /Compound\(children=\[deepcopy\(shape\) for shape in feature_shapes\]\)/u);
  assert.match(prompt, /reparents those exact shape objects/u);
});

test('instrumentation normalizes inherited output literals without changing geometry', () => {
  const code = `SEMANTIC_DIR = "/workspace/semantic_steps"\npart = Box(10,20,30)\nexport_step(part, '/workspace/output.step')\nexport_step(part, "/workspace/semantic_steps/final.step")`;
  const result = parseInstrumentation(JSON.stringify({ instrumented_code: code, modeling_plan: [{ step_id: 'S001', description: 'Base' }] }));
  assert.match(result.instrumented_code, /part = Box\(10,20,30\)/u);
  assert.match(result.instrumented_code, /SEMANTIC_DIR = "semantic_steps"/u);
  assert.match(result.instrumented_code, /export_step\(part, 'output.step'\)/u);
  assert.equal(result.path_normalization.replacements, 3);
  assert.equal(result.instrumented_code.includes('/workspace'), false);
  const relative = parseInstrumentation(JSON.stringify({ instrumented_code: 'export_step(part, "output.step")', modeling_plan: [{}] }));
  assert.equal(relative.path_normalization.replacements, 0);
});

test('instrumentation rejects unresolved root exports before execution', () => {
  for (const code of ['root = "/workspace"\nexport_step(part, root + "/output.step")', 'export_step(part, "/workspace/other.step")', 'export_step(part, "/workspace/semantic_steps/../../output.step")']) {
    assert.throws(() => parseInstrumentation(JSON.stringify({ instrumented_code: code, modeling_plan: [{}] })), /output path/u);
  }
});

test('positive feedback cannot contain discrepancies and negative feedback must be actionable', () => {
  assert.equal(parseCritique('{"aligned":true,"discrepancies":[],"summary":"Aligned"}').aligned, true);
  assert.throws(() => parseCritique('{"aligned":false,"discrepancies":[],"summary":"Bad"}'), /inconsistent/u);
  assert.throws(() => parseCritique('{"aligned":true,"discrepancies":[{"step_id":"FINAL","description":"hole missing","suggested_change":"cut hole"}],"summary":"Wrong"}'), /inconsistent/u);
  assert.equal(parseCritique('{"aligned":false,"discrepancies":[{"step_id":"S001","description":"hole missing","suggested_change":"cut hole"}],"summary":"Fix hole"}').discrepancies.length, 1);
});

test('manifest binds Initial, matches stage order and rejects aliases and escaping paths', async t => {
  const workspace = await mkdtemp(join(tmpdir(), 'seek-cad-adapter-'));
  t.after(() => rm(workspace, { recursive: true, force: true }));
  await mkdir(join(workspace, 'semantic_steps'));
  for (const path of ['S001.step', 'S001-focus.step', 'S002.step', 'S002-focus.step', 'final.step']) await writeFile(join(workspace, 'semantic_steps', path), path);
  const manifest = { version: 1, source_sha256: 'a'.repeat(64), initial_step_sha256: 'b'.repeat(64), modeling_plan: [{ step_id: 'S001', description: 'base' }, { step_id: 'S002', description: 'hole' }], steps: [{ step_id: 'S001', description: 'base', operation: 'add', cumulative_step_path: 'semantic_steps/S001.step', focus_step_path: 'semantic_steps/S001-focus.step' }, { step_id: 'S002', description: 'hole', operation: 'remove', cumulative_step_path: 'semantic_steps/S002.step', focus_step_path: 'semantic_steps/S002-focus.step', prior_step_path: 'semantic_steps/S001.step' }], final_step_path: 'semantic_steps/final.step' };
  const options = { manifest, workspace, sourceSha256: 'a'.repeat(64), initialStepSha256: 'b'.repeat(64) };
  assert.equal((await validateStepManifest(options)).geometry_equivalence_required, true);
  await assert.rejects(() => validateStepManifest({ ...options, sourceSha256: 'c'.repeat(64) }), /does not match/u);
  const escaped = structuredClone(manifest); escaped.steps[0].focus_step_path = '../outside.step';
  await assert.rejects(() => validateStepManifest({ ...options, manifest: escaped }), /Unsafe/u);
  const alias = structuredClone(manifest); alias.steps[1].focus_step_path = alias.steps[1].cumulative_step_path;
  await assert.rejects(() => validateStepManifest({ ...options, manifest: alias }), /alias/u);
  const reordered = structuredClone(manifest); reordered.steps[1].prior_step_path = reordered.steps[1].cumulative_step_path;
  await assert.rejects(() => validateStepManifest({ ...options, manifest: reordered }), /preceding/u);
});

test('does not accept a volume-only equivalence report', () => {
  assert.throws(() => assertGeometryEquivalent({ equivalent: true, method: 'same_volume' }), /changed or could not verify/u);
  assert.equal(assertGeometryEquivalent({ equivalent: true, method: 'brep_symmetric_difference_and_world_bbox' }).equivalent, true);
});
