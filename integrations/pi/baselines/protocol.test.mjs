import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { runCadCodeVerify, runSeekCad } from './container-protocol.mjs';

const sha = value => createHash('sha256').update(value).digest('hex');
const questions = [{ id: 'Q1', question: 'Does the base have the specified outline?' }, { id: 'Q2', question: 'Does the mounting hole exist?' }];
const answers = yes => JSON.stringify({ answers: questions.map(q => ({ id: q.id, answer: yes ? 'Yes' : 'No', evidence: yes ? 'Visible in the current CAD' : 'Visible mismatch with drawing' })) });

function verifyContext(answerStages) {
  const events = [], prompts = [], saves = [], renders = [];
  const ctx = {
    job: { description: 'Reconstruct the plate and hole in the supplied drawing.' },
    drawings: ['drawing-page-1', 'drawing-page-2'],
    executionInstructions: 'Use build123d and export output.step.',
    source: async () => '# current source\nexport_step(part, "output.step")\n',
    renderFour: async iteration => {
      renders.push(iteration);
      return [0, 1, 2, 3].map(id => `render-${iteration}-${id}`);
    },
    save: async (path, value) => saves.push({ path, value }),
    ask: async (label, prompt, images) => {
      events.push(label); prompts.push({ label, prompt, images });
      if (label === 'questions') return JSON.stringify({ questions });
      if (label.startsWith('answers-')) return answers(answerStages[Number(label.split('-')[1])] ?? false);
      if (label.startsWith('feedback-')) return '{"issues":[{"id":"Q2","observation":"Missing hole","correction":"Add specified hole"}]}';
      throw Error(`Unexpected fake model stage: ${label}`);
    },
    revise: async (label, prompt) => { events.push(label); prompts.push({ label, prompt }); },
  };
  return { ctx, events, prompts, saves, renders };
}

test('CADCodeVerify follows Q -> A -> feedback -> revision -> A with fixed questions', async () => {
  const { ctx, events, prompts, renders } = verifyContext([false, true]);
  const result = await runCadCodeVerify(ctx);
  assert.deepEqual(events, ['questions', 'answers-0', 'feedback-0', 'revision-1', 'answers-1']);
  assert.deepEqual(renders, [0, 1]);
  assert.equal(result.stop_reason, 'all_checks_yes');
  assert.equal(result.refinements, 1);
  assert.deepEqual(prompts.find(p => p.label === 'questions').images, ctx.drawings);
  for (const row of prompts.filter(p => p.label.startsWith('answers-'))) {
    assert.equal(row.images.length, 6);
    assert.deepEqual(row.images.slice(0, 2), ctx.drawings);
    for (const q of questions) assert.ok(row.prompt.includes(q.question));
  }
  assert.deepEqual(prompts.find(p => p.label === 'feedback-0').images, []);
});

test('CADCodeVerify stops before feedback or repair when the first answers are all Yes', async () => {
  const { ctx, events, renders } = verifyContext([true]);
  const result = await runCadCodeVerify(ctx);
  assert.deepEqual(events, ['questions', 'answers-0']);
  assert.deepEqual(renders, [0]);
  assert.equal(result.refinements, 0);
});

test('CADCodeVerify obeys two repairs and performs final answers without an extra repair', async () => {
  const { ctx, events, renders } = verifyContext([false, false, false]);
  const result = await runCadCodeVerify(ctx);
  assert.deepEqual(events, ['questions', 'answers-0', 'feedback-0', 'revision-1', 'answers-1', 'feedback-1', 'revision-2', 'answers-2']);
  assert.deepEqual(renders, [0, 1, 2]);
  assert.equal(result.stop_reason, 'refinement_budget');
  assert.equal(result.refinements, 2);
});

async function seekContext(t, { aligned = true, instrumentFailures = 0, equivalence = true, emptySteps = false, critiqueError = false, revisionError = false, unsupported = false } = {}) {
  const workspace = await mkdtemp(join(tmpdir(), 'cad-baseline-protocol-'));
  t.after(() => rm(workspace, { recursive: true, force: true }));
  const source = 'from build123d import *\nexport_step(Box(10, 20, 30), "output.step")\n';
  await writeFile(join(workspace, 'output.step'), 'initial-step-bytes');
  const events = [], imagePaths = [], saves = [], prompts = [];
  let instrumentAttempts = 0;
  const ctx = {
    workspace, job: { fixture_id: '146', description: 'Reconstruct the drawing.' },
    drawings: ['original-drawing'], source: async () => source,
    save: async (name, value) => saves.push({ name, value }),
    ask: async (label, prompt, images) => {
      events.push(label); prompts.push({ label, prompt, images });
      if (label.startsWith('instrument-')) {
        if (unsupported) return '{"unsupported":true,"reason":"No reliable semantic trace"}';
        return JSON.stringify({ instrumented_code: `${source}\n# export genuine stages`, modeling_plan: [{ step_id: 'S001', description: 'Base' }, { step_id: 'S002', description: 'Hole' }] });
      }
      if (label === 'stepwise-critique') {
        if (critiqueError) throw Error('critique API failure');
        return JSON.stringify({ aligned, discrepancies: aligned ? [] : [{ step_id: 'S002', description: 'Missing hole', suggested_change: 'Cut the hole specified in the drawing' }], summary: aligned ? 'Matches' : 'Repair hole' });
      }
      throw Error(`Unexpected stage: ${label}`);
    },
    instrument: async (_code, binding) => {
      events.push('execute-instrumentation');
      instrumentAttempts++;
      if (instrumentAttempts <= instrumentFailures) throw Error('Instrumented final changed Initial geometry');
      const location = join(workspace, `trace-${instrumentAttempts}`);
      await mkdir(join(location, 'semantic_steps'), { recursive: true });
      await mkdir(join(location, 'step_renders'));
      for (const name of ['S001.step', 'S001-focus.step', 'S002.step', 'S002-focus.step', 'final.step']) await writeFile(join(location, 'semantic_steps', name), name);
      for (const name of ['S001.png', 'S002.png', 'FINAL.png']) await writeFile(join(location, 'step_renders', name), name);
      const manifest = {
        version: 1, source_sha256: binding.sourceSha256, initial_step_sha256: binding.stepSha256,
        modeling_plan: emptySteps ? [] : [{ step_id: 'S001', description: 'Base' }, { step_id: 'S002', description: 'Hole' }],
        steps: emptySteps ? [] : [
          { step_id: 'S001', description: 'Base', operation: 'add', cumulative_step_path: 'semantic_steps/S001.step', focus_step_path: 'semantic_steps/S001-focus.step' },
          { step_id: 'S002', description: 'Hole', operation: 'remove', cumulative_step_path: 'semantic_steps/S002.step', focus_step_path: 'semantic_steps/S002-focus.step', prior_step_path: 'semantic_steps/S001.step' },
        ], final_step_path: 'semantic_steps/final.step',
      };
      assert.equal(binding.sourceSha256, sha(source));
      assert.equal(binding.stepSha256, sha('initial-step-bytes'));
      return { workspace: location, semantic_manifest: manifest,
        geometry_equivalence: { equivalent: equivalence, method: 'brep_symmetric_difference_and_world_bbox' },
        step_images: [{ image_path: 'step_renders/S001.png' }, { image_path: 'step_renders/S002.png' }],
        final_image_path: 'step_renders/FINAL.png' };
    },
    image: async path => {
      imagePaths.push(path);
      // Fails on the wrong base (workspace/step_renders instead of trace-N/step_renders).
      const data = await readFile(path, 'utf8');
      return { fakeImage: data };
    },
    revise: async (label, prompt) => {
      events.push(label); prompts.push({ label, prompt });
      if (revisionError) throw Error('revision execution failure after code write');
    },
  };
  return { ctx, events, imagePaths, saves, prompts };
}

test('Seek-CAD uses executed and equivalence-checked stages before critique; positive feedback does not repair', async t => {
  const { ctx, events, imagePaths, prompts } = await seekContext(t);
  const result = await runSeekCad(ctx);
  assert.deepEqual(events, ['instrument-1', 'execute-instrumentation', 'stepwise-critique']);
  assert.equal(result.stop_reason, 'positive_stepwise_feedback');
  assert.equal(result.refinements, 0);
  assert.equal(imagePaths.length, 3);
  assert.ok(imagePaths.every(path => path.startsWith(join(ctx.workspace, 'trace-1', 'step_renders'))));
  const critique = prompts.find(row => row.label === 'stepwise-critique');
  assert.equal(critique.images.length, 4);
  assert.equal(critique.images[0], ctx.drawings[0]);
});

test('Seek-CAD explicit negative feedback triggers exactly one repair', async t => {
  const { ctx, events } = await seekContext(t, { aligned: false });
  const result = await runSeekCad(ctx);
  assert.deepEqual(events, ['instrument-1', 'execute-instrumentation', 'stepwise-critique', 'stepwise-revision']);
  assert.equal(result.refinements, 1);
});

test('Seek-CAD only retries instrumentation failures and retains the original seed', async t => {
  const { ctx, events, saves } = await seekContext(t, { instrumentFailures: 1 });
  const result = await runSeekCad(ctx);
  assert.deepEqual(events, ['instrument-1', 'execute-instrumentation', 'instrument-2', 'execute-instrumentation', 'stepwise-critique']);
  assert.equal(result.refinements, 0);
  assert.equal(saves.filter(row => row.name.startsWith('instrumentation-failure-')).length, 1);
});

test('Seek-CAD rejects changed geometry without final-image-only fallback', async t => {
  const { ctx, events } = await seekContext(t, { equivalence: false });
  await assert.rejects(() => runSeekCad(ctx), /changed or could not verify/u);
  assert.equal(events.filter(x => x === 'execute-instrumentation').length, 2);
  assert.ok(!events.includes('stepwise-critique'));
  assert.ok(!events.includes('stepwise-revision'));
});

test('Seek-CAD rejects an empty trace even when a final image exists', async t => {
  const { ctx, events } = await seekContext(t, { emptySteps: true });
  await assert.rejects(() => runSeekCad(ctx), /semantic steps/u);
  assert.ok(!events.includes('stepwise-critique'));
});

test('Seek-CAD does not retry instrumentation after a critic API failure', async t => {
  const { ctx, events } = await seekContext(t, { critiqueError: true });
  await assert.rejects(() => runSeekCad(ctx), /critique API failure/u);
  assert.deepEqual(events, ['instrument-1', 'execute-instrumentation', 'stepwise-critique']);
});

test('Seek-CAD does not perform another repair after its single revision fails', async t => {
  const { ctx, events } = await seekContext(t, { aligned: false, revisionError: true });
  await assert.rejects(() => runSeekCad(ctx), /revision execution failure/u);
  assert.deepEqual(events, ['instrument-1', 'execute-instrumentation', 'stepwise-critique', 'stepwise-revision']);
});

test('Seek-CAD unsupported trace stops explicitly without a substitute reflection', async t => {
  const { ctx, events } = await seekContext(t, { unsupported: true });
  await assert.rejects(() => runSeekCad(ctx), /unsupported/u);
  assert.deepEqual(events, ['instrument-1']);
});
