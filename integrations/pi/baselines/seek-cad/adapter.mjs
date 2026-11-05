import { createHash } from 'node:crypto';
import { lstat, readFile, realpath } from 'node:fs/promises';
import { isAbsolute, relative, resolve, sep } from 'node:path';

export const METHOD = Object.freeze({
  id: 'seek-cad-svf-drawing-adapted',
  display_name: 'Seek-CAD SVF — engineering-drawing adaptation',
  default_refinement_rounds: 1,
  requires_semantic_step_geometry: true,
  requires_initial_geometry_equivalence: true,
  uses_view_search: false,
  uses_dimension_annotations: false,
  original_repo: 'https://github.com/Sunny-Hack/Seek-CAD',
  original_commit: '230947d70960cb58bea76fe75d46d3796c28ca34',
  fidelity: 'refinement_mechanism_adaptation_not_full_native_reproduction',
});

function jsonObject(text) {
  const stripped = text.trim().replace(/^```(?:json)?\s*/u, '').replace(/\s*```$/u, '');
  const result = JSON.parse(stripped);
  if (!result || Array.isArray(result) || typeof result !== 'object') throw Error('Expected a JSON object');
  return result;
}

function requiredString(value, label) {
  if (typeof value !== 'string' || !value.trim()) throw Error(`Missing ${label}`);
  return value.trim();
}

function validHash(value, label) {
  if (!/^[a-f0-9]{64}$/u.test(value ?? '')) throw Error(`Invalid ${label}`);
}

export function buildInstrumentationPrompt(ctx) {
  requiredString(ctx.source, 'current source');
  validHash(ctx.sourceSha256, 'Initial source hash');
  validHash(ctx.initialStepSha256, 'Initial STEP hash');
  return `Prepare genuine construction-stage evidence for CAD fixture ${ctx.fixtureId ?? 'unknown'}.
This is instrumentation, NOT a repair. Preserve every geometry operation, numeric parameter, unit, feature and final result in the supplied program. Add only snapshot copies, exports and a short public modeling plan. Do not provide private chain-of-thought.
The program executes with its current working directory set to an isolated instrumentation folder, NOT /workspace. All output paths must be relative to that current working directory: output.step and semantic_steps/.... Replace absolute export destinations inherited from the original source with these relative paths; changing only output destinations is allowed and does not change the geometry. Never write to /workspace, a parent directory, or another absolute directory. Use os.makedirs("semantic_steps", exist_ok=True) when needed.
Snapshot creation must not mutate the modeled objects or their assembly ownership. In build123d, Compound(children=[existing_shape, ...]) reparents those exact shape objects: attaching a boss already inside the final part to a focus Compound can remove it from the final part. Use independent copies for every snapshot/focus aggregation, for example: from copy import deepcopy; focus = Compound(children=[deepcopy(shape) for shape in feature_shapes]). Apply the same copy rule to temporary cumulative assemblies. Never attach original model children to a second parent merely to export evidence. Preserve the original final assembly construction and export it only after ensuring snapshot creation has not changed it.

Return JSON {"instrumented_code":"<complete executable Python>","modeling_plan":[{"step_id":"S001","description":"brief public description of the modeled feature"}]}.
The program must export the unchanged complete part as output.step and semantic_steps/final.step. After each coherent feature group, export its cumulative CAD to semantic_steps/S001.step and the actual new/removed/refined feature or operation operand to semantic_steps/S001-focus.step. Later stages must reference their immediately preceding cumulative STEP as prior_step_path. Do not fabricate stages by exporting the final part repeatedly or generating shapes unrelated to the original program.

Write semantic_steps/manifest.json with exactly this structure:
{"version":1,"source_sha256":"${ctx.sourceSha256}","initial_step_sha256":"${ctx.initialStepSha256}","modeling_plan":[{"step_id":"S001","description":"..."}],"steps":[{"step_id":"S001","description":"...","operation":"add","cumulative_step_path":"semantic_steps/S001.step","focus_step_path":"semantic_steps/S001-focus.step"}],"final_step_path":"semantic_steps/final.step"}.
For steps after the first, add prior_step_path naming the preceding cumulative STEP. operation is add, remove, refine, or intersect. Use a maximum of ${ctx.maxSteps ?? 12} meaningful steps. Never change construction solely to make a nicer trace; if reliable instrumentation is impossible, return {"unsupported":true,"reason":"..."}. The controller will compare the instrumented final solid with the Initial STEP using world-coordinate BRep symmetric difference before any feedback is accepted.

CURRENT SOURCE:
${ctx.source}`;
}

export function parseInstrumentation(text) {
  const result = jsonObject(text);
  if (result.unsupported === true) return { unsupported: true, reason: requiredString(result.reason, 'unsupported reason') };
  requiredString(result.instrumented_code, 'instrumented_code');
  if (!Array.isArray(result.modeling_plan) || !result.modeling_plan.length) throw Error('Instrumentation requires a public modeling_plan');
  // The original Initial can export to /workspace/output.step. In a nested
  // execution workspace those inherited literals otherwise overwrite Main CAD
  // while the local worker sees no output. Normalize only known output literals;
  // leave every numeric parameter and geometry operation untouched.
  let replacements = 0;
  const code = result.instrumented_code.replace(/(["'])\/workspace\/(output\.step|semantic_steps(?:\/[^"'\r\n]*)?)\1/gu, (_match, quote, path) => {
    if (path.split(/[\\/]/u).includes('..')) throw Error('Instrumentation output path escapes its isolated execution directory');
    replacements++;
    return `${quote}${path}${quote}`;
  });
  if (code.includes('/workspace')) throw Error('Instrumentation output path contract: use only cwd-relative output.step and semantic_steps/...; absolute /workspace paths would overwrite the Main CAD');
  return { ...result, instrumented_code: code, path_normalization: { policy: 'known-output-literals-relative-to-execution-cwd', replacements } };
}

export async function safeArtifact(workspace, path) {
  requiredString(path, 'artifact path');
  if (isAbsolute(path) || path.split(/[\\/]/u).includes('..')) throw Error(`Unsafe step path: ${path}`);
  const root = await realpath(workspace);
  const absolute = resolve(root, path);
  const metadata = await lstat(absolute);
  const canonical = await realpath(absolute);
  const rel = relative(root, canonical);
  if (!metadata.isFile() || metadata.isSymbolicLink() || rel === '..' || rel.startsWith(`..${sep}`) || isAbsolute(rel)) throw Error(`Unsafe step artifact: ${path}`);
  return absolute;
}

export async function validateStepManifest({ manifest, workspace, sourceSha256, initialStepSha256, maxSteps = 12 }) {
  validHash(sourceSha256, 'Initial source hash');
  validHash(initialStepSha256, 'Initial STEP hash');
  if (manifest.version !== 1 || manifest.source_sha256 !== sourceSha256 || manifest.initial_step_sha256 !== initialStepSha256) throw Error('Step manifest does not match the Initial source/STEP');
  if (!Array.isArray(manifest.steps) || manifest.steps.length < 1 || manifest.steps.length > maxSteps) throw Error('No genuine bounded semantic steps');
  if (!Array.isArray(manifest.modeling_plan) || manifest.modeling_plan.length !== manifest.steps.length) throw Error('Public modeling plan and step evidence differ');
  const ids = new Set();
  const steps = [];
  for (let index = 0; index < manifest.steps.length; index++) {
    const step = manifest.steps[index];
    if (!/^S[0-9]{3,}$/u.test(step.step_id ?? '') || ids.has(step.step_id)) throw Error('Invalid or duplicate semantic step_id');
    ids.add(step.step_id);
    requiredString(step.description, 'step description');
    if (!['add', 'remove', 'refine', 'intersect'].includes(step.operation)) throw Error('Invalid semantic operation');
    if (manifest.modeling_plan[index].step_id !== step.step_id) throw Error('Step order differs from the public plan');
    requiredString(manifest.modeling_plan[index].description, 'public step description');
    if (index && step.prior_step_path !== manifest.steps[index - 1].cumulative_step_path) throw Error('Later steps must identify their actual preceding cumulative geometry');
    const cumulative = await safeArtifact(workspace, step.cumulative_step_path);
    const focus = await safeArtifact(workspace, step.focus_step_path);
    if (cumulative === focus && manifest.steps.length > 1) throw Error('A feature-focus artifact must not simply alias the cumulative model');
    if (index) await safeArtifact(workspace, step.prior_step_path);
    steps.push({ ...step, cumulative_sha256: createHash('sha256').update(await readFile(cumulative)).digest('hex'), focus_sha256: createHash('sha256').update(await readFile(focus)).digest('hex') });
  }
  await safeArtifact(workspace, manifest.final_step_path);
  return { ...manifest, steps, geometry_equivalence_required: true };
}

export function buildCritiquePrompt(ctx) {
  if (!Array.isArray(ctx.steps) || !ctx.steps.length) throw Error('Seek-CAD SVF cannot fall back to final-image-only reflection');
  if (ctx.geometryEquivalent !== true) throw Error('Instrumentation geometry must be independently verified before visual critique');
  return `Evaluate the current CAD against the supplied engineering drawing using real construction-stage evidence.
The first images are the original drawing pages. Each subsequent stage image highlights the current semantic feature/operand, with earlier geometry shown translucently; the last image shows the full final CAD. These are fixed perspective views, not drawing-aligned projections. Match each image with the PUBLIC modeling plan below. This plan is an artifact description, not hidden chain-of-thought.
Check (1) whether the rendered stage expresses its stated operation and location, and (2) whether the plan and complete CAD match structures and dimensions actually specified by the drawing. Distinguish uncertain image evidence from a confirmed discrepancy. Do not infer physical dimensions from uncalibrated apparent scale.
Return only JSON {"aligned":true|false,"discrepancies":[{"step_id":"S001 or FINAL","description":"observed mismatch and drawing evidence","suggested_change":"actionable code/parameter correction"}],"summary":"brief assessment"}.
aligned=true requires no confirmed discrepancy; otherwise list at least one. Do not modify code, use view search, add CAD dimension labels, or propose arbitrary new camera searches.
PUBLIC PLAN AND STEP ORDER:
${JSON.stringify(ctx.steps.map(step => ({ step_id: step.step_id, description: step.description, operation: step.operation })), null, 2)}
DRAWING TASK: ${ctx.drawingDescription ?? 'Reconstruct the object in the attached engineering drawing.'}`;
}

export function parseCritique(text) {
  const result = jsonObject(text);
  if (typeof result.aligned !== 'boolean' || !Array.isArray(result.discrepancies)) throw Error('Invalid SVF critique');
  requiredString(result.summary, 'critique summary');
  for (const item of result.discrepancies) {
    if (!/^(?:S[0-9]{3,}|FINAL)$/u.test(item.step_id ?? '')) throw Error('Critique must identify a real stage or FINAL');
    requiredString(item.description, 'discrepancy');
    requiredString(item.suggested_change, 'suggested_change');
  }
  if (result.aligned === (result.discrepancies.length > 0)) throw Error('Positive/negative SVF feedback is inconsistent');
  return result;
}

export function buildRevisionPrompt(ctx) {
  const feedback = typeof ctx.feedback === 'string' ? parseCritique(ctx.feedback) : ctx.feedback;
  if (!feedback || feedback.aligned !== false || !feedback.discrepancies?.length) throw Error('A code revision requires explicit negative step-wise feedback');
  return `Revise the current CAD program using the independent step-wise visual feedback below. Preserve features the feedback does not identify as wrong, and resolve discrepancies using the original engineering drawing. Work in solution.py and formally execute the result. The instrumentation program is evidence only; begin from the original current CAD program, not from a geometry-altered instrumentation attempt. Do not call view search or dimension-annotation tools.
FEEDBACK:
${JSON.stringify(feedback, null, 2)}
CURRENT CAD SOURCE:
${ctx.source ?? ''}`;
}

export function assertGeometryEquivalent(report) {
  if (report?.equivalent !== true || report.method !== 'brep_symmetric_difference_and_world_bbox') throw Error('Seek-CAD instrumentation changed or could not verify Initial geometry');
  return report;
}
