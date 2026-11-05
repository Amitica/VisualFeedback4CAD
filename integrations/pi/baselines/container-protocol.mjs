// CADCodeVerify and Seek-CAD SVF task adaptations. Executed inside the CAD image.
import { spawn } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import * as verify from "./cadcodeverify/adapter.mjs";
import * as seek from "./seek-cad/adapter.mjs";
import { executionInstructions, buildCompileRepairPrompt } from "./execution-guide.mjs";

const digest = bytes => createHash("sha256").update(bytes).digest("hex");
const readJson = async p => JSON.parse(await readFile(p, "utf8"));
async function save(p, value) { await mkdir(dirname(p), { recursive: true }); await writeFile(p, JSON.stringify(value, null, 2) + "\n"); }
function pythonCode(text) {
  const blocks = [...text.matchAll(/```(?:python|py)?\s*([\s\S]*?)```/giu)];
  const code = (blocks.length ? blocks.at(-1)[1] : text).trim();
  if (!code || !code.includes("output.step")) throw Error("Revision must be executable Python exporting output.step");
  return code + "\n";
}
const valid = response => response.artifact?.status === "valid" && response.artifact.validation?.is_valid === true && response.artifact.validation?.is_watertight === true;

export async function runCadCodeVerify(ctx) {
  const questions = verify.parseQuestions(await ctx.ask("questions", verify.buildQuestionsPrompt({ description: ctx.job.description }), ctx.drawings));
  await ctx.save("questions.json", questions);
  for (let iteration = 0; iteration <= verify.defaults.maxRefinements; iteration++) {
    const images = await ctx.renderFour(iteration);
    const answers = verify.parseAnswers(await ctx.ask(`answers-${iteration}`, verify.buildAnswersPrompt({ questions }), [...ctx.drawings, ...images]), questions);
    await ctx.save(`answers-${iteration}.json`, answers);
    if (verify.allYes(answers)) return { stop_reason: "all_checks_yes", refinements: iteration, questions, answers };
    if (iteration === verify.defaults.maxRefinements) return { stop_reason: "refinement_budget", refinements: iteration, questions, answers };
    const feedback = await ctx.ask(`feedback-${iteration}`, verify.buildFeedbackPrompt({ questions, answers }), []);
    await ctx.revise(`revision-${iteration + 1}`, verify.buildRevisionPrompt({ code: await ctx.source(), feedback, iteration: iteration + 1, executionInstructions: ctx.executionInstructions }));
  }
}

export async function runSeekCad(ctx) {
  const source = await ctx.source();
  const sourceSha256 = digest(source);
  const stepSha256 = digest(await readFile(join(ctx.workspace, "output.step")));
  let instrumentation;
  let evidence;
  let manifest;
  let failure = "";
  for (let attempt = 0; attempt < 2; attempt++) {
    const reply = await ctx.ask(`instrument-${attempt + 1}`, seek.buildInstrumentationPrompt({ fixtureId: ctx.job.fixture_id, source, sourceSha256, initialStepSha256: stepSha256 }) + failure, []);
    try {
      instrumentation = seek.parseInstrumentation(reply);
      if (instrumentation.unsupported) throw Error(`Seek-CAD SVF instrumentation unsupported: ${instrumentation.reason}`);
      if (instrumentation.path_normalization) await ctx.save(`instrumentation-paths-${attempt + 1}.json`, instrumentation.path_normalization);
      evidence = await ctx.instrument(instrumentation.instrumented_code, { sourceSha256, stepSha256 });
      seek.assertGeometryEquivalent(evidence.geometry_equivalence);
      manifest = await seek.validateStepManifest({ manifest: evidence.semantic_manifest, workspace: evidence.workspace, sourceSha256, initialStepSha256: stepSha256 });
      break;
    } catch (error) {
      if (instrumentation?.unsupported) throw error;
      failure = `\nThe previous instrumentation attempt failed controller validation: ${error.message}. Correct the instrumentation only; preserve original geometry.`;
      await ctx.save(`instrumentation-failure-${attempt + 1}.json`, { error: error.message });
      if (attempt === 1) throw error;
    }
  }
  const images = await Promise.all([...evidence.step_images.map(x => x.image_path), evidence.final_image_path].map(p => ctx.image(join(evidence.workspace, p))));
  const critique = seek.parseCritique(await ctx.ask("stepwise-critique", seek.buildCritiquePrompt({ steps: manifest.steps, geometryEquivalent: true, drawingDescription: ctx.job.description }), [...ctx.drawings, ...images]));
  await ctx.save("stepwise-feedback.json", critique);
  if (critique.aligned) return { stop_reason: "positive_stepwise_feedback", refinements: 0, geometry_equivalence: evidence.geometry_equivalence };
  await ctx.revise("stepwise-revision", seek.buildRevisionPrompt({ source, feedback: critique }));
  return { stop_reason: "one_stepwise_refinement", refinements: 1, geometry_equivalence: evidence.geometry_equivalence };
}

async function main() {
  const workspace = "/workspace";
  const job = await readJson(join(workspace, "baseline-job.json"));
  const { resolveVisionModel } = await import("/opt/cadgenbench-pi/dist/src/runner.js");
  const { LocalWorker } = await import("/opt/cadgenbench-pi/dist/src/local-worker.js");
  const { runtime, model } = await resolveVisionModel(job.provider, job.model);
  const worker = new LocalWorker("python");
  const abort = AbortSignal.timeout(job.timeout_minutes * 60_000);
  const root = join(workspace, "protocol");
  await mkdir(root, { recursive: true });
  let calls = 0;
  let turn = 0;
  let last = { artifact: { status: "valid", sha256: job.initial_step_sha256 } };
  const usages = [];
  const image = async p => {
    const bytes = await readFile(p);
    const mimeType = bytes.subarray(0, 8).equals(Buffer.from([137,80,78,71,13,10,26,10])) ? "image/png" : bytes[0] === 255 && bytes[1] === 216 ? "image/jpeg" : bytes.toString("ascii", 0, 4) === "RIFF" && bytes.toString("ascii", 8, 12) === "WEBP" ? "image/webp" : null;
    if (!mimeType) throw Error(`Unsupported image type: ${p}`);
    return { type: "image", mimeType, data: bytes.toString("base64") };
  };
  const drawings = await Promise.all(job.staged_paths.map(p => image(join(workspace, p))));
  const ask = async (label, prompt, images = []) => {
    const response = await runtime.completeSimple(model, { messages: [{ role: "user", content: [{ type: "text", text: prompt }, ...images], timestamp: Date.now() }] }, { reasoning: job.thinking_level, signal: abort });
    calls++;
    const text = response.content.filter(b => b.type === "text").map(b => b.text).join("\n");
    await save(join(root, `${String(calls).padStart(3, "0")}-${label}.json`), { label, prompt, image_count: images.length, image_sha256: images.map(i => digest(Buffer.from(i.data, "base64"))), response_text: text, stop_reason: response.stopReason, usage: response.usage });
    usages.push(response.usage);
    if (response.stopReason !== "stop") throw Error(`Model did not finish ${label}: ${response.stopReason}`);
    return text;
  };
  const execute = async (location, code) => {
    await writeFile(join(location, "solution.py"), code);
    const response = await worker.request(location, { protocol_version: 5, operation: "execute", turn: location === workspace ? ++turn : 1, timeout_s: 120 }, { signal: abort, timeoutMs: 150_000 });
    if (location === workspace) last = response;
    return response;
  };
  const ctx = {
    job, workspace, drawings, executionInstructions, image, ask,
    source: () => readFile(join(workspace, "solution.py"), "utf8"),
    save: (name, value) => save(join(root, name), value),
    renderFour: async iteration => {
      const result = [];
      const binding = { iteration, step_sha256: digest(await readFile(join(workspace, "output.step"))), source_sha256: digest(await readFile(join(workspace, "solution.py"))), views: [] };
      for (const [i, camera] of verify.defaults.renderCameras.entries()) {
        const response = await worker.request(workspace, { protocol_version: 5, operation: "render_drawing_view", step_path: "output.step", region_id: `fixed-${i}`, view_spec: camera, width: 768, height: 768, show_hidden_lines: false, output_dir: `drawing_feedback/baselines/verify-${iteration}-${i}` }, { signal: abort });
        if (!response.render_path) throw Error(`Fixed render failed: ${JSON.stringify(response.error)}`);
        const rendered = await image(join(workspace, response.render_path));
        result.push(rendered);
        binding.views.push({ camera, render_path: response.render_path, image_sha256: digest(Buffer.from(rendered.data, "base64")) });
      }
      if (binding.step_sha256 !== digest(await readFile(join(workspace, "output.step")))) throw Error("CAD changed during fixed-view rendering");
      await save(join(root, `fixed-renders-${iteration}.json`), binding);
      return result;
    },
    revise: async (label, prompt) => {
      const validSource = await readFile(join(workspace, "solution.py"), "utf8");
      let code = pythonCode(await ask(label, prompt.includes(executionInstructions) ? prompt : `${prompt}\n${executionInstructions}`, drawings));
      for (let attempt = 0; attempt < job.max_compile_attempts; attempt++) {
        const response = await execute(workspace, code);
        await save(join(root, `${label}-execute-${attempt + 1}.json`), response);
        if (valid(response)) return;
        if (attempt + 1 === job.max_compile_attempts) throw Error(`${label}: compilation/validity budget exhausted`);
        let diagnostics = { diagnostics_only: true, not_requested: "No mesh topology failure" };
        if (response.execution?.success && response.artifact?.validation?.topology_errors?.some(error => /mesh/u.test(error))) {
          diagnostics = await new Promise(resolve => {
            const p = spawn("python", ["/baseline/geometry_diagnostics.py", join(workspace, "output.step")], { stdio: ["ignore", "pipe", "pipe"] });
            let output = "";
            const timer = setTimeout(() => p.kill("SIGKILL"), 60_000);
            const done = value => { clearTimeout(timer); resolve(value); };
            p.stdout.on("data", data => output += data);
            p.stderr.resume();
            p.once("error", error => done({ diagnostic_error: error.message }));
            p.once("close", () => { try { done(JSON.parse(output)); } catch { done({ diagnostic_error: "Coordinate diagnostics unavailable" }); } });
          });
          await save(join(root, `${label}-geometry-diagnostics-${attempt + 1}.json`), diagnostics);
        }
        code = pythonCode(await ask(`${label}-compile-fix-${attempt + 1}`, buildCompileRepairPrompt({ response, code, validSource, diagnostics }), drawings));
      }
    },
    instrument: async (code, binding) => {
      const location = join(workspace, "instrumentation", String(calls));
      await mkdir(location, { recursive: true });
      const response = await execute(location, code);
      if (!valid(response)) throw Error(`Instrumentation execution failed: ${JSON.stringify(response.execution)}`);
      const manifest = await readJson(join(location, "semantic_steps/manifest.json"));
      await seek.validateStepManifest({ manifest, workspace: location, sourceSha256: binding.sourceSha256, initialStepSha256: binding.stepSha256 });
      await new Promise((yes, no) => {
        const p = spawn("python", ["/baseline/seek-cad/render_steps.py", "--workspace", location, "--manifest", "semantic_steps/manifest.json", "--output-dir", "step_renders", "--source-sha256", binding.sourceSha256, "--initial-step-sha256", binding.stepSha256, "--reference-step", join(workspace, "initial/output.step")], { stdio: "inherit", signal: abort });
        p.once("error", no); p.once("close", code => code === 0 ? yes() : no(Error(`Step rendering/equivalence check failed: ${code}`)));
      });
      return { ...await readJson(join(location, "step_renders/render_manifest.json")), workspace: location, semantic_manifest: manifest };
    },
  };
  let outcome;
  try {
    outcome = job.method === "cadcodeverify" ? await runCadCodeVerify(ctx) : job.method === "seek-cad" ? await runSeekCad(ctx) : (() => { throw Error(`Unknown paper protocol ${job.method}`); })();
    const final = await worker.request(workspace, { protocol_version: 5, operation: "finalize" }, { signal: abort, timeoutMs: 600_000 });
    if (!final.finalized) throw Error("Final CAD is invalid");
    await save(join(workspace, "baseline_result.json"), { ...outcome, status: "completed", method: job.method, implementation_kind: "drawing_task_protocol_adaptation", model: job.model, provider: job.provider, thinking_level: job.thinking_level, initial_source_sha256: job.initial_source_sha256, initial_step_sha256: job.initial_step_sha256, source_sha256: digest(await readFile(join(workspace, "solution.py"))), step_sha256: digest(await readFile(join(workspace, "output.step"))), model_calls: calls, usages, artifact: final.artifact });
  } catch (error) {
    await save(join(workspace, "baseline_result.json"), { status: "error", method: job.method, error: error.message, model_calls: calls, usages, last_execution: last });
    throw error;
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main().catch(error => { console.error(error.message); process.exitCode = 2; });
