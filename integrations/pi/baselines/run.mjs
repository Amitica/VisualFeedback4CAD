#!/usr/bin/env node
import { createHash } from "node:crypto";
import { execFile } from "node:child_process";
import { copyFile, lstat, mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { forEachCaseBounded, validateFormalWorkspace } from "../portable-full-method/run.mjs";
import { command, discoverInitials, exportEvaluation, findInitial, integrationRoot, readJson, repoRoot, stagePaperCase, writeJson } from "./common.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const methods = ["initial", "vanilla", "cadcodeverify", "seek-cad"];
const help = `Engineering-drawing -> CAD baseline runner

node integrations/pi/baselines/run.mjs --method initial --all-generation --skip-unavailable-inputs --output-dir <new-dir>
node integrations/pi/baselines/run.mjs --method vanilla --initial-root <original-Main-workspaces> --output-dir <new-dir>
node integrations/pi/baselines/run.mjs --method cadcodeverify --initial-root <root> --cases 146 --output-dir <new-dir>
node integrations/pi/baselines/run.mjs --method seek-cad --initial-root <root> --cases 146 --output-dir <new-dir>

Options: --config <json>, --cases <ids...>, --all-generation, --skip-unavailable-inputs,
  --initial-root <root>, --data-inputs-dir <dir>, --output-dir <dir>, --image <tag>,
  --parallel <N|all>, --network <Docker-network>, --dry-run.
Initial preserves the generated pre-reflection model. Vanilla continues that exact Main session
using the historical Direct Reflect prompt, with no View Search or dimension SubAgent.
CADCodeVerify / Seek-CAD SVF are explicitly identified paper-protocol task adaptations.
Official IterCAD uses its independent native runner: python integrations/pi/baselines/itercad/native.py --help
`;

function parse(argv) {
  const a = { cases: [] };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (["--help", "-h", "--dry-run", "--all-generation", "--skip-unavailable-inputs"].includes(arg)) { a[arg] = true; continue; }
    if (arg === "--cases") { while (argv[i + 1] && !argv[i + 1].startsWith("--")) a.cases.push(...argv[++i].split(",").filter(Boolean)); if (!a.cases.length) throw Error("--cases requires IDs"); continue; }
    if (!["--method", "--config", "--initial-root", "--output-dir", "--image", "--data-inputs-dir", "--parallel", "--network"].includes(arg) || !argv[i + 1] || argv[i + 1].startsWith("--")) throw Error(`Unknown/incomplete option ${arg}`);
    if (a[arg]) throw Error(`Duplicate option ${arg}`);
    a[arg] = argv[++i];
  }
  if (a.cases.length && a["--all-generation"]) throw Error("Choose --cases or --all-generation");
  return a;
}

export async function availableFixtures(root, skipMissing) {
  const { loadFixture } = await import("../dist/src/fixtures.js");
  const ids = [], excluded = [];
  for (const e of await readdir(root, { withFileTypes: true })) {
    if (!e.isDirectory() || e.name.startsWith(".")) continue;
    try { await loadFixture(root, e.name); ids.push(e.name); }
    catch (error) {
      if (/supports generation only|requires text\+image/u.test(error.message)) continue;
      if (skipMissing && /input is missing/u.test(error.message)) excluded.push({ id: e.name, reason: error.message });
      else throw error;
    }
  }
  return { ids: ids.sort((a, b) => a.localeCompare(b, undefined, { numeric: true })), excluded };
}

export function validate(config) {
  for (const key of ["provider", "model", "image"]) if (!config[key]?.trim()) throw Error(`Missing ${key}`);
  if (!["off", "minimal", "low", "medium", "high"].includes(config.thinking_level)) throw Error("Invalid thinking_level");
  if (!Number.isInteger(config.parallel) || config.parallel < 1 || config.parallel > 64) throw Error("parallel must be in [1,64]");
  if (!Number.isInteger(config.timeout_minutes) || config.timeout_minutes < 1 || config.timeout_minutes > 1440) throw Error("timeout_minutes must be in [1,1440]");
  if (!Number.isInteger(config.max_compile_attempts) || config.max_compile_attempts < 1 || config.max_compile_attempts > 10) throw Error("max_compile_attempts must be in [1,10]");
  if (!config.cases.length || new Set(config.cases).size !== config.cases.length || config.cases.some(id => !/^[A-Za-z0-9][A-Za-z0-9._-]*$/u.test(id))) throw Error("Invalid/duplicate cases");
  if (!/^[A-Za-z0-9][A-Za-z0-9_.-]*$/u.test(config.docker_network)) throw Error("Invalid Docker network");
}

export async function main(argv = process.argv.slice(2)) {
  const args = parse(argv);
  if (args["--help"] || args["-h"]) { console.log(help); return 0; }
  const method = args["--method"];
  if (!methods.includes(method)) throw Error(`--method must be one of ${methods.join(", ")}`);
  if (!args["--output-dir"]) throw Error("--output-dir is required");
  const config = await readJson(resolve(args["--config"] ?? join(here, "config.json")));
  config.data_inputs_dir = resolve(repoRoot, args["--data-inputs-dir"] ?? config.data_inputs_dir);
  config.image = args["--image"] ?? process.env.CAD_BASELINE_IMAGE ?? config.image;
  config.docker_network = args["--network"] ?? process.env.CADGENBENCH_DOCKER_NETWORK ?? config.docker_network;
  const output = resolve(args["--output-dir"]);
  let initialRoot = args["--initial-root"] ? resolve(args["--initial-root"]) : undefined;
  let excluded = [];
  if (args.cases.length) config.cases = args.cases;
  else if (args["--all-generation"]) {
    const selection = await availableFixtures(config.data_inputs_dir, args["--skip-unavailable-inputs"]);
    config.cases = selection.ids; excluded = selection.excluded;
  } else if (initialRoot) config.cases = await discoverInitials(initialRoot);
  config.parallel = args["--parallel"] === "all" ? config.cases.length : args["--parallel"] ? Number(args["--parallel"]) : config.parallel;
  validate(config);
  process.setMaxListeners(Math.max(process.getMaxListeners(), config.parallel + 8));
  process.env.CADGENBENCH_DOCKER_NETWORK = config.docker_network;
  const { loadFixture } = await import("../dist/src/fixtures.js");
  for (const id of config.cases) await loadFixture(config.data_inputs_dir, id);
  const piDir = resolve(process.env.PI_CODING_AGENT_DIR ?? join(homedir(), ".pi/agent"));
  const { buildDockerCreateArgs, providerEnvironment } = await import("../dist/src/docker-worker.js");
  const forwarded = providerEnvironment(config.provider);
  for (const name of forwarded.names) process.env[name] = forwarded.environment[name];
  const generate = [process.execPath, join(integrationRoot, "dist/src/cli.js"), "run", ...config.cases, "--base-agent-only", "--provider", config.provider, "--model", config.model, "--thinking-level", config.thinking_level, "--parallel", String(config.parallel), "--image", config.image, "--data-inputs-dir", config.data_inputs_dir, "--output-dir", join(output, "generation")];
  const initialSources = {};
  const discoveryErrors = {};
  if (initialRoot) for (const id of config.cases) {
    try { initialSources[id] = await findInitial(initialRoot, id); }
    catch (error) { initialSources[id] = join(initialRoot, id); discoveryErrors[id] = error.message; }
  }
  else for (const id of config.cases) initialSources[id] = join(output, "generation", id);
  const vanilla = config.cases.map(id => [process.execPath, join(integrationRoot, "dist/src/main-self-reflect-continuation-cli.js"), initialSources[id], "--output-dir", join(output, "reflection", id), "--provider", config.provider, "--model", config.model, "--thinking-level", config.thinking_level, "--timeout-minutes", String(config.timeout_minutes), "--image", config.image]);
  const sourceFile = { vanilla: "vanilla/source.json", cadcodeverify: "cadcodeverify/source.json", "seek-cad": "seek-cad/source_contract.json" }[method];
  const codeFiles = ["run.mjs", "common.mjs", ...(method === "vanilla" ? ["vanilla/historical-direct-reflect-102.txt"] : ["initial"].includes(method) ? [] : ["container-protocol.mjs", "execution-guide.mjs", "geometry_diagnostics.py", `${method}/adapter.mjs`, ...(method === "seek-cad" ? ["seek-cad/render_steps.py"] : [])])];
  const codeHashes = Object.fromEntries(await Promise.all(codeFiles.map(async file => [file, createHash("sha256").update(await readFile(join(here, file))).digest("hex")])));
  const contract = { schema_version: 1, method, config, excluded_generation_fixtures: excluded, initial_sources: initialSources, initial_is_shared: !!initialRoot, task: "engineering-drawing-to-parametric-CAD", ground_truth_used_for_generation: false, view_search_enabled: false, dimension_subagent_enabled: false, source_contract: sourceFile ? await readJson(join(here, sourceFile)) : null, adapter_code_sha256: codeHashes, commands: { generate: initialRoot ? null : generate, vanilla: method === "vanilla" ? vanilla : null }, evaluation_dir: join(output, "evaluation"), implementation_kind: ["initial", "vanilla"].includes(method) ? "existing_main_agent_protocol" : "paper_mechanism_drawing_adaptation", metrics: { evaluator: "cadgenbench evaluate", shared_metric_protocol: true, evaluation_only_ground_truth: true, scale_normalization: false, paper_reported_scores_reproduced: false } };
  if (args["--dry-run"]) { console.log(JSON.stringify({ dry_run: true, ...contract }, null, 2)); return 0; }
  if (await lstat(output).catch(() => null)) throw Error(`Output exists; choose a new directory: ${output}`);
  if (["cadcodeverify", "seek-cad"].includes(method)) {
    const { stdout } = await promisify(execFile)("docker", ["image", "inspect", config.image, "--format", "{{.Id}}"]);
    contract.docker_image_id = stdout.trim();
    if (!/^sha256:[a-f0-9]{64}$/u.test(contract.docker_image_id)) throw Error("Could not bind the actual CAD runtime image ID");
  }
  await mkdir(output, { recursive: true });
  await writeJson(join(output, "experiment_contract.json"), contract);
  // Keep the requested case set visible to the evaluator, including failed cases.
  for (const id of config.cases) await writeJson(join(output, "evaluation", id, "result.json"), { status: "missing", baseline_method: method, baseline_status: "pending" });
  if (!initialRoot) {
    try { await command(generate, { log: join(output, "generation.log"), timeoutMs: config.timeout_minutes * 60_000 }); }
    catch (error) { await writeJson(join(output, "generation_failure.json"), { error: error.message, note: "Continue checking each case independently; no failed Initial is replaced." }); }
    initialRoot = join(output, "generation");
  }
  const initialBindings = [];
  const initialErrors = {};
  for (const id of config.cases) {
    const source = initialSources[id];
    try {
      if (discoveryErrors[id]) throw Error(discoveryErrors[id]);
      const formal = await validateFormalWorkspace(source, config);
      initialBindings.push(await exportEvaluation(source, join(output, method === "initial" ? "evaluation" : "initial-evaluation", id), { fixture_id: id, method: "initial", model: formal.job.model, thinking_level: formal.job.thinking_level, session_file: formal.result.session_file }));
    } catch (error) {
      initialErrors[id] = error.message;
      await writeJson(join(output, "evaluation", id, "result.json"), { status: "missing", baseline_method: method, baseline_status: "initial_error", error: error.message });
    }
  }
  await writeJson(join(output, "initial_bindings.json"), initialBindings);
  await writeJson(join(output, "initial_errors.json"), initialErrors);
  if (method === "initial") {
    const status = Object.keys(initialErrors).length ? "partial" : "completed";
    await writeJson(join(output, "summary.json"), { status, method, cases: config.cases.map(id => ({ fixture_id: id, status: initialErrors[id] ? "error" : "completed", ...(initialErrors[id] ? { error: initialErrors[id] } : {}) })), evaluation_dir: contract.evaluation_dir });
    return status === "completed" ? 0 : 2;
  }
  if (method === "vanilla") {
    const { mainSelfReflectPrompt } = await import("../dist/src/main-self-reflect.js");
    await writeFile(join(output, "direct-reflect-prompt.txt"), mainSelfReflectPrompt(config.cases[0]) + "\n");
    const results = [];
    await forEachCaseBounded(config.cases, config.parallel, async id => {
      const record = { fixture_id: id, status: "running" };
      results.push(record);
      try {
        if (initialErrors[id]) throw Error(`Unusable Initial: ${initialErrors[id]}`);
        await command(vanilla[config.cases.indexOf(id)], { log: join(output, "reflection", `${id}.log`) });
        await exportEvaluation(join(output, "reflection", id, "run", id), join(output, "evaluation", id), { fixture_id: id, method: "vanilla_historical_direct_reflect", original_source: initialSources[id] });
        record.status = "completed";
      } catch (error) {
        record.status = "error"; record.error = error.message;
        await writeJson(join(output, "evaluation", id, "result.json"), { status: "missing", baseline_method: method, baseline_status: "error", error: error.message });
      }
      return true;
    });
    const status = results.every(r => r.status === "completed") ? "completed" : "partial";
    await writeJson(join(output, "summary.json"), { status, method, cases: results, evaluation_dir: contract.evaluation_dir });
    return status === "completed" ? 0 : 2;
  }
  const results = [];
  const runtimeImage = contract.docker_image_id ?? config.image;
  await forEachCaseBounded(config.cases, config.parallel, async id => {
    const workspace = join(output, "cases", id);
    const record = { fixture_id: id, status: "running" };
    results.push(record);
    try {
      if (initialErrors[id]) throw Error(`Unusable Initial: ${initialErrors[id]}`);
      await stagePaperCase(initialSources[id], workspace, method, config);
      const name = `cad-baseline-${method}-${id}-${createHash("sha256").update(output).digest("hex").slice(0, 10)}`;
      const create = buildDockerCreateArgs(name, workspace, piDir, runtimeImage, forwarded.names, config.docker_network);
      create.pop();
      const run = ["docker", "run", "--rm", ...create.slice(1), "--mount", `type=bind,src=${here},dst=/baseline,readonly`, runtimeImage, "node", "/baseline/container-protocol.mjs"];
      await writeJson(join(workspace, "command.json"), run);
      await command(run, { log: join(workspace, "run.log") });
      await exportEvaluation(workspace, join(output, "evaluation", id), { fixture_id: id, method, original_source: initialSources[id] });
      record.status = "completed";
    } catch (error) {
      record.status = "error"; record.error = error.message;
      await writeJson(join(output, "evaluation", id, "result.json"), { status: "missing", baseline_method: method, baseline_status: "error", error: error.message });
    }
    await writeJson(join(workspace, "case_status.json"), record);
    return true;
  });
  const summary = { status: results.every(r => r.status === "completed") ? "completed" : "partial", method, cases: results, evaluation_dir: contract.evaluation_dir };
  await writeJson(join(output, "summary.json"), summary);
  console.log(JSON.stringify(summary, null, 2));
  return summary.status === "completed" ? 0 : 2;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main().then(code => { process.exitCode = code; }).catch(error => { console.error(error.message); process.exitCode = 2; });
