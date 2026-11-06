#!/usr/bin/env node
// Shared Initial -> one historical Vanilla reflection and N dimension-guided rounds.
import { execFile } from "node:child_process";
import { lstat, mkdir, mkdtemp, readFile, rename, rm, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import {
  digestFile, forEachCaseBounded, freezeAutomaticViews, repositoryRoot, runCommand,
} from "../portable-full-method/run.mjs";
import { exportEvaluation, findInitial, readJson, writeJson } from "../baselines/common.mjs";
import { buildPlan, validateConfig } from "./plan.mjs";
import { validateInitial, validateRound } from "./validation.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const help = `Compare a shared Initial, one Vanilla reflection, and multi-round Ours.

bash integrations/pi/round-comparison/run.sh --setup --cases 140 --rounds 3 --output-dir <new-dir>
bash integrations/pi/round-comparison/run.sh --all-generation --rounds 3 --parallel 4 --output-dir <new-dir>

Options: --config <json>, --cases <ids...> OR --all-generation,
 --rounds <1..20> (Ours only; Vanilla always runs once), --parallel <N|all>,
 --methods <initial,vanilla,ours>, --initial-root <original-Main-root>,
 --output-dir <new-dir>, --data-inputs-dir <dir>, --image <image>,
 --network <Docker-network>, --setup, --dry-run, --help.

Initial is always exported. Ours searches/locks views once, then each round
continues the preceding CAD and Main session with refreshed dimensions.
Every round has a separate workspace and evaluation export. Failed rounds
block only their successors; an earlier candidate is never substituted.
--dry-run has no Docker or model calls and requires the Node entrypoints built.
`;

export function parseArgs(argv) {
  const parsed = { cases: [] };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (["--help", "-h", "--setup", "--dry-run", "--all-generation"].includes(arg)) { parsed[arg] = true; continue; }
    if (arg === "--cases") {
      while (argv[i + 1] && !argv[i + 1].startsWith("--")) parsed.cases.push(...argv[++i].split(",").filter(Boolean));
      if (!parsed.cases.length) throw Error("--cases requires IDs");
      continue;
    }
    if (!["--config", "--rounds", "--parallel", "--methods", "--initial-root", "--output-dir", "--data-inputs-dir", "--image", "--network"].includes(arg) || !argv[i + 1] || argv[i + 1].startsWith("--")) throw Error(`Unknown/incomplete option: ${arg}`);
    if (parsed[arg]) throw Error(`Duplicate option: ${arg}`);
    parsed[arg] = argv[++i];
  }
  if (parsed.cases.length && parsed["--all-generation"]) throw Error("Choose --cases or --all-generation");
  return parsed;
}

async function missing(destination, metadata) {
  // A failed generation or round must never expose a stale evaluation STEP.
  for (const file of ["output.step", "output.stp", "solution.py"]) await rm(join(destination, file), { force: true });
  await writeJson(join(destination, "result.json"), { status: "missing", ...metadata });
}

export async function exportSnapshot(workspace, destination, formal, metadata) {
  await mkdir(dirname(destination), { recursive: true });
  const temporary = await mkdtemp(join(dirname(destination), ".export-"));
  try {
    const binding = await exportEvaluation(workspace, temporary, metadata);
    if (binding.source_sha256 !== formal.source || binding.step_sha256 !== formal.step) throw Error("Export does not match the accepted formal CAD");
    await writeJson(join(temporary, "round.json"), metadata);
    if (await lstat(join(destination, "output.step")).catch(() => null)) throw Error(`Refusing to overwrite an exported result: ${destination}`);
    await rm(destination, { recursive: true, force: true }); // Only this run's missing placeholder.
    await rename(temporary, destination);
    return binding;
  } catch (error) {
    await rm(temporary, { recursive: true, force: true });
    throw error;
  }
}

function predecessor(formal, workspace) {
  return {
    source: formal.source, step: formal.step, workspace,
    last_turn: formal.result.last_turn,
    ...(formal.job.continuation?.id ? { continuation_id: formal.job.continuation.id } : {}),
  };
}

const csv = value => `"${String(value ?? "").replaceAll('"', '""')}"`;
async function writeRoundIndex(output, summary) {
  const columns = ["fixture_id", "method", "round", "status", "source_sha256", "step_sha256", "continuation_id", "evaluation_dir", "error"];
  const rows = summary.cases.flatMap(item => [
    { fixture_id: item.fixture_id, method: "initial", round: 0, ...item.initial },
    ...[...item.vanilla, ...item.ours].map(round => ({ fixture_id: item.fixture_id, ...round })),
  ]);
  await writeFile(join(output, "rounds.csv"), [columns.join(","), ...rows.map(row => columns.map(key => csv(row[key])).join(","))].join("\n") + "\n");
}

/** Dependency injection here exercises scheduling/failure semantics without paid calls. */
export async function runSuite(config, output, plans, services = {}) {
  const execute = services.execute ?? runCommand;
  const checkInitial = services.validateInitial ?? validateInitial;
  const checkRound = services.validateRound ?? validateRound;
  const freezeViews = services.freezeViews ?? freezeAutomaticViews;
  const publish = services.exportSnapshot ?? exportSnapshot;
  if (await lstat(join(output, "summary.json")).catch(() => null)) throw Error("Refusing to overwrite an existing comparison summary");
  await mkdir(output, { recursive: true });
  const summary = {
    schema_version: 1, status: "running", started_at: new Date().toISOString(), config,
    initial_policy: "One original Main Initial, shared by independent Vanilla and Ours branches",
    round_policy: "Vanilla once; Ours dimensions + Main continuation, from the preceding round in the same fixed views",
    geometry_accuracy_scored: false,
    cases: plans.map(plan => ({
      fixture_id: plan.fixture_id, status: "pending",
      initial: { status: "pending", workspace: plan.initial_source, evaluation_dir: plan.initial_evaluation_dir },
      view_search: { status: plan.ours.length ? "pending" : "not_requested" },
      vanilla: plan.vanilla.map(item => ({ method: "vanilla", round: item.round, status: "pending", workspace: item.workspace, evaluation_dir: item.evaluation_dir })),
      ours: plan.ours.map(item => ({ method: "ours", round: item.round, status: "pending", workspace: item.workspace, evaluation_dir: item.evaluation_dir })),
    })),
  };
  let writes = Promise.resolve();
  const persist = () => {
    const snapshot = structuredClone(summary);
    writes = writes.then(() => writeJson(join(output, "summary.json"), snapshot));
    return writes;
  };
  await writeJson(join(output, "experiment_contract.json"), {
    ...summary, cases: plans, runtime: services.runtime ?? null,
    manually_aligned_views: false, view_search_runs_per_case: plans.some(plan => plan.ours.length) ? 1 : 0,
    vanilla_prompt: "historical Direct Reflect, unmodified; ordinary fresh renders allowed",
    vanilla_reflection_rounds: 1, ours_reflection_rounds: config.rounds,
    ground_truth_used_for_generation: false,
  });
  for (const item of summary.cases) {
    await missing(item.initial.evaluation_dir, { method: "initial", round: 0, baseline_status: "pending" });
    for (const round of [...item.vanilla, ...item.ours]) await missing(round.evaluation_dir, { method: round.method, round: round.round, baseline_status: "pending" });
  }
  await persist();
  let interrupted = false;
  const stop = () => { interrupted = true; };
  process.once("SIGINT", stop);
  process.once("SIGTERM", stop);
  const assertRunning = () => { if (interrupted) { const error = Error("Comparison interrupted"); error.interrupted = true; throw error; } };
  const block = async (rounds, reason) => {
    for (const item of rounds) {
      if (item.status !== "pending") continue;
      Object.assign(item, { status: "blocked", error: reason, finished_at: new Date().toISOString() });
      await missing(item.evaluation_dir, { method: item.method, round: item.round, baseline_status: "blocked", error: reason });
    }
    await persist();
  };
  try {
    await forEachCaseBounded(plans, config.case_parallelism ?? 1, async plan => {
      const item = summary.cases.find(record => record.fixture_id === plan.fixture_id);
      item.status = "running";
      item.started_at = new Date().toISOString();
      let initial;
      try {
        assertRunning();
        item.initial.status = "running";
        await persist();
        if (plan.initial_error) throw Error(plan.initial_error);
        if (plan.generate) await execute(plan.generate, { stage: "generation", fixture_id: plan.fixture_id, logPath: join(plan.paths.root, "generation.log"), timeoutMinutes: config.generation_timeout_minutes });
        initial = await checkInitial(plan.initial_source, config);
        Object.assign(item.initial, { status: "completed", source_sha256: initial.source, step_sha256: initial.step, reused: !plan.generate });
        const binding = await publish(plan.initial_source, plan.initial_evaluation_dir, initial, { fixture_id: plan.fixture_id, method: "initial", round: 0, source_sha256: initial.source, step_sha256: initial.step });
        await writeJson(join(plan.paths.root, "initial_binding.json"), binding);
        await persist();
      } catch (error) {
        if (error.interrupted) interrupted = true;
        Object.assign(item.initial, { status: error.interrupted ? "interrupted" : "error", error: error.message });
        await missing(plan.initial_evaluation_dir, { method: "initial", round: 0, baseline_status: item.initial.status, error: error.message });
        await block([...item.vanilla, ...item.ours], `Initial unavailable: ${error.message}`);
        item.view_search.status = plan.ours.length ? "blocked" : "not_requested";
        item.status = item.initial.status;
        item.finished_at = new Date().toISOString();
        await persist();
        return !interrupted;
      }

      const reflect = async (method, rounds, records, frame) => {
        let prior = initial;
        let priorWorkspace = plan.initial_source;
        for (let index = 0; index < rounds.length; index++) {
          const round = rounds[index], record = records[index];
          try {
            assertRunning();
            Object.assign(record, { status: "running", started_at: new Date().toISOString(), predecessor_workspace: priorWorkspace });
            await persist();
            // CLI output roots must be empty at startup; keep host logs beside them.
            await execute(round.command, { stage: "reflection", fixture_id: plan.fixture_id, method, round: round.round, logPath: `${round.output_root}.log`, timeoutMinutes: config.repair_timeout_minutes + 5 });
            const continuation = await readJson(join(round.output_root, "continuation_summary.json"));
            if (continuation.status !== "completed" || continuation.cases?.length !== 1 || continuation.cases[0].fixture_id !== plan.fixture_id || continuation.cases[0].status !== "completed" || resolve(continuation.cases[0].source_workspace ?? "") !== resolve(priorWorkspace) || resolve(continuation.cases[0].workspace ?? "") !== resolve(round.workspace)) throw Error("Continuation summary does not match the requested round and predecessor");
            const checked = await checkRound(round.workspace, config, {
              method, round: round.round,
              predecessor: predecessor(prior, priorWorkspace), initial: { source: initial.source, step: initial.step },
              ...(frame ? { locked_manifest_sha256: frame.fixed_manifest_sha256 } : {}),
            });
            Object.assign(record, checked.record, {
              method, round: round.round, status: "completed", workspace: round.workspace,
              evaluation_dir: round.evaluation_dir, source_sha256: checked.formal.source, step_sha256: checked.formal.step,
              usage_delta: continuation.cases[0].usage_delta ?? null,
              finished_at: new Date().toISOString(),
            });
            await publish(round.workspace, round.evaluation_dir, checked.formal, { fixture_id: plan.fixture_id, ...record, shared_initial_source_sha256: initial.source, shared_initial_step_sha256: initial.step });
            await writeJson(join(round.output_root, "round.json"), { fixture_id: plan.fixture_id, ...record });
            prior = checked.formal;
            priorWorkspace = round.workspace;
            await persist();
          } catch (error) {
            if (error.interrupted) interrupted = true;
            Object.assign(record, { status: error.interrupted ? "interrupted" : "error", error: error.message, finished_at: new Date().toISOString() });
            await missing(round.evaluation_dir, { method, round: round.round, baseline_status: record.status, error: record.error });
            await writeJson(join(round.output_root, "round.json"), { fixture_id: plan.fixture_id, ...record });
            await block(records.slice(index + 1), `${method} round ${round.round} failed: ${error.message}`);
            await persist();
            break;
          }
        }
      };
      const ours = async () => {
        if (!plan.ours.length) return;
        try {
          assertRunning();
          item.view_search.status = "running";
          await persist();
          await execute(plan.search, { stage: "view_search", fixture_id: plan.fixture_id, logPath: join(plan.paths.root, "view_search.log"), timeoutMinutes: config.view_search_timeout_minutes });
          const frame = await freezeViews(plan.initial_source, plan.view_workspace, config);
          Object.assign(item.view_search, { status: "completed", workspace: plan.view_workspace, frame_lock: frame });
          await persist();
          await reflect("ours", plan.ours, item.ours, frame);
        } catch (error) {
          if (error.interrupted) interrupted = true;
          Object.assign(item.view_search, { status: error.interrupted ? "interrupted" : "error", error: error.message });
          await block(item.ours, `View Search unavailable: ${error.message}`);
        }
      };
      await Promise.all([reflect("vanilla", plan.vanilla, item.vanilla), ours()]);
      try {
        item.initial.immutable_after = (await digestFile(join(plan.initial_source, "solution.py"))) === initial.source && (await digestFile(join(plan.initial_source, "output.step"))) === initial.step;
      } catch { item.initial.immutable_after = false; }
      item.status = item.initial.immutable_after && [...item.vanilla, ...item.ours].every(round => round.status === "completed") ? "completed" : interrupted ? "interrupted" : "partial";
      if (!item.initial.immutable_after) item.error = "Shared Initial workspace changed during comparison";
      item.finished_at = new Date().toISOString();
      await persist();
      return !interrupted;
    });
  } finally {
    process.removeListener("SIGINT", stop);
    process.removeListener("SIGTERM", stop);
  }
  await writes;
  summary.status = interrupted ? "interrupted" : summary.cases.every(item => item.status === "completed") ? "completed" : summary.cases.some(item => item.initial.status === "completed") ? "partial" : "error";
  summary.finished_at = new Date().toISOString();
  await persist();
  await writeRoundIndex(output, summary);
  return summary;
}

export async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  if (args["--help"] || args["-h"]) { console.log(help); return 0; }
  if (!args["--output-dir"]) throw Error("--output-dir is required");
  let config = await readJson(resolve(args["--config"] ?? join(here, "config.json")));
  const output = resolve(args["--output-dir"]);
  config.data_inputs_dir = resolve(repositoryRoot, args["--data-inputs-dir"] ?? config.data_inputs_dir);
  config.image = args["--image"] ?? process.env.CAD_ROUND_COMPARISON_IMAGE ?? config.image;
  config.docker_network = args["--network"] ?? process.env.CADGENBENCH_DOCKER_NETWORK ?? config.docker_network;
  if (args.cases.length) config.cases = args.cases;
  if (args["--rounds"]) config.rounds = Number(args["--rounds"]);
  if (args["--methods"]) config.methods = args["--methods"].split(",");
  if (args["--parallel"] && args["--parallel"] !== "all") config.case_parallelism = Number(args["--parallel"]);
  config = validateConfig(config); // Reject invalid options before building images.
  if (!args["--dry-run"] && await lstat(output).catch(() => null)) throw Error(`Output exists; choose a new directory: ${output}`);
  if (args["--setup"] && !args["--dry-run"]) await runCommand(["bash", join(here, "setup.sh")], { env: { ...process.env, CAD_FULL_METHOD_IMAGE: config.image, CAD_BASELINE_IMAGE: config.image } });
  let fixtures;
  try { fixtures = await import("../dist/src/fixtures.js"); }
  catch (error) { if (error.code === "ERR_MODULE_NOT_FOUND") throw Error("Build the Node entrypoints first: bash integrations/pi/round-comparison/setup.sh (or npm --prefix integrations/pi ci && npm --prefix integrations/pi run build)"); throw error; }
  if (args["--all-generation"]) config.cases = await fixtures.discoverGenerationFixtures(config.data_inputs_dir);
  config.case_parallelism = args["--parallel"] === "all" ? config.cases.length : Number(args["--parallel"] ?? config.case_parallelism ?? 1);
  config = validateConfig(config);
  for (const id of config.cases) await fixtures.loadFixture(config.data_inputs_dir, id);
  const piDir = resolve(process.env.PI_CODING_AGENT_DIR || join(homedir(), ".pi/agent"));
  const { providerEnvironment } = await import("../dist/src/docker-worker.js");
  const forwarded = providerEnvironment(config.provider);
  for (const name of forwarded.names) process.env[name] = forwarded.environment[name];
  process.env.PI_CODING_AGENT_DIR = piDir;
  process.env.CADGENBENCH_DOCKER_NETWORK = config.docker_network;
  process.env.CADGENBENCH_SUBAGENT_TIMEOUT_MULTIPLIER = String(config.subagent_timeout_multiplier ?? 1);
  const initialSources = {}, initialErrors = {};
  if (args["--initial-root"]) {
    const root = resolve(args["--initial-root"]);
    for (const id of config.cases) {
      try { initialSources[id] = await findInitial(root, id); }
      catch (error) { initialSources[id] = join(root, id); initialErrors[id] = error.message; }
    }
  }
  const requestedImage = config.image;
  let plans = buildPlan(config, output, piDir, forwarded.names, initialSources);
  for (const plan of plans) if (initialErrors[plan.fixture_id]) plan.initial_error = initialErrors[plan.fixture_id];
  if (args["--dry-run"]) { console.log(JSON.stringify({ dry_run: true, model_calls: 0, docker_invoked: false, output_dir: output, config, cases: plans }, null, 2)); return 0; }
  const inspected = await promisify(execFile)("docker", ["image", "inspect", config.image, "--format", "{{.Id}}"]);
  const imageId = inspected.stdout.trim();
  if (!/^sha256:[a-f0-9]{64}$/u.test(imageId)) throw Error("Cannot bind the actual runtime image ID; run with --setup");
  config = { ...config, image: imageId };
  plans = buildPlan(config, output, piDir, forwarded.names, initialSources);
  for (const plan of plans) if (initialErrors[plan.fixture_id]) plan.initial_error = initialErrors[plan.fixture_id];
  const summary = await runSuite(config, output, plans, { runtime: { requested_image: requestedImage, image_id: imageId } });
  console.log(JSON.stringify({ status: summary.status, output_dir: output, cases: summary.cases.length, vanilla_rounds: 1, ours_rounds: config.rounds, evaluation_dir: join(output, "evaluation") }, null, 2));
  return summary.status === "completed" ? 0 : summary.status === "interrupted" ? 130 : 2;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main().then(code => { process.exitCode = code; }).catch(error => { console.error(error.message); process.exitCode = 2; });
