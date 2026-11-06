import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { copyFile, mkdir, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import test from "node:test";
import { buildPlan, validateConfig } from "./plan.mjs";
import { runSuite } from "./run.mjs";

const sha = value => createHash("sha256").update(value).digest("hex");
const json = async path => JSON.parse(await readFile(path, "utf8"));
const hasFile = path => readFile(path).then(() => true, error => {
  if (error.code === "ENOENT") return false;
  throw error;
});

/**
 * Only host scheduling is simulated. These labelled fake source/STEP bytes are
 * never submitted as CAD/model results and no Docker or model is invoked.
 */
async function fixture(t, options = {}) {
  const root = await mkdtemp(join(tmpdir(), "cad-round-suite-test-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const output = join(root, "experiment");
  const config = validateConfig({
    experiment_id: "unit-test-no-models", provider: "ucloud", model: "gpt-5.6-sol",
    thinking_level: "low", image: "no-docker-unit-test",
    data_inputs_dir: join(root, "inputs"), cases: options.cases ?? ["140"],
    rounds: 3, generation_timeout_minutes: 120, view_search_timeout_minutes: 180,
    repair_timeout_minutes: 120, search_cpus: 2, search_memory: "6g",
    case_parallelism: 2, ...options.config,
  });
  const initialSources = options.reuseInitial
    ? Object.fromEntries(config.cases.map(id => [id, join(root, "existing", id)])) : {};
  const plans = buildPlan(config, output, join(root, "auth"), [], initialSources);
  const byId = new Map(plans.map(plan => [plan.fixture_id, plan]));
  const formals = new Map();
  const events = [];
  const exported = [];
  const roundValidations = [];

  async function materialize(workspace, id, sourceBytes, stepBytes, continuation) {
    await mkdir(join(workspace, "sessions"), { recursive: true });
    await writeFile(join(workspace, "solution.py"), sourceBytes);
    await writeFile(join(workspace, "output.step"), stepBytes);
    await writeFile(join(workspace, "sessions/main.jsonl"), "unit-test-session-not-an-API-log\n");
    const source = sha(sourceBytes), step = sha(stepBytes);
    const job = {
      provider: config.provider, model: config.model, thinking_level: config.thinking_level,
      agent_mode: "base-only", fixture: { id },
      ...(continuation ? { continuation } : {}),
    };
    const result = {
      ...job, fixture_id: id, status: "valid", session_file: "sessions/main.jsonl",
      last_turn: continuation ? continuation.test_round + 1 : 1,
      last_executed_source_sha256: source, last_executed_step_sha256: step,
      artifact: { status: "valid", sha256: step, validation: { is_valid: true, is_watertight: true } },
      tokens: { total: 1 }, cost: 0, tool_calls: 1,
      ...(continuation ? {
        applied_continuation: { id: continuation.id, prompt_sha256: sha(continuation.prompt) },
        optional_reflect_turn: { status: "completed", continuation_id: continuation.id, prompt_sha256: sha(continuation.prompt) },
      } : {}),
      ...(continuation?.dimension_annotation_profile ? {
        dimension_annotations: { profile: continuation.dimension_annotation_profile, status: "completed", source_sha256: source, step_sha256: step, locked_manifest_sha256: continuation.locked_view_manifest_sha256 },
      } : {}),
    };
    await writeFile(join(workspace, ".cadgenbench-pi-job.json"), JSON.stringify(job));
    await writeFile(join(workspace, "agent_result.json"), JSON.stringify(result));
    const formal = { job, result, source, step };
    formals.set(resolve(workspace), formal);
    return formal;
  }

  for (const [id, source] of Object.entries(initialSources)) {
    await materialize(source, id, `# UNIT TEST Initial ${id}\n`, `UNIT TEST Initial STEP ${id}\n`);
  }
  const services = {
    execute: async (command, meta) => {
      events.push({ event: "execute-start", ...meta, command });
      const plan = byId.get(meta.fixture_id);
      assert.ok(plan, "Every launched command must be bound to a requested case");
      // Match the real runner: opening its log creates the parent directory
      // before the continuation CLI validates that output_root is empty.
      assert.equal(typeof meta.logPath, "string");
      await mkdir(dirname(meta.logPath), { recursive: true });
      await writeFile(meta.logPath, "UNIT TEST controller log\n", { flag: "a" });
      if (meta.stage === "reflection") {
        const item = plan[meta.method].find(round => round.round === meta.round);
        const entries = await readdir(item.output_root).catch(error => {
          if (error.code === "ENOENT") return [];
          throw error;
        });
        assert.deepEqual(entries, [], "The real continuation CLI refuses a pre-populated output directory");
      }
      await new Promise(done => setTimeout(done, meta.stage === "reflection" && meta.method === "vanilla" ? 20 : 2));
      if (meta.stage === "generation") {
        await materialize(plan.initial_source, plan.fixture_id, `# UNIT TEST Initial ${plan.fixture_id}\n`, `UNIT TEST Initial STEP ${plan.fixture_id}\n`);
      } else if (meta.stage === "reflection") {
        const item = plan[meta.method].find(round => round.round === meta.round);
        const baseline = formals.get(resolve(item.source));
        assert.ok(baseline, "A round must start from an existing completed predecessor");
        const noOp = meta.method === "ours" && meta.round === 3;
        const sourceBytes = noOp ? await readFile(join(item.source, "solution.py")) : `# UNIT TEST ${meta.method} ${meta.fixture_id} round ${meta.round}\n`;
        const stepBytes = noOp ? await readFile(join(item.source, "output.step")) : `UNIT TEST ${meta.method} STEP ${meta.fixture_id} round ${meta.round}\n`;
        const continuation = {
          id: `${meta.method}-${meta.fixture_id}-${meta.round}`, prompt: "unit-test-continuation",
          kind: meta.method === "ours" ? "main_self_reflect_inspect" : "main_self_reflect",
          test_round: meta.round, baseline_source_sha256: baseline.source, baseline_step_sha256: baseline.step,
          ...(meta.round > 1 ? { previous_continuation_id: baseline.job.continuation.id } : {}),
          ...(meta.method === "ours" ? { dimension_annotation_profile: "aligned-dimension-subagent-v1", locked_view_manifest_sha256: sha(`fixed-manifest-${meta.fixture_id}`) } : {}),
        };
        await materialize(item.workspace, meta.fixture_id, sourceBytes, stepBytes, continuation);
        await writeFile(join(item.output_root, "continuation_summary.json"), JSON.stringify({
          status: "completed",
          cases: [{ fixture_id: meta.fixture_id, status: "completed", source_workspace: item.source, workspace: item.workspace, usage_delta: { tokens: 1, cost: 0, tool_calls: 1 } }],
        }));
      }
      if (options.failExecute?.(meta)) throw Error(`UNIT TEST injected ${meta.stage} failure ${meta.method ?? ""} ${meta.round ?? ""}`);
      events.push({ event: "execute-end", ...meta, command });
    },
    validateInitial: async workspace => {
      events.push({ event: "validate-initial", workspace });
      const formal = formals.get(resolve(workspace));
      assert.ok(formal, "Initial must actually have been generated or supplied");
      if (options.failInitial?.(formal.job.fixture.id)) throw Error("UNIT TEST unusable Initial");
      return formal;
    },
    validateRound: async (workspace, _config, expected) => {
      const formal = formals.get(resolve(workspace));
      assert.ok(formal, "Round validation cannot validate an unexecuted workspace");
      const continuation = formal.job.continuation;
      const method = continuation.kind === "main_self_reflect_inspect" ? "ours" : "vanilla";
      const plan = byId.get(formal.job.fixture.id);
      const item = plan[method].find(round => round.round === continuation.test_round);
      const prior = formals.get(resolve(item.source));
      const initial = formals.get(resolve(plan.initial_source));
      assert.equal(expected.method, method);
      assert.equal(expected.round, item.round);
      assert.equal(expected.predecessor.workspace, item.source);
      assert.equal(expected.predecessor.source, prior.source);
      assert.equal(expected.predecessor.step, prior.step);
      assert.equal(expected.predecessor.last_turn, prior.result.last_turn);
      assert.equal(expected.predecessor.continuation_id, prior.job.continuation?.id);
      assert.deepEqual(expected.initial, { source: initial.source, step: initial.step });
      if (method === "ours") assert.equal(expected.locked_manifest_sha256, sha(`fixed-manifest-${formal.job.fixture.id}`));
      else assert.equal(expected.locked_manifest_sha256, undefined);
      const record = {
        fixture_id: formal.job.fixture.id, status: "completed", continuation_id: continuation.id,
        prompt_sha256: sha(continuation.prompt),
        baseline: {
          source_sha256: continuation.baseline_source_sha256,
          step_sha256: continuation.baseline_step_sha256,
          ...(continuation.previous_continuation_id ? { previous_continuation_id: continuation.previous_continuation_id } : {}),
        },
        after: { source_sha256: formal.source, step_sha256: formal.step, formal_executions_added: 1, is_valid: true, is_watertight: true },
        usage_delta: { tokens: 1, cost: 0, tool_calls: 1 },
      };
      roundValidations.push({ workspace, expected, formal, record });
      return { formal, record };
    },
    freezeViews: async (source, view) => {
      events.push({ event: "freeze-views", source, view });
      const formal = formals.get(resolve(source));
      if (options.failFreeze?.(formal.job.fixture.id)) throw Error("UNIT TEST stale View Search binding");
      return { fixed_manifest_sha256: sha(`fixed-manifest-${formal.job.fixture.id}`), anchor_step_sha256: formal.step, origin: "automatic_view_search", manually_aligned: false };
    },
    exportSnapshot: async (workspace, destination, formal, metadata) => {
      await mkdir(destination, { recursive: true });
      for (const name of ["solution.py", "output.step", "agent_result.json"]) {
        await copyFile(join(workspace, name), join(destination, name));
      }
      const provenance = { ...metadata, source_workspace: workspace, source_sha256: formal.source, step_sha256: formal.step };
      await writeFile(join(destination, "baseline_provenance.json"), JSON.stringify(provenance));
      await writeFile(join(destination, "result.json"), JSON.stringify({ status: "valid" }));
      exported.push({ workspace, destination, formal, metadata });
      return provenance;
    },
  };
  return { root, output, config, plans, services, events, exported, formals, roundValidations };
}

test("a shared Initial fans out into one Vanilla reflection and three serial Ours rounds", async t => {
  const f = await fixture(t);
  const summary = await runSuite(f.config, f.output, f.plans, f.services);
  assert.equal(summary.status, "completed");
  const item = summary.cases[0], plan = f.plans[0];
  assert.equal(item.initial.status, "completed");
  assert.equal(item.vanilla.length, 1);
  assert.equal(item.ours.length, 3);
  assert.ok([...item.vanilla, ...item.ours].every(round => round.status === "completed"));
  const starts = f.events.filter(event => event.event === "execute-start");
  assert.equal(starts.filter(event => event.stage === "generation").length, 1);
  assert.equal(starts.filter(event => event.stage === "view_search").length, 1);
  assert.equal(f.events.filter(event => event.event === "freeze-views").length, 1);
  assert.equal(starts.filter(event => event.stage === "reflection").length, 4);
  const initial = f.formals.get(resolve(plan.initial_source));
  for (const first of [plan.vanilla[0], plan.ours[0]]) {
    const formal = f.formals.get(resolve(first.workspace));
    assert.equal(formal.job.continuation.baseline_source_sha256, initial.source);
    assert.equal(formal.job.continuation.baseline_step_sha256, initial.step);
  }
  for (let i = 1; i < plan.ours.length; i++) {
    const previous = f.formals.get(resolve(plan.ours[i - 1].workspace));
    const current = f.formals.get(resolve(plan.ours[i].workspace));
    assert.equal(current.job.continuation.baseline_source_sha256, previous.source);
    assert.equal(current.job.continuation.baseline_step_sha256, previous.step);
    assert.equal(current.job.continuation.previous_continuation_id, previous.job.continuation.id);
  }
  const vanillaEnd = f.events.findIndex(event => event.event === "execute-end" && event.method === "vanilla");
  const searchStart = f.events.findIndex(event => event.event === "execute-start" && event.stage === "view_search");
  assert.ok(searchStart >= 0 && searchStart < vanillaEnd, "View Search must not wait for the other branch to finish");
  assert.equal(f.exported.length, 5);
  assert.equal((await json(join(f.output, "summary.json"))).status, "completed");
  const csv = await readFile(join(f.output, "rounds.csv"), "utf8");
  assert.match(csv, /initial/u);
  assert.match(csv, /vanilla/u);
  assert.match(csv, /ours/u);
});

test("an unchanged Ours round still has its own completed reflection and snapshot", async t => {
  const f = await fixture(t);
  const summary = await runSuite(f.config, f.output, f.plans, f.services);
  const [, previous, current] = f.plans[0].ours;
  const before = f.formals.get(resolve(previous.workspace));
  const after = f.formals.get(resolve(current.workspace));
  assert.equal(before.source, after.source);
  assert.equal(before.step, after.step);
  assert.notEqual(before.job.continuation.id, after.job.continuation.id);
  assert.equal(summary.cases[0].ours[2].status, "completed");
  assert.equal((await json(join(current.output_root, "round.json"))).status, "completed");
  assert.equal(await readFile(join(current.evaluation_dir, "output.step"), "utf8"), await readFile(join(previous.evaluation_dir, "output.step"), "utf8"));
  assert.notEqual(current.evaluation_dir, previous.evaluation_dir);
});

test("a failed Ours round preserves its partial workspace but blocks later rounds without a STEP fallback", async t => {
  const f = await fixture(t, { cases: ["140", "146"], failExecute: meta => meta.stage === "reflection" && meta.fixture_id === "140" && meta.method === "ours" && meta.round === 2 });
  const summary = await runSuite(f.config, f.output, f.plans, f.services);
  assert.notEqual(summary.status, "completed");
  const failed = summary.cases.find(item => item.fixture_id === "140");
  const good = summary.cases.find(item => item.fixture_id === "146");
  assert.equal(failed.vanilla[0].status, "completed");
  assert.equal(failed.ours[0].status, "completed");
  assert.equal(failed.ours[1].status, "error");
  assert.equal(failed.ours[2].status, "blocked");
  assert.equal(good.status, "completed");
  assert.ok(good.ours.every(item => item.status === "completed"));
  const [, rejected, blocked] = f.plans[0].ours;
  assert.equal(await hasFile(join(rejected.workspace, "output.step")), true, "A failed execution can leave real partial files in its raw workspace");
  for (const round of [rejected, blocked]) {
    assert.equal(await hasFile(join(round.evaluation_dir, "output.step")), false);
    assert.equal((await json(join(round.evaluation_dir, "result.json"))).status, "missing");
  }
  assert.ok(!f.events.some(event => event.event === "execute-start" && event.method === "ours" && event.fixture_id === "140" && event.round === 3));
  assert.ok(!f.exported.some(item => item.workspace === rejected.workspace));
});

test("View Search binding failure blocks Ours while Vanilla and other cases finish", async t => {
  const f = await fixture(t, { cases: ["140", "146"], failFreeze: id => id === "140" });
  const summary = await runSuite(f.config, f.output, f.plans, f.services);
  const failed = summary.cases.find(item => item.fixture_id === "140");
  assert.equal(failed.initial.status, "completed");
  assert.equal(failed.vanilla[0].status, "completed");
  assert.equal(failed.view_search.status, "error");
  assert.ok(failed.ours.every(item => item.status === "blocked"));
  assert.ok(!f.events.some(event => event.event === "execute-start" && event.fixture_id === "140" && event.method === "ours"));
  assert.equal(summary.cases.find(item => item.fixture_id === "146").status, "completed");
  for (const round of f.plans[0].ours) assert.equal(await hasFile(join(round.evaluation_dir, "output.step")), false);
});

test("an unusable Initial blocks both branches without discarding another case", async t => {
  const f = await fixture(t, { cases: ["140", "146"], failInitial: id => id === "140" });
  const summary = await runSuite(f.config, f.output, f.plans, f.services);
  const failed = summary.cases.find(item => item.fixture_id === "140");
  assert.equal(failed.initial.status, "error");
  assert.ok([...failed.vanilla, ...failed.ours].every(round => round.status === "blocked"));
  assert.equal(summary.cases.find(item => item.fixture_id === "146").status, "completed");
  assert.ok(!f.events.some(event => event.event === "execute-start" && event.fixture_id === "140" && event.stage !== "generation"));
  assert.equal(await hasFile(join(f.plans[0].initial_evaluation_dir, "output.step")), false);
  assert.equal((await json(join(f.plans[0].initial_evaluation_dir, "result.json"))).status, "missing");
});

test("reusing an existing Initial exports it once without starting generation", async t => {
  const f = await fixture(t, { reuseInitial: true });
  const source = await readFile(join(f.plans[0].initial_source, "solution.py"));
  const step = await readFile(join(f.plans[0].initial_source, "output.step"));
  const summary = await runSuite(f.config, f.output, f.plans, f.services);
  assert.equal(summary.status, "completed");
  assert.ok(!f.events.some(event => event.stage === "generation"));
  assert.equal(f.exported.filter(item => item.destination === f.plans[0].initial_evaluation_dir).length, 1);
  assert.deepEqual(await readFile(join(f.plans[0].initial_source, "solution.py")), source);
  assert.deepEqual(await readFile(join(f.plans[0].initial_source, "output.step")), step);
});

test("Initial-only selection does not invoke either reflection branch or View Search", async t => {
  const f = await fixture(t, { config: { methods: ["initial"] } });
  const summary = await runSuite(f.config, f.output, f.plans, f.services);
  assert.equal(summary.status, "completed");
  assert.deepEqual(summary.cases[0].vanilla, []);
  assert.deepEqual(summary.cases[0].ours, []);
  assert.equal(f.events.filter(event => event.event === "execute-start").length, 1);
  assert.equal(f.exported.length, 1);
});

test("controller logs exist without pre-populating a continuation output root", async t => {
  const f = await fixture(t, { config: { rounds: 1 } });
  const summary = await runSuite(f.config, f.output, f.plans, f.services);
  assert.equal(summary.status, "completed");
  for (const event of f.events.filter(event => event.event === "execute-start" && event.stage === "reflection")) {
    assert.equal(await readFile(event.logPath, "utf8"), "UNIT TEST controller log\n");
  }
});
