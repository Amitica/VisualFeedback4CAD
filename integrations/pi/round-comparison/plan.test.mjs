import assert from "node:assert/strict";
import { join } from "node:path";
import test from "node:test";
import { casePaths } from "../portable-full-method/run.mjs";
import { buildPlan, roundName, validateConfig } from "./plan.mjs";

const config = {
  experiment_id: "round-comparison-test",
  provider: "ucloud", model: "gpt-5.6-sol", thinking_level: "low",
  cases: ["140", "146"], image: "cadgenbench-pi-agent:round-comparison-test",
  data_inputs_dir: "/data/inputs", generation_timeout_minutes: 120,
  view_search_timeout_minutes: 180, repair_timeout_minutes: 120,
  search_cpus: 2, search_memory: "6g", docker_network: "bridge",
};
const output = "/tmp/cad round experiment";
const auth = "/tmp/pi agent";
const option = (command, key) => command[command.indexOf(key) + 1];

test("defaults to a shared Initial, one Vanilla reflection, and three Ours rounds", () => {
  const normalized = validateConfig(config);
  assert.equal(normalized.rounds, 3);
  assert.deepEqual(normalized.methods, ["initial", "vanilla", "ours"]);
  assert.equal(config.rounds, undefined, "Defaults must not mutate a shared config object");
  const plans = buildPlan(config, output, auth, ["UCLOUD_API_KEY"]);
  assert.deepEqual(plans.map(plan => plan.fixture_id), config.cases);
  for (const plan of plans) {
    assert.deepEqual(plan.paths, casePaths(output, plan.fixture_id));
    assert.equal(plan.initial_source, plan.paths.source);
    assert.equal(plan.initial_evaluation_dir, join(output, "evaluation/initial", plan.fixture_id));
    assert.ok(plan.generate.includes("--base-agent-only"));
    assert.equal(option(plan.generate, "--parallel"), "1");
    assert.equal(plan.vanilla.length, 1);
    assert.equal(plan.ours.length, 3);
    assert.equal(plan.vanilla[0].source, plan.initial_source);
    assert.equal(plan.ours[0].source, plan.initial_source);
    assert.equal(plan.search_root, plan.paths.search);
    assert.equal(plan.view_workspace, plan.paths.view);
    assert.equal(plan.search[plan.search.indexOf("/opt/cadgenbench-pi/dist/src/drawing-search-cli.js") + 1], plan.initial_source);
  }
});

test("Ours chains immediately preceding workspaces while searching and locking views only once", () => {
  const [plan] = buildPlan(config, output, auth);
  for (const [index, item] of plan.ours.entries()) {
    assert.equal(item.method, "ours");
    assert.equal(item.round, index + 1);
    assert.equal(item.source, index ? plan.ours[index - 1].workspace : plan.initial_source);
    assert.equal(item.command[2], item.source);
    assert.equal(item.workspace, join(item.output_root, "run", plan.fixture_id));
    assert.equal(item.evaluation_dir, join(output, "evaluation/ours", roundName(index + 1), plan.fixture_id));
    assert.equal(option(item.command, "--dimension-annotation-profile"), "aligned-dimension-subagent-v1");
    if (index === 0) {
      assert.equal(option(item.command, "--inspect-view-search-root"), plan.search_root);
      assert.ok(!item.command.includes("--allow-chained-self-reflect"));
      assert.ok(!item.command.includes("--reuse-fixed-view-from-source"));
    } else {
      assert.ok(item.command.includes("--allow-chained-self-reflect"));
      assert.ok(item.command.includes("--reuse-fixed-view-from-source"));
      assert.ok(!item.command.includes("--inspect-view-search-root"));
    }
    assert.ok(!item.command.some(arg => arg.endsWith("drawing-search-cli.js")));
  }
  assert.equal(new Set(plan.ours.map(item => item.output_root)).size, 3);
});

test("Vanilla keeps a single historical Direct Reflect without aligned tools or chaining", () => {
  const [plan] = buildPlan({ ...config, rounds: 20 }, output, auth);
  assert.equal(plan.vanilla.length, 1);
  const item = plan.vanilla[0];
  assert.equal(item.round, 1);
  assert.equal(item.evaluation_dir, join(output, "evaluation/vanilla/round-01", plan.fixture_id));
  for (const forbidden of ["--allow-chained-self-reflect", "--reuse-fixed-view-from-source", "--no-new-images", "--inspect-view-search-root", "--dimension-annotation-profile"]) {
    assert.ok(!item.command.includes(forbidden), forbidden);
  }
});

test("Ours round overrides preserve the same model, image, timeout and independent output for every round", () => {
  const [plan] = buildPlan({ ...config, rounds: 5 }, output, auth);
  assert.equal(plan.ours.length, 5);
  for (const item of [...plan.vanilla, ...plan.ours]) {
    assert.equal(option(item.command, "--provider"), config.provider);
    assert.equal(option(item.command, "--model"), config.model);
    assert.equal(option(item.command, "--thinking-level"), "low");
    assert.equal(option(item.command, "--image"), config.image);
    assert.equal(option(item.command, "--timeout-minutes"), "120");
    assert.equal(option(item.command, "--output-dir"), item.output_root);
  }
  assert.equal(roundName(1), "round-01");
  assert.equal(roundName(20), "round-20");
});

test("an existing external Initial skips generation and is visible read-only to View Search", () => {
  const initial = "/tmp/existing experiment/initial/140";
  const [plan, fresh] = buildPlan(config, output, auth, [], { "140": initial });
  assert.equal(plan.initial_source, initial);
  assert.equal(plan.generate, null);
  assert.ok(plan.search.includes(`type=bind,src=${initial},dst=${initial},readonly`));
  assert.equal(plan.search[plan.search.indexOf("/opt/cadgenbench-pi/dist/src/drawing-search-cli.js") + 1], initial);
  assert.equal(plan.vanilla[0].source, initial);
  assert.equal(plan.ours[0].source, initial);
  assert.ok(Array.isArray(fresh.generate), "One reused case must not suppress another case's fresh generation");
  const internal = join(output, "seeds/140");
  const [inside] = buildPlan(config, output, auth, [], { "140": internal });
  assert.ok(!inside.search.includes(`type=bind,src=${internal},dst=${internal},readonly`));
});

test("method selection keeps Initial evidence implicit and omits unrequested branches", () => {
  const [ours] = buildPlan({ ...config, methods: ["ours"] }, output, auth);
  assert.equal(ours.vanilla.length, 0);
  assert.equal(ours.ours.length, 3);
  assert.ok(ours.initial_source && ours.initial_evaluation_dir && ours.generate);
  const [initial] = buildPlan({ ...config, methods: ["initial"] }, output, auth);
  assert.deepEqual(initial.vanilla, []);
  assert.deepEqual(initial.ours, []);
  const [vanilla] = buildPlan({ ...config, methods: ["vanilla"] }, output, auth);
  assert.equal(vanilla.vanilla.length, 1);
  assert.deepEqual(vanilla.ours, []);
});

test("commands retain spaces and shell metacharacters as literal argv rather than shell source", () => {
  const unsafeForShell = "/tmp/CAD $(touch SHOULD_NOT_EXIST) ; literal";
  const [plan] = buildPlan(config, unsafeForShell, "/tmp/auth with spaces", [], { "140": "/tmp/seeds/$(false) literal/140" });
  assert.equal(plan.ours[0].command[2], "/tmp/seeds/$(false) literal/140");
  assert.equal(option(plan.ours[0].command, "--output-dir"), join(unsafeForShell, "cases/140/ours/round-01"));
  assert.equal(plan.search[0], "docker");
  for (const command of [plan.search, ...plan.ours.map(item => item.command), plan.vanilla[0].command]) {
    assert.ok(Array.isArray(command));
    assert.ok(!["bash", "sh"].includes(command[0]));
    assert.ok(!command.includes("-c"));
  }
});

test("rejects ambiguous budgets, unsafe mount paths and invalid method selections", () => {
  for (const rounds of [0, -1, 21, 1.2, "3", null, NaN]) {
    assert.throws(() => validateConfig({ ...config, rounds }), /rounds/u);
  }
  for (const methods of [[], ["ours", "ours"], ["cadcodeverify"], "ours", null]) {
    assert.throws(() => validateConfig({ ...config, methods }), /methods/u);
  }
  assert.throws(() => validateConfig({ ...config, vanilla_rounds: 1 }), /Vanilla runs exactly one/u);
  for (const number of [0, 21, "1"]) assert.throws(() => roundName(number), /rounds/u);
  for (const bad of ["/tmp/out,extra", "/tmp/out\nextra"]) {
    assert.throws(() => buildPlan(config, bad, auth), /Docker mount/u);
    assert.throws(() => buildPlan(config, output, bad), /Docker mount/u);
    assert.throws(() => buildPlan(config, output, auth, [], { "140": bad }), /Docker mount/u);
  }
  assert.throws(() => buildPlan(config, output, auth, [], { "140": "relative/140" }), /absolute path/u);
  assert.throws(() => validateConfig({ ...config, cases: ["../140"] }), /cases/u);
  assert.throws(() => validateConfig({ ...config, case_parallelism: 0 }), /case_parallelism/u);
});
