import assert from "node:assert/strict";
import test from "node:test";
import { allYes, buildFeedbackPrompt, buildRevisionPrompt, defaults, parseAnswers, parseQuestions } from "./adapter.mjs";
const questions = parseQuestions('{"questions":[{"id":"Q1","question":"Does it have two holes?"},{"id":"Q2","question":"Is the plate thickness correct?"}]}');

test("preserves fixed question identity and does not mistake uncertainty for success", () => {
  const answers = parseAnswers('{"answers":[{"id":"Q2","answer":"Unclear","evidence":"No physical scale"},{"id":"Q1","answer":"Yes","evidence":"Both holes are visible"}]}', questions);
  assert.deepEqual(answers.map(a => a.id), ["Q1", "Q2"]);
  assert.equal(allYes(answers), false);
  const prompt = buildFeedbackPrompt({ questions, answers });
  assert.match(prompt, /No physical scale/u);
  assert.doesNotMatch(prompt, /Both holes are visible/u);
});

test("rejects malformed counts, duplicate identities and missing evidence", () => {
  assert.throws(() => parseQuestions('{"questions":["One?"]}'), /2–5/u);
  assert.throws(() => parseAnswers('{"answers":[{"id":"Q1","answer":"Yes","evidence":"x"},{"id":"Q1","answer":"No","evidence":"y"}]}', questions), /Duplicate/u);
  assert.throws(() => parseAnswers('{"answers":[{"id":"Q1","answer":"Yes","evidence":"x"},{"id":"Q2","answer":"Yes"}]}', questions), /evidence/u);
});

test("stops all-Yes sets and enforces two refinement rounds", () => {
  const answers = questions.map(q => ({ id: q.id, answer: "Yes", evidence: "Observed" }));
  assert.equal(allYes(answers), true);
  assert.throws(() => buildFeedbackPrompt({ questions, answers }), /stop/u);
  assert.throws(() => buildRevisionPrompt({ code: "x", feedback: "x", iteration: 3 }), /two/u);
  assert.equal(defaults.renderCameras.length, 4);
});
