/** CADCodeVerify's Q -> A -> feedback -> code protocol, adapted to drawings. */
export const identity = Object.freeze({
  method: "cadcodeverify-drawing-adaptation",
  paper: "Generating CAD Code with Vision-Language Models for 3D Designs",
  paper_url: "https://openreview.net/forum?id=BLWaTeucYX",
  repository: "https://github.com/Kamel773/CAD_Code_Generation",
  source_commit: "33dcecd6087ff7b8a4454b1ba4d56504a6364399",
  implementation_kind: "paper_protocol_reimplementation",
  source_repository_contains_inference_code: false,
});

export const defaults = Object.freeze({
  minQuestions: 2,
  maxQuestions: 5,
  maxRefinements: 2,
  renderCameras: [
    { name: "azimuth_0", direction: [1, 1, 1], up: [-1, -1, 2] },
    { name: "azimuth_90", direction: [-1, 1, 1], up: [1, -1, 2] },
    { name: "azimuth_180", direction: [-1, -1, 1], up: [1, 1, 2] },
    { name: "azimuth_270", direction: [1, -1, 1], up: [-1, 1, 2] },
  ],
});

function jsonResponse(text) {
  if (typeof text !== "string") throw Error("Expected model text");
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/iu);
  try { return JSON.parse((fenced?.[1] ?? text).trim()); }
  catch { throw Error("CADCodeVerify stage must return valid JSON"); }
}

function nonblank(value, name) {
  if (typeof value !== "string" || !value.trim()) throw Error(`Missing ${name}`);
  return value.trim();
}

export function parseQuestions(text) {
  const value = jsonResponse(text);
  const rows = Array.isArray(value) ? value : value.questions;
  if (!Array.isArray(rows) || rows.length < 2 || rows.length > 5) throw Error("Expected 2–5 verification questions");
  const questions = rows.map((row, index) => ({
    id: typeof row === "string" ? `Q${index + 1}` : nonblank(row.id, "question id"),
    question: nonblank(typeof row === "string" ? row : row.question, "question"),
  }));
  if (new Set(questions.map(q => q.id)).size !== questions.length) throw Error("Duplicate question id");
  return questions;
}

export function parseAnswers(text, questions) {
  const value = jsonResponse(text);
  const rows = Array.isArray(value) ? value : value.answers;
  if (!Array.isArray(rows) || rows.length !== questions.length) throw Error("Every question requires exactly one answer");
  const byId = new Map();
  for (const row of rows) {
    const id = nonblank(row.id, "answer id");
    if (byId.has(id) || !questions.some(q => q.id === id)) throw Error("Duplicate or unknown answer id");
    const answer = { yes: "Yes", no: "No", unclear: "Unclear" }[String(row.answer).trim().toLowerCase()];
    if (!answer) throw Error("Answers must be Yes, No, or Unclear");
    byId.set(id, { id, answer, evidence: nonblank(row.evidence ?? row.reason, "observable answer evidence") });
  }
  return questions.map(q => byId.get(q.id));
}

export function allYes(answers) {
  return Array.isArray(answers) && answers.length >= 2 && answers.every(a => a.answer === "Yes");
}

export function unresolvedAnswers(answers) {
  return answers.filter(answer => answer.answer === "No" || answer.answer === "Unclear");
}

export function buildQuestionsPrompt({ description = "Reconstruct the part in the supplied engineering drawing." } = {}) {
  return `You are defining CADCodeVerify checks from a reference engineering drawing. Read the supplied drawing pages and this request: ${description}
Generate 2 to 5 distinct verification questions about its most important geometric requirements. Each must be a positive Yes/No question: Yes means the requirement is satisfied. Cover the part structure and clearly specified dimensions when these can be checked. Questions must be grounded in the reference, not invented from the current model. Do not answer yet.
Return only JSON: {"questions":[{"id":"Q1","question":"Does the model ...?"}, ...]}.`;
}

export function buildAnswersPrompt({ questions }) {
  return `The reference engineering drawings and four fixed-direction renders of the current generated CAD are supplied. Answer every fixed CADCodeVerify question by comparing this CAD against the reference. Do not modify, omit, or add questions. Use Yes only when the current model satisfies the requirement, No when the evidence establishes a discrepancy, and Unclear when projection or absent information prevents a reliable judgment. A render does not establish an absolute dimension merely because its outline looks similar.
Questions: ${JSON.stringify(questions)}
Return only JSON: {"answers":[{"id":"Q1","answer":"Yes|No|Unclear","evidence":"Brief observable evidence or the specific uncertainty"}, ...]}.`;
}

export function buildFeedbackPrompt({ questions, answers }) {
  const unresolved = unresolvedAnswers(answers).map(a => ({ ...a, question: questions.find(q => q.id === a.id)?.question }));
  if (unresolved.length === 0) throw Error("All checks are Yes; stop without requesting corrective feedback");
  return `Convert only these unresolved CADCodeVerify checks into concise, actionable feedback for CAD program correction. Already satisfied checks are deliberately excluded. For a No answer, describe the supported discrepancy and the needed correction. For Unclear, retain the uncertainty; do not manufacture a geometric defect or unsupported numeric value. Preserve unrelated geometry. Do not generate code in this stage.
Unresolved checks: ${JSON.stringify(unresolved)}
Return JSON {"issues":[{"id":"question id","observation":"Evidence or uncertainty","correction":"Supported corrective guidance, or cautious verification guidance when unclear"}]}.`;
}

export function buildRevisionPrompt({ code, feedback, iteration = 1, executionInstructions = "Generate a complete executable build123d Python program and export its model to output.step." }) {
  if (!Number.isInteger(iteration) || iteration < 1 || iteration > defaults.maxRefinements) throw Error("CADCodeVerify permits two refinement rounds in this protocol");
  return `Revise the current CAD program using the supplied CADCodeVerify feedback and reference drawings. This is refinement ${iteration} of ${defaults.maxRefinements}. Correct supported discrepancies while preserving requirements already satisfied. Do not invent values for uncertain checks. ${executionInstructions}
Feedback: ${typeof feedback === "string" ? feedback : JSON.stringify(feedback)}
Current program:
\`\`\`python
${code}
\`\`\`
Return one complete corrected Python code block.`;
}
