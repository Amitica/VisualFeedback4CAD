import assert from "node:assert/strict";
import test from "node:test";
import { buildCompileRepairPrompt, executionInstructions } from "./execution-guide.mjs";

test("authoring guide covers observed API failures and retains strict geometry validation", () => {
  assert.match(executionInstructions, /RectangleRounded\(width, height, radius\)/u);
  assert.match(executionInstructions, /ShapeList itself has no wrapped attribute/u);
  assert.match(executionInstructions, /Do not discard components/u);
  assert.match(executionInstructions, /rather than disabling validation or changing tessellation/u);
});

test("compile repair separates failed code, last valid reference and structured world-coordinate evidence", () => {
  const response = { artifact: { status: "invalid", validation: { topology_errors: ["mesh non-manifold"] } } };
  const diagnostics = { geometry_modified: false, boundary_edge_locations: [{ endpoints_mm: [[-120, 0, 25], [-90, 0, 25]], incident_triangles: 4 }] };
  const prompt = buildCompileRepairPrompt({ response, diagnostics, validSource: "# LAST_VALID_REFERENCE\npart = previous_part", code: "# FAILED_PROPOSAL\npart = changed_part" });
  assert.ok(prompt.includes(JSON.stringify(response)));
  assert.ok(prompt.includes(JSON.stringify(diagnostics)));
  assert.ok(prompt.indexOf("LAST VALID PRE-REVISION PROGRAM:") < prompt.indexOf("# LAST_VALID_REFERENCE"));
  assert.ok(prompt.indexOf("# LAST_VALID_REFERENCE") < prompt.indexOf("CURRENT PROPOSED REVISION:"));
  assert.ok(prompt.indexOf("CURRENT PROPOSED REVISION:") < prompt.indexOf("# FAILED_PROPOSAL"));
  assert.match(prompt, /not as a replacement answer/u);
  assert.match(prompt, /Do not simply revert the requested correction/u);
});
