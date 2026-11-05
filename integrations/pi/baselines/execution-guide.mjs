export const executionInstructions = `Return ONLY a complete executable build123d 0.10.0 Python program in a python code block. Export the complete final part to output.step. Units are millimetres. Do not use View Search, annotation tools, GT, or external files.
AUTHORING API REFERENCE:
- Import from build123d. The rounded rectangle sketch is RectangleRounded(width, height, radius), not RoundedRectangle.
- BuildPart.part and standalone Shape/Part/Compound objects can be exported. A boolean subtraction may return ShapeList: preserve all intended components and wrap that list in Compound(children=list(value)) before export_step; ShapeList itself has no wrapped attribute. Do not discard components merely to make export succeed.
- Temporary feature objects inside a BuildPart context can be added implicitly. Construct helper cutters privately or outside the active builder, and use their intended boolean operation.
- Validity includes a closed manifold boundary mesh, beyond a BRep reporting closed shells. Exact tangent contacts between two bores, or between separate pieces, can produce zero-thickness edges. All intended joins must have valid material continuity; correct boolean construction rather than disabling validation or changing tessellation.
Preserve geometry that is already correct and apply only the changes required by the supplied feedback.`;

export function buildCompileRepairPrompt({ response, code, validSource, diagnostics }) {
  return `${executionInstructions}
Repair only the execution/topology error in this proposed revision. Keep its intended drawing corrections and avoid introducing additional features. The pre-revision program below executed validly; use it to identify which new construction caused the error, not as a replacement answer. Do not simply revert the requested correction.
WORKER ERROR:
${JSON.stringify(response)}
READ-ONLY GEOMETRY DIAGNOSTICS:
${JSON.stringify(diagnostics)}
LAST VALID PRE-REVISION PROGRAM:
${validSource}
CURRENT PROPOSED REVISION:
${code}`;
}
