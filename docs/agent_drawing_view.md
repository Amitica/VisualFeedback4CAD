# 2D drawing projection and Pi self-reflection

CADGenBench has two related 2D interfaces:

- `cadgenbench.tools.drawing_view` is the deterministic geometry layer. It
  crops drawing regions, creates exact OpenCascade hidden-line projections,
  aligns them to raster drawings, scores candidates, and writes comparison
  images.
- The Pi integration exposes one no-argument Agent tool,
  `self_reflect_2d_drawing({})`. It uses three restricted SubAgents to split
  merged pages, find cameras for the persistent crops, and diagnose the current
  CAD model. The main Agent never supplies a crop or camera to this public
  tool.

The 2D result is feedback. Formal acceptance still depends only on a current
`execute_build123d({})` result whose STEP is valid and watertight and whose
source and STEP hashes still match the workspace files.

## Deterministic geometry layer

For a selected view, the Python implementation:

1. reads the STEP B-Rep with OpenCascade;
2. creates visible and hidden curves with exact HLR;
3. aligns visible CAD curves to the reference using uniform scale, small
   in-plane rotation, and translation;
4. computes a score in `[0, 1]`, where higher is better; and
5. writes the original drawing beside the aligned CAD line view.

The registration never uses independent X/Y scaling, shear, or a homography.
Those transforms could hide an incorrect CAD aspect ratio. Reference-to-CAD
distance has a lower weight because engineering drawings also contain
dimensions, text, centre lines, and title blocks.

The main Python APIs are:

- `crop_normalized_region()` for a region in a merged drawing;
- `render_step_view()` for a clean standalone HLR PNG;
- `rank_view_candidates_detailed()` for a complete, persisted candidate
  ranking, including the best-three list and winner/runner-up margin;
- `compare_step_to_drawing()` and `compare_step_to_drawings()` for selected
  views; and
- `write_comparison_sheet()` for aligned per-region rows.

Standard views are `front`, `rear`, `left`, `right`, `top`, `bottom`, and
`iso`. A custom view supplies both `direction` and `up`. A simple straight
section can use a model-space `section_origin` or a relative
`section_offset_fraction` strictly inside `(-1, 1)` plus `section_keep` set to
`front` or `back`.

## Persistent segmentation of merged pages

A dedicated segmentation SubAgent splits every merged page before view search.
It selects regions with normalized coordinates:

```text
[left, top, right, bottom], each coordinate in 0..1000
```

The worker converts that box to pixels. During segmentation, each call writes a
draft crop under the input-SHA cache. Reusing a `region_id` replaces that draft,
which lets the SubAgent inspect and tighten a poor boundary. Normalized
coordinates remain stable when an input page is resized for model context.

`submit_region_manifest` accepts only region IDs and classification metadata.
The controller copies the chosen drafts into canonical
`regions/<region-id>/crop.png` paths, verifies their pixels, then publishes a
`DrawingRegionManifest`. This is the persistent input to view search. The
original merged page is never modified, and view search cannot recrop it.

The four worker operations are internal controller APIs:

| Operation | Purpose |
| --- | --- |
| `crop_drawing_region` | Crop one normalized region from a merged page. |
| `render_drawing_view` | Render one controller-approved view of the current STEP. |
| `rank_drawing_views` | Align every proposed view to a crop and return the full ranking, margin, and aligned comparisons. |
| `compare_drawing_sheet` | Stack the selected aligned comparisons into the final sheet. |

All input paths must be safe paths relative to the fixture workspace. Output
directories must stay under `drawing_feedback/`. Traversal and symbolic-link
outputs are rejected. Expensive 2D work runs in a killable child process and
returns structured drawing or timeout errors.

Regions marked `error` by the controller do not need a CAD render. Their sheet
row contains the original crop and a clear status placeholder, so one failed
region does not discard the rest of the comparison.

## Three restricted SubAgents

The public workflow is:

```text
main Agent calls self_reflect_2d_drawing({})
    -> segmentation SubAgent turns merged pages into persistent crops
    -> controller locks DrawingRegionManifest
    -> controller starts one independent view-search Agent per region concurrently
    -> each Agent sees its crop, its one source page, and the labelled current-CAD reference
    -> deterministic capacity-OT worker renders, aligns, and ranks every candidate
    -> controller locks the view manifest and builds one comparison sheet
    -> reflection SubAgent diagnoses the locked comparisons
    -> main Agent receives findings and may edit solution.py
```

### Segmentation SubAgent

The first SubAgent sees only the complete staged drawing pages. It has exactly
two tools:

- `crop_region`, which creates or replaces a controller-owned draft crop; and
- `submit_region_manifest`, which selects drafts and records a view-type hint,
  confidence, and note.

It cannot inspect the STEP, render CAD, search a camera, run bash, read arbitrary
files, or edit `solution.py`. It preserves every separately drawn geometry view,
including repeated or similar views elsewhere on a page or on another page.
Each crop keeps the full geometry plus its associated dimension text, dimension
lines, arrows, and leaders without clipping. Every annotation belongs to one
region, and same-page crop areas must not overlap; touching boundaries are
allowed. It may crop the same `region_id` several times; only the latest
submitted draft is promoted. Unsubmitted drafts remain work artifacts and never
become view-search input.

The locked manifest records controller-derived page/ROI/path, PNG width and
height, and crop SHA256. The SubAgent cannot supply or change those fields. It
has at most 32 tool calls, with call 32 reserved for submission, ten minutes of
active Agent time, and a twenty-minute wall-clock ceiling. Controlled Python
crop time does not consume the active budget.

### Independent concurrent view-search Agents

The controller starts one independent Agent for every locked region with
`Promise.allSettled`; there is no shared view-search Agent and no concurrency
limit. Each Agent receives exactly three initial images: its target crop, only
the complete source page identified by the locked region's `page`, and a fixed
isometric line render of the current CAD. The prompt gives the CAD render's
exact `direction` and `up` vectors and marks it as shape-only context, not a
ranked candidate. No unrelated drawing page is sent. The Agent has exactly
three tools, all bound by the controller to that one region:

- `render_candidate_view`
- `rank_view_candidates`
- `submit_view_result`

The Agent session receives each custom tool's name, description, and complete
JSON Schema automatically. Those schemas document every field, including the
camera-vector convention, straight-section pair, the mutually exclusive
`candidates`/`candidate_ids` rank modes, and all final submission fields. The
normal call sequence is a batch `rank_view_candidates({candidates:[...]})`,
optional preview/refinement calls, then
`submit_view_result({view_type,status,selected_candidate_id,confidence,note})`.

The tools expose no `region_id`, so an Agent cannot cross region boundaries or
alter the locked page, ROI, crop path, dimensions, or hash. It also cannot run
bash, read arbitrary files, or edit the CAD source. Direction and up vectors
may contain any finite values, but both must be non-zero and perpendicular.
Only simple straight sections are supported.

Each Agent has its own session, call budget, candidate pool, ranking state,
candidate-ID namespace, and `view-search/<region-id>/` artifact directory. A
rank call accepts 1--32 new cameras, with at most 32 unique candidates across
the region's whole search. Calls within one region are sequential; different
regions may infer, render, and score at the same time. The worker returns the
complete numeric ranking, while the Agent sees comparison images only for the
current eligible Top-2. Each visible comparison is directly labelled with its
candidate ID, numeric rank, score, `direction`, `up`, and section parameters.
The Agent may select either Top-1 or Top-2 or add refined candidates. It must
eventually submit `matched` or `ambiguous`; both statuses require one candidate
from the latest eligible Top-2. No fixed margin changes the Agent's status.
Work calls stop at 23 so call 24 remains available for `submit_view_result`.

Every Agent has ten minutes of active inference time and a 120-minute wall
clock ceiling. Controlled render/rank time pauses the active timer. A batch
rank timeout is `60 + 180 * new-candidate-count` seconds. A failed or timed-out
region becomes `error` without cancelling the other searches; Ctrl-C aborts all
active sessions and workers.

Production scoring is exclusively `capacity_transport_v1`: 256 maximum edge
tokens (96 at working resolution), distance scale
`max(3 px, 0.018 * max image side)`, unmatched CAD cost `1.0`, and unmatched
reference cost `0.25`. It computes one-to-one partial transport diagnostics and
does not compute or blend a multiscale pyramid or spatial-grid score.

### Reflection SubAgent

The controller does not let the third SubAgent change the locked camera. It
receives the comparison sheet, manifest, current `solution.py` text, STEP
measurements, validation data, and source/STEP hashes. It has exactly two
tools:

- `inspect_region`, which opens an already locked region; and
- `submit_reflection`, which returns structured findings and next actions.

It can inspect at most eight regions, has ten minutes of active Agent time, and
a twelve-minute wall-clock ceiling. It cannot write files or invoke a renderer.
The main Agent remains the only Agent that edits `solution.py`, and any edit
must be followed by another formal `execute_build123d({})` call.

All restricted Agents use the run's provider, model, thinking level, and
credential setup. Their role, session path, result path, token usage, cost,
duration, tool calls, start/end times, and terminal status are persisted and
included in the fixture totals. Ctrl-C aborts the main session, all restricted
sessions, and active worker processes.

## Cache and artifacts

The segmentation cache key is the SHA256 of the ordered original input paths and
bytes; it deliberately does not include the CAD source or STEP. Before reuse,
the controller recomputes that input digest, checks each canonical crop has its
expected path, opens every PNG, and verifies its recorded width, height, and
SHA256. Any mismatch causes segmentation to run again instead of feeding stale
or edited crops to view search.

The later exact cache key includes the feedback schema/scorer profile,
input-image SHA, current STEP SHA, provider, and model. An exact hit reuses the
complete report. Region results are cached separately below that exact run by
region and crop hash, so restart reruns only missing or invalid regions. Camera
rankings are never reused across STEP hashes; a changed solid starts a fresh
candidate search instead of inheriting an old Top-3 direction bias.

Artifacts for one fixture use this layout:

```text
drawing_feedback/
├── input_cache/<input-sha>/
│   ├── region_manifest.json
│   ├── regions/<region-id>/crop.png
│   ├── work/regions/<region-id>/crop.png
│   └── sessions/segmentation/*.jsonl
└── runs/<cache-key>/
    ├── run.json
    ├── view_manifest.json
    ├── reflection.json
    ├── current-cad-reference/
    │   ├── render.png
    │   └── metadata.json
    ├── view-search/<region-id>/
    │   ├── result.json
    │   ├── candidates/<candidate-id>/render.png
    │   ├── rankings/comparison.png
    │   └── sessions/*.jsonl
    ├── sheet/
    │   ├── comparison.png
    │   └── rows/*.png
    └── sessions/
        └── reflection/*.jsonl
```

`work/regions/` contains replaceable segmentation drafts.
`input_cache/.../regions/` and `region_manifest.json` are the canonical split;
each view-search Agent receives its own crop, only its locked source page, and
the current-CAD reference with explicit camera metadata. The run-level
`run.json` is the exact-cache envelope, its `view_manifest.json` locks regions
and cameras for one STEP, and `reflection.json` is the source-level diagnosis.

## Command-line diagnostics

The standalone CLI remains useful for testing a view selected by a human or
another controller. Its ROI uses pixel `(X, Y, W, H)`, not the worker's
normalized left/top/right/bottom format:

```bash
cadgenbench drawing-view candidate.step reference.png \
  --view top \
  --roi 145 200 500 380 \
  --output-dir feedback
```

For a tightly cropped reference, use `--full-image`. To rank named presets:

```bash
cadgenbench drawing-view candidate.step reference.png \
  --candidates top bottom front rear left right \
  --full-image \
  --output-dir feedback
```

The compact CLI deliberately returns only the winning `score` and
`comparison_png`. Use `rank_view_candidates_detailed()` or the Pi worker when
the full ranking and per-candidate comparisons are required.

For several already selected views, create a manifest:

```json
{
  "views": [
    {
      "name": "main orthographic",
      "reference": "04_main.png",
      "view": "top",
      "roi": [145, 200, 500, 380]
    },
    {
      "name": "side view",
      "reference": "05_side.png",
      "view": "left",
      "roi": [150, 45, 480, 155]
    }
  ]
}
```

Then run:

```bash
cadgenbench drawing-sheet candidate.step views.json --output-dir feedback
```

The returned score is the arithmetic mean across selected views. Rows are
scaled uniformly and padded; they are not stretched or cropped.

## Doctor, image version, and tests

`cadgenbench-pi doctor` builds a box and a synthetic merged drawing containing
multiple view regions and a title block. Without calling an LLM, it verifies
normalized cropping, standalone HLR rendering, candidate ranking, aligned
comparison generation, and final sheet composition in the container.

The current runtime image is `cadgenbench-pi-agent:0.5.0`. Keep older tagged
images available for persisted runs: resume uses the
recorded image ID and never silently upgrades it.

Run the deterministic and integration tests from the repository root:

```bash
python -m pytest tests/tools/test_drawing_view.py \
  tests/integrations/test_pi_worker.py
```

Run the Pi TypeScript checks from `integrations/pi/`:

```bash
npm run check
npm test
```

To test only the new split stage, prepare a fixture workspace with
`.cadgenbench-pi-job.json`, then mount that workspace and the focused harness
into the normal image. Pass the same Pi-directory mount and provider credentials
as a normal run:

```bash
docker run --rm <normal-security-and-credential-options> \
  --mount type=bind,src=<fixture-workspace>,dst=/workspace \
  --mount type=bind,src=$PWD/manual/segment-drawing-smoke.mjs,dst=/segment-drawing-smoke.mjs,readonly \
  cadgenbench-pi-agent:0.5.0 node /segment-drawing-smoke.mjs
```

For a fixture with a known expected count, add
`-e CADGENBENCH_EXPECTED_REGION_COUNT=<n>`. This catches missing or extra
regions; crop boundaries still require visual inspection.

The harness invokes `runDrawingSegmentation()` twice. The first call must write
and verify the canonical crops; the second must be a cache hit without another
worker call. It allows only `prepare_images` and `crop_drawing_region`, so a
passing result proves that camera rendering, ranking, and reflection were not
accidentally started. Its machine-readable
result is also saved as `segmentation_smoke_result.json` in the fixture
workspace.

After `npm run image:build`, run the opt-in Docker checks explicitly against
0.5.0:

```bash
CADGENBENCH_RUN_DOCKER_TESTS=1 \
CADGENBENCH_PI_IMAGE=cadgenbench-pi-agent:0.5.0 \
python -m pytest tests/integrations/test_pi_worker.py
```
