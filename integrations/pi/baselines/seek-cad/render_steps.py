"""Render executed semantic CAD stages for the Seek-CAD SVF drawing adaptation.

Run inside the CAD container, after executing instrumentation in an isolated
workspace. The reference Initial must be mounted read-only or copied unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_artifact(workspace: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in name.replace("\\", "/").split("/"):
        raise ValueError(f"Unsafe step artifact: {name}")
    path = workspace / name
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing or symlinked step artifact: {name}")
    path.resolve().relative_to(workspace.resolve())
    return path


def _volume(wrapped) -> float:
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(wrapped, props)
    return abs(float(props.Mass()))


def _bbox(shape) -> list[float]:
    box = shape.bounding_box()
    return [box.min.X, box.min.Y, box.min.Z, box.max.X, box.max.Y, box.max.Z]


def geometry_equivalence(reference: Path, candidate: Path, *, relative_volume_tolerance=1e-8):
    """Check identical world-coordinate solids; never fit, scale, or rigidly align."""
    from build123d import import_step
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    a, b = import_step(str(reference)), import_step(str(candidate))
    if a is None or b is None or not a.solids() or not b.solids():
        raise ValueError("Geometry comparison requires two nonempty solids")
    volumes = [_volume(a.wrapped), _volume(b.wrapped)]
    if any(not math.isfinite(v) or v <= 0 for v in volumes):
        raise ValueError("Geometry comparison requires finite positive solid volumes")
    differences = []
    for left, right in [(a, b), (b, a)]:
        cut = BRepAlgoAPI_Cut(left.wrapped, right.wrapped)
        cut.SetFuzzyValue(1e-7)
        cut.Build()
        if not cut.IsDone():
            raise ValueError("BRep symmetric-difference operation failed")
        differences.append(0.0 if cut.Shape().IsNull() else _volume(cut.Shape()))
    ba, bb = _bbox(a), _bbox(b)
    diagonal = math.sqrt(sum((ba[i + 3] - ba[i]) ** 2 for i in range(3)))
    bbox_error = max(abs(x - y) for x, y in zip(ba, bb))
    volume_tolerance = max(1e-7, volumes[0] * relative_volume_tolerance)
    bbox_tolerance = max(1e-5, diagonal * 1e-7)
    return {
        "method": "brep_symmetric_difference_and_world_bbox",
        "equivalent": sum(differences) <= volume_tolerance and bbox_error <= bbox_tolerance,
        "reference_step_sha256": sha256(reference), "candidate_step_sha256": sha256(candidate),
        "reference_volume_mm3": volumes[0], "candidate_volume_mm3": volumes[1],
        "symmetric_difference_volume_mm3": sum(differences), "bbox_linf_error_mm": bbox_error,
        "volume_tolerance_mm3": volume_tolerance, "bbox_tolerance_mm": bbox_tolerance,
        "alignment_applied": False,
    }


def validate_manifest(workspace: Path, manifest: dict, source_sha256: str, initial_step_sha256: str, max_steps: int):
    if manifest.get("version") != 1 or manifest.get("source_sha256") != source_sha256 or manifest.get("initial_step_sha256") != initial_step_sha256:
        raise ValueError("Step manifest does not match Initial source/STEP")
    if not re.fullmatch(r"[a-f0-9]{64}", source_sha256) or not re.fullmatch(r"[a-f0-9]{64}", initial_step_sha256):
        raise ValueError("Invalid Initial source/STEP hashes")
    steps, plan = manifest.get("steps"), manifest.get("modeling_plan")
    if not isinstance(steps, list) or not 1 <= len(steps) <= max_steps:
        raise ValueError("Missing genuine bounded semantic steps")
    if not isinstance(plan, list) or len(plan) != len(steps):
        raise ValueError("Public modeling plan and step evidence differ")
    ids = set()
    for index, step in enumerate(steps):
        sid = step.get("step_id", "")
        if not re.fullmatch(r"S[0-9]{3,}", sid) or sid in ids:
            raise ValueError("Invalid or duplicate semantic step ID")
        ids.add(sid)
        if not isinstance(step.get("description"), str) or not step["description"].strip():
            raise ValueError("Missing step description")
        if step.get("operation") not in {"add", "remove", "refine", "intersect"}:
            raise ValueError("Invalid semantic operation")
        if plan[index].get("step_id") != sid or not str(plan[index].get("description", "")).strip():
            raise ValueError("Public plan does not match stage order")
        cumulative = safe_artifact(workspace, step.get("cumulative_step_path"))
        focus = safe_artifact(workspace, step.get("focus_step_path"))
        if len(steps) > 1 and cumulative == focus:
            raise ValueError("Current-feature focus must not alias the cumulative model")
        if index:
            if step.get("prior_step_path") != steps[index - 1]["cumulative_step_path"]:
                raise ValueError("Step does not identify its preceding cumulative model")
            safe_artifact(workspace, step["prior_step_path"])
    return safe_artifact(workspace, manifest.get("final_step_path"))


def render_scene(paths, colors, output: Path, bbox, caption, width, height):
    import numpy as np
    import pyvista as pv
    from cadgenbench.common.mesh import tessellate_step
    minimum, maximum = np.asarray(bbox[:3]), np.asarray(bbox[3:])
    diagonal = float(np.linalg.norm(maximum - minimum))
    center = (minimum + maximum) / 2
    plotter = pv.Plotter(off_screen=True, window_size=(width, height))
    try:
        plotter.set_background("white")
        for path, color in zip(paths, colors):
            mesh = tessellate_step(path, max(0.01, diagonal * 0.001))
            faces = np.column_stack((np.full(len(mesh.triangles), 3), mesh.triangles)).ravel()
            body = pv.PolyData(mesh.vertices, faces)
            # Mesh triangle boundaries are tessellation artefacts, not CAD
            # edges. Render shaded faces, then the actual B-Rep edge curves.
            plotter.add_mesh(body, color=color[:3], opacity=color[3], show_edges=False,
                             smooth_shading=False)
            edge_points, edge_cells = brep_edge_polylines(path, max(0.05, diagonal / 200))
            if edge_points:
                lines = pv.PolyData(np.asarray(edge_points, dtype=float),
                                    lines=np.asarray(edge_cells, dtype=np.int64))
                plotter.add_mesh(lines, color=(0.28, 0.29, 0.31), opacity=color[3],
                                 line_width=1.2, lighting=False)
        plotter.camera.position = tuple(center + np.array([1.0, -1.0, 0.8]) / np.linalg.norm([1.0, -1.0, 0.8]) * diagonal * 2.4)
        plotter.camera.focal_point = tuple(center)
        plotter.camera.up = (0, 0, 1)
        plotter.camera.parallel_projection = False
        plotter.camera.view_angle = 30
        plotter.reset_camera_clipping_range()
        plotter.add_text(caption, position="upper_left", font_size=10, color="black")
        plotter.screenshot(str(output))
    finally:
        plotter.close()


def brep_edge_polylines(path, segment_length):
    """Sample topological edge curves independently of face triangulation."""
    from build123d import GeomType, import_step
    shape = import_step(str(path))
    points, cells = [], []
    for edge in shape.edges():
        length = float(edge.length)
        if not math.isfinite(length) or length <= 0:
            continue
        count = 2 if edge.geom_type == GeomType.LINE else max(16, min(256, math.ceil(length / segment_length) + 1))
        start = len(points)
        for index in range(count):
            position = edge.position_at(index / (count - 1))
            points.append((position.X, position.Y, position.Z))
        cells.extend([count, *range(start, start + count)])
    return points, cells


def render_semantic_steps(*, workspace: Path, manifest_path: str, output_dir: str,
                          source_sha256: str, initial_step_sha256: str,
                          reference_step: Path, max_steps=12, width=896, height=672):
    from build123d import import_step
    workspace = workspace.resolve()
    manifest_file = safe_artifact(workspace, manifest_path)
    manifest = json.loads(manifest_file.read_text())
    final = validate_manifest(workspace, manifest, source_sha256, initial_step_sha256, max_steps)
    if sha256(reference_step) != initial_step_sha256:
        raise ValueError("Reference Initial STEP no longer matches the controller hash")
    equivalence = geometry_equivalence(reference_step, final)
    if not equivalence["equivalent"]:
        raise ValueError("Instrumentation changed the Initial geometry; no SVF critique may run")
    if Path(output_dir).is_absolute() or ".." in Path(output_dir).parts:
        raise ValueError("Output must remain inside the trace workspace")
    output = workspace / output_dir
    output.resolve().relative_to(workspace)
    output.mkdir(parents=True, exist_ok=False)
    box = _bbox(import_step(str(final)))
    rendered = []
    for step in manifest["steps"]:
        paths, colors = [], []
        if step.get("prior_step_path"):
            paths.append(safe_artifact(workspace, step["prior_step_path"]))
            colors.append((0.62, 0.65, 0.70, 0.16))
        paths.append(safe_artifact(workspace, step["focus_step_path"]))
        colors.append((0.92, 0.39, 0.14, 1.0) if step["operation"] == "remove" else (0.16, 0.44, 0.83, 1.0))
        image = output / f"{step['step_id']}.png"
        render_scene(paths, colors, image, box, f"{step['step_id']}: {step['description']}", width, height)
        rendered.append({**step, "image_path": str(image.relative_to(workspace)), "image_sha256": sha256(image),
                         "cumulative_sha256": sha256(safe_artifact(workspace, step["cumulative_step_path"])),
                         "focus_sha256": sha256(safe_artifact(workspace, step["focus_step_path"]))})
    final_image = output / "FINAL.png"
    render_scene([final], [(0.78, 0.83, 0.90, 1)], final_image, box, "FINAL CAD", width, height)
    report = {"baseline_id": "seek-cad-svf-drawing-adapted", "manifest_sha256": sha256(manifest_file),
              "source_sha256": source_sha256, "initial_step_sha256": initial_step_sha256,
              "geometry_equivalence": equivalence, "step_images": rendered,
              "final_image_path": str(final_image.relative_to(workspace)), "final_image_sha256": sha256(final_image),
              "camera": {"projection": "perspective", "fixed_to_final_bbox": True, "view_angle_deg": 30},
              "renderer": {"source_sha256": sha256(Path(__file__)), "triangle_edges_visible": False,
                           "edge_representation": "sampled BRep topological curves"},
              "trace_provenance": "model-instrumented execution; final solid independently verified; semantic step attribution remains critic evidence"}
    (output / "render_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--manifest", default="semantic_steps/manifest.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--initial-step-sha256", required=True)
    parser.add_argument("--reference-step", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=12)
    args = parser.parse_args()
    result = render_semantic_steps(workspace=args.workspace, manifest_path=args.manifest,
        output_dir=args.output_dir, source_sha256=args.source_sha256,
        initial_step_sha256=args.initial_step_sha256, reference_step=args.reference_step,
        max_steps=args.max_steps)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
