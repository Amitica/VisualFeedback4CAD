from pathlib import Path
import importlib.util
import hashlib
import json

import pytest

spec = importlib.util.spec_from_file_location("seek_cad_render_steps", Path(__file__).with_name("render_steps.py"))
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


def export(path, shape):
    from build123d import export_step
    export_step(shape, str(path))
    return path


def test_symmetric_difference_rejects_same_volume_and_bbox_with_wrong_hole(tmp_path):
    from build123d import Box, Cylinder, Pos
    plate = Box(30, 20, 4)
    a = export(tmp_path / "a.step", plate - Pos(5, 0, 0) * Cylinder(2, 8))
    b = export(tmp_path / "b.step", plate - Pos(-5, 0, 0) * Cylinder(2, 8))
    same = export(tmp_path / "same.step", plate - Pos(5, 0, 0) * Cylinder(2, 8))
    assert renderer.geometry_equivalence(a, same)["equivalent"]
    report = renderer.geometry_equivalence(a, b)
    assert abs(report["reference_volume_mm3"] - report["candidate_volume_mm3"]) < 1e-6
    assert report["bbox_linf_error_mm"] < 1e-6
    assert report["symmetric_difference_volume_mm3"] > 1
    assert not report["equivalent"]


def test_executes_real_semantic_geometry_and_produces_highlighted_perspective_views(tmp_path):
    from build123d import Box, Cylinder, Pos
    from PIL import Image
    folder = tmp_path / "semantic_steps"
    folder.mkdir()
    plate = Box(30, 20, 4)
    tool = Pos(5, 0, 0) * Cylinder(2, 4)
    final = plate - tool
    export(folder / "S001.step", plate)
    export(folder / "S001-focus.step", plate)
    export(folder / "S002.step", final)
    export(folder / "S002-focus.step", tool)
    export(folder / "final.step", final)
    reference = export(tmp_path / "initial.step", final)
    initial_hash = hashlib.sha256(reference.read_bytes()).hexdigest()
    manifest = {
        "version": 1, "source_sha256": "a" * 64, "initial_step_sha256": initial_hash,
        "modeling_plan": [{"step_id": "S001", "description": "Base plate"}, {"step_id": "S002", "description": "Cut mounting hole"}],
        "steps": [
            {"step_id": "S001", "description": "Base plate", "operation": "add", "cumulative_step_path": "semantic_steps/S001.step", "focus_step_path": "semantic_steps/S001-focus.step"},
            {"step_id": "S002", "description": "Cut mounting hole", "operation": "remove", "cumulative_step_path": "semantic_steps/S002.step", "focus_step_path": "semantic_steps/S002-focus.step", "prior_step_path": "semantic_steps/S001.step"},
        ], "final_step_path": "semantic_steps/final.step",
    }
    (folder / "manifest.json").write_text(json.dumps(manifest))
    report = renderer.render_semantic_steps(workspace=tmp_path, manifest_path="semantic_steps/manifest.json", output_dir="rendered", source_sha256="a" * 64, initial_step_sha256=initial_hash, reference_step=reference, width=480, height=360)
    assert report["geometry_equivalence"]["equivalent"]
    assert report["camera"]["projection"] == "perspective"
    assert len(report["step_images"]) == 2
    for item in report["step_images"]:
        with Image.open(tmp_path / item["image_path"]) as image:
            assert image.size == (480, 360)
            assert image.convert("RGB").getextrema() != ((255, 255),) * 3
    with Image.open(tmp_path / report["final_image_path"]) as image:
        assert image.size == (480, 360)
    # An altered final must fail before any critique images are published.
    export(folder / "final.step", plate)
    with pytest.raises(ValueError, match="changed the Initial geometry"):
        renderer.render_semantic_steps(workspace=tmp_path, manifest_path="semantic_steps/manifest.json", output_dir="bad-rendered", source_sha256="a" * 64, initial_step_sha256=initial_hash, reference_step=reference, width=480, height=360)
    assert not (tmp_path / "bad-rendered").exists()


def test_rejects_final_only_trace_and_path_escape(tmp_path):
    with pytest.raises(ValueError, match="Initial source/STEP"):
        renderer.validate_manifest(tmp_path, {"version": 1}, "a" * 64, "b" * 64, 12)
    with pytest.raises(ValueError, match="Unsafe step"):
        renderer.safe_artifact(tmp_path, "../private.step")
