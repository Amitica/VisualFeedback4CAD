"""Container-side worker for the Pi CADGenBench integration.

The process reads one JSON request from stdin and writes one JSON response to
stdout. Generated Python runs in a child process; its stdout and stderr never
share the worker protocol stream.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

# OCCT/build123d may load a system libexpat before Python's pyexpat extension.
# Re-exec with the active environment's library directory first, then preload
# pyexpat before importing PIL, artifact helpers, or any lazy CAD module.
from cadgenbench.tools._cli import _reexec_with_environment_libraries

_reexec_with_environment_libraries()
import xml.parsers.expat  # noqa: E402,F401

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps

from cadgenbench.common.artifacts import StepArtifacts
from cadgenbench.integrations.dimension_annotations import DIMENSION_OPERATIONS

PROTOCOL_VERSION = 5
DEFAULT_WORKSPACE = Path("/workspace")
SOLUTION_FILE = "solution.py"
DEFAULT_EXECUTION_TIMEOUT_S = 120.0
DEFAULT_DRAWING_TIMEOUT_S = 600.0
MAX_DRAWING_EDGE_PX = 4096
MAX_DRAWING_SHEET_HEIGHT_PX = 8192
MAX_DRAWING_SHEET_PIXELS = 24_000_000
DRAWING_FEEDBACK_DIR = "drawing_feedback"
DRAWING_OPERATIONS = frozenset(
    {
        "crop_drawing_region",
        "render_drawing_view",
        "rank_drawing_views",
        "compare_drawing_sheet",
        "compose_reflection_inspection",
        "render_fixed_projection",
        "compose_repair_comparison",
        "compose_code_construction_highlight",
        "compose_code_ablation_highlight",
        "compose_visual_diff",
        "compose_code_edit_visualization",
        "compose_repair_candidate_preview",
        "compose_shape_feedback",
        "compare_candidate_topology",
    }
)
SUPPORTED_OPERATIONS = frozenset(
    {
        "doctor",
        "prepare_images",
        "execute",
        "execute_repair_candidate",
        "execute_code_construction_probe",
        "inspect",
        "finalize",
        *DIMENSION_OPERATIONS,
        *DRAWING_OPERATIONS,
    }
)
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_GRID_CELL_PATTERN = re.compile(r"^[A-D][1-4]$")
_DRAWING_REGION_STATUSES = frozenset({"matched", "ambiguous", "error"})
_ACTIVE_SCRIPT_PROCESS: subprocess.Popen[str] | None = None
_ACTIVE_DRAWING_PROCESS: multiprocessing.Process | None = None


class ProtocolError(ValueError):
    """Raised for malformed or unsupported worker requests."""


def _terminate_active_children(signum: int, _frame: Any) -> None:
    """Stop nested CAD work before the one-request worker exits on a signal."""

    script = _ACTIVE_SCRIPT_PROCESS
    if script is not None and script.poll() is None:
        try:
            os.killpg(script.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    drawing = _ACTIVE_DRAWING_PROCESS
    if drawing is not None and drawing.is_alive():
        drawing.terminate()
        drawing.join(0.5)
        if drawing.is_alive():
            drawing.kill()
            drawing.join(0.5)
    raise SystemExit(128 + signum)


def _capture_native_output(function: Any) -> tuple[Any, Exception | None, str]:
    """Run *function* while reserving stdout for the JSON protocol.

    OCC and VTK sometimes write through native file descriptors, bypassing
    ``contextlib.redirect_stdout``. Redirect both process descriptors to a
    temporary file, then restore them before serializing the response.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(sys.stdout.fileno())
    saved_stderr = os.dup(sys.stderr.fileno())
    result: Any = None
    error: Exception | None = None
    diagnostics = b""
    try:
        with tempfile.TemporaryFile() as captured:
            os.dup2(captured.fileno(), sys.stdout.fileno())
            os.dup2(captured.fileno(), sys.stderr.fileno())
            try:
                result = function()
            except Exception as exc:  # noqa: BLE001 - restored then re-raised
                error = exc
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
                captured.seek(0)
                diagnostics = captured.read()
    finally:
        os.dup2(saved_stdout, sys.stdout.fileno())
        os.dup2(saved_stderr, sys.stderr.fileno())
        os.close(saved_stdout)
        os.close(saved_stderr)
    return result, error, diagnostics.decode("utf-8", errors="replace")[-65536:]


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError(f"{label} must be a JSON object")
    return value


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProtocolError(f"{label} must be an integer >= {minimum}")
    return value


def _require_number(value: Any, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{label} must be a number")
    result = float(value)
    if not result >= minimum:
        raise ProtocolError(f"{label} must be >= {minimum}")
    return result


def _require_finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolError(f"{label} must be finite")
    return result


def _require_string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value.strip()):
        suffix = "a non-empty string" if nonempty else "a string"
        raise ProtocolError(f"{label} must be {suffix}")
    return value


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ProtocolError(f"{label} must be a boolean")
    return value


def _require_identifier(value: Any, label: str) -> str:
    identifier = _require_string(value, label).strip()
    if not _IDENTIFIER_PATTERN.fullmatch(identifier):
        raise ProtocolError(
            f"{label} must contain only letters, digits, '.', '_', or '-' "
            "and must start with a letter or digit"
        )
    return identifier


def _require_python_identifier(value: Any, label: str) -> str:
    identifier = _require_string(value, label).strip()
    if len(identifier) > 128 or not identifier.isidentifier() or identifier.startswith("__"):
        raise ProtocolError(
            f"{label} must be a Python variable name of at most 128 characters"
        )
    return identifier


def _require_sha256(value: Any, label: str) -> str:
    digest = _require_string(value, label).strip()
    if not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ProtocolError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _require_grid_cell(value: Any, label: str) -> str:
    cell = _require_string(value, label).strip().upper()
    if not _GRID_CELL_PATTERN.fullmatch(cell):
        raise ProtocolError(f"{label} must be one fixed 4x4 grid cell from A1 through D4")
    return cell


def _grid_cell_roi(cell: str) -> tuple[int, int, int, int]:
    column = ord(cell[0]) - ord("A")
    row = int(cell[1]) - 1
    return (column * 250, row * 250, (column + 1) * 250, (row + 1) * 250)


def _require_vector3(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ProtocolError(f"{label} must be an array of three numbers")
    result = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ProtocolError(f"{label}[{index}] must be a number")
        number = float(item)
        if not math.isfinite(number):
            raise ProtocolError(f"{label}[{index}] must be finite")
        result.append(number)
    if math.sqrt(sum(number * number for number in result)) <= 1e-12:
        raise ProtocolError(f"{label} must be non-zero")
    return (result[0], result[1], result[2])


def _require_normalized_roi(value: Any, label: str = "roi") -> tuple[int, int, int, int]:
    """Validate ``[left, top, right, bottom]`` in the stable 0..1000 space."""

    if not isinstance(value, list) or len(value) != 4:
        raise ProtocolError(
            f"{label} must be [left, top, right, bottom] in normalized 0..1000 coordinates"
        )
    coordinates = tuple(
        _require_int(item, f"{label}[{index}]") for index, item in enumerate(value)
    )
    left, top, right, bottom = coordinates
    if any(coordinate > 1000 for coordinate in coordinates):
        raise ProtocolError(f"{label} coordinates must be <= 1000")
    if left >= right or top >= bottom:
        raise ProtocolError(f"{label} must satisfy left < right and top < bottom")
    return coordinates


def _require_normalized_point(value: Any, label: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ProtocolError(f"{label} must be [x, y] in normalized 0..1000 coordinates")
    point = tuple(
        _require_int(item, f"{label}[{index}]") for index, item in enumerate(value)
    )
    if any(coordinate > 1000 for coordinate in point):
        raise ProtocolError(f"{label} coordinates must be <= 1000")
    return point


def _require_normalized_polyline(
    value: Any,
    label: str,
    *,
    minimum_points: int,
    maximum_points: int = 64,
) -> list[list[int]]:
    if not isinstance(value, list) or not minimum_points <= len(value) <= maximum_points:
        raise ProtocolError(
            f"{label} must contain {minimum_points} to {maximum_points} normalized points"
        )
    points = [
        list(_require_normalized_point(item, f"{label}[{index}]"))
        for index, item in enumerate(value)
    ]
    if len(set(map(tuple, points))) < minimum_points:
        raise ProtocolError(f"{label} must contain {minimum_points} distinct points")
    return points


def _safe_relative_path(workspace: Path, value: Any, label: str) -> tuple[Path, str]:
    relative = Path(_require_string(value, label))
    if relative.is_absolute() or ".." in relative.parts:
        raise ProtocolError(f"{label} must stay inside the workspace")
    resolved_workspace = workspace.resolve()
    resolved = (resolved_workspace / relative).resolve()
    if resolved != resolved_workspace and resolved_workspace not in resolved.parents:
        raise ProtocolError(f"{label} must stay inside the workspace")
    return resolved, relative.as_posix()


def _safe_regular_file(workspace: Path, value: Any, label: str) -> tuple[Path, str]:
    path, relative = _safe_relative_path(workspace, value, label)
    if not path.is_file() or path.is_symlink():
        raise ProtocolError(f"{label} must be a regular file in the workspace")
    return path, relative


def _safe_feedback_output_dir(
    workspace: Path, value: Any, label: str = "output_dir"
) -> tuple[Path, str]:
    """Resolve an output directory constrained to ``drawing_feedback/``."""

    output, relative = _safe_relative_path(workspace, value, label)
    relative_path = Path(relative)
    if not relative_path.parts or relative_path.parts[0] != DRAWING_FEEDBACK_DIR:
        raise ProtocolError(f"{label} must be inside {DRAWING_FEEDBACK_DIR}/")

    workspace_resolved = workspace.resolve()
    current = workspace_resolved
    for part in relative_path.parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ProtocolError(f"{label} must not traverse symbolic links")
    if output.exists() and not output.is_dir():
        raise ProtocolError(f"{label} must name a directory")
    return output, relative


def _relative_to_workspace(workspace: Path, path: str | Path) -> str:
    resolved_workspace = workspace.resolve()
    resolved = Path(path).resolve()
    if resolved != resolved_workspace and resolved_workspace not in resolved.parents:
        raise RuntimeError(f"Drawing output escaped workspace: {resolved}")
    return resolved.relative_to(resolved_workspace).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_files(workspace: Path) -> dict[str, tuple[int, int]]:
    snapshot: dict[str, tuple[int, int]] = {}
    for path in workspace.iterdir():
        if path.is_file() and not path.is_symlink():
            stat = path.stat()
            snapshot[path.name] = (stat.st_mtime_ns, stat.st_size)
    return snapshot


def _run_python(
    script_path: Path,
    workspace: Path,
    timeout_s: float,
) -> dict[str, Any]:
    global _ACTIVE_SCRIPT_PROCESS

    before = _snapshot_files(workspace)
    started = time.monotonic()
    process = subprocess.Popen(
        [sys.executable, str(script_path)],
        cwd=workspace,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    _ACTIVE_SCRIPT_PROCESS = process
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
    duration_s = time.monotonic() - started
    after = _snapshot_files(workspace)
    changed = {
        name: size
        for name, (mtime_ns, size) in after.items()
        if name not in before or before[name] != (mtime_ns, size)
    }
    if timed_out:
        error = f"Script timed out after {timeout_s:g}s"
        if stderr.strip():
            error += f"\n--- partial stderr before timeout ---\n{stderr.strip()}"
        stderr = error
    return {
        "success": process.returncode == 0 and not timed_out,
        "exit_code": None if timed_out else process.returncode,
        "timed_out": timed_out,
        "stdout": stdout,
        "stderr": stderr,
        "files_produced": changed,
        "duration_s": round(duration_s, 6),
    }


def _validation_payload(step_path: Path) -> tuple[dict[str, Any] | None, Any, str | None]:
    try:
        artifacts = StepArtifacts(step_path)
        analysis = artifacts.analysis
        validation = analysis.validation
        measurements = analysis.measurements
        bbox = measurements.bounding_box
        return (
            {
                "is_valid": validation.is_valid,
                "is_watertight": validation.is_watertight,
                "topology_errors": list(validation.topology_errors),
                "solid_count": measurements.solid_count,
                "shell_count": measurements.shell_count,
                "face_count": measurements.face_count,
                "volume": measurements.volume,
                "bbox": {
                    "x": bbox.size_x,
                    "y": bbox.size_y,
                    "z": bbox.size_z,
                    "x_min": bbox.x_min,
                    "x_max": bbox.x_max,
                    "y_min": bbox.y_min,
                    "y_max": bbox.y_max,
                    "z_min": bbox.z_min,
                    "z_max": bbox.z_max,
                },
            },
            artifacts,
            None,
        )
    except Exception as exc:  # noqa: BLE001 - invalid CAD is model feedback
        return None, None, f"{type(exc).__name__}: {exc}"


def _render_iso(artifacts: Any, destination: Path) -> tuple[str | None, str | None]:
    if artifacts is None:
        return None, None
    try:
        from cadgenbench.common.viewer import render_mesh

        images = render_mesh(artifacts.mesh(), views=["iso"])
        if not images:
            return None, "Renderer returned no images"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(images[0].data)
        return destination.as_posix(), None
    except Exception as exc:  # noqa: BLE001 - preserve validation result
        return None, f"{type(exc).__name__}: {exc}"


def _inspect_artifact(
    workspace: Path,
    *,
    render_destination: Path | None = None,
    previous_sha256: str | None = None,
) -> dict[str, Any]:
    candidate = workspace / "output.step"
    if not candidate.exists() or not candidate.is_file() or candidate.is_symlink():
        return {
            "status": "missing",
            "path": None,
            "sha256": None,
            "validation": None,
            "validation_error": None,
            "render_path": None,
            "render_error": None,
        }
    digest = _sha256(candidate)
    validation, artifacts, validation_error = _validation_payload(candidate)
    render_path: str | None = None
    render_error: str | None = None
    if render_destination is not None and validation is not None:
        rendered, render_error = _render_iso(artifacts, render_destination)
        if rendered is not None:
            render_path = render_destination.relative_to(workspace).as_posix()
    if previous_sha256 is not None and digest == previous_sha256:
        status = "stale"
    elif validation is not None and validation["is_valid"] and validation["is_watertight"]:
        status = "valid"
    else:
        status = "invalid"
    return {
        "status": status,
        "path": "output.step",
        "sha256": digest,
        "validation": validation,
        "validation_error": validation_error,
        "render_path": render_path,
        "render_error": render_error,
    }


def _doctor(workspace: Path) -> dict[str, Any]:
    doctor_dir = workspace / ".doctor"
    if doctor_dir.exists():
        shutil.rmtree(doctor_dir)
    doctor_dir.mkdir(parents=True)
    script = doctor_dir / "box.py"
    script.write_text(
        "from build123d import Box, export_step\n"
        "part = Box(10, 20, 30)\n"
        "export_step(part, 'output.step')\n",
        encoding="utf-8",
    )
    execution = _run_python(script, doctor_dir, DEFAULT_EXECUTION_TIMEOUT_S)
    artifact = _inspect_artifact(
        doctor_dir,
        render_destination=doctor_dir / "iso.png",
    )
    try:
        version = importlib.metadata.version("build123d")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    drawing_2d: dict[str, Any]
    if execution["success"] and artifact["status"] == "valid":
        try:
            drawing_2d = _doctor_drawing_2d(doctor_dir)
        except Exception as exc:  # noqa: BLE001 - doctor must remain diagnostic
            drawing_2d = {
                "ok": False,
                "error": {
                    "kind": "drawing",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
            }
    else:
        drawing_2d = {
            "ok": False,
            "error": {
                "kind": "prerequisite",
                "error_type": "InvalidArtifactError",
                "message": "2D doctor requires the synthetic STEP box",
            },
        }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "operation": "doctor",
        "ok": (
            execution["success"]
            and artifact["status"] == "valid"
            and artifact["render_path"] is not None
            and drawing_2d["ok"]
        ),
        "python": sys.version.split()[0],
        "build123d": version,
        "execution": execution,
        "artifact": artifact,
        "drawing_2d": drawing_2d,
    }


def _prepare_images(request: dict[str, Any], workspace: Path) -> dict[str, Any]:
    raw_paths = request.get("paths")
    if not isinstance(raw_paths, list) or not raw_paths:
        raise ProtocolError("paths must be a non-empty array")
    max_edge = _require_int(request.get("max_edge", 2000), "max_edge", minimum=1)
    output_dir = workspace / "prepared_inputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, Any]] = []
    for index, raw_path in enumerate(raw_paths):
        source, source_relative = _safe_relative_path(workspace, raw_path, f"paths[{index}]")
        if not source.is_file() or source.is_symlink():
            raise ProtocolError(f"Input image does not exist: {source_relative}")
        with Image.open(source) as image:
            image.load()
            original_width, original_height = image.size
            scale = min(1.0, max_edge / max(original_width, original_height))
            width = max(1, round(original_width * scale))
            height = max(1, round(original_height * scale))
            prepared = image.convert("RGB")
            if (width, height) != prepared.size:
                prepared = prepared.resize((width, height), Image.Resampling.LANCZOS)
            destination = output_dir / f"{index:02d}.png"
            prepared.save(destination, format="PNG", optimize=True)
        outputs.append(
            {
                "source_path": source_relative,
                "path": destination.relative_to(workspace).as_posix(),
                "mime_type": "image/png",
                "original_width": original_width,
                "original_height": original_height,
                "width": width,
                "height": height,
            }
        )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "operation": "prepare_images",
        "images": outputs,
    }


def _canonical_view_spec(value: Any, label: str = "view_spec") -> dict[str, Any]:
    spec = _require_mapping(value, label)
    allowed_fields = {
        "name",
        "direction",
        "up",
        "section_offset_fraction",
        "section_keep",
    }
    unknown_fields = sorted(set(spec) - allowed_fields)
    if unknown_fields:
        raise ProtocolError(
            f"{label} contains unsupported fields: {', '.join(unknown_fields)}"
        )
    name = _require_identifier(spec.get("name"), f"{label}.name")
    direction = _require_vector3(spec.get("direction"), f"{label}.direction")
    up = _require_vector3(spec.get("up"), f"{label}.up")
    dot = sum(left * right for left, right in zip(direction, up, strict=True))
    direction_length = math.sqrt(sum(value * value for value in direction))
    up_length = math.sqrt(sum(value * value for value in up))
    if abs(dot / (direction_length * up_length)) > 1e-8:
        raise ProtocolError(f"{label}.direction and {label}.up must be perpendicular")

    offset = spec.get("section_offset_fraction")
    if offset is not None:
        offset = _require_number(
            offset,
            f"{label}.section_offset_fraction",
            minimum=-1.0,
        )
        if not -1.0 < offset < 1.0:
            raise ProtocolError(
                f"{label}.section_offset_fraction must be strictly between -1 and 1"
            )
    keep = spec.get("section_keep")
    if keep is not None:
        keep = _require_string(keep, f"{label}.section_keep").strip()
        if keep not in {"front", "back"}:
            raise ProtocolError(f"{label}.section_keep must be 'front' or 'back'")
    if (offset is None) != (keep is None):
        raise ProtocolError(
            f"{label} requires both section_offset_fraction and section_keep"
        )
    result = {
        "name": name,
        "direction": list(direction),
        "up": list(up),
    }
    if offset is not None:
        result["section_offset_fraction"] = offset
        result["section_keep"] = keep
    return result


def _canonical_similarity_transform(
    value: Any, label: str = "transform"
) -> dict[str, Any]:
    transform = _require_mapping(value, label)
    expected_fields = {
        "scale_px_per_unit",
        "rotation_deg",
        "center_x_px",
        "center_y_px",
        "model_center_x",
        "model_center_y",
        "reflected",
    }
    unknown_fields = sorted(set(transform) - expected_fields)
    missing_fields = sorted(expected_fields - set(transform))
    if unknown_fields or missing_fields:
        details = []
        if unknown_fields:
            details.append(f"unsupported: {', '.join(unknown_fields)}")
        if missing_fields:
            details.append(f"missing: {', '.join(missing_fields)}")
        raise ProtocolError(f"{label} fields are invalid ({'; '.join(details)})")
    result = {
        key: _require_finite_number(transform[key], f"{label}.{key}")
        for key in (
            "scale_px_per_unit",
            "rotation_deg",
            "center_x_px",
            "center_y_px",
            "model_center_x",
            "model_center_y",
        )
    }
    if result["scale_px_per_unit"] <= 0:
        raise ProtocolError(f"{label}.scale_px_per_unit must be positive")
    result["reflected"] = _require_bool(transform["reflected"], f"{label}.reflected")
    if result["reflected"]:
        raise ProtocolError(f"{label}.reflected must be false for causal repair comparisons")
    return result


def _prepare_drawing_request(
    request: dict[str, Any], workspace: Path
) -> dict[str, Any]:
    """Validate a drawing request before starting its killable child process."""

    operation = request["operation"]
    canonical: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "operation": operation,
    }
    _, output_relative = _safe_feedback_output_dir(
        workspace, request.get("output_dir")
    )
    canonical["output_dir"] = output_relative

    if operation == "crop_drawing_region":
        _, input_relative = _safe_regular_file(
            workspace, request.get("input_path"), "input_path"
        )
        canonical.update(
            {
                "input_path": input_relative,
                "region_id": _require_identifier(
                    request.get("region_id"), "region_id"
                ),
                "roi": list(_require_normalized_roi(request.get("roi"))),
            }
        )
        return canonical

    if operation == "render_drawing_view":
        _, step_relative = _safe_regular_file(
            workspace, request.get("step_path"), "step_path"
        )
        width = _require_int(request.get("width", 1200), "width", minimum=64)
        height = _require_int(request.get("height", 1200), "height", minimum=64)
        if width > MAX_DRAWING_EDGE_PX or height > MAX_DRAWING_EDGE_PX:
            raise ProtocolError(
                f"width and height must be <= {MAX_DRAWING_EDGE_PX} pixels"
            )
        canonical.update(
            {
                "step_path": step_relative,
                "region_id": _require_identifier(
                    request.get("region_id"), "region_id"
                ),
                "view_spec": _canonical_view_spec(request.get("view_spec")),
                "width": width,
                "height": height,
                "show_hidden_lines": _require_bool(
                    request.get("show_hidden_lines", True), "show_hidden_lines"
                ),
            }
        )
        return canonical

    if operation == "render_fixed_projection":
        _, crop_relative = _safe_regular_file(
            workspace, request.get("crop_path"), "crop_path"
        )
        _, step_relative = _safe_regular_file(
            workspace, request.get("step_path"), "step_path"
        )
        anchor_path = request.get("projection_anchor_step_path")
        anchor_relative = None
        if anchor_path is not None:
            _, anchor_relative = _safe_regular_file(
                workspace,
                anchor_path,
                "projection_anchor_step_path",
            )
        canonical.update(
            {
                "region_id": _require_identifier(
                    request.get("region_id"), "region_id"
                ),
                "crop_path": crop_relative,
                "step_path": step_relative,
                "view_spec": _canonical_view_spec(request.get("view_spec")),
                "transform": _canonical_similarity_transform(
                    request.get("transform")
                ),
                **(
                    {"projection_anchor_step_path": anchor_relative}
                    if anchor_relative is not None
                    else {}
                ),
            }
        )
        return canonical

    if operation == "compose_code_ablation_highlight":
        _, reference_relative = _safe_regular_file(
            workspace, request.get("reference_path"), "reference_path"
        )
        _, current_relative = _safe_regular_file(
            workspace, request.get("current_render_path"), "current_render_path"
        )
        _, ablated_relative = _safe_regular_file(
            workspace, request.get("ablated_render_path"), "ablated_render_path"
        )
        canonical.update(
            {
                "region_id": _require_identifier(
                    request.get("region_id"), "region_id"
                ),
                "construction_ref": _require_identifier(
                    request.get("construction_ref"), "construction_ref"
                ),
                "reference_path": reference_relative,
                "current_render_path": current_relative,
                "ablated_render_path": ablated_relative,
            }
        )
        return canonical

    if operation == "compose_visual_diff":
        _, reference_relative = _safe_regular_file(
            workspace, request.get("reference_path"), "reference_path"
        )
        _, current_relative = _safe_regular_file(
            workspace, request.get("current_render_path"), "current_render_path"
        )
        _, ablated_relative = _safe_regular_file(
            workspace, request.get("ablated_render_path"), "ablated_render_path"
        )
        canonical.update(
            {
                "region_id": _require_identifier(request.get("region_id"), "region_id"),
                "diff_id": _require_identifier(request.get("diff_id"), "diff_id"),
                "construction_ref": _require_identifier(
                    request.get("construction_ref"), "construction_ref"
                ),
                "target_roi": list(_require_normalized_roi(request.get("target_roi"))),
                "reference_path": reference_relative,
                "current_render_path": current_relative,
                "ablated_render_path": ablated_relative,
            }
        )
        return canonical

    if operation == "compose_code_edit_visualization":
        paths = {}
        for field in (
            "reference_path",
            "current_render_path",
            "edited_render_path",
            "current_step_path",
            "edited_step_path",
        ):
            _, relative_path = _safe_regular_file(
                workspace, request.get(field), field
            )
            paths[field] = relative_path
        raw_origin = request.get("projection_origin")
        if not isinstance(raw_origin, list) or len(raw_origin) != 3:
            raise ProtocolError("projection_origin must contain three finite numbers")
        projection_origin = [
            _require_finite_number(value, f"projection_origin[{index}]")
            for index, value in enumerate(raw_origin)
        ]
        canonical.update(
            {
                "region_id": _require_identifier(request.get("region_id"), "region_id"),
                "hypothesis_id": _require_identifier(
                    request.get("hypothesis_id"), "hypothesis_id"
                ),
                "target_roi": list(
                    _require_normalized_roi(request.get("target_roi"), "target_roi")
                ),
                **paths,
                "view_spec": _canonical_view_spec(request.get("view_spec")),
                "transform": _canonical_similarity_transform(request.get("transform")),
                "projection_origin": projection_origin,
            }
        )
        return canonical

    if operation == "compare_candidate_topology":
        _, current_relative = _safe_regular_file(
            workspace, request.get("current_step_path"), "current_step_path"
        )
        raw_candidates = request.get("candidates")
        if not isinstance(raw_candidates, list) or not 1 <= len(raw_candidates) <= 3:
            raise ProtocolError("candidates must contain one to three items")
        candidates = []
        seen_ids: set[str] = set()
        for index, raw_candidate in enumerate(raw_candidates):
            item = _require_mapping(raw_candidate, f"candidates[{index}]")
            unknown_fields = sorted(set(item) - {"candidate_id", "step_path"})
            if unknown_fields:
                raise ProtocolError(
                    f"candidates[{index}] contains unsupported fields: "
                    f"{', '.join(unknown_fields)}"
                )
            candidate_id = _require_identifier(
                item.get("candidate_id"), f"candidates[{index}].candidate_id"
            )
            if candidate_id in seen_ids:
                raise ProtocolError("candidate_id values must be unique")
            seen_ids.add(candidate_id)
            _, step_relative = _safe_regular_file(
                workspace, item.get("step_path"), f"candidates[{index}].step_path"
            )
            candidates.append(
                {"candidate_id": candidate_id, "step_path": step_relative}
            )
        canonical.update(
            {"current_step_path": current_relative, "candidates": candidates}
        )
        return canonical

    if operation == "compose_shape_feedback":
        grouped = "candidates" in request
        legacy_fields = {"candidate_render_path", "candidate_id", "hypothesis"}
        if grouped and legacy_fields.intersection(request):
            raise ProtocolError(
                "candidates must not be mixed with legacy candidate fields"
            )
        paths = {}
        for field in ("reference_path", "current_render_path"):
            _, relative_path = _safe_regular_file(workspace, request.get(field), field)
            paths[field] = relative_path
        if grouped:
            raw_candidates = request["candidates"]
            if (
                not isinstance(raw_candidates, list)
                or not 1 <= len(raw_candidates) <= 3
            ):
                raise ProtocolError("candidates must contain one to three items")
            candidates = []
            seen_ids: set[str] = set()
            for index, raw_candidate in enumerate(raw_candidates):
                label = f"candidates[{index}]"
                item = _require_mapping(raw_candidate, label)
                unknown_fields = sorted(
                    set(item)
                    - {"candidate_id", "render_path", "hypothesis", "step_path"}
                )
                if unknown_fields:
                    raise ProtocolError(
                        f"{label} contains unsupported fields: {', '.join(unknown_fields)}"
                    )
                candidate_id = _require_identifier(
                    item.get("candidate_id"), f"{label}.candidate_id"
                )
                if candidate_id in seen_ids:
                    raise ProtocolError("candidate_id values must be unique")
                seen_ids.add(candidate_id)
                _, render_relative = _safe_regular_file(
                    workspace, item.get("render_path"), f"{label}.render_path"
                )
                hypothesis = _require_string(
                    item.get("hypothesis", ""), f"{label}.hypothesis", nonempty=False
                ).strip()
                if len(hypothesis) > 2000:
                    raise ProtocolError(
                        f"{label}.hypothesis must contain at most 2000 characters"
                    )
                candidates.append(
                    {
                        "candidate_id": candidate_id,
                        "render_path": render_relative,
                        "hypothesis": hypothesis,
                    }
                )
                if "step_path" in item:
                    _, step_relative = _safe_regular_file(
                        workspace, item["step_path"], f"{label}.step_path"
                    )
                    candidates[-1]["step_path"] = step_relative
            canonical["candidates"] = candidates
        else:
            _, render_relative = _safe_regular_file(
                workspace, request.get("candidate_render_path"), "candidate_render_path"
            )
            hypothesis = _require_string(
                request.get("hypothesis", ""), "hypothesis", nonempty=False
            ).strip()
            if len(hypothesis) > 2000:
                raise ProtocolError("hypothesis must contain at most 2000 characters")
            canonical.update(
                {
                    "candidate_render_path": render_relative,
                    "candidate_id": _require_identifier(
                        request.get("candidate_id"), "candidate_id"
                    ),
                    "hypothesis": hypothesis,
                }
            )
        target_roi = _require_normalized_roi(
            request.get("target_roi", [0, 0, 1000, 1000]), "target_roi"
        )
        if target_roi[2] - target_roi[0] < 50 or target_roi[3] - target_roi[1] < 50:
            raise ProtocolError("target_roi width and height must each be at least 50")
        canonical.update(
            {
                "region_id": _require_identifier(request.get("region_id"), "region_id"),
                "target_roi": list(target_roi),
                **paths,
            }
        )
        # Deprecated STEP/frame inputs remain validated for saved callers. The
        # image compositor no longer uses them to calculate BRep highlights.
        brep_fields = {
            "current_step_path",
            "view_spec",
            "transform",
            "projection_origin",
        }
        with_brep = bool(brep_fields.intersection(request)) or any(
            "step_path" in item for item in canonical.get("candidates", [])
        )
        if with_brep:
            if (
                not grouped
                or not brep_fields.issubset(request)
                or any(
                    "step_path" not in item for item in canonical.get("candidates", [])
                )
            ):
                raise ProtocolError(
                    "BRep highlights require Current and all candidate STEPs plus a complete frozen frame"
                )
            _, current_step_relative = _safe_regular_file(
                workspace, request["current_step_path"], "current_step_path"
            )
            view_spec = _canonical_view_spec(request["view_spec"])
            if view_spec.get("section_offset_fraction") is not None:
                raise ProtocolError("BRep highlights require a non-section frozen view")
            raw_origin = request["projection_origin"]
            if not isinstance(raw_origin, list) or len(raw_origin) != 3:
                raise ProtocolError(
                    "projection_origin must contain three finite numbers"
                )
            canonical.update(
                current_step_path=current_step_relative,
                view_spec=view_spec,
                transform=_canonical_similarity_transform(request["transform"]),
                projection_origin=[
                    _require_finite_number(value, f"projection_origin[{index}]")
                    for index, value in enumerate(raw_origin)
                ],
            )
        return canonical

    if operation == "compose_repair_candidate_preview":
        paths = {}
        for field in (
            "reference_path",
            "current_render_path",
            "current_step_path",
        ):
            _, relative_path = _safe_regular_file(
                workspace, request.get(field), field
            )
            paths[field] = relative_path
        raw_candidates = request.get("candidates")
        if not isinstance(raw_candidates, list) or not 2 <= len(raw_candidates) <= 3:
            raise ProtocolError("candidates must contain two or three items")
        candidates = []
        seen_ids: set[str] = set()
        for index, raw_candidate in enumerate(raw_candidates):
            item = _require_mapping(raw_candidate, f"candidates[{index}]")
            unknown_fields = sorted(
                set(item) - {"candidate_id", "hypothesis", "render_path", "step_path"}
            )
            if unknown_fields:
                raise ProtocolError(
                    f"candidates[{index}] contains unsupported fields: "
                    f"{', '.join(unknown_fields)}"
                )
            candidate_id = _require_identifier(
                item.get("candidate_id"), f"candidates[{index}].candidate_id"
            )
            if candidate_id in seen_ids:
                raise ProtocolError("candidate_id values must be unique")
            seen_ids.add(candidate_id)
            hypothesis = _require_string(
                item.get("hypothesis"), f"candidates[{index}].hypothesis"
            ).strip()
            if not hypothesis or len(hypothesis) > 2000:
                raise ProtocolError(
                    f"candidates[{index}].hypothesis must contain 1-2000 characters"
                )
            _, render_relative = _safe_regular_file(
                workspace,
                item.get("render_path"),
                f"candidates[{index}].render_path",
            )
            _, step_relative = _safe_regular_file(
                workspace,
                item.get("step_path"),
                f"candidates[{index}].step_path",
            )
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "hypothesis": hypothesis,
                    "render_path": render_relative,
                    "step_path": step_relative,
                }
            )
        raw_origin = request.get("projection_origin")
        if not isinstance(raw_origin, list) or len(raw_origin) != 3:
            raise ProtocolError("projection_origin must contain three finite numbers")
        canonical.update(
            {
                "region_id": _require_identifier(request.get("region_id"), "region_id"),
                **paths,
                "view_spec": _canonical_view_spec(request.get("view_spec")),
                "transform": _canonical_similarity_transform(request.get("transform")),
                "projection_origin": [
                    _require_finite_number(value, f"projection_origin[{index}]")
                    for index, value in enumerate(raw_origin)
                ],
                "candidates": candidates,
            }
        )
        return canonical

    if operation == "compose_code_construction_highlight":
        _, reference_relative = _safe_regular_file(
            workspace, request.get("reference_path"), "reference_path"
        )
        _, current_relative = _safe_regular_file(
            workspace, request.get("current_render_path"), "current_render_path"
        )
        raw_groups = request.get("groups")
        if not isinstance(raw_groups, list) or not 1 <= len(raw_groups) <= 4:
            raise ProtocolError("groups must contain one to four items")
        allowed_colors = {"red", "blue", "orange", "magenta"}
        groups = []
        seen_names: set[str] = set()
        seen_colors: set[str] = set()
        for index, raw_group in enumerate(raw_groups):
            item = _require_mapping(raw_group, f"groups[{index}]")
            unknown_fields = sorted(
                set(item) - {"group_name", "color", "construction_render_path"}
            )
            if unknown_fields:
                raise ProtocolError(
                    f"groups[{index}] contains unsupported fields: "
                    f"{', '.join(unknown_fields)}"
                )
            group_name = _require_python_identifier(
                item.get("group_name"),
                f"groups[{index}].group_name",
            )
            color = _require_string(
                item.get("color"), f"groups[{index}].color"
            ).strip()
            if color not in allowed_colors:
                raise ProtocolError(
                    f"groups[{index}].color must be one of: "
                    f"{', '.join(sorted(allowed_colors))}"
                )
            if group_name in seen_names:
                raise ProtocolError("group_name values must be unique")
            if color in seen_colors:
                raise ProtocolError("construction group colors must be unique")
            seen_names.add(group_name)
            seen_colors.add(color)
            _, construction_relative = _safe_regular_file(
                workspace,
                item.get("construction_render_path"),
                f"groups[{index}].construction_render_path",
            )
            groups.append(
                {
                    "group_name": group_name,
                    "color": color,
                    "construction_render_path": construction_relative,
                }
            )
        canonical.update(
            {
                "region_id": _require_identifier(
                    request.get("region_id"), "region_id"
                ),
                "reference_path": reference_relative,
                "current_render_path": current_relative,
                "groups": groups,
            }
        )
        return canonical

    if operation == "compose_repair_comparison":
        raw_candidate_ids = request.get("candidate_ids")
        if not isinstance(raw_candidate_ids, list) or not 2 <= len(raw_candidate_ids) <= 3:
            raise ProtocolError("candidate_ids must contain two or three identifiers")
        candidate_ids = [
            _require_identifier(value, f"candidate_ids[{index}]")
            for index, value in enumerate(raw_candidate_ids)
        ]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ProtocolError("candidate_ids must be unique")
        raw_rows = request.get("rows")
        if not isinstance(raw_rows, list) or not 1 <= len(raw_rows) <= 3:
            raise ProtocolError("rows must contain one to three comparison regions")
        rows = []
        seen_regions = set()
        allowed_statuses = {"valid", "invalid", "missing", "error", "timeout"}
        for row_index, raw_row in enumerate(raw_rows):
            row = _require_mapping(raw_row, f"rows[{row_index}]")
            region_id = _require_identifier(
                row.get("region_id"), f"rows[{row_index}].region_id"
            )
            if region_id in seen_regions:
                raise ProtocolError(f"Duplicate comparison region: {region_id}")
            seen_regions.add(region_id)
            _, reference_relative = _safe_regular_file(
                workspace,
                row.get("reference_path"),
                f"rows[{row_index}].reference_path",
            )
            _, current_relative = _safe_regular_file(
                workspace,
                row.get("current_render_path"),
                f"rows[{row_index}].current_render_path",
            )
            raw_candidates = row.get("candidates")
            if not isinstance(raw_candidates, list) or len(raw_candidates) != len(candidate_ids):
                raise ProtocolError(
                    f"rows[{row_index}].candidates must match candidate_ids"
                )
            candidates = []
            for candidate_index, raw_candidate in enumerate(raw_candidates):
                item = _require_mapping(
                    raw_candidate,
                    f"rows[{row_index}].candidates[{candidate_index}]",
                )
                candidate_id = _require_identifier(
                    item.get("candidate_id"),
                    f"rows[{row_index}].candidates[{candidate_index}].candidate_id",
                )
                if candidate_id != candidate_ids[candidate_index]:
                    raise ProtocolError(
                        f"rows[{row_index}] candidate order must match candidate_ids"
                    )
                status = _require_identifier(
                    item.get("status"),
                    f"rows[{row_index}].candidates[{candidate_index}].status",
                )
                if status not in allowed_statuses:
                    raise ProtocolError(
                        f"Unsupported candidate comparison status: {status}"
                    )
                prepared = {"candidate_id": candidate_id, "status": status}
                if item.get("render_path") is not None:
                    _, render_relative = _safe_regular_file(
                        workspace,
                        item["render_path"],
                        f"rows[{row_index}].candidates[{candidate_index}].render_path",
                    )
                    prepared["render_path"] = render_relative
                elif status == "valid":
                    raise ProtocolError("A valid candidate requires render_path")
                candidates.append(prepared)
            rows.append(
                {
                    "region_id": region_id,
                    "reference_path": reference_relative,
                    "current_render_path": current_relative,
                    "candidates": candidates,
                }
            )
        canonical["candidate_ids"] = candidate_ids
        canonical["rows"] = rows
        return canonical

    if operation == "rank_drawing_views":
        _, crop_relative = _safe_regular_file(
            workspace, request.get("crop_path"), "crop_path"
        )
        _, step_relative = _safe_regular_file(
            workspace, request.get("step_path", "output.step"), "step_path"
        )
        raw_candidates = request.get("candidates")
        if not isinstance(raw_candidates, list) or len(raw_candidates) < 1:
            raise ProtocolError("candidates must be a non-empty array")
        if len(raw_candidates) > 32:
            raise ProtocolError("candidates must contain at most 32 items")
        candidates = []
        candidate_ids = set()
        spec_names = set()
        for index, raw_candidate in enumerate(raw_candidates):
            candidate = _require_mapping(raw_candidate, f"candidates[{index}]")
            candidate_id = _require_identifier(
                candidate.get("candidate_id"), f"candidates[{index}].candidate_id"
            )
            if candidate_id in candidate_ids:
                raise ProtocolError(f"Duplicate candidate_id: {candidate_id}")
            view_spec = _canonical_view_spec(
                candidate.get("view_spec"), f"candidates[{index}].view_spec"
            )
            if view_spec["name"] in spec_names:
                raise ProtocolError(
                    f"Candidate view names must be unique: {view_spec['name']}"
                )
            candidate_ids.add(candidate_id)
            spec_names.add(view_spec["name"])
            prepared_candidate = {
                "candidate_id": candidate_id,
                "view_spec": view_spec,
            }
            if candidate.get("render_path") is not None:
                _, render_relative = _safe_regular_file(
                    workspace,
                    candidate["render_path"],
                    f"candidates[{index}].render_path",
                )
                prepared_candidate["render_path"] = render_relative
            candidates.append(prepared_candidate)
        top_k = _require_int(request.get("top_k", 3), "top_k", minimum=1)
        if top_k > len(candidates):
            raise ProtocolError("top_k must not exceed the number of candidates")
        max_iterations = _require_int(
            request.get("max_iterations", 45), "max_iterations", minimum=1
        )
        if max_iterations > 200:
            raise ProtocolError("max_iterations must be <= 200")
        canonical.update(
            {
                "region_id": _require_identifier(
                    request.get("region_id"), "region_id"
                ),
                "crop_path": crop_relative,
                "step_path": step_relative,
                "candidates": candidates,
                "top_k": top_k,
                "allow_reflection": _require_bool(
                    request.get("allow_reflection", False), "allow_reflection"
                ),
                "max_rotation_deg": _require_number(
                    request.get("max_rotation_deg", 6.0), "max_rotation_deg"
                ),
                "min_roi_fill": _require_number(
                    request.get("min_roi_fill", 0.70), "min_roi_fill", minimum=0.001
                ),
                "seed": _require_int(request.get("seed", 0), "seed"),
                "max_iterations": max_iterations,
            }
        )
        if canonical["max_rotation_deg"] > 180.0:
            raise ProtocolError("max_rotation_deg must be <= 180")
        if canonical["min_roi_fill"] > 1.08:
            raise ProtocolError("min_roi_fill must be <= 1.08")
        return canonical

    if operation == "compose_reflection_inspection":
        _, crop_relative = _safe_regular_file(
            workspace, request.get("crop_path"), "crop_path"
        )
        _, aligned_relative = _safe_regular_file(
            workspace,
            request.get("aligned_render_path"),
            "aligned_render_path",
        )
        mode = _require_identifier(request.get("mode"), "mode")
        if mode not in {
            "grid_overview",
            "coordinate_overview",
            "grid_comparison",
            "dual_roi",
        }:
            raise ProtocolError(
                "mode must be grid_overview, coordinate_overview, grid_comparison, or dual_roi"
            )
        canonical.update(
            {
                "region_id": _require_identifier(
                    request.get("region_id"), "region_id"
                ),
                "crop_path": crop_relative,
                "aligned_render_path": aligned_relative,
                "mode": mode,
            }
        )
        if mode in {"grid_overview", "coordinate_overview"}:
            if request.get("reference_roi") is not None or request.get("cad_roi") is not None:
                raise ProtocolError(f"{mode} does not accept ROIs")
            if request.get("reference_cell") is not None or request.get("cad_cell") is not None:
                raise ProtocolError(f"{mode} does not accept selected cells")
            if mode == "grid_overview":
                if request.get("target_annotations") not in (None, []):
                    raise ProtocolError(
                        "target_annotations are supported only by coordinate_overview"
                    )
                return canonical
            raw_annotations = request.get("target_annotations", [])
            if not isinstance(raw_annotations, list) or len(raw_annotations) > 50:
                raise ProtocolError("target_annotations must be an array with at most 50 items")
            annotations: list[dict[str, Any]] = []
            annotation_names: set[str] = set()
            annotation_marks: set[int] = set()
            for index, raw_annotation in enumerate(raw_annotations):
                annotation = _require_mapping(
                    raw_annotation, f"target_annotations[{index}]"
                )
                mark = _require_int(
                    annotation.get("mark"),
                    f"target_annotations[{index}].mark",
                    minimum=1,
                )
                if mark in annotation_marks:
                    raise ProtocolError(f"Duplicate target annotation mark: {mark}")
                name = _require_string(
                    annotation.get("name"), f"target_annotations[{index}].name"
                ).strip()
                if len(name) > 80 or any(ord(character) < 32 for character in name):
                    raise ProtocolError(
                        f"target_annotations[{index}].name must be one line with at most 80 characters"
                    )
                name_key = name.casefold()
                if name_key in annotation_names:
                    raise ProtocolError(f"Duplicate target annotation name: {name}")
                has_point = annotation.get("point") is not None
                has_box = annotation.get("box") is not None
                if has_point == has_box:
                    raise ProtocolError(
                        f"target_annotations[{index}] must contain exactly one of point or box"
                    )
                prepared_annotation: dict[str, Any] = {"mark": mark, "name": name}
                if has_point:
                    prepared_annotation["point"] = list(
                        _require_normalized_point(
                            annotation.get("point"),
                            f"target_annotations[{index}].point",
                        )
                    )
                else:
                    box = _require_normalized_roi(
                        annotation.get("box"), f"target_annotations[{index}].box"
                    )
                    if box[2] - box[0] < 10 or box[3] - box[1] < 10:
                        raise ProtocolError(
                            f"target_annotations[{index}].box width and height must each be at least 10"
                        )
                    prepared_annotation["box"] = list(box)
                annotation_marks.add(mark)
                annotation_names.add(name_key)
                annotations.append(prepared_annotation)
            canonical["target_annotations"] = annotations
            return canonical
        if mode == "dual_roi":
            if request.get("reference_cell") is not None or request.get("cad_cell") is not None:
                raise ProtocolError("dual_roi does not accept grid cells")
            reference_roi = _require_normalized_roi(
                request.get("reference_roi"), "reference_roi"
            )
            cad_roi = _require_normalized_roi(request.get("cad_roi"), "cad_roi")
            for label, roi in (("reference_roi", reference_roi), ("cad_roi", cad_roi)):
                if roi[2] - roi[0] < 50 or roi[3] - roi[1] < 50:
                    raise ProtocolError(f"{label} width and height must each be at least 50")
            canonical.update(
                {"reference_roi": reference_roi, "cad_roi": cad_roi}
            )
            return canonical
        if request.get("reference_roi") is not None or request.get("cad_roi") is not None:
            raise ProtocolError("grid_comparison does not accept free ROIs")
        reference_cell = _require_grid_cell(
            request.get("reference_cell"), "reference_cell"
        )
        cad_cell = _require_grid_cell(request.get("cad_cell"), "cad_cell")
        canonical["reference_cell"] = reference_cell
        canonical["cad_cell"] = cad_cell
        canonical["reference_roi"] = list(_grid_cell_roi(reference_cell))
        canonical["cad_roi"] = list(_grid_cell_roi(cad_cell))
        return canonical

    raw_regions = request.get("regions")
    if not isinstance(raw_regions, list) or not raw_regions:
        raise ProtocolError("regions must be a non-empty array")
    if len(raw_regions) > 12:
        raise ProtocolError("regions must contain at most 12 items")
    regions = []
    region_ids = set()
    for index, raw_region in enumerate(raw_regions):
        region = _require_mapping(raw_region, f"regions[{index}]")
        region_id = _require_identifier(
            region.get("region_id"), f"regions[{index}].region_id"
        )
        if region_id in region_ids:
            raise ProtocolError(f"Duplicate region_id: {region_id}")
        region_ids.add(region_id)
        score = region.get("score")
        if score is not None:
            score = _require_number(score, f"regions[{index}].score")
            if score > 1.0:
                raise ProtocolError(f"regions[{index}].score must be <= 1")
        margin = region.get("margin")
        if margin is not None:
            margin = _require_number(margin, f"regions[{index}].margin")
            if margin > 1.0:
                raise ProtocolError(f"regions[{index}].margin must be <= 1")
        status = _require_identifier(
            region.get("status", "matched"), f"regions[{index}].status"
        )
        if status not in _DRAWING_REGION_STATUSES:
            raise ProtocolError(
                f"regions[{index}].status must be one of "
                f"{sorted(_DRAWING_REGION_STATUSES)}"
            )
        selected: dict[str, Any] = {
            "region_id": region_id,
            "status": status,
        }
        if score is not None:
            selected["score"] = score
        if margin is not None:
            selected["margin"] = margin
        comparison_path = region.get("comparison_path")
        if comparison_path is not None:
            _, comparison_relative = _safe_regular_file(
                workspace,
                comparison_path,
                f"regions[{index}].comparison_path",
            )
            selected["comparison_path"] = comparison_relative
        crop_path = region.get("crop_path")
        if crop_path is not None:
            _, crop_relative = _safe_regular_file(
                workspace, crop_path, f"regions[{index}].crop_path"
            )
            selected["crop_path"] = crop_relative
        render_path = region.get("render_path")
        if render_path is not None:
            _, render_relative = _safe_regular_file(
                workspace,
                render_path,
                f"regions[{index}].render_path",
            )
            selected["render_path"] = render_relative
        if comparison_path is None and "crop_path" not in selected:
            raise ProtocolError(
                f"regions[{index}] requires comparison_path or crop_path"
            )
        if region.get("view_spec") is not None:
            selected["view_spec"] = _canonical_view_spec(
                region["view_spec"], f"regions[{index}].view_spec"
            )
        regions.append(selected)
    canonical["regions"] = regions
    return canonical


def _safe_output_file(output_dir: Path, filename: str) -> Path:
    destination = output_dir / filename
    if destination.exists() and (destination.is_symlink() or destination.is_dir()):
        raise RuntimeError(f"Refusing unsafe drawing output: {destination}")
    return destination


def _public_view_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Return the canonical components expected by the TS search controller."""

    result = {
        "name": spec["name"],
        "direction": [float(value) for value in spec["direction"]],
        "up": [float(value) for value in spec["up"]],
    }
    if spec.get("section_offset_fraction") is not None:
        result["section_offset_fraction"] = float(spec["section_offset_fraction"])
        result["section_keep"] = spec["section_keep"]
    return result


def _execute_crop_drawing_region(
    request: dict[str, Any], workspace: Path
) -> dict[str, Any]:
    source, _ = _safe_regular_file(workspace, request["input_path"], "input_path")
    output_dir, _ = _safe_feedback_output_dir(workspace, request["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = _safe_output_file(output_dir, "crop.png")
    with Image.open(source) as opened:
        opened.load()
        image = opened.convert("RGB")
        width, height = image.size
        left, top, right, bottom = request["roi"]
        pixel_roi = (
            max(0, math.floor(left * width / 1000.0)),
            max(0, math.floor(top * height / 1000.0)),
            min(width, math.ceil(right * width / 1000.0)),
            min(height, math.ceil(bottom * height / 1000.0)),
        )
        if pixel_roi[0] >= pixel_roi[2] or pixel_roi[1] >= pixel_roi[3]:
            raise ValueError(
                f"Normalized ROI {request['roi']} is empty for image size {image.size}"
            )
        cropped = image.crop(pixel_roi)
        cropped.save(destination, format="PNG", optimize=True)
    return {
        "region_id": request["region_id"],
        "input_path": request["input_path"],
        "roi": request["roi"],
        "pixel_roi": list(pixel_roi),
        "crop_path": _relative_to_workspace(workspace, destination),
        "width": cropped.width,
        "height": cropped.height,
        "mime_type": "image/png",
        "sha256": _sha256(destination),
    }


def _execute_render_drawing_view(
    request: dict[str, Any], workspace: Path
) -> dict[str, Any]:
    from cadgenbench.tools.drawing_view import ViewSpec, render_step_view

    step_path, _ = _safe_regular_file(workspace, request["step_path"], "step_path")
    output_dir, _ = _safe_feedback_output_dir(workspace, request["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = _safe_output_file(output_dir, "render.png")
    raw_spec = request["view_spec"]
    spec = ViewSpec(
        name=raw_spec["name"],
        direction=tuple(raw_spec["direction"]),
        up=tuple(raw_spec["up"]),
        section_offset_fraction=raw_spec.get("section_offset_fraction"),
        section_keep=raw_spec.get("section_keep"),
    )
    render_step_view(
        step_path,
        destination,
        view=spec.name,
        direction=spec.direction,
        up=spec.up,
        section_offset_fraction=spec.section_offset_fraction,
        section_keep=spec.section_keep,
        width=request["width"],
        height=request["height"],
        show_hidden_lines=request["show_hidden_lines"],
    )
    if not destination.is_file() or destination.is_symlink():
        raise RuntimeError("2D renderer did not produce a regular PNG file")
    with Image.open(destination) as rendered:
        rendered.load()
        rendered_size = rendered.size
    return {
        "region_id": request["region_id"],
        "step_path": request["step_path"],
        "step_sha256": _sha256(step_path),
        "view_spec": _public_view_spec(raw_spec),
        "render_path": _relative_to_workspace(workspace, destination),
        "width": rendered_size[0],
        "height": rendered_size[1],
        "mime_type": "image/png",
        "sha256": _sha256(destination),
    }


def _execute_render_fixed_projection(
    request: dict[str, Any], workspace: Path
) -> dict[str, Any]:
    from cadgenbench.tools.drawing_view import (
        SimilarityTransform,
        _normalise3,
        _relative_section_origin,
        _render_cad_view,
        project_step,
    )

    step_path, step_relative = _safe_regular_file(
        workspace, request["step_path"], "step_path"
    )
    crop_path, _ = _safe_regular_file(
        workspace, request["crop_path"], "crop_path"
    )
    output_dir, _ = _safe_feedback_output_dir(workspace, request["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = _safe_output_file(output_dir, "aligned-render.png")
    view_spec = request["view_spec"]
    transform = SimilarityTransform(**request["transform"])
    projection_origin = None
    section_origin = None
    projection_anchor_sha256 = None
    anchor_relative = request.get("projection_anchor_step_path")
    if anchor_relative is not None:
        from build123d import import_step

        anchor_path, _ = _safe_regular_file(
            workspace,
            anchor_relative,
            "projection_anchor_step_path",
        )
        anchor = import_step(str(anchor_path))
        try:
            wrapped = None if anchor is None else anchor.wrapped
        except AssertionError:
            wrapped = None
        if wrapped is None:
            raise RuntimeError("Projection anchor STEP contains no shape")
        center = anchor.center()
        projection_origin = (float(center.X), float(center.Y), float(center.Z))
        if view_spec.get("section_offset_fraction") is not None:
            # Existing calibrated frames explicitly use the full anchor's
            # centroid, including section views. Freeze the physical plane
            # against that same anchor, never against a changed candidate bbox.
            section_origin = _relative_section_origin(
                anchor,
                _normalise3(view_spec["direction"]),
                view_spec["section_offset_fraction"],
            )
        projection_anchor_sha256 = _sha256(anchor_path)
    projection = project_step(
        step_path,
        view=view_spec["name"],
        direction=tuple(view_spec["direction"]),
        up=tuple(view_spec["up"]),
        look_at=projection_origin,
        section_origin=section_origin,
        section_offset_fraction=(
            view_spec.get("section_offset_fraction")
            if section_origin is None
            else None
        ),
        section_keep=view_spec.get("section_keep"),
    )
    with Image.open(crop_path) as opened:
        reference_size = opened.size
    rendered = _render_cad_view(
        reference_size,
        projection,
        transform,
        show_hidden_lines=False,
    )
    rendered.save(destination, format="PNG", optimize=True)
    # Preserve the native HLR curves in exactly the rendered pixel frame.
    # The compositor binds this sidecar to the PNG hash before using it.
    contours_path = _safe_output_file(output_dir, "aligned-render.contours.json")
    contours_path.write_text(json.dumps({
        "schema_version": 1,
        "kind": "projected_cad_contours",
        "render_sha256": _sha256(destination),
        "width": rendered.width,
        "height": rendered.height,
        "visible_polylines": [
            transform.apply(line).tolist() for line in projection.visible_polylines
        ],
        "step_sha256": _sha256(step_path),
        "view_spec": _public_view_spec(view_spec),
        "transform": _transform_payload(transform),
        "projection_origin": projection_origin,
        **(
            {"projection_anchor_step_sha256": projection_anchor_sha256}
            if projection_anchor_sha256 is not None
            else {}
        ),
        **(
            {"section_origin": list(section_origin)}
            if section_origin is not None
            else {}
        ),
    }, allow_nan=False) + "\n")
    return {
        "region_id": request["region_id"],
        "step_path": step_relative,
        "step_sha256": _sha256(step_path),
        "view_spec": _public_view_spec(view_spec),
        "transform": _transform_payload(transform),
        "render_path": _relative_to_workspace(workspace, destination),
        "width": rendered.width,
        "height": rendered.height,
        "mime_type": "image/png",
        "sha256": _sha256(destination),
        "contours_path": _relative_to_workspace(workspace, contours_path),
        "contours_sha256": _sha256(contours_path),
        "projection_bounds": [float(value) for value in projection.bounds],
        "projection_width_units": float(projection.width),
        "projection_height_units": float(projection.height),
        **(
            {"section_origin": list(section_origin)}
            if section_origin is not None
            else {}
        ),
        **(
            {
                "projection_origin": list(projection_origin),
                "projection_anchor_step_sha256": projection_anchor_sha256,
            }
            if projection_origin is not None
            else {}
        ),
    }


def _repair_comparison_panel(
    source: Path | None,
    *,
    title: str,
    status: str,
    size: tuple[int, int] = (600, 430),
) -> Image.Image:
    panel = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(panel)
    draw.text((12, 7), title, fill=(8, 12, 18), font=_reflection_font(24, bold=True))
    status_color = (36, 122, 72) if status in {"original", "current", "valid"} else (176, 52, 52)
    draw.text((12, 36), status, fill=status_color, font=_reflection_font(15, bold=True))
    image_box = (12, 62, size[0] - 12, size[1] - 12)
    image_size = (image_box[2] - image_box[0], image_box[3] - image_box[1])
    if source is None:
        draw.rectangle(image_box, fill=(235, 238, 242), outline=(150, 158, 168), width=2)
        label = "No candidate rendering"
        bounds = draw.textbbox((0, 0), label, font=_reflection_font(18, bold=True))
        draw.text(
            (
                image_box[0] + (image_size[0] - (bounds[2] - bounds[0])) // 2,
                image_box[1] + (image_size[1] - (bounds[3] - bounds[1])) // 2,
            ),
            label,
            fill=(85, 94, 104),
            font=_reflection_font(18, bold=True),
        )
        return panel
    with Image.open(source) as opened:
        image = ImageOps.contain(opened.convert("RGB"), image_size, Image.Resampling.LANCZOS)
    offset = (
        image_box[0] + (image_size[0] - image.width) // 2,
        image_box[1] + (image_size[1] - image.height) // 2,
    )
    panel.paste(image, offset)
    draw.rectangle(
        (offset[0] - 1, offset[1] - 1, offset[0] + image.width, offset[1] + image.height),
        outline=(150, 158, 168),
        width=1,
    )
    return panel


def _execute_compose_repair_comparison(
    request: dict[str, Any], workspace: Path
) -> dict[str, Any]:
    output_dir, _ = _safe_feedback_output_dir(workspace, request["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    margin, gap, title_height, row_label_height = 16, 12, 58, 34
    panel_size = (600, 430)
    column_count = 2 + len(request["candidate_ids"])
    width = margin * 2 + column_count * panel_size[0] + (column_count - 1) * gap
    height = (
        title_height
        + len(request["rows"]) * (row_label_height + panel_size[1])
        + max(0, len(request["rows"]) - 1) * gap
        + margin
    )
    if width > MAX_DRAWING_EDGE_PX or height > MAX_DRAWING_SHEET_HEIGHT_PX:
        raise RuntimeError("Causal repair comparison exceeds image limits")
    if width * height > MAX_DRAWING_SHEET_PIXELS:
        raise RuntimeError("Causal repair comparison exceeds pixel limit")
    canvas = Image.new("RGB", (width, height), (240, 243, 247))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (margin, 15),
        "Counterfactual repair verification — one frozen 2D frame per region",
        fill=(25, 32, 42),
        font=_reflection_font(22, bold=True),
    )
    y = title_height
    for row in request["rows"]:
        draw.text(
            (margin, y + 7),
            f"Region {row['region_id']}",
            fill=(35, 72, 115),
            font=_reflection_font(17, bold=True),
        )
        y += row_label_height
        sources: list[tuple[str, str, Path | None]] = [
            (
                "Original",
                "original",
                _safe_regular_file(workspace, row["reference_path"], "reference_path")[0],
            ),
            (
                "Current",
                "current",
                _safe_regular_file(workspace, row["current_render_path"], "current_render_path")[0],
            ),
        ]
        for candidate in row["candidates"]:
            render_path = candidate.get("render_path")
            sources.append(
                (
                    candidate["candidate_id"],
                    candidate["status"],
                    _safe_regular_file(workspace, render_path, "render_path")[0]
                    if render_path is not None
                    else None,
                )
            )
        x = margin
        for title, status, source in sources:
            panel = _repair_comparison_panel(source, title=title, status=status)
            canvas.paste(panel, (x, y))
            x += panel_size[0] + gap
        y += panel_size[1] + gap
    destination = _safe_output_file(output_dir, "comparison.png")
    canvas.save(destination, format="PNG", optimize=True)
    return {
        "sheet_path": _relative_to_workspace(workspace, destination),
        "width": canvas.width,
        "height": canvas.height,
        "mime_type": "image/png",
        "sha256": _sha256(destination),
    }


def _projection_line_mask(image: Image.Image) -> Image.Image:
    """Return an antialias-tolerant mask for dark projected CAD lines."""

    grayscale = image.convert("L")
    return grayscale.point(lambda value: 255 if value < 245 else 0, mode="L")


def _mask_pixel_count(mask: Image.Image) -> int:
    histogram = mask.histogram()
    return int(sum(histogram[1:]))


_CODE_CONSTRUCTION_COLORS = {
    "red": (216, 27, 44),
    "blue": (21, 101, 192),
    "orange": (239, 108, 0),
    "magenta": (142, 36, 170),
}


def _append_code_construction_legend(
    comparison: Image.Image,
    groups: list[dict[str, Any]],
) -> Image.Image:
    margin = 18
    legend_height = 108
    canvas = Image.new(
        "RGB",
        (comparison.width, comparison.height + legend_height),
        (244, 246, 248),
    )
    canvas.paste(comparison, (0, 0))
    draw = ImageDraw.Draw(canvas)
    title_y = comparison.height + 14
    draw.text(
        (margin, title_y),
        "Highlighted solution.py construction groups:",
        fill=(25, 32, 42),
        font=_reflection_font(17, bold=True),
    )
    entries_y = title_y + 36
    entries_x = margin
    available_width = max(1, comparison.width - entries_x - margin)
    entry_width = available_width // max(1, len(groups))
    for index, item in enumerate(groups):
        x = entries_x + index * entry_width
        color = _CODE_CONSTRUCTION_COLORS[item["color"]]
        draw.rounded_rectangle(
            (x, entries_y + 1, x + 24, entries_y + 25), radius=4, fill=color
        )
        draw.text(
            (x + 34, entries_y + 1),
            item["group_name"],
            fill=(25, 32, 42),
            font=_reflection_font(16, bold=True),
        )
    return canvas


def _execute_compose_code_ablation_highlight(
    request: dict[str, Any], workspace: Path
) -> dict[str, Any]:
    """Highlight final visible Current edges that disappear after ablation."""

    output_dir, _ = _safe_feedback_output_dir(workspace, request["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        name: _safe_regular_file(workspace, request[field], field)[0]
        for name, field in (
            ("reference", "reference_path"),
            ("current", "current_render_path"),
            ("ablated", "ablated_render_path"),
        )
    }
    with Image.open(paths["reference"]) as opened:
        reference = opened.convert("RGB")
    with Image.open(paths["current"]) as opened:
        current = opened.convert("RGB")
    with Image.open(paths["ablated"]) as opened:
        ablated = opened.convert("RGB")
    if reference.size != current.size or current.size != ablated.size:
        raise RuntimeError(
            "Code-ablation highlight inputs must have identical pixel dimensions"
        )

    current_mask = _projection_line_mask(current)
    ablated_mask = _projection_line_mask(ablated)
    current_pixels = _mask_pixel_count(current_mask)
    # Treat an Ablated edge within three pixels as the same final visible edge.
    # This absorbs antialiasing and HLR sampling noise without moving the red
    # highlight away from pixels that genuinely belong to Current.
    ablated_near = ablated_mask.filter(ImageFilter.MaxFilter(7))
    current_near = current_mask.filter(ImageFilter.MaxFilter(7))
    removed_mask = ImageChops.subtract(current_mask, ablated_near)
    revealed_mask = ImageChops.subtract(ablated_mask, current_near)
    removed_pixels = _mask_pixel_count(removed_mask)
    revealed_pixels = _mask_pixel_count(revealed_mask)
    common = {
        "region_id": request["region_id"],
        "construction_ref": request["construction_ref"],
        "current_pixel_count": current_pixels,
        "removed_pixel_count": removed_pixels,
        "revealed_pixel_count": revealed_pixels,
    }
    minimum_change = max(20, round(max(1, current_pixels) * 0.001))
    if removed_pixels < minimum_change:
        return {**common, "visual_status": "no_visible_change"}

    reference_highlighted = reference.copy()
    highlighted = current.copy()
    red_halo = removed_mask.filter(ImageFilter.MaxFilter(5)).point(
        lambda value: round(value * 0.88), mode="L"
    )
    reference_overlay = red_halo.point(
        lambda value: round(value * 0.55), mode="L"
    )
    reference_highlighted.paste(
        Image.new("RGB", reference.size, _CODE_CONSTRUCTION_COLORS["red"]),
        mask=reference_overlay,
    )
    highlighted.paste(
        Image.new("RGB", current.size, _CODE_CONSTRUCTION_COLORS["red"]),
        mask=red_halo,
    )
    reference_panel = _render_reflection_coordinate_panel(
        reference_highlighted,
        f"Original drawing — projected footprint of current {request['construction_ref']}",
        roi=None,
    )
    current_panel = _render_reflection_coordinate_panel(
        highlighted,
        f"Current CAD — {request['construction_ref']} highlighted by code ablation",
        roi=None,
    )
    comparison = _compose_reflection_pair(
        reference_panel,
        current_panel,
        header_text=(
            f"{request['region_id']} · {request['construction_ref']} code-block localization · "
            "complete aligned view"
        ),
    )
    destination = _safe_output_file(output_dir, "code-ablation-highlight.png")
    comparison.save(destination, format="PNG", optimize=True)
    return {
        **common,
        "visual_status": "highlighted",
        "comparison_path": _relative_to_workspace(workspace, destination),
        "width": comparison.width,
        "height": comparison.height,
        "mime_type": "image/png",
        "sha256": _sha256(destination),
    }


_VISUAL_DIFF_OVERLAP = (86, 94, 101)
_VISUAL_DIFF_TARGET_ONLY = (183, 116, 62)
_VISUAL_DIFF_CURRENT_ONLY = (62, 112, 139)
_VISUAL_DIFF_MUTED = (126, 133, 139)


def _expanded_pixel_box(
    box: tuple[int, int, int, int],
    size: tuple[int, int],
    *,
    padding_fraction: float = 0.14,
) -> tuple[int, int, int, int]:
    width = max(1, box[2] - box[0])
    height = max(1, box[3] - box[1])
    padding_x = max(12, round(width * padding_fraction))
    padding_y = max(12, round(height * padding_fraction))
    return (
        max(0, box[0] - padding_x),
        max(0, box[1] - padding_y),
        min(size[0], box[2] + padding_x),
        min(size[1], box[3] + padding_y),
    )


def _mask_centroid(mask: Image.Image) -> tuple[float, float]:
    box = mask.getbbox()
    if box is None:
        raise RuntimeError("Visual-diff mask is empty")
    pixels = mask.load()
    count = 0
    total_x = 0
    total_y = 0
    for y in range(box[1], box[3]):
        for x in range(box[0], box[2]):
            if pixels[x, y]:
                count += 1
                total_x += x
                total_y += y
    if count == 0:
        raise RuntimeError("Visual-diff mask is empty")
    return total_x / count, total_y / count


def _draw_displacement_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: tuple[int, int, int] = _VISUAL_DIFF_MUTED,
) -> None:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length < 5:
        return
    ux = dx / length
    uy = dy / length
    tip = (round(end[0]), round(end[1]))
    base = (end[0] - ux * 15, end[1] - uy * 15)
    perpendicular = (-uy * 8, ux * 8)
    draw.line((*start, *end), fill=color, width=3)
    draw.polygon(
        [
            tip,
            (round(base[0] + perpendicular[0]), round(base[1] + perpendicular[1])),
            (round(base[0] - perpendicular[0]), round(base[1] - perpendicular[1])),
        ],
        fill=color,
    )


def _execute_compose_visual_diff(
    request: dict[str, Any], workspace: Path
) -> dict[str, Any]:
    """Render one same-coordinate Target-versus-Current geometry residual."""

    output_dir, _ = _safe_feedback_output_dir(workspace, request["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_path = _safe_regular_file(
        workspace, request["reference_path"], "reference_path"
    )[0]
    current_path = _safe_regular_file(
        workspace, request["current_render_path"], "current_render_path"
    )[0]
    with Image.open(reference_path) as opened:
        reference = opened.convert("RGB")
    with Image.open(current_path) as opened:
        current = opened.convert("RGB")
    if reference.size != current.size:
        raise RuntimeError("Visual-link inputs must have identical pixel dimensions")
    if reference.width > MAX_DRAWING_EDGE_PX or reference.height > MAX_DRAWING_EDGE_PX:
        raise RuntimeError("Visual-link inputs exceed the drawing edge limit")

    target_roi = request["target_roi"]
    target_pixel_roi = _pixel_roi(target_roi, reference.size)
    ablated_path = _safe_regular_file(
        workspace, request["ablated_render_path"], "ablated_render_path"
    )[0]
    with Image.open(ablated_path) as opened:
        ablated = opened.convert("RGB")
    if ablated.size != current.size:
        raise RuntimeError("Visual-diff ablated render must match the Current frame")

    current_line_mask = _projection_line_mask(current)
    ablated_line_mask = _projection_line_mask(ablated)
    current_mask = ImageChops.subtract(
        current_line_mask, ablated_line_mask.filter(ImageFilter.MaxFilter(7))
    )
    current_geometry_box = current_mask.getbbox()
    current_pixel_count = _mask_pixel_count(current_mask)
    if current_geometry_box is None or current_pixel_count < 20:
        raise RuntimeError("Visual-diff construction has no reliable visible geometry")

    target_roi_mask = Image.new("L", reference.size, 0)
    ImageDraw.Draw(target_roi_mask).rectangle(target_pixel_roi, fill=255)
    reference_line_mask = _projection_line_mask(reference)
    geometry_extent = max(
        current_geometry_box[2] - current_geometry_box[0],
        current_geometry_box[3] - current_geometry_box[1],
    )
    support_radius = max(6, min(16, round(geometry_extent * 0.035)))
    support_filter = support_radius * 2 + 1
    current_support = current_mask.filter(ImageFilter.MaxFilter(support_filter))
    target_mask = ImageChops.multiply(
        ImageChops.multiply(reference_line_mask, target_roi_mask), current_support
    )
    if _mask_pixel_count(target_mask) < max(20, round(current_pixel_count * 0.08)):
        target_mask = ImageChops.multiply(reference_line_mask, target_roi_mask)
    target_geometry_box = target_mask.getbbox()
    if target_geometry_box is None or _mask_pixel_count(target_mask) < 20:
        raise RuntimeError("Visual-diff Target ROI has no reliable visible geometry")

    tolerance = 5
    target_halo = target_mask.filter(ImageFilter.MaxFilter(tolerance))
    current_halo = current_mask.filter(ImageFilter.MaxFilter(tolerance))
    target_overlap = ImageChops.multiply(target_mask, current_halo)
    current_overlap = ImageChops.multiply(current_mask, target_halo)
    overlap_mask = ImageChops.lighter(target_overlap, current_overlap)
    target_only_mask = ImageChops.subtract(target_mask, current_halo)
    current_only_mask = ImageChops.subtract(current_mask, target_halo)
    overlap_pixel_count = _mask_pixel_count(target_overlap)
    target_only_pixel_count = _mask_pixel_count(target_only_mask)
    current_only_pixel_count = _mask_pixel_count(current_only_mask)
    union_pixel_count = (
        overlap_pixel_count + target_only_pixel_count + current_only_pixel_count
    )
    iou = overlap_pixel_count / union_pixel_count if union_pixel_count else 1.0
    target_centroid = _mask_centroid(target_mask)
    current_centroid = _mask_centroid(current_mask)
    centroid_delta = (
        current_centroid[0] - target_centroid[0],
        current_centroid[1] - target_centroid[1],
    )

    overlay = Image.blend(Image.new("RGB", current.size, "white"), current, 0.14)
    overlap_visible = overlap_mask.filter(ImageFilter.MaxFilter(3))
    target_only_visible = target_only_mask.filter(ImageFilter.MaxFilter(3))
    current_only_visible = current_only_mask.filter(ImageFilter.MaxFilter(3))
    overlay.paste(
        Image.new("RGB", current.size, _VISUAL_DIFF_OVERLAP), mask=overlap_visible
    )
    overlay.paste(
        Image.new("RGB", current.size, _VISUAL_DIFF_TARGET_ONLY),
        mask=target_only_visible,
    )
    overlay.paste(
        Image.new("RGB", current.size, _VISUAL_DIFF_CURRENT_ONLY),
        mask=current_only_visible,
    )
    overlay_draw = ImageDraw.Draw(overlay)
    _draw_displacement_arrow(overlay_draw, target_centroid, current_centroid)

    union_box = (
        min(target_pixel_roi[0], current_geometry_box[0]),
        min(target_pixel_roi[1], current_geometry_box[1]),
        max(target_pixel_roi[2], current_geometry_box[2]),
        max(target_pixel_roi[3], current_geometry_box[3]),
    )
    shared_crop_box = _expanded_pixel_box(
        union_box, current.size, padding_fraction=0.08
    )
    shared_overlay = overlay.crop(shared_crop_box)
    rendered = ImageOps.contain(
        shared_overlay, (1320, 690), method=Image.Resampling.LANCZOS
    )
    margin = 30
    header_height = 72
    footer_height = 72
    canvas = Image.new(
        "RGB",
        (
            max(900, rendered.width + margin * 2),
            header_height + rendered.height + footer_height,
        ),
        (243, 245, 247),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (margin, 20),
        f"{request['diff_id']} · {request['region_id']} · {request['construction_ref']}",
        fill=(30, 38, 47),
        font=_reflection_font(24, bold=True),
    )
    image_x = (canvas.width - rendered.width) // 2
    canvas.paste(rendered, (image_x, header_height))
    draw.rectangle(
        (
            image_x - 1,
            header_height - 1,
            image_x + rendered.width,
            header_height + rendered.height,
        ),
        outline=(185, 191, 198),
        width=1,
    )
    legend_y = header_height + rendered.height + 22
    legend_font = _reflection_font(16, bold=True)
    legend = [
        (_VISUAL_DIFF_OVERLAP, "T ∩ C"),
        (_VISUAL_DIFF_TARGET_ONLY, "T − C"),
        (_VISUAL_DIFF_CURRENT_ONLY, "C − T"),
    ]
    legend_x = margin
    for color, label in legend:
        draw.ellipse((legend_x, legend_y, legend_x + 16, legend_y + 16), fill=color)
        draw.text(
            (legend_x + 24, legend_y - 2), label, fill=(55, 63, 71), font=legend_font
        )
        legend_x += 125
    metric_text = f"IoU {iou:.3f}   Δcenter ({centroid_delta[0]:+.1f}, {centroid_delta[1]:+.1f}) px"
    metric_bounds = draw.textbbox((0, 0), metric_text, font=legend_font)
    draw.text(
        (canvas.width - margin - (metric_bounds[2] - metric_bounds[0]), legend_y - 2),
        metric_text,
        fill=(70, 79, 88),
        font=legend_font,
    )

    target_mask_path = _safe_output_file(output_dir, "target-mask.png")
    current_mask_path = _safe_output_file(output_dir, "current-mask.png")
    comparison_path = _safe_output_file(output_dir, "visual-diff.png")
    target_mask.save(target_mask_path, format="PNG", optimize=True)
    current_mask.save(current_mask_path, format="PNG", optimize=True)
    canvas.save(comparison_path, format="PNG", optimize=True)
    return {
        "region_id": request["region_id"],
        "diff_id": request["diff_id"],
        "construction_ref": request["construction_ref"],
        "target_roi": target_roi,
        "target_pixel_roi": list(target_pixel_roi),
        "target_geometry_pixel_bbox": list(target_geometry_box),
        "current_geometry_pixel_bbox": list(current_geometry_box),
        "comparison_path": _relative_to_workspace(workspace, comparison_path),
        "target_mask_path": _relative_to_workspace(workspace, target_mask_path),
        "current_mask_path": _relative_to_workspace(workspace, current_mask_path),
        "metrics": {
            "overlap_pixel_count": overlap_pixel_count,
            "target_only_pixel_count": target_only_pixel_count,
            "current_only_pixel_count": current_only_pixel_count,
            "iou": iou,
            "centroid_delta_px": [centroid_delta[0], centroid_delta[1]],
        },
        "width": canvas.width,
        "height": canvas.height,
        "mime_type": "image/png",
        "sha256": {
            "comparison": _sha256(comparison_path),
            "target_mask": _sha256(target_mask_path),
            "current_mask": _sha256(current_mask_path),
        },
    }


_CODE_EDIT_ADD = (28, 145, 86)
_CODE_EDIT_REMOVE = (215, 45, 55)


def _coalesce_shape_effect(shape: Any) -> Any | None:
    """Return one exportable Shape for a single- or multi-part B-rep effect."""

    try:
        if shape.wrapped is not None:
            return shape
    except (AssertionError, AttributeError):
        pass

    try:
        candidates = list(shape)
    except TypeError:
        return None
    parts = []
    for candidate in candidates:
        try:
            if candidate.wrapped is not None:
                parts.append(candidate)
        except (AssertionError, AttributeError):
            continue
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]

    from build123d import Compound

    return Compound(children=parts)


def _shape_effect_payload(shape: Any, relation: str) -> dict[str, Any] | None:
    try:
        wrapped = shape.wrapped
    except (AssertionError, AttributeError):
        return None
    if wrapped is None:
        return None
    faces = list(shape.faces())
    edges = list(shape.edges())
    if not faces and not edges:
        return None
    bounds = shape.bounding_box()
    center = shape.center()
    surface_types = sorted(
        {
            str(face.geom_type).removeprefix("GeomType.").lower()
            for face in faces
        }
    )
    return {
        "relation": relation,
        "solid_count": len(shape.solids()),
        "face_count": len(faces),
        "edge_count": len(edges),
        "volume_mm3": round(float(shape.volume), 6),
        "surface_area_mm2": round(sum(float(face.area) for face in faces), 6),
        "bbox_mm": {
            "x": round(float(bounds.size.X), 6),
            "y": round(float(bounds.size.Y), 6),
            "z": round(float(bounds.size.Z), 6),
            "x_min": round(float(bounds.min.X), 6),
            "x_max": round(float(bounds.max.X), 6),
            "y_min": round(float(bounds.min.Y), 6),
            "y_max": round(float(bounds.max.Y), 6),
            "z_min": round(float(bounds.min.Z), 6),
            "z_max": round(float(bounds.max.Z), 6),
        },
        "centroid_mm": [
            round(float(center.X), 6),
            round(float(center.Y), 6),
            round(float(center.Z), 6),
        ],
        "surface_types": surface_types,
    }


def _visible_code_edit_masks(
    current_render: Image.Image,
    edited_render: Image.Image,
) -> tuple[Image.Image, Image.Image]:
    """Return visible lines removed and added between two frozen CAD views."""

    current_mask = _projection_line_mask(current_render)
    edited_mask = _projection_line_mask(edited_render)
    # Both renders share one exact camera and transform. A one-pixel tolerance
    # absorbs antialiasing/HLR sampling without turning hidden B-rep faces into
    # visible edit marks.
    edited_near = edited_mask.filter(ImageFilter.MaxFilter(3))
    current_near = current_mask.filter(ImageFilter.MaxFilter(3))
    removed_mask = ImageChops.subtract(current_mask, edited_near)
    added_mask = ImageChops.subtract(edited_mask, current_near)
    return removed_mask, added_mask


def _code_edit_panel(
    image: Image.Image,
    title: str,
    *,
    size: tuple[int, int] = (600, 430),
) -> Image.Image:
    panel = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(panel)
    draw.text((12, 9), title, fill=(42, 48, 55), font=_reflection_font(18, bold=True))
    image_box = (12, 42, size[0] - 12, size[1] - 12)
    contained = ImageOps.contain(
        image.convert("RGB"),
        (image_box[2] - image_box[0], image_box[3] - image_box[1]),
        Image.Resampling.LANCZOS,
    )
    offset = (
        image_box[0] + (image_box[2] - image_box[0] - contained.width) // 2,
        image_box[1] + (image_box[3] - image_box[1] - contained.height) // 2,
    )
    panel.paste(contained, offset)
    draw.rectangle(
        (
            offset[0] - 1,
            offset[1] - 1,
            offset[0] + contained.width,
            offset[1] + contained.height,
        ),
        outline=(188, 194, 201),
        width=1,
    )
    return panel


def _execute_compose_code_edit_visualization(
    request: dict[str, Any], workspace: Path
) -> dict[str, Any]:
    """Render visible edit lines and report exact B-rep additions/removals."""

    from build123d import export_step, import_step

    output_dir, _ = _safe_feedback_output_dir(workspace, request["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = {
        field: _safe_regular_file(workspace, request[field], field)[0]
        for field in (
            "reference_path",
            "current_render_path",
            "edited_render_path",
            "current_step_path",
            "edited_step_path",
        )
    }
    with Image.open(resolved["reference_path"]) as opened:
        reference = opened.convert("RGB")
    with Image.open(resolved["current_render_path"]) as opened:
        current_render = opened.convert("RGB")
    with Image.open(resolved["edited_render_path"]) as opened:
        edited_render = opened.convert("RGB")
    if reference.size != current_render.size or current_render.size != edited_render.size:
        raise RuntimeError("Code-hypothesis frames must have identical dimensions")

    current = import_step(str(resolved["current_step_path"]))
    edited = import_step(str(resolved["edited_step_path"]))
    removed_shape = current - edited
    added_shape = edited - current
    shape_effects = [
        (removed_shape, "removed_by_proposal", "removed"),
        (added_shape, "added_by_proposal", "added"),
    ]
    geometry: list[dict[str, Any]] = []
    for raw_shape, relation, name in shape_effects:
        shape = _coalesce_shape_effect(raw_shape)
        if shape is None:
            continue
        payload = _shape_effect_payload(shape, relation)
        if payload is None or (
            payload["face_count"] == 0 and payload["volume_mm3"] <= 1e-9
        ):
            continue
        geometry.append(payload)
        step_path = _safe_output_file(output_dir, f"{name}.step")
        export_step(shape, str(step_path))
    if not geometry:
        raise RuntimeError("The isolated code edit produced no B-rep geometry change")

    target_roi = request["target_roi"]
    pixel_roi = _pixel_roi(target_roi, reference.size)
    target_crop = reference.crop(pixel_roi)
    delta_full = Image.blend(
        Image.new("RGB", current_render.size, "white"), current_render, 0.16
    )
    current_lines = _projection_line_mask(current_render).filter(ImageFilter.MaxFilter(3))
    delta_full.paste(
        Image.new("RGB", current_render.size, (108, 116, 125)),
        mask=current_lines.point(lambda value: round(value * 0.5), mode="L"),
    )
    removed_mask, added_mask = _visible_code_edit_masks(
        current_render,
        edited_render,
    )
    removed_visible = removed_mask.filter(ImageFilter.MaxFilter(5))
    added_visible = added_mask.filter(ImageFilter.MaxFilter(5))
    delta_full.paste(
        Image.new("RGB", current_render.size, _CODE_EDIT_REMOVE),
        mask=removed_visible,
    )
    delta_full.paste(
        Image.new("RGB", current_render.size, _CODE_EDIT_ADD),
        mask=added_visible,
    )
    delta_crop = delta_full.crop(pixel_roi)

    panels = [
        _code_edit_panel(target_crop, "TARGET ROI"),
        _code_edit_panel(delta_crop, "EDIT PREVIEW"),
    ]
    margin, gap = 16, 16
    canvas = Image.new(
        "RGB",
        (
            margin * 2 + sum(panel.width for panel in panels) + gap * (len(panels) - 1),
            margin * 2 + panels[0].height,
        ),
        (244, 246, 248),
    )
    x = margin
    for panel in panels:
        canvas.paste(panel, (x, margin))
        x += panel.width + gap

    card_path = _safe_output_file(output_dir, "code-hypothesis.png")
    canvas.save(card_path, format="PNG", optimize=True)
    return {
        "region_id": request["region_id"],
        "hypothesis_id": request["hypothesis_id"],
        "target_roi": target_roi,
        "target_pixel_roi": list(pixel_roi),
        "visual_encoding": {
            "current_cad": "gray",
            "removed_by_proposal": "red",
            "added_by_proposal": "green",
        },
        "geometry": geometry,
        "card_path": _relative_to_workspace(workspace, card_path),
        "width": canvas.width,
        "height": canvas.height,
        "mime_type": "image/png",
        "sha256": _sha256(card_path),
    }


def _render_brep_effect(
    step_path: Path,
    *,
    image_size: tuple[int, int],
    view_spec: dict[str, Any],
    transform: dict[str, Any],
    projection_origin: list[float],
) -> Image.Image:
    """Render an exact B-rep effect in the frozen Current projection."""

    from cadgenbench.tools.drawing_view import (
        SimilarityTransform,
        _render_cad_view,
        project_step,
    )

    projection = project_step(
        step_path,
        view=view_spec["name"],
        direction=tuple(view_spec["direction"]),
        up=tuple(view_spec["up"]),
        look_at=tuple(projection_origin),
        section_offset_fraction=view_spec.get("section_offset_fraction"),
        section_keep=view_spec.get("section_keep"),
    )
    return _render_cad_view(
        image_size,
        projection,
        SimilarityTransform(**transform),
        show_hidden_lines=True,
    )


def _candidate_effect_panel(
    candidate_render: Image.Image,
    *,
    title: str,
    topology: dict[str, Any],
    removed_render: Image.Image | None,
    added_render: Image.Image | None,
) -> Image.Image:
    """Compose one readable full-view candidate panel from exact B-rep effects."""

    candidate_lines = _projection_line_mask(candidate_render).filter(
        ImageFilter.MaxFilter(3)
    )
    highlighted = Image.new("RGB", candidate_render.size, "white")
    highlighted.paste(
        Image.new("RGB", candidate_render.size, (105, 113, 122)),
        mask=candidate_lines,
    )
    if removed_render is not None:
        removed_lines = _projection_line_mask(removed_render).filter(
            ImageFilter.MaxFilter(5)
        )
        highlighted.paste(
            Image.new("RGB", candidate_render.size, _CODE_EDIT_REMOVE),
            mask=removed_lines,
        )
    if added_render is not None:
        added_lines = _projection_line_mask(added_render).filter(
            ImageFilter.MaxFilter(5)
        )
        highlighted.paste(
            Image.new("RGB", candidate_render.size, _CODE_EDIT_ADD),
            mask=added_lines,
        )
    panel = _code_edit_panel(highlighted, title, size=(720, 510))
    draw = ImageDraw.Draw(panel)
    topology_text = (
        f"b0={topology['b0']}   b1={topology['b1']}   b2={topology['b2']}"
    )
    text_bounds = draw.textbbox(
        (0, 0), topology_text, font=_reflection_font(16, bold=True)
    )
    draw.rectangle(
        (
            12,
            panel.height - 40,
            28 + text_bounds[2] - text_bounds[0],
            panel.height - 10,
        ),
        fill=(255, 255, 255),
        outline=(188, 194, 201),
        width=1,
    )
    draw.text(
        (20, panel.height - 35),
        topology_text,
        fill=(45, 53, 62),
        font=_reflection_font(16, bold=True),
    )
    return panel


def _execute_compose_repair_candidate_preview(
    request: dict[str, Any], workspace: Path
) -> dict[str, Any]:
    """Compare isolated repairs using exact B-rep effects and Betti changes."""

    from build123d import export_step, import_step

    from cadgenbench.eval.topo_match import compute_betti_for_step

    output_dir, _ = _safe_feedback_output_dir(workspace, request["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_path = _safe_regular_file(
        workspace, request["reference_path"], "reference_path"
    )[0]
    current_render_path = _safe_regular_file(
        workspace, request["current_render_path"], "current_render_path"
    )[0]
    current_step_path = _safe_regular_file(
        workspace, request["current_step_path"], "current_step_path"
    )[0]
    with Image.open(reference_path) as opened:
        reference = opened.convert("RGB")
    with Image.open(current_render_path) as opened:
        current_render = opened.convert("RGB")
    if reference.size != current_render.size:
        raise RuntimeError("Original and Current repair-preview frames must match")

    current_shape = import_step(str(current_step_path))
    current_topology_result = compute_betti_for_step(current_step_path)
    current_topology = current_topology_result.to_dict()
    common_deflection = current_topology_result.linear_deflection_mm
    candidate_results: list[dict[str, Any]] = []
    candidate_panels: list[Image.Image] = []

    for index, candidate in enumerate(request["candidates"]):
        candidate_render_path = _safe_regular_file(
            workspace,
            candidate["render_path"],
            f"candidates[{index}].render_path",
        )[0]
        candidate_step_path = _safe_regular_file(
            workspace,
            candidate["step_path"],
            f"candidates[{index}].step_path",
        )[0]
        with Image.open(candidate_render_path) as opened:
            candidate_render = opened.convert("RGB")
        if candidate_render.size != current_render.size:
            raise RuntimeError(
                f"Candidate {candidate['candidate_id']} does not use the frozen Current frame"
            )

        candidate_shape = import_step(str(candidate_step_path))
        raw_effects = [
            (current_shape - candidate_shape, "removed_by_proposal", "removed"),
            (candidate_shape - current_shape, "added_by_proposal", "added"),
        ]
        geometry: list[dict[str, Any]] = []
        effect_renders: dict[str, Image.Image] = {}
        candidate_output = output_dir / f"{index + 1:02d}-{candidate['candidate_id']}"
        if candidate_output.exists() and (
            candidate_output.is_symlink() or not candidate_output.is_dir()
        ):
            raise RuntimeError(f"Refusing unsafe candidate output: {candidate_output}")
        candidate_output.mkdir(parents=True, exist_ok=True)
        for raw_shape, relation, name in raw_effects:
            shape = _coalesce_shape_effect(raw_shape)
            if shape is None:
                continue
            payload = _shape_effect_payload(shape, relation)
            if payload is None or (
                payload["face_count"] == 0 and payload["volume_mm3"] <= 1e-9
            ):
                continue
            geometry.append(payload)
            effect_path = _safe_output_file(candidate_output, f"{name}.step")
            export_step(shape, str(effect_path))
            effect_renders[name] = _render_brep_effect(
                effect_path,
                image_size=current_render.size,
                view_spec=request["view_spec"],
                transform=request["transform"],
                projection_origin=request["projection_origin"],
            )
        if not geometry:
            raise RuntimeError(
                f"Candidate {candidate['candidate_id']} produced no B-rep geometry change"
            )

        topology_result = compute_betti_for_step(
            candidate_step_path,
            linear_deflection_mm=common_deflection,
        )
        topology = topology_result.to_dict()
        topology_delta = {
            key: topology[key] - current_topology[key]
            for key in ("b0", "b1", "b2")
        }
        candidate_results.append(
            {
                "candidate_id": candidate["candidate_id"],
                "hypothesis": candidate["hypothesis"],
                "topology": topology,
                "topology_delta": topology_delta,
                "geometry": geometry,
            }
        )
        candidate_panels.append(
            _candidate_effect_panel(
                candidate_render,
                title=f"CANDIDATE {candidate['candidate_id']}",
                topology=topology,
                removed_render=effect_renders.get("removed"),
                added_render=effect_renders.get("added"),
            )
        )

    panels = [
        _code_edit_panel(reference, "ORIGINAL DRAWING", size=(720, 510)),
        _code_edit_panel(current_render, "CURRENT CAD", size=(720, 510)),
        *candidate_panels,
    ]
    columns = 2
    rows = math.ceil(len(panels) / columns)
    margin, gap = 18, 18
    panel_width, panel_height = panels[0].size
    canvas = Image.new(
        "RGB",
        (
            margin * 2 + columns * panel_width + (columns - 1) * gap,
            margin * 2 + rows * panel_height + (rows - 1) * gap + 58,
        ),
        (244, 246, 248),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (margin, 14),
        f"{request['region_id']} · isolated repair candidates",
        fill=(31, 39, 48),
        font=_reflection_font(23, bold=True),
    )
    draw.text(
        (canvas.width - 430, 18),
        "gray candidate · green added · red removed",
        fill=(72, 80, 89),
        font=_reflection_font(15, bold=True),
    )
    for index, panel in enumerate(panels):
        x = margin + (index % columns) * (panel_width + gap)
        y = margin + 58 + (index // columns) * (panel_height + gap)
        canvas.paste(panel, (x, y))

    card_path = _safe_output_file(output_dir, "repair-candidate-preview.png")
    canvas.save(card_path, format="PNG", optimize=True)
    return {
        "region_id": request["region_id"],
        "current_topology": current_topology,
        "candidates": candidate_results,
        "visual_encoding": {
            "current_cad": "gray",
            "removed_by_proposal": "red",
            "added_by_proposal": "green",
        },
        "card_path": _relative_to_workspace(workspace, card_path),
        "width": canvas.width,
        "height": canvas.height,
        "mime_type": "image/png",
        "sha256": _sha256(card_path),
    }


def _execute_compose_code_construction_highlight(
    request: dict[str, Any], workspace: Path
) -> dict[str, Any]:
    output_dir, _ = _safe_feedback_output_dir(workspace, request["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_path = _safe_regular_file(
        workspace, request["reference_path"], "reference_path"
    )[0]
    current_path = _safe_regular_file(
        workspace, request["current_render_path"], "current_render_path"
    )[0]
    with Image.open(reference_path) as reference_opened, Image.open(
        current_path
    ) as current_opened:
        reference = reference_opened.convert("RGB")
        current = current_opened.convert("RGB")
    if reference.size != current.size:
        raise RuntimeError(
            "Code-construction highlight inputs must have identical pixel dimensions"
        )

    current_mask = _projection_line_mask(current)
    current_pixels = _mask_pixel_count(current_mask)
    full_model_tolerance = max(20, round(max(1, current_pixels) * 0.003))
    highlighted = current.copy()
    results: list[dict[str, Any]] = []
    visible_groups: list[dict[str, Any]] = []
    for index, item in enumerate(request["groups"]):
        construction_path = _safe_regular_file(
            workspace,
            item["construction_render_path"],
            f"groups[{index}].construction_render_path",
        )[0]
        with Image.open(construction_path) as construction_opened:
            construction = construction_opened.convert("RGB")
        if construction.size != current.size:
            raise RuntimeError(
                "Code-construction highlight inputs must have identical pixel dimensions"
            )
        construction_mask = _projection_line_mask(construction)
        visible_pixels = _mask_pixel_count(construction_mask)
        difference_pixels = _mask_pixel_count(
            ImageChops.difference(construction_mask, current_mask)
        )
        if visible_pixels < 10:
            visual_status = "invisible"
        elif difference_pixels <= full_model_tolerance:
            visual_status = "full_model"
        else:
            visual_status = "highlighted"
            thick_mask = construction_mask.filter(ImageFilter.MaxFilter(5)).point(
                lambda value: round(value * 0.82), mode="L"
            )
            highlighted.paste(
                Image.new("RGB", current.size, _CODE_CONSTRUCTION_COLORS[item["color"]]),
                mask=thick_mask,
            )
            visible_groups.append(item)
        results.append(
            {
                "group_name": item["group_name"],
                "color": item["color"],
                "visual_status": visual_status,
                "visible_pixel_count": visible_pixels,
                "difference_pixel_count": difference_pixels,
            }
        )

    common = {
        "region_id": request["region_id"],
        "groups": results,
    }
    if not visible_groups:
        return {**common, "visual_status": "no_visible_highlights"}

    reference_panel = _render_reflection_coordinate_panel(
        reference,
        "Original engineering drawing",
        roi=None,
    )
    current_panel = _render_reflection_coordinate_panel(
        highlighted,
        "Current CAD — selected construction groups highlighted",
        roi=None,
    )
    comparison = _compose_reflection_pair(
        reference_panel,
        current_panel,
        header_text=(
            f"{request['region_id']} · code construction localization · "
            "complete aligned view"
        ),
    )
    comparison = _append_code_construction_legend(
        comparison, visible_groups
    )
    destination = _safe_output_file(output_dir, "construction-highlight.png")
    comparison.save(destination, format="PNG", optimize=True)
    return {
        **common,
        "visual_status": "highlighted",
        "comparison_path": _relative_to_workspace(workspace, destination),
        "width": comparison.width,
        "height": comparison.height,
        "mime_type": "image/png",
        "sha256": _sha256(destination),
    }


def _transform_payload(transform: Any) -> dict[str, Any]:
    return {
        "scale_px_per_unit": float(transform.scale_px_per_unit),
        "rotation_deg": float(transform.rotation_deg),
        "center_x_px": float(transform.center_x_px),
        "center_y_px": float(transform.center_y_px),
        "model_center_x": float(transform.model_center_x),
        "model_center_y": float(transform.model_center_y),
        "reflected": bool(transform.reflected),
    }


def _capacity_transport_payload(metrics: Any) -> dict[str, Any]:
    return {
        "version": int(metrics.version),
        "score": float(metrics.score),
        "normalized_cost": float(metrics.normalized_cost),
        "distance_scale_px": float(metrics.distance_scale_px),
        "unmatched_cad_cost": float(metrics.unmatched_cad_cost),
        "unmatched_reference_cost": float(metrics.unmatched_reference_cost),
        "reference_edge_pixels": int(metrics.reference_edge_pixels),
        "cad_edge_pixels": int(metrics.cad_edge_pixels),
        "reference_tokens": int(metrics.reference_tokens),
        "cad_tokens": int(metrics.cad_tokens),
        "matched_pairs": int(metrics.matched_pairs),
        "unmatched_cad_tokens": int(metrics.unmatched_cad_tokens),
        "unmatched_reference_tokens": int(metrics.unmatched_reference_tokens),
        "total_cost": float(metrics.total_cost),
        "matched_cad_fraction": float(metrics.matched_cad_fraction),
        "matched_reference_fraction": float(metrics.matched_reference_fraction),
        "mean_matched_distance_px": float(metrics.mean_matched_distance_px),
        "p90_matched_distance_px": float(metrics.p90_matched_distance_px),
    }


def _execute_rank_drawing_views(
    request: dict[str, Any], workspace: Path
) -> dict[str, Any]:
    from cadgenbench.tools.drawing_view import (
        CAPACITY_TRANSPORT_SCORER_V1_CONFIG,
        ViewCandidate,
        _render_cad_view,
        _write_comparison,
        evaluate_projection_v2,
        extract_envelope_geometry_v2,
        project_step,
        render_step_view,
    )

    step_path, _ = _safe_regular_file(workspace, request["step_path"], "step_path")
    crop_path, _ = _safe_regular_file(workspace, request["crop_path"], "crop_path")
    output_dir, _ = _safe_feedback_output_dir(workspace, request["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    previews_dir = output_dir / "previews"
    resolved_candidates = []
    for index, original in enumerate(request["candidates"]):
        candidate = dict(original)
        render_path = candidate.get("render_path")
        if render_path is not None:
            _, render_relative = _safe_regular_file(
                workspace,
                render_path,
                f"candidates[{index}].render_path",
            )
        else:
            if previews_dir.exists() and (
                previews_dir.is_symlink() or not previews_dir.is_dir()
            ):
                raise RuntimeError(f"Refusing unsafe preview output: {previews_dir}")
            previews_dir.mkdir(parents=True, exist_ok=True)
            destination = _safe_output_file(
                previews_dir, f"{candidate['candidate_id']}.png"
            )
            spec = candidate["view_spec"]
            render_step_view(
                step_path,
                destination,
                view=spec["name"],
                direction=tuple(spec["direction"]),
                up=tuple(spec["up"]),
                section_offset_fraction=spec.get("section_offset_fraction"),
                section_keep=spec.get("section_keep"),
                show_hidden_lines=True,
            )
            if not destination.is_file() or destination.is_symlink():
                raise RuntimeError(
                    f"2D renderer did not produce preview for {candidate['candidate_id']}"
                )
            render_relative = _relative_to_workspace(workspace, destination)
        candidate["render_path"] = render_relative
        resolved_candidates.append(candidate)
    candidates = tuple(
        ViewCandidate(
            name=item["view_spec"]["name"],
            direction=tuple(item["view_spec"]["direction"]),
            up=tuple(item["view_spec"]["up"]),
            section_offset_fraction=item["view_spec"].get(
                "section_offset_fraction"
            ),
            section_keep=item["view_spec"].get("section_keep"),
        )
        for item in resolved_candidates
    )
    source_by_name = {
        item["view_spec"]["name"]: item for item in resolved_candidates
    }
    reference_geometry = extract_envelope_geometry_v2(
        crop_path,
        config=CAPACITY_TRANSPORT_SCORER_V1_CONFIG,
    )
    evaluated = []
    for candidate in candidates:
        projection = project_step(
            step_path,
            view=candidate.name,
            direction=candidate.direction,
            up=candidate.up,
            section_origin=candidate.section_origin,
            section_offset_fraction=candidate.section_offset_fraction,
            section_keep=candidate.section_keep,
        )
        evaluation = evaluate_projection_v2(
            projection,
            crop_path,
            reference_geometry=reference_geometry,
            max_rotation_deg=request["max_rotation_deg"],
            seed=request["seed"],
            max_iterations=request["max_iterations"],
            config=CAPACITY_TRANSPORT_SCORER_V1_CONFIG,
        )
        if evaluation.metrics.mode != "capacity_transport":
            raise RuntimeError(
                "Production view ranker returned a non-capacity scoring mode"
            )
        source = source_by_name[candidate.name]
        candidate_dir = output_dir / "candidates" / source["candidate_id"]
        candidate_dir.mkdir(parents=True, exist_ok=True)
        comparison_path = _safe_output_file(candidate_dir, "comparison.png")
        aligned_render_path = _safe_output_file(
            candidate_dir, "aligned-render.png"
        )
        with Image.open(crop_path) as opened:
            reference = opened.convert("RGB")
        cad_view = _render_cad_view(
            reference.size,
            projection,
            evaluation.transform,
            show_hidden_lines=False,
        )
        cad_view.save(aligned_render_path, format="PNG", optimize=True)
        _write_comparison(
            reference,
            cad_view,
            comparison_path,
            view_name=candidate.name,
            score=float(evaluation.metrics.score),
        )
        evaluated.append(
            (source, evaluation, comparison_path, aligned_render_path)
        )

    evaluated.sort(
        key=lambda item: (
            bool(item[1].metrics.eligible),
            float(item[1].metrics.raw_score),
        ),
        reverse=True,
    )
    rankings = []
    for rank, (
        source,
        evaluation,
        comparison_path,
        aligned_render_path,
    ) in enumerate(evaluated, 1):
        metrics = evaluation.metrics
        rankings.append(
            {
                "rank": rank,
                "candidate_id": source["candidate_id"],
                "render_path": source["render_path"],
                "view_spec": _public_view_spec(source["view_spec"]),
                "score": float(metrics.score),
                "raw_score": float(metrics.raw_score),
                "eligible": bool(metrics.eligible),
                "failed_gates": list(metrics.failed_gates),
                "scoring_mode": "capacity_transport",
                "capacity_transport": _capacity_transport_payload(
                    metrics.capacity_transport
                ),
                "comparison_path": _relative_to_workspace(
                    workspace, comparison_path
                ),
                "aligned_render_path": _relative_to_workspace(
                    workspace, aligned_render_path
                ),
                "transform": _transform_payload(evaluation.transform),
                "alignment": {
                    "objective": float(evaluation.alignment.objective),
                    "mode": evaluation.alignment.mode,
                    "scale_at_bound": bool(evaluation.alignment.scale_at_bound),
                    "rotation_at_limit": bool(
                        evaluation.alignment.rotation_at_limit
                    ),
                    "translation_at_limit": bool(
                        evaluation.alignment.translation_at_limit
                    ),
                },
            }
        )
    top = rankings[: request["top_k"]]
    comparison_path = _safe_output_file(output_dir, "comparison.png")
    shutil.copy2(workspace / top[0]["comparison_path"], comparison_path)
    margin = (
        float(rankings[0]["score"] - rankings[1]["score"])
        if len(rankings) >= 2
        else None
    )
    return {
        "region_id": request["region_id"],
        "crop_path": request["crop_path"],
        "step_path": request["step_path"],
        "step_sha256": _sha256(step_path),
        "best": top[0],
        "top": top,
        "rankings": rankings,
        "margin": margin,
        "comparison_path": _relative_to_workspace(workspace, comparison_path),
        "scoring_mode": "capacity_transport",
        "scorer_profile": "capacity_transport_v1",
    }


def _fit_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    panel = Image.new("RGB", size, "white")
    contained = ImageOps.contain(image.convert("RGB"), size, Image.Resampling.LANCZOS)
    panel.paste(
        contained,
        ((size[0] - contained.width) // 2, (size[1] - contained.height) // 2),
    )
    return panel


def _compose_unaligned_comparison(
    crop_path: Path,
    render_path: Path | None,
    destination: Path,
    *,
    region_id: str,
    status: str,
) -> Path:
    with Image.open(crop_path) as crop_opened:
        crop = crop_opened.convert("RGB")
        if render_path is None:
            render = Image.new("RGB", crop.size, (232, 235, 239))
            placeholder = ImageDraw.Draw(render)
            try:
                placeholder_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
                detail_font = ImageFont.truetype("DejaVuSans.ttf", 17)
            except OSError:
                placeholder_font = ImageFont.load_default()
                detail_font = ImageFont.load_default()
            placeholder.rectangle(
                (8, 8, render.width - 9, render.height - 9),
                outline=(145, 153, 163),
                width=2,
            )
            placeholder.text(
                (24, max(24, render.height // 2 - 38)),
                status.upper(),
                fill=(74, 83, 94),
                font=placeholder_font,
            )
            placeholder.text(
                (24, max(56, render.height // 2 + 2)),
                "No supported CAD comparison is available for this region.",
                fill=(90, 99, 110),
                font=detail_font,
            )
            right_title = f"No CAD rendering ({status})"
        else:
            with Image.open(render_path) as render_opened:
                render = render_opened.convert("RGB")
            right_title = "Candidate CAD (unaligned)"
        panel_width = max(crop.width, render.width)
        panel_height = max(crop.height, render.height)
        panel_size = (panel_width, panel_height)
        margin, gap, header = 16, 16, 52
        canvas = Image.new(
            "RGB",
            (2 * margin + 2 * panel_width + gap, 2 * margin + header + panel_height),
            (244, 246, 248),
        )
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
        except OSError:
            font = ImageFont.load_default()
        draw.text((margin, margin), f"{region_id} · Original crop", fill=(25, 32, 42), font=font)
        draw.text(
            (margin + panel_width + gap, margin),
            right_title,
            fill=(25, 32, 42),
            font=font,
        )
        image_y = margin + header
        canvas.paste(_fit_panel(crop, panel_size), (margin, image_y))
        canvas.paste(
            _fit_panel(render, panel_size),
            (margin + panel_width + gap, image_y),
        )
        canvas.save(destination, format="PNG", optimize=True)
    return destination


def _stack_comparisons(
    paths: list[Path],
    destination: Path,
    *,
    score: float | None,
    regions: list[dict[str, Any]],
) -> Path:
    opened = [Image.open(path).convert("RGB") for path in paths]
    try:
        gap = 16
        row_header = 40
        header = 52 if score is not None else 0
        fixed_height = (
            header + row_header * len(opened) + gap * (len(opened) - 1)
        )
        aspect_sum = sum(image.height / image.width for image in opened)
        target_width = max(image.width for image in opened)
        height_width_limit = math.floor(
            (MAX_DRAWING_SHEET_HEIGHT_PX - fixed_height - len(opened))
            / aspect_sum
        )
        effective_fixed_height = fixed_height + len(opened)
        pixel_width_limit = math.floor(
            (
                -effective_fixed_height
                + math.sqrt(
                    effective_fixed_height**2
                    + 4 * aspect_sum * MAX_DRAWING_SHEET_PIXELS
                )
            )
            / (2 * aspect_sum)
        )
        target_width = max(
            1, min(target_width, height_width_limit, pixel_width_limit)
        )
        rows = [
            image.copy()
            if image.width == target_width
            else image.resize(
                (
                    target_width,
                    max(1, round(image.height * target_width / image.width)),
                ),
                Image.Resampling.LANCZOS,
            )
            for image in opened
        ]
        height = fixed_height + sum(row.height for row in rows)
        if (
            height > MAX_DRAWING_SHEET_HEIGHT_PX
            or target_width * height > MAX_DRAWING_SHEET_PIXELS
        ):
            raise RuntimeError(
                "Comparison sheet could not be fitted within the image limits"
            )
        canvas = Image.new("RGB", (target_width, height), (238, 241, 244))
        draw = ImageDraw.Draw(canvas)
        try:
            title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
            row_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 17)
        except OSError:
            title_font = ImageFont.load_default()
            row_font = ImageFont.load_default()
        if score is not None:
            draw.text(
                (16, 12),
                f"Overall geometry score = {score:.4f}",
                fill=(35, 72, 115),
                font=title_font,
            )
        y = header
        for row, region in zip(rows, regions, strict=True):
            view_name = region.get("view_spec", {}).get("name", "none")
            region_score = region.get("score")
            margin = region.get("margin")
            score_text = "n/a" if region_score is None else f"{region_score:.4f}"
            margin_text = "n/a" if margin is None else f"{margin:.4f}"
            draw.text(
                (16, y + 9),
                (
                    f"{region['region_id']} · status={region['status']} · "
                    f"view={view_name} · score={score_text} · margin={margin_text}"
                ),
                fill=(35, 72, 115),
                font=row_font,
            )
            y += row_header
            canvas.paste(row, (0, y))
            y += row.height + gap
        canvas.save(destination, format="PNG", optimize=True)
    finally:
        for image in opened:
            image.close()
    return destination


def _execute_compare_drawing_sheet(
    request: dict[str, Any], workspace: Path
) -> dict[str, Any]:
    output_dir, _ = _safe_feedback_output_dir(workspace, request["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    row_dir = output_dir / "rows"
    if row_dir.exists() and row_dir.is_symlink():
        raise RuntimeError(f"Refusing unsafe drawing output: {row_dir}")
    row_dir.mkdir(parents=True, exist_ok=True)
    comparison_paths = []
    selected_regions = []
    scores = []
    for index, region in enumerate(request["regions"]):
        if "comparison_path" in region:
            comparison, _ = _safe_regular_file(
                workspace, region["comparison_path"], "comparison_path"
            )
            aligned = True
        else:
            comparison = _safe_output_file(
                row_dir, f"{index:02d}-{region['region_id']}.png"
            )
            render_path = (
                _safe_regular_file(
                    workspace, region["render_path"], "render_path"
                )[0]
                if "render_path" in region
                else None
            )
            _compose_unaligned_comparison(
                _safe_regular_file(workspace, region["crop_path"], "crop_path")[0],
                render_path,
                comparison,
                region_id=region["region_id"],
                status=region["status"],
            )
            aligned = False
        comparison_paths.append(comparison)
        selected_region = {
            key: region[key]
            for key in (
                "region_id",
                "status",
                "crop_path",
                "render_path",
                "view_spec",
                "score",
                "margin",
            )
            if key in region
        }
        selected_region.update(
            {
                "comparison_path": _relative_to_workspace(workspace, comparison),
                "aligned": aligned,
            }
        )
        selected_regions.append(selected_region)
        if region.get("score") is not None:
            scores.append(float(region["score"]))
    overall_score = sum(scores) / len(scores) if scores else None
    destination = _safe_output_file(output_dir, "comparison.png")
    _stack_comparisons(
        comparison_paths,
        destination,
        score=overall_score,
        regions=request["regions"],
    )
    return {
        "regions": selected_regions,
        "score": overall_score,
        "sheet_path": _relative_to_workspace(workspace, destination),
        "mime_type": "image/png",
        "sha256": _sha256(destination),
    }


def _reflection_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(filename, size)
    except OSError:
        return ImageFont.load_default()


def _pixel_roi(
    roi: tuple[int, int, int, int] | list[int],
    size: tuple[int, int],
) -> tuple[int, int, int, int]:
    width, height = size
    left = max(0, min(width - 1, math.floor(roi[0] * width / 1000)))
    top = max(0, min(height - 1, math.floor(roi[1] * height / 1000)))
    right = max(left + 1, min(width, math.ceil(roi[2] * width / 1000)))
    bottom = max(top + 1, min(height, math.ceil(roi[3] * height / 1000)))
    return left, top, right, bottom


def _pixel_grid_cell_roi(
    cell: str, size: tuple[int, int]
) -> tuple[int, int, int, int]:
    width, height = size
    if width < 4 or height < 4:
        raise RuntimeError("Fixed 4x4 reflection grids require images at least 4x4 pixels")
    column = ord(cell[0]) - ord("A")
    row = int(cell[1]) - 1
    return (
        column * width // 4,
        row * height // 4,
        (column + 1) * width // 4,
        (row + 1) * height // 4,
    )


def _render_reflection_context_panel(
    image: Image.Image,
    title: str,
    *,
    roi: tuple[int, int, int, int] | list[int] | None,
    size: tuple[int, int] = (720, 390),
    show_grid: bool = True,
) -> Image.Image:
    panel = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(panel)
    title_height = 38
    margin = 10
    draw.text((12, 8), title, fill=(25, 32, 42), font=_reflection_font(20, bold=True))
    available = (size[0] - margin * 2, size[1] - title_height - margin)
    scale = min(available[0] / image.width, available[1] / image.height)
    rendered_size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    rendered = image.resize(rendered_size, Image.Resampling.LANCZOS)
    offset_x = (size[0] - rendered.width) // 2
    offset_y = title_height + (size[1] - title_height - rendered.height) // 2
    panel.paste(rendered, (offset_x, offset_y))
    draw.rectangle(
        (
            offset_x - 1,
            offset_y - 1,
            offset_x + rendered.width,
            offset_y + rendered.height,
        ),
        outline=(160, 168, 178),
        width=1,
    )
    if show_grid:
        grid_font = _reflection_font(max(12, min(18, rendered.width // 30)), bold=True)
        for index in range(1, 4):
            x = offset_x + round(index * rendered.width / 4)
            y = offset_y + round(index * rendered.height / 4)
            draw.line((x, offset_y, x, offset_y + rendered.height), fill=(126, 162, 199), width=1)
            draw.line((offset_x, y, offset_x + rendered.width, y), fill=(126, 162, 199), width=1)
        for row in range(4):
            for column in range(4):
                x = offset_x + round(column * rendered.width / 4) + 5
                y = offset_y + round(row * rendered.height / 4) + 4
                label = f"{chr(ord('A') + column)}{row + 1}"
                box = draw.textbbox((x, y), label, font=grid_font)
                draw.rectangle((box[0] - 2, box[1] - 1, box[2] + 2, box[3] + 1), fill=(255, 255, 255))
                draw.text((x, y), label, fill=(35, 72, 115), font=grid_font)
    if roi is not None:
        left, top, right, bottom = roi
        x0 = offset_x + round(left * rendered.width / 1000)
        y0 = offset_y + round(top * rendered.height / 1000)
        x1 = offset_x + round(right * rendered.width / 1000)
        y1 = offset_y + round(bottom * rendered.height / 1000)
        draw.rectangle((x0, y0, x1, y1), outline=(210, 45, 45), width=4)
    return panel


def _target_mark_badge_size(
    draw: ImageDraw.ImageDraw, mark: int, font: ImageFont.ImageFont
) -> tuple[int, int, tuple[int, int, int, int]]:
    text_box = draw.textbbox((0, 0), str(mark), font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    return max(22, text_width + 10), max(22, text_height + 8), text_box


def _clamped_badge_rect(
    center: tuple[int, int],
    size: tuple[int, int],
    bounds: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    center_x, center_y = center
    width, height = size
    bound_left, bound_top, bound_right, bound_bottom = bounds
    left = min(max(bound_left, round(center_x - width / 2)), bound_right - width)
    top = min(max(bound_top, round(center_y - height / 2)), bound_bottom - height)
    return left, top, left + width, top + height


def _rect_overlap_area(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> int:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    return max(0, right - left) * max(0, bottom - top)


def _target_mark_ink_score(
    clean_panel: Image.Image, rect: tuple[int, int, int, int]
) -> float:
    grayscale = clean_panel.crop(rect).convert("L")
    histogram = grayscale.histogram()
    pixels = max(1, grayscale.width * grayscale.height)
    return sum((255 - value) * count for value, count in enumerate(histogram)) / pixels


def _choose_target_mark_badge_rect(
    clean_panel: Image.Image,
    candidates: list[tuple[tuple[int, int], int]],
    size: tuple[int, int],
    bounds: tuple[int, int, int, int],
    occupied: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int]:
    scored: list[tuple[float, int, tuple[int, int, int, int]]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for order, (center, placement_priority) in enumerate(candidates):
        rect = _clamped_badge_rect(center, size, bounds)
        if rect in seen:
            continue
        seen.add(rect)
        overlap = sum(_rect_overlap_area(rect, other) for other in occupied)
        score = (
            overlap * 1_000_000
            + placement_priority * 10_000
            + _target_mark_ink_score(clean_panel, rect)
        )
        scored.append((score, order, rect))
    if not scored:
        raise RuntimeError("No valid Target mark badge position")
    return min(scored, key=lambda item: (item[0], item[1]))[2]


def _draw_target_mark_badge(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    mark: int,
    font: ImageFont.ImageFont,
    text_box: tuple[int, int, int, int],
) -> None:
    color = (0, 132, 104, 235)
    draw.rounded_rectangle(
        rect,
        radius=min(8, (rect[3] - rect[1]) // 2),
        fill=color,
        outline=(255, 255, 255, 245),
        width=2,
    )
    text = str(mark)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    text_x = rect[0] + (rect[2] - rect[0] - text_width) // 2 - text_box[0]
    text_y = rect[1] + (rect[3] - rect[1] - text_height) // 2 - text_box[1]
    draw.text((text_x, text_y), text, fill=(255, 255, 255, 255), font=font)


def _draw_target_box(
    draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]
) -> None:
    draw.rectangle(box, outline=(0, 132, 104, 140), width=2)


def _render_reflection_coordinate_panel(
    image: Image.Image,
    title: str,
    *,
    roi: tuple[int, int, int, int] | list[int] | None,
    annotations: list[dict[str, Any]] | None = None,
    size: tuple[int, int] = (720, 450),
) -> Image.Image:
    """Render one clean comparison panel without coordinate decorations."""
    panel = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(panel)
    title_height = 38
    margin = 10
    draw.text((12, 8), title, fill=(25, 32, 42), font=_reflection_font(20, bold=True))
    usable_width = size[0] - margin * 2
    usable_height = size[1] - title_height - margin
    scale = min(usable_width / image.width, usable_height / image.height)
    rendered_size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    rendered = image.resize(rendered_size, Image.Resampling.LANCZOS)
    offset_x = margin + (usable_width - rendered.width) // 2
    offset_y = title_height + (usable_height - rendered.height) // 2
    panel.paste(rendered, (offset_x, offset_y))
    if roi is not None:
        left, top, right, bottom = roi
        x0 = offset_x + round(left * rendered.width / 1000)
        y0 = offset_y + round(top * rendered.height / 1000)
        x1 = offset_x + round(right * rendered.width / 1000)
        y1 = offset_y + round(bottom * rendered.height / 1000)
        draw.rectangle((x0, y0, x1, y1), outline=(210, 45, 45), width=4)
    clean_panel = panel.copy()
    annotation_layer = Image.new("RGBA", panel.size, (0, 0, 0, 0))
    annotation_draw = ImageDraw.Draw(annotation_layer)
    annotation_font = _reflection_font(14, bold=True)
    image_bounds = (
        offset_x,
        offset_y,
        offset_x + rendered.width,
        offset_y + rendered.height,
    )
    occupied_badges: list[tuple[int, int, int, int]] = []
    ordered_annotations = sorted(
        annotations or [],
        key=lambda annotation: (
            (annotation["box"][2] - annotation["box"][0])
            * (annotation["box"][3] - annotation["box"][1])
            if "box" in annotation
            else -1
        ),
        reverse=True,
    )
    for annotation in ordered_annotations:
        mark = annotation["mark"]
        badge_width, badge_height, text_box = _target_mark_badge_size(
            draw, mark, annotation_font
        )
        if "point" in annotation:
            point_x, point_y = annotation["point"]
            anchor_x = offset_x + round(point_x * rendered.width / 1000)
            anchor_y = offset_y + round(point_y * rendered.height / 1000)
            horizontal_offset = badge_width + 3
            vertical_offset = badge_height + 3
            candidates = [
                ((anchor_x + horizontal_offset, anchor_y), 0),
                ((anchor_x - horizontal_offset, anchor_y), 0),
                ((anchor_x, anchor_y + vertical_offset), 0),
                ((anchor_x, anchor_y - vertical_offset), 0),
                ((anchor_x + horizontal_offset, anchor_y + vertical_offset), 1),
                ((anchor_x - horizontal_offset, anchor_y + vertical_offset), 1),
                ((anchor_x + horizontal_offset, anchor_y - vertical_offset), 1),
                ((anchor_x - horizontal_offset, anchor_y - vertical_offset), 1),
                ((anchor_x, anchor_y), 2),
            ]
            badge_rect = _choose_target_mark_badge_rect(
                clean_panel,
                candidates,
                (badge_width, badge_height),
                image_bounds,
                occupied_badges,
            )
            badge_center = (
                (badge_rect[0] + badge_rect[2]) // 2,
                (badge_rect[1] + badge_rect[3]) // 2,
            )
            if badge_center != (anchor_x, anchor_y):
                annotation_draw.line(
                    (anchor_x, anchor_y, badge_center[0], badge_center[1]),
                    fill=(0, 132, 104, 150),
                    width=1,
                )
                annotation_draw.ellipse(
                    (anchor_x - 2, anchor_y - 2, anchor_x + 2, anchor_y + 2),
                    fill=(0, 132, 104, 190),
                )
        else:
            left, top, right, bottom = annotation["box"]
            box = (
                offset_x + round(left * rendered.width / 1000),
                offset_y + round(top * rendered.height / 1000),
                offset_x + round(right * rendered.width / 1000),
                offset_y + round(bottom * rendered.height / 1000),
            )
            _draw_target_box(annotation_draw, box)
            half_width = badge_width // 2
            half_height = badge_height // 2
            candidates = [
                ((box[0] - half_width + 3, box[1] - half_height + 3), 0),
                ((box[2] + half_width - 3, box[1] - half_height + 3), 0),
                ((box[0] - half_width + 3, box[3] + half_height - 3), 0),
                ((box[2] + half_width - 3, box[3] + half_height - 3), 0),
            ]
            badge_rect = _choose_target_mark_badge_rect(
                clean_panel,
                candidates,
                (badge_width, badge_height),
                image_bounds,
                occupied_badges,
            )
        _draw_target_mark_badge(
            annotation_draw, badge_rect, mark, annotation_font, text_box
        )
        occupied_badges.append(badge_rect)
    return Image.alpha_composite(panel.convert("RGBA"), annotation_layer).convert("RGB")


def _render_reflection_local_panel(
    crop: Image.Image,
    title: str,
    shared_size: tuple[int, int],
    *,
    size: tuple[int, int] = (720, 460),
) -> Image.Image:
    panel = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(panel)
    title_height = 38
    margin = 10
    draw.text((12, 8), title, fill=(25, 32, 42), font=_reflection_font(20, bold=True))
    shared = Image.new("RGB", shared_size, (255, 255, 255))
    paste_x = (shared.width - crop.width) // 2
    paste_y = (shared.height - crop.height) // 2
    shared.paste(crop, (paste_x, paste_y))
    shared_draw = ImageDraw.Draw(shared)
    shared_draw.rectangle(
        (paste_x, paste_y, paste_x + crop.width - 1, paste_y + crop.height - 1),
        outline=(160, 168, 178),
        width=max(1, round(max(shared_size) / 500)),
    )
    available = (size[0] - margin * 2, size[1] - title_height - margin)
    scale = min(available[0] / shared.width, available[1] / shared.height)
    rendered_size = (
        max(1, round(shared.width * scale)),
        max(1, round(shared.height * scale)),
    )
    rendered = shared.resize(rendered_size, Image.Resampling.LANCZOS)
    offset_x = (size[0] - rendered.width) // 2
    offset_y = title_height + (size[1] - title_height - rendered.height) // 2
    panel.paste(rendered, (offset_x, offset_y))
    return panel


def _compose_reflection_pair(
    left: Image.Image,
    right: Image.Image,
    *,
    header_text: str,
) -> Image.Image:
    margin, gap, header = 16, 16, 64
    canvas = Image.new(
        "RGB",
        (margin * 2 + left.width + right.width + gap, header + max(left.height, right.height) + margin),
        (244, 246, 248),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 17), header_text, fill=(25, 32, 42), font=_reflection_font(23, bold=True))
    canvas.paste(left, (margin, header))
    canvas.paste(right, (margin + left.width + gap, header))
    return canvas


def _execute_compose_reflection_inspection(
    request: dict[str, Any], workspace: Path
) -> dict[str, Any]:
    output_dir, _ = _safe_feedback_output_dir(workspace, request["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    crop_path = _safe_regular_file(workspace, request["crop_path"], "crop_path")[0]
    aligned_path = _safe_regular_file(
        workspace, request["aligned_render_path"], "aligned_render_path"
    )[0]
    with Image.open(crop_path) as reference_opened, Image.open(aligned_path) as cad_opened:
        reference = reference_opened.convert("RGB")
        cad = cad_opened.convert("RGB")
    if reference.size != cad.size:
        raise RuntimeError(
            "Aligned reflection inputs must have exactly the same pixel dimensions"
        )
    if reference.width > MAX_DRAWING_EDGE_PX or reference.height > MAX_DRAWING_EDGE_PX:
        raise RuntimeError("Aligned reflection inputs exceed the drawing edge limit")

    region_id = request["region_id"]
    mode = request["mode"]
    if mode == "coordinate_overview":
        reference_context = _render_reflection_coordinate_panel(
            reference,
            "Original engineering drawing",
            roi=None,
            annotations=request.get("target_annotations", []),
        )
        cad_context = _render_reflection_coordinate_panel(
            cad,
            "Current aligned CAD",
            roi=None,
        )
        context = _compose_reflection_pair(
            reference_context,
            cad_context,
            header_text=f"{region_id} · complete aligned comparison",
        )
        context_path = _safe_output_file(output_dir, "coordinate-overview.png")
        context.save(context_path, format="PNG", optimize=True)
        return {
            "region_id": region_id,
            "mode": mode,
            "width": reference.width,
            "height": reference.height,
            "context_path": _relative_to_workspace(workspace, context_path),
            "sha256": {"context": _sha256(context_path)},
        }

    if mode == "grid_overview":
        reference_context = _render_reflection_context_panel(
            reference, "1  Original drawing · fixed 4×4 grid", roi=None
        )
        cad_context = _render_reflection_context_panel(
            cad, "2  Aligned CAD · fixed 4×4 grid", roi=None
        )
        context = _compose_reflection_pair(
            reference_context,
            cad_context,
            header_text=f"{region_id} · complete locked comparison · cells A1-D4",
        )
        context_path = _safe_output_file(output_dir, "grid-overview.png")
        context.save(context_path, format="PNG", optimize=True)
        return {
            "region_id": region_id,
            "mode": mode,
            "width": reference.width,
            "height": reference.height,
            "context_path": _relative_to_workspace(workspace, context_path),
            "sha256": {"context": _sha256(context_path)},
        }

    if mode == "dual_roi":
        reference_roi = request["reference_roi"]
        cad_roi = request["cad_roi"]
        reference_pixel_roi = _pixel_roi(reference_roi, reference.size)
        cad_pixel_roi = _pixel_roi(cad_roi, cad.size)
        reference_crop = reference.crop(reference_pixel_roi)
        cad_crop = cad.crop(cad_pixel_roi)
        reference_crop_path = _safe_output_file(output_dir, "reference-crop.png")
        cad_crop_path = _safe_output_file(output_dir, "cad-crop.png")
        reference_crop.save(reference_crop_path, format="PNG", optimize=True)
        cad_crop.save(cad_crop_path, format="PNG", optimize=True)

        reference_context = _render_reflection_coordinate_panel(
            reference,
            f"Original drawing — reference_roi {list(reference_roi)}",
            roi=reference_roi,
        )
        cad_context = _render_reflection_coordinate_panel(
            cad,
            f"Aligned CAD — cad_roi {list(cad_roi)}",
            roi=cad_roi,
        )
        context = _compose_reflection_pair(
            reference_context,
            cad_context,
            header_text=(
                f"{region_id} · red boxes use each panel's independent 0–1000 coordinates"
            ),
        )
        context_path = _safe_output_file(output_dir, "context.png")
        context.save(context_path, format="PNG", optimize=True)

        shared_size = (
            max(reference_crop.width, cad_crop.width),
            max(reference_crop.height, cad_crop.height),
        )
        reference_local = _render_reflection_local_panel(
            reference_crop,
            f"3  Original local crop · {reference_crop.width}×{reference_crop.height}px",
            shared_size,
        )
        cad_local = _render_reflection_local_panel(
            cad_crop,
            f"4  CAD local crop · {cad_crop.width}×{cad_crop.height}px",
            shared_size,
        )
        margin, gap = 16, 16
        canvas = Image.new(
            "RGB",
            (
                context.width,
                context.height
                + gap
                + max(reference_local.height, cad_local.height)
                + margin,
            ),
            (244, 246, 248),
        )
        canvas.paste(context, (0, 0))
        local_y = context.height + gap
        canvas.paste(reference_local, (margin, local_y))
        canvas.paste(cad_local, (margin + reference_local.width + gap, local_y))
        comparison_path = _safe_output_file(output_dir, "comparison.png")
        canvas.save(comparison_path, format="PNG", optimize=True)
        return {
            "region_id": region_id,
            "mode": mode,
            "reference_roi": list(reference_roi),
            "cad_roi": list(cad_roi),
            "reference_pixel_roi": list(reference_pixel_roi),
            "cad_pixel_roi": list(cad_pixel_roi),
            "width": reference.width,
            "height": reference.height,
            "reference_crop_path": _relative_to_workspace(
                workspace, reference_crop_path
            ),
            "cad_crop_path": _relative_to_workspace(workspace, cad_crop_path),
            "context_path": _relative_to_workspace(workspace, context_path),
            "comparison_path": _relative_to_workspace(
                workspace, comparison_path
            ),
            "sha256": {
                "reference_crop": _sha256(reference_crop_path),
                "cad_crop": _sha256(cad_crop_path),
                "context": _sha256(context_path),
                "comparison": _sha256(comparison_path),
            },
        }

    reference_cell = request["reference_cell"]
    cad_cell = request["cad_cell"]
    reference_roi = request["reference_roi"]
    cad_roi = request["cad_roi"]
    reference_pixel_roi = _pixel_grid_cell_roi(reference_cell, reference.size)
    cad_pixel_roi = _pixel_grid_cell_roi(cad_cell, cad.size)
    reference_crop = reference.crop(reference_pixel_roi)
    cad_crop = cad.crop(cad_pixel_roi)
    reference_crop_path = _safe_output_file(output_dir, "reference-crop.png")
    cad_crop_path = _safe_output_file(output_dir, "cad-crop.png")
    reference_crop.save(reference_crop_path, format="PNG", optimize=True)
    cad_crop.save(cad_crop_path, format="PNG", optimize=True)

    reference_context = _render_reflection_context_panel(
        reference, f"1  Original drawing · selected {reference_cell}", roi=reference_roi
    )
    cad_context = _render_reflection_context_panel(
        cad, f"2  Aligned CAD · selected {cad_cell}", roi=cad_roi
    )
    context = _compose_reflection_pair(
        reference_context,
        cad_context,
        header_text=(
            f"{region_id} · fixed-cell comparison · "
            f"original {reference_cell} / CAD {cad_cell}"
        ),
    )
    context_path = _safe_output_file(output_dir, "context.png")
    context.save(context_path, format="PNG", optimize=True)

    shared_size = (
        max(reference_crop.width, cad_crop.width),
        max(reference_crop.height, cad_crop.height),
    )
    reference_local = _render_reflection_local_panel(
        reference_crop,
        f"3  Original local crop · {reference_crop.width}×{reference_crop.height}px",
        shared_size,
    )
    cad_local = _render_reflection_local_panel(
        cad_crop,
        f"4  CAD local crop · {cad_crop.width}×{cad_crop.height}px",
        shared_size,
    )
    margin, gap, header = 16, 16, 64
    canvas = Image.new(
        "RGB",
        (
            context.width,
            context.height + gap + max(reference_local.height, cad_local.height) + margin,
        ),
        (244, 246, 248),
    )
    canvas.paste(context, (0, 0))
    local_y = context.height + gap
    canvas.paste(reference_local, (margin, local_y))
    canvas.paste(cad_local, (margin + reference_local.width + gap, local_y))
    comparison_path = _safe_output_file(output_dir, "comparison.png")
    canvas.save(comparison_path, format="PNG", optimize=True)
    return {
        "region_id": region_id,
        "mode": mode,
        "reference_cell": reference_cell,
        "cad_cell": cad_cell,
        "reference_roi": list(reference_roi),
        "cad_roi": list(cad_roi),
        "reference_pixel_roi": list(reference_pixel_roi),
        "cad_pixel_roi": list(cad_pixel_roi),
        "width": reference.width,
        "height": reference.height,
        "reference_crop_path": _relative_to_workspace(
            workspace, reference_crop_path
        ),
        "cad_crop_path": _relative_to_workspace(workspace, cad_crop_path),
        "context_path": _relative_to_workspace(workspace, context_path),
        "comparison_path": _relative_to_workspace(workspace, comparison_path),
        "sha256": {
            "reference_crop": _sha256(reference_crop_path),
            "cad_crop": _sha256(cad_crop_path),
            "context": _sha256(context_path),
            "comparison": _sha256(comparison_path),
        },
    }


def _execute_compose_shape_feedback(
    request: dict[str, Any], workspace: Path
) -> dict[str, Any]:
    """Compare fixed-frame candidate projections against the same drawing."""
    from cadgenbench.integrations.shape_feedback import compose_shape_feedback

    paths = {
        field: _safe_regular_file(workspace, request[field], field)[0]
        for field in ("reference_path", "current_render_path")
    }
    candidate_kwargs: dict[str, Any]
    if "candidates" in request:
        candidates = [
            {
                "candidate_id": candidate["candidate_id"],
                "hypothesis": candidate["hypothesis"],
                "render_path": _safe_regular_file(
                    workspace,
                    candidate["render_path"],
                    f"candidates[{index}].render_path",
                )[0],
            }
            for index, candidate in enumerate(request["candidates"])
        ]
        candidate_kwargs = {"candidates": candidates}
        candidate_paths = [candidate["render_path"] for candidate in candidates]
    else:
        candidate_path = _safe_regular_file(
            workspace, request["candidate_render_path"], "candidate_render_path"
        )[0]
        candidate_kwargs = {
            "candidate_render_path": candidate_path,
            "candidate_id": request["candidate_id"],
            "hypothesis": request["hypothesis"],
        }
        candidate_paths = [candidate_path]
    # Bound raster work before constructing the shared display canvas.
    for path in [*paths.values(), *candidate_paths]:
        with Image.open(path) as raster:
            if (
                raster.width > MAX_DRAWING_EDGE_PX
                or raster.height > MAX_DRAWING_SHEET_HEIGHT_PX
                or raster.width * raster.height > MAX_DRAWING_SHEET_PIXELS
            ):
                raise ProtocolError("Shape-feedback raster exceeds drawing size limits")
    output_dir = _safe_feedback_output_dir(workspace, request["output_dir"])[0]
    response = compose_shape_feedback(
        **paths,
        **candidate_kwargs,
        output_dir=output_dir,
        region_id=request["region_id"],
        target_roi=request["target_roi"],
    )
    response["card_path"] = _relative_to_workspace(
        workspace, Path(response["card_path"])
    )
    for card in response["cards"]:
        card["card_path"] = _relative_to_workspace(workspace, Path(card["card_path"]))
    for item in response["files"].values():
        item["path"] = _relative_to_workspace(workspace, Path(item["path"]))
    return response


def _execute_compare_candidate_topology(
    request: dict[str, Any], workspace: Path
) -> dict[str, Any]:
    """Report candidate-minus-Current Betti changes without a target CAD or score."""
    from cadgenbench.eval.topo_match import compute_betti_for_step

    _safe_feedback_output_dir(workspace, request["output_dir"])
    current_step, current_relative = _safe_regular_file(
        workspace, request["current_step_path"], "current_step_path"
    )
    current_hash = _sha256(current_step)
    current_topology = None
    current_error = None
    linear_deflection_mm = None
    try:
        current_result = compute_betti_for_step(current_step)
        current_topology = current_result.to_dict()
        linear_deflection_mm = current_result.linear_deflection_mm
    except Exception as exc:  # noqa: BLE001 - topology must remain optional feedback
        current_error = f"{type(exc).__name__}: {exc}"

    candidates = []
    for index, candidate in enumerate(request["candidates"]):
        step_path, step_relative = _safe_regular_file(
            workspace, candidate["step_path"], f"candidates[{index}].step_path"
        )
        result = {
            "candidate_id": candidate["candidate_id"],
            "step_path": step_relative,
            "step_sha256": _sha256(step_path),
            "status": "error",
            "topology": None,
            "topology_delta": None,
            "error": None,
        }
        try:
            # If Current failed, each candidate can still report its own topology,
            # using its default deflection. A relative change remains unavailable.
            topology = compute_betti_for_step(
                step_path, linear_deflection_mm=linear_deflection_mm
            ).to_dict()
            result["topology"] = topology
            result["status"] = "ok"
            if current_topology is not None:
                result["topology_delta"] = {
                    key: int(topology[key]) - int(current_topology[key])
                    for key in ("b0", "b1", "b2")
                }
        except Exception as exc:  # noqa: BLE001 - retain successful sibling results
            result["error"] = f"{type(exc).__name__}: {exc}"
        candidates.append(result)

    return {
        "current_step_path": current_relative,
        "current_step_sha256": current_hash,
        "current_topology": current_topology,
        "current_error": current_error,
        "candidates": candidates,
    }


def _execute_drawing_request(
    request: dict[str, Any], workspace: Path
) -> dict[str, Any]:
    operation = request["operation"]
    if operation == "crop_drawing_region":
        return _execute_crop_drawing_region(request, workspace)
    if operation == "render_drawing_view":
        return _execute_render_drawing_view(request, workspace)
    if operation == "render_fixed_projection":
        return _execute_render_fixed_projection(request, workspace)
    if operation == "rank_drawing_views":
        return _execute_rank_drawing_views(request, workspace)
    if operation == "compose_reflection_inspection":
        return _execute_compose_reflection_inspection(request, workspace)
    if operation == "compose_repair_comparison":
        return _execute_compose_repair_comparison(request, workspace)
    if operation == "compose_code_construction_highlight":
        return _execute_compose_code_construction_highlight(request, workspace)
    if operation == "compose_code_ablation_highlight":
        return _execute_compose_code_ablation_highlight(request, workspace)
    if operation == "compose_visual_diff":
        return _execute_compose_visual_diff(request, workspace)
    if operation == "compose_code_edit_visualization":
        return _execute_compose_code_edit_visualization(request, workspace)
    if operation == "compose_repair_candidate_preview":
        return _execute_compose_repair_candidate_preview(request, workspace)
    if operation == "compose_shape_feedback":
        return _execute_compose_shape_feedback(request, workspace)
    if operation == "compare_candidate_topology":
        return _execute_compare_candidate_topology(request, workspace)
    return _execute_compare_drawing_sheet(request, workspace)


def _drawing_process_entry(
    sender: Any, request: dict[str, Any], workspace: Path
) -> None:
    try:
        sender.send({"ok": True, "result": _execute_drawing_request(request, workspace)})
    except Exception as exc:  # noqa: BLE001 - serialized to the parent worker
        sender.send(
            {
                "ok": False,
                "error": {
                    "kind": "drawing",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc()[-16384:],
                },
            }
        )
    finally:
        sender.close()


def _run_drawing_operation(
    request: dict[str, Any], workspace: Path
) -> dict[str, Any]:
    global _ACTIVE_DRAWING_PROCESS

    canonical = _prepare_drawing_request(request, workspace)
    timeout_s = _require_number(
        request.get("timeout_s", DEFAULT_DRAWING_TIMEOUT_S),
        "timeout_s",
        minimum=0.001,
    )
    started = time.monotonic()
    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_drawing_process_entry,
        args=(sender, canonical, workspace),
        daemon=False,
    )
    process.start()
    _ACTIVE_DRAWING_PROCESS = process
    sender.close()
    deadline = time.monotonic() + timeout_s
    child_response: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        if receiver.poll(min(0.05, max(0.0, remaining))):
            child_response = receiver.recv()
            break
        if not process.is_alive():
            break
    process.join(max(0.0, deadline - time.monotonic()))
    if process.is_alive():
        process.terminate()
        process.join(1.0)
        if process.is_alive():
            process.kill()
            process.join()
        receiver.close()
        return {
            "protocol_version": PROTOCOL_VERSION,
            "operation": request["operation"],
            "status": "error",
            "duration_s": round(time.monotonic() - started, 6),
            "error": {
                "kind": "timeout",
                "error_type": "TimeoutError",
                "message": f"Drawing operation timed out after {timeout_s:g}s",
            },
        }

    if child_response is None and receiver.poll():
        child_response = receiver.recv()
    if child_response is None:
        child_response = {
            "ok": False,
            "error": {
                "kind": "worker",
                "error_type": "ChildProcessError",
                "message": f"Drawing child exited with code {process.exitcode}",
            },
        }
    receiver.close()
    response = {
        "protocol_version": PROTOCOL_VERSION,
        "operation": request["operation"],
        "status": "ok" if child_response["ok"] else "error",
        "duration_s": round(time.monotonic() - started, 6),
    }
    if child_response["ok"]:
        response.update(child_response["result"])
    else:
        error = dict(child_response["error"])
        error.pop("traceback", None)
        response["error"] = error
    return response


def _doctor_drawing_2d(doctor_workspace: Path) -> dict[str, Any]:
    """Exercise the deterministic merged-sheet 2D pipeline without an LLM."""

    common_render = {
        "operation": "render_drawing_view",
        "step_path": "output.step",
        "width": 320,
        "height": 240,
        "show_hidden_lines": True,
    }
    top = _execute_render_drawing_view(
        {
            **common_render,
            "region_id": "top",
            "view_spec": {
                "name": "top",
                "direction": [0.0, 0.0, 1.0],
                "up": [0.0, 1.0, 0.0],
                "section_offset_fraction": None,
                "section_keep": None,
            },
            "output_dir": "drawing_feedback/doctor/candidates/top",
        },
        doctor_workspace,
    )
    front = _execute_render_drawing_view(
        {
            **common_render,
            "region_id": "front",
            "view_spec": {
                "name": "front",
                "direction": [0.0, -1.0, 0.0],
                "up": [0.0, 0.0, 1.0],
                "section_offset_fraction": None,
                "section_keep": None,
            },
            "output_dir": "drawing_feedback/doctor/candidates/front",
        },
        doctor_workspace,
    )

    merged_path = doctor_workspace / "synthetic-merged-drawing.png"
    merged = Image.new("RGB", (1000, 1000), "white")
    with Image.open(doctor_workspace / top["render_path"]) as top_image:
        merged.paste(top_image.convert("RGB"), (100, 100))
    with Image.open(doctor_workspace / front["render_path"]) as front_image:
        secondary = ImageOps.contain(
            front_image.convert("RGB"), (280, 210), Image.Resampling.LANCZOS
        )
        merged.paste(secondary, (600, 100))
    draw = ImageDraw.Draw(merged)
    draw.rectangle((60, 60, 460, 380), outline=(80, 90, 100), width=2)
    draw.rectangle((560, 60, 920, 380), outline=(80, 90, 100), width=2)
    draw.rectangle((580, 820, 960, 960), outline=(40, 45, 50), width=3)
    draw.line((760, 820, 760, 960), fill=(40, 45, 50), width=2)
    draw.text((600, 845), "CADGenBench 2D DOCTOR", fill=(20, 25, 30))
    merged.save(merged_path, format="PNG", optimize=True)

    crop = _execute_crop_drawing_region(
        {
            "operation": "crop_drawing_region",
            "input_path": merged_path.name,
            "region_id": "top",
            "roi": [100, 100, 420, 340],
            "output_dir": "drawing_feedback/doctor/regions/top",
        },
        doctor_workspace,
    )
    ranked = _execute_rank_drawing_views(
        {
            "operation": "rank_drawing_views",
            "region_id": "top",
            "crop_path": crop["crop_path"],
            "step_path": "output.step",
            "candidates": [
                {
                    "candidate_id": "top",
                    "render_path": top["render_path"],
                    "view_spec": top["view_spec"],
                },
                {
                    "candidate_id": "front",
                    "render_path": front["render_path"],
                    "view_spec": front["view_spec"],
                },
            ],
            "top_k": 2,
            "allow_reflection": False,
            "max_rotation_deg": 3.0,
            "min_roi_fill": 0.70,
            "seed": 0,
            "max_iterations": 10,
            "output_dir": "drawing_feedback/doctor/ranking/top",
        },
        doctor_workspace,
    )
    sheet = _execute_compare_drawing_sheet(
        {
            "operation": "compare_drawing_sheet",
            "regions": [
                {
                    "region_id": "top",
                    "status": "matched",
                    "crop_path": crop["crop_path"],
                    "render_path": ranked["best"]["render_path"],
                    "comparison_path": ranked["comparison_path"],
                    "view_spec": ranked["best"]["view_spec"],
                    "score": ranked["best"]["score"],
                    "margin": ranked["margin"],
                }
            ],
            "output_dir": "drawing_feedback/doctor/sheet",
        },
        doctor_workspace,
    )

    def doctor_path(relative: str) -> str:
        return f".doctor/{relative}"

    return {
        "ok": True,
        "input_path": doctor_path(merged_path.name),
        "crop_path": doctor_path(crop["crop_path"]),
        "render_path": doctor_path(top["render_path"]),
        "comparison_path": doctor_path(ranked["comparison_path"]),
        "sheet_path": doctor_path(sheet["sheet_path"]),
        "best_view": ranked["best"]["view_spec"],
        "score": ranked["best"]["score"],
        "margin": ranked["margin"],
    }


def _restore_regular_file(path: Path, contents: bytes) -> None:
    temporary = path.with_name(f".{path.name}.restore-{os.getpid()}")
    temporary.write_bytes(contents)
    os.replace(temporary, path)


def _execute_code_construction_probe(
    request: dict[str, Any], workspace: Path
) -> dict[str, Any]:
    group_name = _require_python_identifier(
        request.get("group_name"), "group_name"
    )
    raw_construction_names = request.get("construction_names")
    if (
        not isinstance(raw_construction_names, list)
        or not 1 <= len(raw_construction_names) <= 8
    ):
        raise ProtocolError("construction_names must contain one to eight names")
    construction_names = [
        _require_python_identifier(value, f"construction_names[{index}]")
        for index, value in enumerate(raw_construction_names)
    ]
    if len(set(construction_names)) != len(construction_names):
        raise ProtocolError("construction_names must be unique within one group")
    output_dir, output_relative = _safe_feedback_output_dir(
        workspace, request.get("output_dir")
    )
    expected_source_sha256 = _require_sha256(
        request.get("formal_source_sha256"), "formal_source_sha256"
    )
    expected_step_sha256 = _require_sha256(
        request.get("formal_step_sha256"), "formal_step_sha256"
    )
    timeout_s = _require_number(
        request.get("timeout_s", DEFAULT_EXECUTION_TIMEOUT_S),
        "timeout_s",
        minimum=0.001,
    )
    formal_source = workspace / SOLUTION_FILE
    formal_step = workspace / "output.step"
    if (
        not formal_source.is_file()
        or formal_source.is_symlink()
        or not formal_step.is_file()
        or formal_step.is_symlink()
    ):
        raise ProtocolError("Formal solution.py and output.step must be regular files")
    source_contents = formal_source.read_bytes()
    step_contents = formal_step.read_bytes()
    if hashlib.sha256(source_contents).hexdigest() != expected_source_sha256:
        raise ProtocolError("Formal solution.py changed before construction inspection")
    if hashlib.sha256(step_contents).hexdigest() != expected_step_sha256:
        raise ProtocolError("Formal output.step changed before construction inspection")

    output_dir.mkdir(parents=True, exist_ok=True)
    executed_source = output_dir / "solution.py"
    shutil.copy2(formal_source, executed_source)
    probe_step = output_dir / "output.step"
    result_path = output_dir / "probe-result.json"
    for stale in (probe_step, result_path):
        if stale.exists():
            if stale.is_symlink() or not stale.is_file():
                raise ProtocolError("Construction probe outputs must be regular files")
            stale.unlink()
    runner = output_dir / "probe-runner.py"
    probe_arguments = ["--source", "solution.py"]
    for construction_name in construction_names:
        probe_arguments.extend(["--construction", construction_name])
    probe_arguments.extend(
        ["--output", "output.step", "--result", "probe-result.json"]
    )
    probe_argv = json.dumps(probe_arguments)
    runner.write_text(
        "from cadgenbench.integrations.code_construction_probe import main\n"
        f"raise SystemExit(main({probe_argv}))\n",
        encoding="utf-8",
    )
    execution = _run_python(runner, output_dir, timeout_s)
    (output_dir / "stdout.txt").write_text(execution["stdout"], encoding="utf-8")
    (output_dir / "stderr.txt").write_text(execution["stderr"], encoding="utf-8")

    formal_changed = (
        not formal_source.is_file()
        or formal_source.is_symlink()
        or not formal_step.is_file()
        or formal_step.is_symlink()
        or _sha256(formal_source) != expected_source_sha256
        or _sha256(formal_step) != expected_step_sha256
    )
    if formal_changed:
        _restore_regular_file(formal_source, source_contents)
        _restore_regular_file(formal_step, step_contents)
        execution["success"] = False
        violation = (
            "Construction probe attempted to modify formal solution.py or output.step; "
            "the controller restored both files."
        )
        execution["stderr"] = "\n".join(
            value for value in (execution["stderr"].strip(), violation) if value
        )
        (output_dir / "stderr.txt").write_text(
            execution["stderr"], encoding="utf-8"
        )

    probe_result: dict[str, Any]
    if result_path.is_file() and not result_path.is_symlink():
        try:
            loaded = json.loads(result_path.read_text(encoding="utf-8"))
            probe_result = _require_mapping(loaded, "probe-result.json")
        except (json.JSONDecodeError, UnicodeDecodeError, ProtocolError) as exc:
            probe_result = {
                "probe_status": "execution_error",
                "reason": f"construction probe returned an invalid result: {exc}",
                "available_names": [],
                "instance_count": 0,
            }
    else:
        probe_result = {
            "probe_status": "execution_error",
            "reason": "isolated source execution did not produce a construction result",
            "available_names": [],
            "instance_count": 0,
        }
    allowed_statuses = {
        "renderable",
        "not_found",
        "not_renderable",
        "execution_error",
    }
    probe_status = probe_result.get("probe_status")
    if probe_status not in allowed_statuses:
        probe_status = "execution_error"
    reason = str(probe_result.get("reason") or "construction probe failed")[:4000]
    if formal_changed:
        probe_status = "execution_error"
        reason = "the isolated construction probe attempted to modify formal files"
    raw_names = probe_result.get("available_names")
    available_names = (
        [
            value
            for value in raw_names
            if isinstance(value, str) and value.isidentifier() and len(value) <= 128
        ][:200]
        if isinstance(raw_names, list)
        else []
    )
    raw_instance_count = probe_result.get("instance_count", 0)
    instance_count = (
        raw_instance_count
        if isinstance(raw_instance_count, int)
        and not isinstance(raw_instance_count, bool)
        and raw_instance_count >= 0
        else 0
    )
    raw_instance_counts = probe_result.get("instance_counts")
    instance_counts = {
        name: (
            value
            if isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
            else 0
        )
        for name in construction_names
        for value in [
            raw_instance_counts.get(name, 0)
            if isinstance(raw_instance_counts, dict)
            else 0
        ]
    }
    artifact = _inspect_artifact(output_dir)
    if artifact["path"] is not None:
        artifact["path"] = f"{output_relative}/output.step"
    if probe_status == "renderable" and artifact["path"] is None:
        probe_status = "not_renderable"
        reason = "the captured construction did not produce a STEP projection artifact"
    return {
        "protocol_version": PROTOCOL_VERSION,
        "operation": "execute_code_construction_probe",
        "group_name": group_name,
        "construction_names": construction_names,
        "probe_status": probe_status,
        "reason": reason,
        "available_names": available_names,
        "instance_count": instance_count,
        "instance_counts": instance_counts,
        "source_path": f"{output_relative}/solution.py",
        "source_sha256": _sha256(executed_source),
        "formal_source_sha256": _sha256(formal_source),
        "formal_step_sha256": _sha256(formal_step),
        "formal_unchanged": True,
        "execution": execution,
        "artifact": artifact,
    }


def _execute_repair_candidate(
    request: dict[str, Any], workspace: Path
) -> dict[str, Any]:
    candidate_id = _require_identifier(request.get("candidate_id"), "candidate_id")
    source_path, source_relative = _safe_regular_file(
        workspace, request.get("source_path"), "source_path"
    )
    output_dir, output_relative = _safe_feedback_output_dir(
        workspace, request.get("output_dir")
    )
    expected_source_sha256 = _require_sha256(
        request.get("formal_source_sha256"), "formal_source_sha256"
    )
    expected_step_sha256 = _require_sha256(
        request.get("formal_step_sha256"), "formal_step_sha256"
    )
    timeout_s = _require_number(
        request.get("timeout_s", DEFAULT_EXECUTION_TIMEOUT_S),
        "timeout_s",
        minimum=0.001,
    )
    formal_source = workspace / SOLUTION_FILE
    formal_step = workspace / "output.step"
    if (
        not formal_source.is_file()
        or formal_source.is_symlink()
        or not formal_step.is_file()
        or formal_step.is_symlink()
    ):
        raise ProtocolError("Formal solution.py and output.step must be regular files")
    source_contents = formal_source.read_bytes()
    step_contents = formal_step.read_bytes()
    if hashlib.sha256(source_contents).hexdigest() != expected_source_sha256:
        raise ProtocolError("Formal solution.py changed before candidate execution")
    if hashlib.sha256(step_contents).hexdigest() != expected_step_sha256:
        raise ProtocolError("Formal output.step changed before candidate execution")

    output_dir.mkdir(parents=True, exist_ok=True)
    executed_source = output_dir / "solution.py"
    shutil.copy2(source_path, executed_source)
    candidate_step = output_dir / "output.step"
    if candidate_step.exists():
        if candidate_step.is_symlink() or not candidate_step.is_file():
            raise ProtocolError("Candidate output.step must be a regular file")
        candidate_step.unlink()
    runner = output_dir / "candidate-runner.py"
    runner.write_text(
        "import importlib\n"
        "import runpy\n"
        "from pathlib import Path\n"
        "import build123d\n"
        "_candidate_dir = Path(__file__).resolve().parent\n"
        "_candidate_output = _candidate_dir / 'output.step'\n"
        "_real_export_step = build123d.export_step\n"
        "def _redirect_export_step(*args, **kwargs):\n"
        "    positional = list(args)\n"
        "    if len(positional) >= 2:\n"
        "        positional[1] = str(_candidate_output)\n"
        "    else:\n"
        "        for key in ('file_name', 'file_path', 'filename'):\n"
        "            if key in kwargs:\n"
        "                kwargs[key] = str(_candidate_output)\n"
        "                break\n"
        "        else:\n"
        "            raise RuntimeError('Candidate export_step call has no output argument')\n"
        "    return _real_export_step(*positional, **kwargs)\n"
        "build123d.export_step = _redirect_export_step\n"
        "try:\n"
        "    importlib.import_module('build123d.exporters3d').export_step = _redirect_export_step\n"
        "except (ImportError, AttributeError):\n"
        "    pass\n"
        "runpy.run_path(str(_candidate_dir / 'solution.py'), run_name='__main__')\n",
        encoding="utf-8",
    )
    execution = _run_python(runner, output_dir, timeout_s)
    (output_dir / "stdout.txt").write_text(execution["stdout"], encoding="utf-8")
    (output_dir / "stderr.txt").write_text(execution["stderr"], encoding="utf-8")

    formal_changed = (
        not formal_source.is_file()
        or formal_source.is_symlink()
        or not formal_step.is_file()
        or formal_step.is_symlink()
        or _sha256(formal_source) != expected_source_sha256
        or _sha256(formal_step) != expected_step_sha256
    )
    if formal_changed:
        _restore_regular_file(formal_source, source_contents)
        _restore_regular_file(formal_step, step_contents)
        execution["success"] = False
        violation = (
            "Candidate attempted to modify formal solution.py or output.step; "
            "the controller restored both files."
        )
        execution["stderr"] = "\n".join(
            value for value in (execution["stderr"].strip(), violation) if value
        )
        (output_dir / "stderr.txt").write_text(
            execution["stderr"], encoding="utf-8"
        )

    artifact = _inspect_artifact(output_dir)
    if artifact["path"] is not None:
        artifact["path"] = f"{output_relative}/output.step"
    return {
        "protocol_version": PROTOCOL_VERSION,
        "operation": "execute_repair_candidate",
        "candidate_id": candidate_id,
        "source_path": source_relative,
        "source_sha256": _sha256(source_path),
        "formal_source_sha256": _sha256(formal_source),
        "formal_step_sha256": _sha256(formal_step),
        "formal_unchanged": True,
        "execution": execution,
        "artifact": artifact,
    }


def _execute(request: dict[str, Any], workspace: Path) -> dict[str, Any]:
    turn = _require_int(request.get("turn"), "turn", minimum=1)
    timeout_s = _require_number(
        request.get("timeout_s", DEFAULT_EXECUTION_TIMEOUT_S),
        "timeout_s",
        minimum=0.001,
    )
    turn_dir = workspace / "turns" / f"{turn:04d}"
    if turn_dir.exists():
        raise ProtocolError(f"Turn {turn} already exists")
    turn_dir.mkdir(parents=True)

    solution_path = workspace / SOLUTION_FILE
    if not solution_path.exists() or not solution_path.is_file() or solution_path.is_symlink():
        raise ProtocolError(f"{SOLUTION_FILE} must be a regular file in the workspace")
    source_sha256 = _sha256(solution_path)

    candidate = workspace / "output.step"
    if candidate.exists() and candidate.is_file() and not candidate.is_symlink():
        shutil.copy2(candidate, turn_dir / "output.before.step")
        candidate.unlink()
    elif candidate.exists():
        if candidate.is_dir() and not candidate.is_symlink():
            raise ProtocolError("output.step is a directory")
        candidate.unlink()

    script_path = turn_dir / "code.py"
    shutil.copy2(solution_path, script_path)
    # Execute the canonical file from the workspace root. Executing the archived
    # turns/<n>/code.py would put that directory first on sys.path, where
    # `code.py` can shadow Python's stdlib `code` module during build123d imports.
    execution = _run_python(solution_path, workspace, timeout_s)
    (turn_dir / "stdout.txt").write_text(execution["stdout"], encoding="utf-8")
    (turn_dir / "stderr.txt").write_text(execution["stderr"], encoding="utf-8")

    artifact = _inspect_artifact(
        workspace,
        render_destination=turn_dir / "iso.png",
    )
    if candidate.exists() and candidate.is_file() and not candidate.is_symlink():
        shutil.copy2(candidate, turn_dir / "output.step")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "operation": "execute",
        "turn": turn,
        "source_sha256": source_sha256,
        "execution": execution,
        "artifact": artifact,
    }


def _inspect(workspace: Path) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "operation": "inspect",
        "artifact": _inspect_artifact(workspace),
    }


def _finalize(workspace: Path) -> dict[str, Any]:
    artifact = _inspect_artifact(workspace)
    if artifact["status"] != "valid":
        return {
            "protocol_version": PROTOCOL_VERSION,
            "operation": "finalize",
            "artifact": artifact,
            "finalized": False,
        }
    from cadgenbench.eval.evaluate import evaluate_candidate_only

    evaluate_candidate_only(workspace / "output.step", workspace)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "operation": "finalize",
        "artifact": artifact,
        "finalized": True,
        "result_path": "result.json",
        "renders_path": "renders",
    }


def handle_request(request: dict[str, Any], workspace: Path) -> dict[str, Any]:
    version = _require_int(request.get("protocol_version"), "protocol_version", minimum=1)
    if version != PROTOCOL_VERSION:
        raise ProtocolError(
            f"Unsupported protocol_version {version}; expected {PROTOCOL_VERSION}"
        )
    operation = _require_string(request.get("operation"), "operation")
    if operation not in SUPPORTED_OPERATIONS:
        raise ProtocolError(f"Unsupported operation: {operation}")
    workspace.mkdir(parents=True, exist_ok=True)
    if operation == "doctor":
        return _doctor(workspace)
    if operation == "prepare_images":
        return _prepare_images(request, workspace)
    if operation == "execute":
        return _execute(request, workspace)
    if operation == "execute_repair_candidate":
        return _execute_repair_candidate(request, workspace)
    if operation == "execute_code_construction_probe":
        return _execute_code_construction_probe(request, workspace)
    if operation == "inspect":
        return _inspect(workspace)
    if operation in DIMENSION_OPERATIONS:
        from cadgenbench.integrations.dimension_annotations import handle_dimension_request

        return handle_dimension_request(request, workspace)
    if operation in DRAWING_OPERATIONS:
        return _run_drawing_operation(request, workspace)
    return _finalize(workspace)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(os.environ.get("CADGENBENCH_PI_WORKSPACE", DEFAULT_WORKSPACE)),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    signal.signal(signal.SIGINT, _terminate_active_children)
    signal.signal(signal.SIGTERM, _terminate_active_children)
    args = _parse_args(argv)
    diagnostics = ""
    try:
        request = _require_mapping(json.load(sys.stdin), "request")
        response, captured_error, diagnostics = _capture_native_output(
            lambda: handle_request(request, args.workspace)
        )
        if captured_error is not None:
            raise captured_error
    except (ProtocolError, json.JSONDecodeError) as exc:
        response = {
            "protocol_version": PROTOCOL_VERSION,
            "error": {"kind": "protocol", "message": str(exc)},
        }
        if diagnostics.strip():
            response["worker_diagnostics"] = diagnostics
        json.dump(response, sys.stdout, separators=(",", ":"))
        sys.stdout.write("\n")
        return 2
    except Exception as exc:  # noqa: BLE001 - one machine-readable failure path
        response = {
            "protocol_version": PROTOCOL_VERSION,
            "error": {
                "kind": "worker",
                "message": f"{type(exc).__name__}: {exc}",
            },
        }
        if diagnostics.strip():
            response["worker_diagnostics"] = diagnostics
        json.dump(response, sys.stdout, separators=(",", ":"))
        sys.stdout.write("\n")
        return 1
    if diagnostics.strip():
        response["worker_diagnostics"] = diagnostics
    json.dump(response, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
