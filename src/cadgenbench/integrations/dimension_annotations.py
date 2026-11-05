"""Version-bound dimension evidence for the Pi reflection subagent.

The caller supplies a fixed, manually aligned camera. Entity selection and
annotation placement are agent choices; every displayed value is calculated
from the STEP snapshot by CADInspector.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

DIMENSION_OPERATIONS = frozenset({"prepare_dimension_view", "measure_dimension_view"})
_QUANTITIES = {
    "diameter": 1,
    "radius": 1,
    "edge_length": 1,
    "center_distance": 2,
    "directional_distance": 2,
    "face_distance": 2,
    "angle": 2,
}
_TRANSFORM_FIELDS = {
    "scale_px_per_unit",
    "rotation_deg",
    "center_x_px",
    "center_y_px",
    "model_center_x",
    "model_center_y",
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _mapping(value: Any, label: str) -> dict:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    return value


def _number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _vector(
    value: Any, label: str, *, size: int = 3, nonzero: bool = False
) -> list[float]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{label} must contain {size} finite numbers")
    result = [_number(v, label) for v in value]
    if nonzero and sum(v * v for v in result) <= 1e-24:
        raise ValueError(f"{label} must be nonzero")
    return result


def _path(
    workspace: Path,
    value: Any,
    label: str,
    *,
    internal: bool = False,
    feedback: bool = False,
    exists: bool = True,
) -> Path:
    raw = Path(_string(value, label))
    if not internal and (raw.is_absolute() or ".." in raw.parts):
        raise ValueError(f"{label} must be a workspace-relative path without '..'")
    result = (workspace / raw).resolve()
    if not result.is_relative_to(workspace) or result == workspace:
        raise ValueError(f"{label} must stay inside the workspace")
    if feedback and (
        not result.is_relative_to(workspace / "drawing_feedback")
        or result == workspace / "drawing_feedback"
    ):
        raise ValueError(f"{label} must be inside drawing_feedback")
    if exists and not result.is_file():
        raise ValueError(f"{label} must be an existing regular file")
    return result


def _output(workspace: Path, value: Any) -> Path:
    result = _path(workspace, value, "output_dir", feedback=True, exists=False)
    if result.exists():
        raise ValueError(
            "output_dir already exists; use a fresh directory for each operation"
        )
    return result


def _relative(workspace: Path, path: Path | str) -> str:
    return str(Path(path).resolve().relative_to(workspace))


def _frame(request: dict) -> dict:
    spec = _mapping(request.get("view_spec"), "view_spec")
    if request.get("view_type") == "section" or any(
        spec.get(k) is not None or request.get(k) is not None
        for k in ("section_origin", "section_offset_fraction", "section_keep")
    ):
        raise ValueError("Section views are unsupported until cut topology is mapped")
    name = _string(spec.get("name"), "view_spec.name")
    direction = _vector(spec.get("direction"), "view_spec.direction", nonzero=True)
    up = _vector(spec.get("up"), "view_spec.up", nonzero=True)
    origin = _vector(request.get("projection_origin"), "projection_origin")
    transform = _mapping(request.get("transform"), "transform")
    if set(transform) - (
        _TRANSFORM_FIELDS | {"reflected"}
    ) or not _TRANSFORM_FIELDS <= set(transform):
        raise ValueError(
            "transform must contain the six similarity-transform fields and optional reflected"
        )
    transform = {**transform}
    for key in _TRANSFORM_FIELDS:
        transform[key] = _number(transform[key], f"transform.{key}")
    if transform["scale_px_per_unit"] <= 0:
        raise ValueError("transform.scale_px_per_unit must be positive")
    if "reflected" in transform and not isinstance(transform["reflected"], bool):
        raise ValueError("transform.reflected must be a boolean")
    size = request.get("image_size")
    if (
        not isinstance(size, list)
        or len(size) != 2
        or any(
            isinstance(v, bool) or not isinstance(v, int) or not 32 <= v <= 4096
            for v in size
        )
    ):
        raise ValueError("image_size must contain two integers in [32, 4096]")
    return {
        "view": name,
        "direction": direction,
        "up": up,
        "origin": origin,
        "transform": transform,
        "image_size": size,
    }


def _artifacts(workspace: Path, paths: list[Path]) -> dict[str, str]:
    return {_relative(workspace, path): _digest(path) for path in paths}


def _prepare(request: dict, workspace: Path) -> dict:
    from cadgenbench.tools.cad_inspector import CADInspector

    step = _path(workspace, request.get("step_path"), "step_path")
    expected = _sha(request.get("expected_step_sha256"), "expected_step_sha256")
    reference = _path(workspace, request.get("reference_path"), "reference_path")
    output = _output(workspace, request.get("output_dir"))
    region = _string(request.get("region_id"), "region_id")
    frame = _frame(request)
    kind = request.get("kind", "edge")
    if kind not in {"edge", "face", "all"}:
        raise ValueError("kind must be edge, face or all")
    geometry_type = request.get("geometry_type")
    if geometry_type is not None:
        geometry_type = _string(geometry_type, "geometry_type").upper()
        from build123d import GeomType

        if geometry_type not in GeomType.__members__:
            raise ValueError(f"Unknown geometry_type: {geometry_type}")
    step_bytes = step.read_bytes()
    if hashlib.sha256(step_bytes).hexdigest() != expected:
        raise ValueError(
            "STEP hash changed; prepare against the current expected_step_sha256"
        )
    # Freeze the checked bytes, so edits to the live CAD cannot change this session.
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "snapshot.step"
    snapshot.write_bytes(step_bytes)
    inspector = CADInspector(
        snapshot, reference_path=reference, expected_sha256=expected, **frame
    )
    prepared = inspector.prepare(output, kind=kind, geometry_type=geometry_type)
    session = output / "session.json"
    saved = json.loads(session.read_text())
    saved["region_id"] = region
    saved["source_step_path"] = _relative(workspace, step)
    session.write_text(json.dumps(saved, indent=2, allow_nan=False) + "\n")
    numbered = [output / page["image"] for page in prepared["pages"]]
    artifacts = [
        session,
        snapshot,
        output / "entities.json",
        output / "cad.png",
        output / "reference.png",
        *numbered,
    ]
    return {
        "region_id": region,
        "step_sha256": expected,
        "session_path": _relative(workspace, session),
        "entities_path": _relative(workspace, output / "entities.json"),
        "cad_path": _relative(workspace, output / "cad.png"),
        "reference_path": _relative(workspace, output / "reference.png"),
        "numbered_paths": [_relative(workspace, p) for p in numbered],
        "visible_selected_count": prepared["visible_selected_count"],
        "objects": inspector.objects,
        "artifact_sha256": _artifacts(workspace, artifacts),
    }


def _measurement_requests(value: Any) -> list[dict]:
    if not isinstance(value, list) or not 1 <= len(value) <= 64:
        raise ValueError(
            "requests must be a nonempty array with at most 64 measurements"
        )
    allowed = {
        "quantity",
        "entity_ids",
        "direction",
        "offset_px",
        "label_position_px",
        "annotation_style",
    }
    result = []
    for index, item in enumerate(value):
        item = dict(_mapping(item, f"requests[{index}]"))
        if set(item) - allowed:
            raise ValueError(
                "Measurement requests accept only geometry selection and label placement; arbitrary values or labels are forbidden"
            )
        quantity = item.get("quantity")
        if not isinstance(quantity, str) or quantity not in _QUANTITIES:
            raise ValueError(f"Unsupported quantity: {quantity}")
        ids = item.get("entity_ids")
        if (
            not isinstance(ids, list)
            or len(ids) != _QUANTITIES[quantity]
            or any(
                not isinstance(v, str) or not re.fullmatch(r"[EF][0-9]{3,}", v)
                for v in ids
            )
        ):
            raise ValueError(
                f"{quantity} requires {_QUANTITIES[quantity]} E/F entity IDs"
            )
        if "direction" in item:
            item["direction"] = _vector(item["direction"], "direction", nonzero=True)
        if "offset_px" in item:
            item["offset_px"] = _number(item["offset_px"], "offset_px")
        if "label_position_px" in item:
            item["label_position_px"] = _vector(
                item["label_position_px"], "label_position_px", size=2
            )
        if "annotation_style" in item and item["annotation_style"] != "leader":
            raise ValueError(
                "annotation_style supports only leader; omit it for automatic placement"
            )
        result.append(item)
    return result


def _measure(request: dict, workspace: Path) -> dict:
    from cadgenbench.tools.cad_inspector import CADInspector

    session = _path(
        workspace, request.get("session_path"), "session_path", feedback=True
    )
    expected = _sha(request.get("expected_step_sha256"), "expected_step_sha256")
    output = _output(workspace, request.get("output_dir"))
    requests = _measurement_requests(request.get("requests"))
    saved = _mapping(json.loads(session.read_text()), "session")
    if saved.get("version") != 1:
        raise ValueError("Unsupported inspector session version")
    if _sha(saved.get("model_sha256"), "session.model_sha256") != expected:
        raise ValueError(
            "Stale session: expected_step_sha256 differs; prepare a new session and reselect entity IDs"
        )
    config = _mapping(saved.get("configuration"), "session.configuration")
    allowed_config = {
        "step_path",
        "view",
        "direction",
        "up",
        "origin",
        "transform",
        "image_size",
        "reference_path",
    }
    if set(config) != allowed_config:
        raise ValueError("Invalid session configuration")
    step = _path(
        workspace,
        config.get("step_path"),
        "session.step_path",
        internal=True,
        feedback=True,
    )
    reference = _path(
        workspace,
        config.get("reference_path"),
        "session.reference_path",
        internal=True,
        feedback=True,
    )
    if step.parent != session.parent or reference.parent != session.parent:
        raise ValueError(
            "Session STEP and reference must belong to the same prepared directory"
        )
    # Validate stored camera and every filesystem input before loading the session.
    _frame(
        {
            "view_spec": {
                "name": config["view"],
                "direction": config["direction"],
                "up": config["up"],
            },
            "projection_origin": config["origin"],
            "transform": config["transform"],
            "image_size": config["image_size"],
        }
    )
    if _digest(step) != expected:
        raise ValueError("STEP hash changed; prepare a new session and reselect IDs")
    inspector = CADInspector.load(session)
    report = inspector.measure_and_render(requests, output)
    paths = {
        name: output / filename
        for name, filename in {
            "comparison_path": "comparison.png",
            "annotated_path": "annotated.png",
            "selection_path": "selection.png",
            "measurements_path": "measurements.json",
            "requests_path": "requests.json",
        }.items()
    }
    return {
        "region_id": saved.get("region_id"),
        "step_sha256": expected,
        "session_path": _relative(workspace, session),
        "entities_path": _relative(workspace, session.parent / "entities.json"),
        "cad_path": _relative(workspace, session.parent / "cad.png"),
        "reference_path": _relative(workspace, reference),
        **{key: _relative(workspace, path) for key, path in paths.items()},
        "results": report["results"],
        "artifact_sha256": _artifacts(workspace, list(paths.values())),
    }


def handle_dimension_request(request: dict, workspace: Path) -> dict:
    """Return the same success/error envelope as existing drawing operations."""
    operation = request.get("operation")
    base = {"protocol_version": 5, "operation": operation}
    try:
        workspace = workspace.resolve()
        if operation == "prepare_dimension_view":
            response = _prepare(request, workspace)
        elif operation == "measure_dimension_view":
            response = _measure(request, workspace)
        else:
            raise ValueError(f"Unsupported dimension operation: {operation}")
        return {**base, "status": "ok", **response}
    except Exception as exc:  # noqa: BLE001 - preserve recoverable drawing protocol
        return {
            **base,
            "status": "error",
            "error": {"kind": "drawing", "message": f"{type(exc).__name__}: {exc}"},
        }
