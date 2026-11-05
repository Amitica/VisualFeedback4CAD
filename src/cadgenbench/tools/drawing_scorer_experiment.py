# Copyright 2026 Hugging Face
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Offline, replayable baseline-versus-v2 drawing scorer experiment.

This module intentionally has no LLM or network dependency.  It validates a
pinned STEP/crop fixture, expands every reviewed camera over the configured
page rolls, and runs the legacy and v2 scorers against exactly the same
candidates.  Human-reviewed bounding boxes in the fixture are audit metadata;
neither scorer receives them.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import dataclasses
import hashlib
import importlib
import importlib.metadata
import json
import math
import multiprocessing
import os
import platform
import re
import shlex
import shutil
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

EXPERIMENT_SCHEMA_VERSION = 1
CANDIDATE_CACHE_SCHEMA_VERSION = 1
RESUME_IDENTITY_SCHEMA_VERSION = 1
MODULE_NAME = "cadgenbench.tools.drawing_scorer_experiment"
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")
DEFAULT_SCORER_VARIANT = "default"
NO_ENVELOPE_MATCH_VARIANT = "no-envelope-match"
CAPACITY_TRANSPORT_VARIANT = "capacity-transport"
SCORER_VARIANTS = (
    DEFAULT_SCORER_VARIANT,
    NO_ENVELOPE_MATCH_VARIANT,
    CAPACITY_TRANSPORT_VARIANT,
)
FULL_FIXTURE_REGION_IDS = (
    "p0-main-orthographic",
    "p0-isometric",
    "p0-section-a-a",
    "p1-bottom-view",
    "p1-isometric",
)
MAIN_ORTHOGRAPHIC_REGION_IDS = (
    "p0-main-orthographic",
    "p1-bottom-view",
)
ISOMETRIC_REGION_IDS = ("p0-isometric", "p1-isometric")
SECTION_REGION_IDS = ("p0-section-a-a",)
HISTORICAL_EDGE_ON_CANDIDATES = {
    "p0-main-orthographic": "axis-front-r000",
    "p1-bottom-view": "axis-right-r000",
}
EDGE_ON_CANDIDATE_BASES = {
    "p0-main-orthographic": (
        "axis-front",
        "axis-rear",
        "axis-left",
        "axis-right",
    ),
    "p1-bottom-view": ("axis-right",),
}
EDGE_ON_REJECTION_GATES = frozenset(
    {"aspect_ratio", "width_fill_low", "height_fill_low"}
)
REPEAT_EXACT_COMPONENT_FIELDS = (
    "eligible",
    "failed_gates",
    "warnings",
    "mode",
    "confidence",
    "fallback_used",
    "gross",
    "edge",
    "envelope_score",
    "edge_score",
    "spatial_edge_mass",
    "reference_envelope",
    "cad_envelope",
    "alignment",
)
REQUIRED_TOP_LEVEL_ARTIFACTS = (
    "baseline_rankings.json",
    "command.txt",
    "config.json",
    "environment.json",
    "overlay.png",
    "report.json",
    "report.md",
    "resume_identity.json",
    "results.json",
    "run.log",
    "scorer_v2_rankings.json",
    "top2.json",
)
CACHED_V2_REQUIRED_FIELDS = frozenset(
    {
        "candidate_id",
        "base_candidate_id",
        "page_roll_deg",
        "view_spec",
        "status",
        "score",
        "raw_score",
        "eligible",
        "failed_gates",
        "warnings",
        "mode",
        "confidence",
        "fallback_used",
        "transform",
        "alignment",
        "gross",
        "edge",
        "envelope_score",
        "edge_score",
        "spatial_edge_mass",
        "reference_envelope",
        "cad_envelope",
        "comparison_png",
        "cad_png",
        "aligned_overlay_png",
        "request_identity",
        "request_identity_sha256",
    }
)
WORKER_SINGLE_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)

_WORKER_THREADPOOL_LIMITER: Any | None = None


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


DEFAULT_FIXTURE_CONFIG = (
    _repository_root()
    / "tests"
    / "fixtures"
    / "drawing_scorer"
    / "fixture_145"
    / "config.json"
)
DEFAULT_OUTPUT_ROOT = (
    _repository_root() / "results" / "drawing-scorer-v2" / "fixture-145"
)


class DuplicateJsonKeyError(ValueError):
    """Raised before JSON can silently replace a repeated object key."""


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKeyError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def load_json_strict(path: str | Path) -> dict[str, Any]:
    """Load one JSON object and reject duplicate keys at every depth."""

    source = Path(path)
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {source}: {error}") from error
    if not isinstance(value, dict):
        raise TypeError(
            f"Expected a JSON object in {source}, got {type(value).__name__}"
        )
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fixture_path(fixture_root: Path, relative_path: Any) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise TypeError(
            f"Fixture path must be a non-empty string, got {relative_path!r}"
        )
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError(f"Fixture payload path must be relative: {relative_path!r}")
    candidate = (fixture_root / relative).resolve()
    try:
        candidate.relative_to(fixture_root.resolve())
    except ValueError as error:
        raise ValueError(f"Fixture path escapes its root: {relative_path!r}") from error
    return candidate


def _number_vector(value: Any, *, field: str) -> tuple[float, float, float]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in value
        )
    ):
        raise TypeError(f"{field} must be an array of three finite numbers")
    vector = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in vector):
        raise ValueError(f"{field} must contain only finite numbers")
    return vector


def _normalise(vector: Sequence[float], *, field: str) -> tuple[float, float, float]:
    norm = math.sqrt(sum(float(value) ** 2 for value in vector))
    if norm <= 1e-12:
        raise ValueError(f"{field} must not be the zero vector")
    return tuple(float(value) / norm for value in vector)  # type: ignore[return-value]


def _camera_basis(
    view_spec: Mapping[str, Any], *, field: str
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    direction = _normalise(
        _number_vector(view_spec.get("direction"), field=f"{field}.direction"),
        field=f"{field}.direction",
    )
    up = _normalise(
        _number_vector(view_spec.get("up"), field=f"{field}.up"),
        field=f"{field}.up",
    )
    cosine = sum(a * b for a, b in zip(direction, up, strict=True))
    if abs(cosine) > 1e-7:
        raise ValueError(
            f"{field}.up must be perpendicular to direction; dot={cosine:.9g}"
        )
    return direction, up


def _rotate_up(
    direction: Sequence[float],
    up: Sequence[float],
    roll_deg: int,
) -> tuple[float, float, float]:
    """Apply a page roll around the observer direction using Rodrigues' rule."""

    axis = _normalise(direction, field="direction")
    source = _normalise(up, field="up")
    radians = math.radians(float(roll_deg))
    cosine = math.cos(radians)
    sine = math.sin(radians)
    cross = (
        axis[1] * source[2] - axis[2] * source[1],
        axis[2] * source[0] - axis[0] * source[2],
        axis[0] * source[1] - axis[1] * source[0],
    )
    dot = sum(a * b for a, b in zip(axis, source, strict=True))
    rotated = tuple(
        source[index] * cosine
        + cross[index] * sine
        + axis[index] * dot * (1.0 - cosine)
        for index in range(3)
    )
    normalised = _normalise(rotated, field="rotated up")
    return tuple(0.0 if abs(value) < 1e-12 else value for value in normalised)


def _configured_rolls(config: Mapping[str, Any]) -> tuple[int, ...]:
    defaults = config.get("defaults")
    if not isinstance(defaults, dict):
        raise TypeError("Fixture defaults must be an object")
    raw_rolls = defaults.get("page_rolls_deg")
    if not isinstance(raw_rolls, list) or not raw_rolls:
        raise TypeError("defaults.page_rolls_deg must be a non-empty array")
    rolls: list[int] = []
    for value in raw_rolls:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("Page rolls must be integer degrees")
        if value % 90 != 0:
            raise ValueError(f"Page roll must be a multiple of 90 degrees, got {value}")
        normalised = value % 360
        if normalised in rolls:
            raise ValueError(f"Duplicate page roll after normalisation: {value}")
        rolls.append(normalised)
    return tuple(rolls)


def _regions(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = config.get("regions")
    if not isinstance(value, list) or not value:
        raise TypeError("Fixture regions must be a non-empty array")
    if not all(isinstance(item, dict) for item in value):
        raise TypeError("Every fixture region must be an object")
    return value


def expand_region_candidates(
    config: Mapping[str, Any],
    region: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Resolve a region's base cameras and every configured page roll."""

    candidate_sets = config.get("candidate_sets")
    if not isinstance(candidate_sets, dict):
        raise TypeError("Fixture candidate_sets must be an object")
    set_name = region.get("candidate_set")
    raw_candidates = candidate_sets.get(set_name)
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError(f"Unknown or empty candidate set {set_name!r}")

    expanded: list[dict[str, Any]] = []
    seen_base: set[str] = set()
    seen_expanded: set[str] = set()
    for index, candidate in enumerate(raw_candidates):
        if not isinstance(candidate, dict):
            raise TypeError(f"candidate_sets.{set_name}[{index}] must be an object")
        base_id = candidate.get("candidate_id")
        if not isinstance(base_id, str) or not base_id.strip():
            raise TypeError(
                f"candidate_sets.{set_name}[{index}].candidate_id is invalid"
            )
        if base_id in seen_base:
            raise ValueError(f"Duplicate base candidate ID in {set_name}: {base_id}")
        seen_base.add(base_id)
        view_spec = candidate.get("view_spec")
        if not isinstance(view_spec, dict):
            raise TypeError(f"Candidate {base_id!r} view_spec must be an object")
        direction, base_up = _camera_basis(view_spec, field=f"candidate {base_id!r}")

        section_offset = view_spec.get("section_offset_fraction")
        section_keep = view_spec.get("section_keep")
        if section_offset is not None:
            if isinstance(section_offset, bool) or not isinstance(
                section_offset, (int, float)
            ):
                raise TypeError(f"Candidate {base_id!r} section offset must be numeric")
            section_offset = float(section_offset)
            if not math.isfinite(section_offset) or not -1.0 < section_offset < 1.0:
                raise ValueError(f"Candidate {base_id!r} has invalid section offset")
            if section_keep not in {"front", "back"}:
                raise ValueError(f"Candidate {base_id!r} has invalid section_keep")
        elif section_keep is not None:
            raise ValueError(
                f"Candidate {base_id!r} has section_keep without an offset"
            )

        for roll in _configured_rolls(config):
            candidate_id = f"{base_id}-r{roll:03d}"
            if candidate_id in seen_expanded:
                raise ValueError(f"Duplicate expanded candidate ID: {candidate_id}")
            seen_expanded.add(candidate_id)
            resolved_spec: dict[str, Any] = {
                "direction": list(direction),
                "up": list(_rotate_up(direction, base_up, roll)),
            }
            if section_offset is not None:
                resolved_spec["section_offset_fraction"] = section_offset
                resolved_spec["section_keep"] = section_keep
            expanded.append(
                {
                    "candidate_id": candidate_id,
                    "base_candidate_id": base_id,
                    "page_roll_deg": roll,
                    "view_spec": resolved_spec,
                }
            )
    return expanded


def accepted_oracle_candidate_ids(
    config: Mapping[str, Any],
    region: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[str, ...]:
    """Expand the reviewed oracle equivalence families to concrete IDs."""

    resolved = list(candidates or expand_region_candidates(config, region))
    available = {str(candidate["candidate_id"]) for candidate in resolved}
    oracle = region.get("oracle")
    if not isinstance(oracle, dict):
        raise TypeError(f"Region {region.get('region_id')!r} oracle must be an object")
    families = oracle.get("accepted_families")
    if not isinstance(families, list) or not families:
        raise TypeError("oracle.accepted_families must be a non-empty array")
    accepted: list[str] = []
    for family_index, family in enumerate(families):
        if not isinstance(family, dict):
            raise TypeError(
                f"oracle.accepted_families[{family_index}] must be an object"
            )
        base_ids = family.get("base_candidate_ids")
        rolls = family.get("rolls_deg")
        if not isinstance(base_ids, list) or not base_ids:
            raise TypeError("Oracle base_candidate_ids must be a non-empty array")
        if not isinstance(rolls, list) or not rolls:
            raise TypeError("Oracle rolls_deg must be a non-empty array")
        for base_id in base_ids:
            if not isinstance(base_id, str):
                raise TypeError("Oracle base candidate IDs must be strings")
            for roll in rolls:
                if isinstance(roll, bool) or not isinstance(roll, int):
                    raise TypeError("Oracle rolls must be integer degrees")
                candidate_id = f"{base_id}-r{roll % 360:03d}"
                if candidate_id not in available:
                    raise ValueError(
                        f"Oracle candidate {candidate_id!r} is absent from region candidates"
                    )
                if candidate_id not in accepted:
                    accepted.append(candidate_id)
    canonical = oracle.get("canonical_candidate_id")
    if not isinstance(canonical, str) or canonical not in available:
        raise ValueError(f"Invalid oracle canonical_candidate_id: {canonical!r}")
    if canonical not in accepted:
        raise ValueError(f"Oracle canonical candidate {canonical!r} is not accepted")
    return tuple(accepted)


def load_and_validate_fixture(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    """Verify schema, hashes, dimensions, camera bases, and oracle families."""

    from PIL import Image

    path = Path(config_path).resolve()
    config = load_json_strict(path)
    fixture_root = path.parent
    if config.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported fixture schema: {config.get('schema_version')!r}"
        )
    if not isinstance(config.get("fixture_id"), str):
        raise TypeError("fixture_id must be a string")

    model = config.get("model")
    if not isinstance(model, dict):
        raise TypeError("Fixture model must be an object")
    model_path = _fixture_path(fixture_root, model.get("path"))
    if not model_path.is_file():
        raise FileNotFoundError(f"Pinned STEP is missing: {model_path}")
    if _sha256(model_path) != model.get("sha256"):
        raise ValueError(f"Pinned STEP SHA-256 mismatch: {model_path}")

    region_ids: set[str] = set()
    global_base_ids: set[str] = set()
    candidate_sets = config.get("candidate_sets")
    if not isinstance(candidate_sets, dict) or not candidate_sets:
        raise TypeError("candidate_sets must be a non-empty object")
    for set_name, entries in candidate_sets.items():
        if (
            not isinstance(set_name, str)
            or not isinstance(entries, list)
            or not entries
        ):
            raise TypeError("Each candidate set must be a named non-empty array")
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(
                entry.get("candidate_id"), str
            ):
                raise TypeError(f"Invalid candidate entry in set {set_name!r}")
            base_id = entry["candidate_id"]
            if base_id in global_base_ids:
                raise ValueError(f"Base candidate ID is not globally unique: {base_id}")
            global_base_ids.add(base_id)

    _configured_rolls(config)
    for region in _regions(config):
        region_id = region.get("region_id")
        if not isinstance(region_id, str) or not region_id:
            raise TypeError("Every region needs a non-empty region_id")
        if region_id in region_ids:
            raise ValueError(f"Duplicate region_id: {region_id}")
        region_ids.add(region_id)
        reference = _fixture_path(fixture_root, region.get("reference"))
        if not reference.is_file():
            raise FileNotFoundError(f"Pinned drawing crop is missing: {reference}")
        if _sha256(reference) != region.get("reference_sha256"):
            raise ValueError(f"Pinned crop SHA-256 mismatch: {reference}")
        with Image.open(reference) as image:
            if image.format != "PNG":
                raise ValueError(f"Pinned crop must be PNG: {reference}")
            if list(image.size) != region.get("image_size"):
                raise ValueError(
                    f"Pinned crop size mismatch for {region_id}: "
                    f"actual={list(image.size)}, configured={region.get('image_size')!r}"
                )
        candidates = expand_region_candidates(config, region)
        accepted_oracle_candidate_ids(config, region, candidates)
    return config, fixture_root


def _scorer_common_options(
    defaults: Mapping[str, Any],
    *,
    max_iterations: int,
) -> dict[str, Any]:
    """Return scorer inputs; deliberately and testably excludes the audit bbox."""

    return {
        "roi": None,
        "max_rotation_deg": float(defaults["max_rotation_deg"]),
        "seed": int(defaults["seed"]),
        "max_iterations": int(max_iterations),
    }


def _baseline_options(
    defaults: Mapping[str, Any],
    *,
    max_iterations: int,
) -> dict[str, Any]:
    return {
        **_scorer_common_options(defaults, max_iterations=max_iterations),
        "allow_reflection": bool(defaults["allow_reflection"]),
        "min_roi_fill": float(defaults["min_roi_fill"]),
        "seed": int(defaults["seed"]),
    }


def _validate_scorer_variant(value: str) -> str:
    if value not in SCORER_VARIANTS:
        choices = ", ".join(repr(item) for item in SCORER_VARIANTS)
        raise ValueError(f"Unknown scorer variant {value!r}; expected one of {choices}")
    return value


def _v2_config_for_variant(drawing_view: ModuleType, variant: str) -> Any:
    """Resolve one explicit V2 variant without silently approximating it.

    The no-envelope-match experiment is only valid when the scorer itself
    exposes the end-to-end switch.  Falling back to ``envelope_weight=0``
    would still let the envelope influence registration seeds, bounds, gates,
    and the spatial comparison box, so a missing API is a hard error.
    """

    selected = _validate_scorer_variant(variant)
    default_config = drawing_view.DEFAULT_SCORER_V2_CONFIG
    if selected == DEFAULT_SCORER_VARIANT:
        return default_config
    if not hasattr(default_config, "use_envelope_matching"):
        raise RuntimeError(
            "ScorerV2Config does not expose use_envelope_matching; refusing "
            "to run a non-strict no-envelope-match approximation"
        )
    replacement = {"use_envelope_matching": False}
    if selected == CAPACITY_TRANSPORT_VARIANT:
        if not hasattr(default_config, "use_capacity_transport"):
            raise RuntimeError("ScorerV2Config does not expose use_capacity_transport")
        replacement["use_capacity_transport"] = True
    resolved = dataclasses.replace(default_config, **replacement)
    if bool(getattr(resolved, "use_envelope_matching", True)):
        raise RuntimeError("Failed to disable envelope matching for the ablation")
    if selected == CAPACITY_TRANSPORT_VARIANT and not bool(
        getattr(resolved, "use_capacity_transport", False)
    ):
        raise RuntimeError("Failed to enable capacity transport")
    return resolved


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _jsonable(item_method())
        except (TypeError, ValueError):
            pass
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _verify_top_level_artifacts(run_dir: Path) -> tuple[str, ...]:
    missing = tuple(
        name for name in REQUIRED_TOP_LEVEL_ARTIFACTS if not (run_dir / name).is_file()
    )
    if missing:
        raise RuntimeError(f"Run is missing required artifact(s): {', '.join(missing)}")
    return REQUIRED_TOP_LEVEL_ARTIFACTS


def _artifact_path(path: str | Path, run_dir: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(run_dir.resolve()))
    except ValueError:
        return str(resolved)


def _candidate_metadata(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {str(candidate["candidate_id"]): dict(candidate) for candidate in candidates}


def _view_candidates(
    drawing_view: ModuleType, candidates: Sequence[Mapping[str, Any]]
) -> list[Any]:
    result = []
    for candidate in candidates:
        spec = candidate["view_spec"]
        result.append(
            drawing_view.ViewCandidate(
                name=candidate["candidate_id"],
                direction=tuple(spec["direction"]),
                up=tuple(spec["up"]),
                section_offset_fraction=spec.get("section_offset_fraction"),
                section_keep=spec.get("section_keep"),
            )
        )
    return result


def _oracle_summary(
    region: Mapping[str, Any],
    rankings: Sequence[Mapping[str, Any]],
    accepted_ids: Sequence[str],
) -> dict[str, Any]:
    accepted = set(accepted_ids)

    def is_eligible_accepted(record: Mapping[str, Any]) -> bool:
        # Baseline records and older unit fixtures predate explicit status and
        # eligibility fields; their documented default is a successful,
        # eligible ranking.  V2 always writes both fields.
        return (
            str(record["candidate_id"]) in accepted
            and record.get("status", "ok") == "ok"
            and bool(record.get("eligible", True))
        )

    rank_by_id = {
        str(record["candidate_id"]): int(record["rank"]) for record in rankings
    }
    canonical = str(region["oracle"]["canonical_candidate_id"])
    accepted_ranks = {
        candidate_id: rank_by_id[candidate_id]
        for candidate_id in accepted_ids
        if candidate_id in rank_by_id
    }
    eligible_accepted_ranks = {
        str(record["candidate_id"]): int(record["rank"])
        for record in rankings
        if is_eligible_accepted(record)
    }
    best_accepted_rank = min(accepted_ranks.values(), default=None)
    best_eligible_accepted_rank = min(
        eligible_accepted_ranks.values(),
        default=None,
    )
    top_records = list(rankings[:2])
    top_ids = [str(record["candidate_id"]) for record in top_records]
    top_eligible_accepted = [is_eligible_accepted(record) for record in top_records]
    return {
        "canonical_candidate_id": canonical,
        "canonical_rank": rank_by_id.get(canonical),
        "accepted_candidate_ids": list(accepted_ids),
        "accepted_ranks": accepted_ranks,
        "best_accepted_rank": best_accepted_rank,
        "eligible_accepted_ranks": eligible_accepted_ranks,
        "best_eligible_accepted_rank": best_eligible_accepted_rank,
        "top1_accepted": bool(top_eligible_accepted and top_eligible_accepted[0]),
        "top2_contains_accepted": any(top_eligible_accepted),
        "top2_exact_accepted_family": (
            len(top_ids) == len(accepted_ids) == 2
            and set(top_ids) == accepted
            and all(top_eligible_accepted)
        ),
        "oracle_confidence": region["oracle"].get("confidence"),
    }


def _run_baseline_region(
    drawing_view: ModuleType,
    *,
    model_path: Path,
    reference_path: Path,
    output_dir: Path,
    candidates: Sequence[Mapping[str, Any]],
    region: Mapping[str, Any],
    options: Mapping[str, Any],
    accepted_ids: Sequence[str],
    run_dir: Path,
) -> dict[str, Any]:
    result = drawing_view.rank_view_candidates_detailed(
        model_path,
        reference_path,
        output_dir,
        candidates=_view_candidates(drawing_view, candidates),
        roi=None,
        allow_reflection=options["allow_reflection"],
        max_rotation_deg=options["max_rotation_deg"],
        min_roi_fill=options["min_roi_fill"],
        seed=options["seed"],
        max_iterations=options["max_iterations"],
    )
    metadata = _candidate_metadata(candidates)
    rankings: list[dict[str, Any]] = []
    for item in result.rankings:
        candidate = metadata[item.name]
        record = {
            "rank": item.rank,
            "candidate_id": item.name,
            "base_candidate_id": candidate["base_candidate_id"],
            "page_roll_deg": candidate["page_roll_deg"],
            "view_spec": candidate["view_spec"],
            "status": "ok",
            "score": item.score,
            "raw_score": item.score,
            "eligible": True,
            "failed_gates": [],
            "transform": _jsonable(item.transform),
            "alignment": _jsonable(item.alignment_metrics),
            "geometry": _jsonable(item.geometry_metrics),
            "comparison_png": _artifact_path(item.comparison_png, run_dir),
        }
        if item.rank <= 2:
            spec = candidate["view_spec"]
            projection = drawing_view.project_step(
                model_path,
                view=item.name,
                direction=tuple(spec["direction"]),
                up=tuple(spec["up"]),
                section_offset_fraction=spec.get("section_offset_fraction"),
                section_keep=spec.get("section_keep"),
            )
            from PIL import Image

            with Image.open(reference_path) as source:
                image_size = source.size
            cad_view = drawing_view._render_cad_view(
                image_size,
                projection,
                item.transform,
                show_hidden_lines=False,
            )
            cad_path = output_dir / "top2_candidates" / f"{item.name}.png"
            cad_path.parent.mkdir(parents=True, exist_ok=True)
            cad_view.save(cad_path, format="PNG", optimize=True)
            record["cad_png"] = _artifact_path(cad_path, run_dir)
        rankings.append(record)
    overlay = output_dir / "overlay.png"
    shutil.copy2(result.comparison_png, overlay)
    top2_contact = output_dir / "top2.png"
    _write_top2_contact_sheet(
        reference_path,
        [
            (
                f"#{record['rank']} {record['candidate_id']} score={record['score']:.4f}",
                run_dir / record["cad_png"],
            )
            for record in rankings[:2]
        ],
        top2_contact,
    )
    payload = {
        "scorer": "baseline",
        "region_id": region["region_id"],
        "scorer_inputs": dict(options),
        "rankings": rankings,
        "top2": rankings[:2],
        "margin": result.margin,
        "oracle": _oracle_summary(region, rankings, accepted_ids),
        "overlay_png": _artifact_path(overlay, run_dir),
        "top2_contact_sheet_png": _artifact_path(top2_contact, run_dir),
    }
    _write_json(output_dir / "ranking.json", payload)
    _write_json(
        output_dir / "top2.json", {"top2": rankings[:2], "oracle": payload["oracle"]}
    )
    return payload


def _v2_record_sort_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    status_order = 0 if record.get("status") == "ok" else 1
    eligible_order = 0 if record.get("eligible") else 1
    selected_score = (
        record.get("score") if record.get("eligible") else record.get("raw_score")
    )
    numeric_score = (
        float(selected_score) if isinstance(selected_score, (int, float)) else -1.0
    )
    return (status_order, eligible_order, -numeric_score, str(record["candidate_id"]))


def _available_cpu_count() -> int:
    """Return the CPUs this process can actually schedule on."""

    affinity = getattr(os, "sched_getaffinity", None)
    if callable(affinity):
        try:
            count = len(affinity(0))
        except OSError:
            count = 0
        if count > 0:
            return count
    return max(1, int(os.cpu_count() or 1))


def _available_physical_cpu_count() -> int:
    """Count physical cores represented by the current Linux CPU affinity."""

    affinity = getattr(os, "sched_getaffinity", None)
    if not callable(affinity):
        return _available_cpu_count()
    try:
        cpus = affinity(0)
    except OSError:
        return _available_cpu_count()
    cores: set[tuple[str, str]] = set()
    for cpu in cpus:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            package = (
                (topology / "physical_package_id").read_text(encoding="utf-8").strip()
            )
            core = (topology / "core_id").read_text(encoding="utf-8").strip()
        except OSError:
            return _available_cpu_count()
        cores.add((package, core))
    return max(1, len(cores))


def _resolve_parallel_jobs(requested_jobs: int, *, task_count: int) -> int:
    """Resolve ``0=auto`` without creating workers that cannot receive work."""

    if isinstance(requested_jobs, bool) or not isinstance(requested_jobs, int):
        raise TypeError("jobs must be an integer")
    if requested_jobs < 0:
        raise ValueError("jobs must be zero (auto) or a positive integer")
    if task_count <= 0:
        return 0
    logical_available = _available_cpu_count()
    requested = logical_available if requested_jobs == 0 else requested_jobs
    return max(1, min(int(requested), int(task_count), logical_available))


def _projection_work_key(model_path: Path, view_spec: Mapping[str, Any]) -> str:
    """Canonical key for one STEP HLR projection shared across regions."""

    payload = {
        "model_path": str(model_path.resolve()),
        "direction": [float(value) for value in view_spec["direction"]],
        "up": [float(value) for value in view_spec["up"]],
        "section_offset_fraction": view_spec.get("section_offset_fraction"),
        "section_keep": view_spec.get("section_keep"),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


@contextlib.contextmanager
def _single_thread_worker_environment() -> Any:
    """Make spawned workers import every numeric backend in single-thread mode."""

    previous = {name: os.environ.get(name) for name in WORKER_SINGLE_THREAD_ENVIRONMENT}
    try:
        for name in WORKER_SINGLE_THREAD_ENVIRONMENT:
            os.environ[name] = "1"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _initialise_v2_worker() -> None:
    """Pin native numeric libraries to one thread inside every worker process."""

    global _WORKER_THREADPOOL_LIMITER
    for name in WORKER_SINGLE_THREAD_ENVIRONMENT:
        os.environ[name] = "1"
    try:
        from threadpoolctl import threadpool_limits

        # Retain the controller for the worker lifetime; otherwise its finalizer
        # may restore the inherited parent limits while candidate work is active.
        _WORKER_THREADPOOL_LIMITER = threadpool_limits(limits=1)
    except ImportError:
        _WORKER_THREADPOOL_LIMITER = None


def _parallel_process_context() -> multiprocessing.context.BaseContext:
    """Isolate OCCT state; forking an initialized native kernel is unsafe."""

    return multiprocessing.get_context("spawn")


@contextlib.contextmanager
def _managed_v2_process_pool(
    *,
    max_workers: int,
) -> Any:
    """Cancel and terminate isolated workers promptly when the run is interrupted."""

    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=_parallel_process_context(),
        initializer=_initialise_v2_worker,
    )
    try:
        yield executor
    except KeyboardInterrupt:
        process_map = getattr(executor, "_processes", None) or {}
        processes = list(process_map.values())
        executor.shutdown(wait=False, cancel_futures=True)
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=1.0)
        raise
    except BaseException:
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)


def _write_json_atomic(path: Path, value: Any) -> None:
    """Persist a completed candidate without exposing a truncated checkpoint."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_runtime_fingerprint() -> dict[str, Any]:
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "packages": {
            name: _package_version(name)
            for name in (
                "cadgenbench",
                "numpy",
                "scipy",
                "Pillow",
                "build123d",
                "cadquery-ocp",
            )
        },
    }


def _candidate_request_identity(
    *,
    model_sha256: str,
    reference_sha256: str,
    drawing_view_sha256: str,
    experiment_runner_sha256: str,
    runtime_fingerprint: Mapping[str, Any],
    candidate: Mapping[str, Any],
    options: Mapping[str, Any],
    selected_variant: str,
    scorer_config: Any,
) -> tuple[dict[str, Any], str]:
    identity = {
        "schema_version": CANDIDATE_CACHE_SCHEMA_VERSION,
        "model_sha256": model_sha256,
        "reference_sha256": reference_sha256,
        "drawing_view_sha256": drawing_view_sha256,
        "experiment_runner_sha256": experiment_runner_sha256,
        "runtime_fingerprint": _jsonable(runtime_fingerprint),
        "candidate_id": str(candidate["candidate_id"]),
        "view_spec": _jsonable(candidate["view_spec"]),
        "options": _jsonable(options),
        "scorer_variant": selected_variant,
        "scorer_config": _jsonable(scorer_config),
    }
    return identity, _canonical_json_sha256(identity)


def _safe_cached_artifact(path_value: Any, *, run_dir: Path) -> Path | None:
    if not isinstance(path_value, str) or not path_value:
        return None
    relative = Path(path_value)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    source = run_dir / relative
    try:
        resolved = source.resolve(strict=True)
        resolved.relative_to(run_dir.resolve())
    except (OSError, ValueError):
        return None
    if source.is_symlink() or not resolved.is_file():
        return None
    if resolved.suffix.lower() == ".png":
        try:
            from PIL import Image

            with Image.open(resolved) as image:
                image.verify()
        except (OSError, ValueError):
            return None
    return resolved


def _v2_render_artifacts_complete(
    record: Mapping[str, Any],
    *,
    run_dir: Path,
) -> bool:
    return all(
        _safe_cached_artifact(record.get(field), run_dir=run_dir) is not None
        for field in ("comparison_png", "cad_png", "aligned_overlay_png")
    )


def _load_cached_v2_record(
    path: Path,
    *,
    candidate: Mapping[str, Any],
    request_identity: Mapping[str, Any],
    request_identity_sha256: str,
    run_dir: Path,
) -> dict[str, Any] | None:
    """Return only a complete, successful record for the exact same request."""

    if not path.is_file():
        return None
    try:
        record = load_json_strict(path)
    except (OSError, TypeError, ValueError):
        return None
    if not CACHED_V2_REQUIRED_FIELDS.issubset(record):
        return None
    if record.get("status") != "ok":
        return None
    if record.get("candidate_id") != candidate["candidate_id"]:
        return None
    if record.get("base_candidate_id") != candidate["base_candidate_id"]:
        return None
    if record.get("page_roll_deg") != candidate["page_roll_deg"]:
        return None
    if record.get("view_spec") != _jsonable(candidate["view_spec"]):
        return None
    if record.get("request_identity") != _jsonable(request_identity):
        return None
    if record.get("request_identity_sha256") != request_identity_sha256:
        return None
    if _canonical_json_sha256(record["request_identity"]) != request_identity_sha256:
        return None
    for field in ("score", "raw_score", "envelope_score", "edge_score"):
        value = record.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            return None
    transform = record.get("transform")
    required_transform = {
        "scale_px_per_unit",
        "rotation_deg",
        "center_x_px",
        "center_y_px",
        "model_center_x",
        "model_center_y",
        "reflected",
    }
    if not isinstance(transform, Mapping) or not required_transform.issubset(transform):
        return None
    for field in required_transform - {"reflected"}:
        value = transform[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            return None
    for artifact_field in ("comparison_png", "cad_png", "aligned_overlay_png"):
        artifact = record.get(artifact_field)
        if (
            isinstance(artifact, str)
            and _safe_cached_artifact(artifact, run_dir=run_dir) is None
        ):
            record[artifact_field] = None
        elif artifact is not None and not isinstance(artifact, str):
            return None
    record.pop("rank", None)
    return record


def _expected_candidate_domain(
    expected_candidate_ids: Sequence[str],
    records: Sequence[Mapping[str, Any]],
) -> None:
    expected = [str(candidate_id) for candidate_id in expected_candidate_ids]
    actual = [str(record.get("candidate_id")) for record in records]
    duplicate_expected = sorted(
        candidate_id
        for candidate_id in set(expected)
        if expected.count(candidate_id) > 1
    )
    duplicate_actual = sorted(
        candidate_id for candidate_id in set(actual) if actual.count(candidate_id) > 1
    )
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if duplicate_expected or duplicate_actual or missing or unexpected:
        raise RuntimeError(
            "V2 candidate domain mismatch: "
            f"duplicate_expected={duplicate_expected} "
            f"duplicate_actual={duplicate_actual} missing={missing} "
            f"unexpected={unexpected}"
        )


def _project_v2_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    """Generate one unique HLR projection and strip non-pickleable OCCT shapes."""

    drawing_view = importlib.import_module("cadgenbench.tools.drawing_view")
    spec = task["view_spec"]
    projection = drawing_view.project_step(
        Path(task["model_path"]),
        view=str(task["view_name"]),
        direction=tuple(spec["direction"]),
        up=tuple(spec["up"]),
        section_offset_fraction=spec.get("section_offset_fraction"),
        section_keep=spec.get("section_keep"),
    )
    projection = dataclasses.replace(
        projection,
        visible_shape=None,
        hidden_shape=None,
    )
    return {
        "projection_key": str(task["projection_key"]),
        "projection": projection,
        "worker_pid": os.getpid(),
    }


def _v2_candidate_error_record(
    candidate: Mapping[str, Any],
    error: BaseException,
    *,
    request_identity: Mapping[str, Any] | None = None,
    request_identity_sha256: str | None = None,
) -> dict[str, Any]:
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "base_candidate_id": candidate["base_candidate_id"],
        "page_roll_deg": candidate["page_roll_deg"],
        "view_spec": candidate["view_spec"],
        "status": "error",
        "score": 0.0,
        "raw_score": 0.0,
        "eligible": False,
        "failed_gates": ["evaluation_error"],
        "envelope_score": None,
        "edge_score": None,
        "spatial_edge_mass": None,
        "capacity_transport": None,
        "error_type": type(error).__name__,
        "message": str(error),
        "request_identity": _jsonable(request_identity),
        "request_identity_sha256": request_identity_sha256,
    }


def _evaluate_v2_candidate(
    drawing_view: ModuleType,
    *,
    projection: Any,
    candidate: Mapping[str, Any],
    reference_path: Path,
    reference_geometry: Any,
    output_dir: Path,
    run_dir: Path,
    options: Mapping[str, Any],
    selected_variant: str,
    scorer_config: Any,
    render_artifacts: bool = True,
    request_identity: Mapping[str, Any] | None = None,
    request_identity_sha256: str | None = None,
) -> dict[str, Any]:
    """Evaluate and durably checkpoint one candidate in its unique directory."""

    candidate_id = str(candidate["candidate_id"])
    candidate_dir = output_dir / "candidates" / candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    evaluation = drawing_view.evaluate_projection_v2(
        projection,
        reference_path,
        roi=None,
        reference_geometry=reference_geometry,
        max_rotation_deg=options["max_rotation_deg"],
        seed=options["seed"],
        max_iterations=options["max_iterations"],
        config=scorer_config,
    )
    metrics = evaluation.metrics
    if (
        selected_variant == NO_ENVELOPE_MATCH_VARIANT
        and getattr(metrics, "mode", None) != "no_envelope_match"
    ):
        raise RuntimeError(
            "Strict no-envelope-match evaluation returned mode "
            f"{getattr(metrics, 'mode', None)!r}"
        )
    if (
        selected_variant == CAPACITY_TRANSPORT_VARIANT
        and getattr(metrics, "mode", None) != "capacity_transport"
    ):
        raise RuntimeError(
            "Capacity-transport evaluation returned mode "
            f"{getattr(metrics, 'mode', None)!r}"
        )
    record = {
        "candidate_id": candidate_id,
        "base_candidate_id": candidate["base_candidate_id"],
        "page_roll_deg": candidate["page_roll_deg"],
        "view_spec": candidate["view_spec"],
        "status": "ok",
        "score": float(metrics.score),
        "raw_score": float(metrics.raw_score),
        "eligible": bool(metrics.eligible),
        "failed_gates": list(metrics.failed_gates),
        "warnings": list(getattr(metrics, "warnings", ())),
        "mode": getattr(metrics, "mode", None),
        "confidence": getattr(metrics, "confidence", None),
        "fallback_used": getattr(metrics, "fallback_used", None),
        "transform": _jsonable(evaluation.transform),
        "alignment": _jsonable(evaluation.alignment),
        "gross": _jsonable(metrics.gross),
        "edge": _jsonable(metrics.edge),
        "envelope_score": float(metrics.envelope_score),
        "edge_score": float(metrics.edge_score),
        "spatial_edge_mass": _jsonable(metrics.spatial_edge_mass),
        "capacity_transport": _jsonable(getattr(metrics, "capacity_transport", None)),
        "reference_envelope": _jsonable(metrics.reference),
        "cad_envelope": _jsonable(metrics.cad),
        "comparison_png": None,
        "cad_png": None,
        "aligned_overlay_png": None,
        "request_identity": _jsonable(request_identity),
        "request_identity_sha256": request_identity_sha256,
        "cache_hit": False,
    }
    if render_artifacts:
        comparison = candidate_dir / "comparison.png"
        cad_path = candidate_dir / "cad.png"
        aligned_overlay = candidate_dir / "aligned_overlay.png"
        _write_v2_comparison(
            drawing_view,
            reference_path=reference_path,
            projection=projection,
            transform=evaluation.transform,
            score=float(metrics.score),
            candidate_id=candidate_id,
            output_path=comparison,
            cad_output_path=cad_path,
            aligned_overlay_path=aligned_overlay,
        )
        record.update(
            {
                "comparison_png": _artifact_path(comparison, run_dir),
                "cad_png": _artifact_path(cad_path, run_dir),
                "aligned_overlay_png": _artifact_path(aligned_overlay, run_dir),
            }
        )
    _write_json_atomic(candidate_dir / "result.json", record)
    return record


def _evaluate_v2_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    """Process-pool entry point for one reference/camera candidate pair."""

    drawing_view = importlib.import_module("cadgenbench.tools.drawing_view")
    candidate = task["candidate"]
    candidate_dir = (
        Path(task["output_dir"]) / "candidates" / str(candidate["candidate_id"])
    )
    try:
        record = _evaluate_v2_candidate(
            drawing_view,
            projection=task["projection"],
            candidate=candidate,
            reference_path=Path(task["reference_path"]),
            reference_geometry=task["reference_geometry"],
            output_dir=Path(task["output_dir"]),
            run_dir=Path(task["run_dir"]),
            options=task["options"],
            selected_variant=str(task["selected_variant"]),
            scorer_config=task["scorer_config"],
            render_artifacts=bool(task.get("render_artifacts", False)),
            request_identity=task.get("request_identity"),
            request_identity_sha256=task.get("request_identity_sha256"),
        )
    except Exception as error:  # noqa: BLE001 - preserve the rest of the batch
        record = _v2_candidate_error_record(
            candidate,
            error,
            request_identity=task.get("request_identity"),
            request_identity_sha256=task.get("request_identity_sha256"),
        )
        _write_json_atomic(candidate_dir / "result.json", record)
    return {"record": record, "worker_pid": os.getpid()}


def _extract_reference_v2_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    """Extract one crop's reference evidence concurrently with STEP HLR work."""

    drawing_view = importlib.import_module("cadgenbench.tools.drawing_view")
    reference_path = Path(task["reference_path"])
    geometry = drawing_view.extract_envelope_geometry_v2(
        reference_path,
        roi=None,
        config=task["scorer_config"],
    )
    _write_reference_envelope_overlay(
        reference_path,
        geometry,
        Path(task["reference_overlay"]),
    )
    return {
        "region_id": str(task["region_id"]),
        "reference_geometry": geometry,
        "worker_pid": os.getpid(),
    }


def _render_v2_record(
    drawing_view: ModuleType,
    *,
    record: Mapping[str, Any],
    projection: Any,
    reference_path: Path,
    output_dir: Path,
    run_dir: Path,
) -> dict[str, Any]:
    rendered = dict(record)
    candidate_id = str(rendered["candidate_id"])
    transform_payload = rendered.get("transform")
    if not isinstance(transform_payload, Mapping):
        raise TypeError(f"Missing saved transform for {candidate_id}")
    transform_type = getattr(drawing_view, "SimilarityTransform", SimpleNamespace)
    transform = transform_type(**dict(transform_payload))
    candidate_dir = output_dir / "candidates" / candidate_id
    comparison = candidate_dir / "comparison.png"
    cad_path = candidate_dir / "cad.png"
    aligned_overlay = candidate_dir / "aligned_overlay.png"
    _write_v2_comparison(
        drawing_view,
        reference_path=reference_path,
        projection=projection,
        transform=transform,
        score=float(rendered["score"]),
        candidate_id=candidate_id,
        output_path=comparison,
        cad_output_path=cad_path,
        aligned_overlay_path=aligned_overlay,
    )
    rendered.update(
        {
            "comparison_png": _artifact_path(comparison, run_dir),
            "cad_png": _artifact_path(cad_path, run_dir),
            "aligned_overlay_png": _artifact_path(aligned_overlay, run_dir),
        }
    )
    _write_json_atomic(candidate_dir / "result.json", rendered)
    return rendered


def _render_v2_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    drawing_view = importlib.import_module("cadgenbench.tools.drawing_view")
    rendered = _render_v2_record(
        drawing_view,
        record=task["record"],
        projection=task["projection"],
        reference_path=Path(task["reference_path"]),
        output_dir=Path(task["output_dir"]),
        run_dir=Path(task["run_dir"]),
    )
    return {"record": rendered, "worker_pid": os.getpid()}


def _write_v2_comparison(
    drawing_view: ModuleType,
    *,
    reference_path: Path,
    projection: Any,
    transform: Any,
    score: float,
    candidate_id: str,
    output_path: Path,
    cad_output_path: Path,
    aligned_overlay_path: Path,
) -> None:
    from PIL import Image, ImageOps

    with Image.open(reference_path) as source:
        reference = source.convert("RGB")
    # V2 currently scores visible HLR edges only.  Keep the audit overlay on
    # the exact same evidence instead of drawing gray hidden lines that did
    # not contribute to the numeric result.
    cad_view = drawing_view._render_cad_view(
        reference.size,
        projection,
        transform,
        show_hidden_lines=False,
    )
    cad_view.save(cad_output_path, format="PNG", optimize=True)
    reference_strength = ImageOps.invert(ImageOps.grayscale(reference))
    cad_strength = ImageOps.invert(ImageOps.grayscale(cad_view))
    reference_alpha = reference_strength.point(lambda value: round(value * 0.72))
    cad_alpha = cad_strength.point(lambda value: round(value * 0.78))
    overlay = Image.new("RGBA", reference.size, (255, 255, 255, 255))
    reference_layer = Image.new("RGBA", reference.size, (220, 45, 45, 0))
    reference_layer.putalpha(reference_alpha)
    cad_layer = Image.new("RGBA", reference.size, (20, 105, 230, 0))
    cad_layer.putalpha(cad_alpha)
    overlay.alpha_composite(reference_layer)
    overlay.alpha_composite(cad_layer)
    aligned_overlay_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.convert("RGB").save(
        aligned_overlay_path,
        format="PNG",
        optimize=True,
    )
    drawing_view._write_comparison(
        reference,
        cad_view,
        output_path,
        view_name=candidate_id,
        score=score,
    )


def _write_reference_envelope_overlay(
    reference_path: Path,
    geometry: Any,
    output_path: Path,
) -> None:
    """Show exactly which pixels v2 treated as the coarse part envelope."""

    from PIL import Image, ImageDraw

    with Image.open(reference_path) as source:
        canvas = source.convert("RGBA")
    envelope_alpha = Image.fromarray(
        geometry.envelope_mask.astype("uint8") * 92,
        mode="L",
    )
    boundary_alpha = Image.fromarray(
        geometry.boundary_mask.astype("uint8") * 230,
        mode="L",
    )
    envelope = Image.new("RGBA", canvas.size, (0, 155, 230, 0))
    envelope.putalpha(envelope_alpha)
    boundary = Image.new("RGBA", canvas.size, (220, 35, 35, 0))
    boundary.putalpha(boundary_alpha)
    canvas.alpha_composite(envelope)
    canvas.alpha_composite(boundary)
    diagnostics = geometry.diagnostics
    draw = ImageDraw.Draw(canvas)
    if diagnostics.bbox_px is not None:
        draw.rectangle(diagnostics.bbox_px, outline=(255, 125, 0, 255), width=3)
    caption = (
        f"envelope confidence={diagnostics.confidence} "
        f"stable={diagnostics.stable} threshold={diagnostics.selected_threshold} "
        f"closing={diagnostics.closing_radius_px}"
    )
    draw.rectangle((0, 0, min(canvas.width, 900), 34), fill=(255, 255, 255, 225))
    draw.text((10, 9), caption, fill=(20, 20, 20, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output_path, format="PNG", optimize=True)


def _write_top2_contact_sheet(
    reference_path: Path,
    candidates: Sequence[tuple[str, Path]],
    output_path: Path,
) -> None:
    """Persist original crop plus the two clean, aligned top-ranked CAD views."""

    from PIL import Image, ImageDraw

    with Image.open(reference_path) as source:
        reference = source.convert("RGB")
    panels: list[tuple[str, Any]] = [("Original crop", reference)]
    for label, path in candidates[:2]:
        with Image.open(path) as source:
            panels.append((label, source.convert("RGB")))
    panel_width = max(panel.width for _, panel in panels)
    panel_height = max(panel.height for _, panel in panels)
    gap = 16
    header = 48
    sheet = Image.new(
        "RGB",
        (
            len(panels) * panel_width + (len(panels) + 1) * gap,
            panel_height + header + 2 * gap,
        ),
        (240, 243, 247),
    )
    draw = ImageDraw.Draw(sheet)
    for index, (label, panel) in enumerate(panels):
        x = gap + index * (panel_width + gap)
        draw.text((x, gap + 10), label, fill=(20, 25, 35))
        sheet.paste(panel, (x, gap + header))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG", optimize=True)


def _transform_delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    fields = (
        "scale_px_per_unit",
        "rotation_deg",
        "center_x_px",
        "center_y_px",
        "model_center_x",
        "model_center_y",
    )
    return max(abs(float(left[field]) - float(right[field])) for field in fields)


def _structured_differences(
    nominal: Any,
    repeated: Any,
    *,
    path: str,
    absolute_tolerance: float = 1e-12,
) -> list[dict[str, Any]]:
    """Return exact structural/non-numeric and tolerant numeric differences."""

    if isinstance(nominal, Mapping) and isinstance(repeated, Mapping):
        differences: list[dict[str, Any]] = []
        for key in sorted(set(nominal) | set(repeated), key=str):
            child_path = f"{path}.{key}" if path else str(key)
            if key not in nominal or key not in repeated:
                differences.append(
                    {
                        "path": child_path,
                        "nominal": nominal.get(key, "<missing>"),
                        "repeated": repeated.get(key, "<missing>"),
                        "reason": "missing_key",
                    }
                )
                continue
            differences.extend(
                _structured_differences(
                    nominal[key],
                    repeated[key],
                    path=child_path,
                    absolute_tolerance=absolute_tolerance,
                )
            )
        return differences
    if isinstance(nominal, (list, tuple)) and isinstance(repeated, (list, tuple)):
        differences = []
        if len(nominal) != len(repeated):
            differences.append(
                {
                    "path": path,
                    "nominal": len(nominal),
                    "repeated": len(repeated),
                    "reason": "sequence_length",
                }
            )
        for index, (left, right) in enumerate(zip(nominal, repeated, strict=False)):
            differences.extend(
                _structured_differences(
                    left,
                    right,
                    path=f"{path}[{index}]",
                    absolute_tolerance=absolute_tolerance,
                )
            )
        return differences
    numeric = (
        isinstance(nominal, (int, float))
        and not isinstance(nominal, bool)
        and isinstance(repeated, (int, float))
        and not isinstance(repeated, bool)
    )
    if numeric:
        difference = abs(float(nominal) - float(repeated))
        if difference <= absolute_tolerance:
            return []
        return [
            {
                "path": path,
                "nominal": nominal,
                "repeated": repeated,
                "absolute_difference": difference,
                "absolute_tolerance": absolute_tolerance,
                "reason": "numeric_difference",
            }
        ]
    if nominal == repeated:
        return []
    return [
        {
            "path": path,
            "nominal": nominal,
            "repeated": repeated,
            "reason": "value_difference",
        }
    ]


def _repeat_component_differences(
    nominal: Mapping[str, Any],
    repeated: Mapping[str, Any],
) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []
    for field in REPEAT_EXACT_COMPONENT_FIELDS:
        differences.extend(
            _structured_differences(
                nominal.get(field, "<missing>"),
                repeated.get(field, "<missing>"),
                path=field,
            )
        )
    return differences


def _multistart_coverage(record: Mapping[str, Any]) -> dict[str, Any]:
    """Prove that one nominal result evaluated the complete configured grid."""

    if record.get("status") != "ok":
        return {
            "candidate_id": record["candidate_id"],
            "complete": False,
            "reason": "nominal_candidate_error",
        }
    diagnostics = record["alignment"]["diagnostics"]
    expected = {
        (round(float(scale), 8), round(float(rotation), 8))
        for scale in diagnostics["scale_seeds_px_per_unit"]
        for rotation in diagnostics["rotation_seeds_deg"]
    }
    actual = {
        (
            round(float(trial["start_scale_px_per_unit"]), 8),
            round(float(trial["start_rotation_deg"]), 8),
        )
        for trial in diagnostics["trials"]
    }
    return {
        "candidate_id": record["candidate_id"],
        "seed": diagnostics["seed"],
        "expected_start_count": len(expected),
        "actual_trial_count": len(diagnostics["trials"]),
        "unique_actual_start_count": len(actual),
        "missing_starts": [list(pair) for pair in sorted(expected - actual)],
        "unexpected_starts": [list(pair) for pair in sorted(actual - expected)],
        "complete": (
            len(diagnostics["trials"]) == len(expected) and actual == expected
        ),
    }


def _run_v2_perturbations(
    drawing_view: ModuleType,
    *,
    reference_path: Path,
    reference_geometry: Any,
    projections: Mapping[str, Any],
    nominal_records: Sequence[Mapping[str, Any]],
    accepted_ids: Sequence[str],
    options: Mapping[str, Any],
    log: RunLog,
    scorer_config: Any | None = None,
) -> dict[str, Any]:
    """Fresh-align and re-rank the complete domain for every perturbation.

    Reusing nominal transforms does not test registration, and rerunning only
    nominal top-two candidates cannot detect a lower-ranked candidate moving
    ahead.  This audit deliberately pays the full deterministic compute cost.
    """

    audit_started_at = datetime.now(UTC)
    audit_started = time.perf_counter()
    default_config = (
        drawing_view.DEFAULT_SCORER_V2_CONFIG
        if scorer_config is None
        else scorer_config
    )
    nominal_by_id = {str(record["candidate_id"]): record for record in nominal_records}
    domain_ids = sorted(nominal_by_id)
    nominal_order = [str(record["candidate_id"]) for record in nominal_records]
    accepted = set(accepted_ids)

    def run_variant(
        variant_id: str,
        *,
        seed: int,
        variant_config: Any,
        variant_geometry: Any,
        threshold: int | None,
        change: Mapping[str, Any],
    ) -> dict[str, Any]:
        started_at = datetime.now(UTC)
        started = time.perf_counter()
        log.write(
            f"v2 perturbation {variant_id} fresh-align-and-rerank "
            f"all_candidates={len(domain_ids)} seed={seed}"
        )
        records: list[dict[str, Any]] = []
        for candidate_id in domain_ids:
            nominal = nominal_by_id[candidate_id]
            try:
                evaluation = drawing_view.evaluate_projection_v2(
                    projections[candidate_id],
                    reference_path,
                    roi=None,
                    threshold=threshold,
                    reference_geometry=variant_geometry,
                    max_rotation_deg=options["max_rotation_deg"],
                    seed=seed,
                    max_iterations=options["max_iterations"],
                    config=variant_config,
                )
                metrics = evaluation.metrics
                transform = _jsonable(evaluation.transform)
                record = {
                    "candidate_id": candidate_id,
                    "status": "ok",
                    "score": float(metrics.score),
                    "raw_score": float(metrics.raw_score),
                    "eligible": bool(metrics.eligible),
                    "failed_gates": list(metrics.failed_gates),
                    "warnings": list(getattr(metrics, "warnings", ())),
                    "mode": getattr(metrics, "mode", None),
                    "confidence": getattr(metrics, "confidence", None),
                    "fallback_used": getattr(metrics, "fallback_used", None),
                    "score_delta_from_nominal": float(
                        metrics.score - float(nominal["score"])
                    ),
                    "raw_score_delta_from_nominal": float(
                        metrics.raw_score - float(nominal["raw_score"])
                    ),
                    "transform_delta_from_nominal": _transform_delta(
                        transform, nominal["transform"]
                    ),
                    "transform": transform,
                    # Persist every start/final transform so the fresh search
                    # can be audited without trusting this runner.
                    "alignment": _jsonable(evaluation.alignment),
                    "gross": _jsonable(metrics.gross),
                    "edge": _jsonable(metrics.edge),
                    "envelope_score": float(metrics.envelope_score),
                    "edge_score": float(metrics.edge_score),
                    "spatial_edge_mass": _jsonable(metrics.spatial_edge_mass),
                    "capacity_transport": _jsonable(
                        getattr(metrics, "capacity_transport", None)
                    ),
                    "reference_envelope": _jsonable(metrics.reference),
                    "cad_envelope": _jsonable(metrics.cad),
                }
            except Exception as error:  # noqa: BLE001 - preserve audit evidence
                record = {
                    "candidate_id": candidate_id,
                    "status": "error",
                    "score": 0.0,
                    "raw_score": 0.0,
                    "eligible": False,
                    "failed_gates": ["evaluation_error"],
                    "envelope_score": None,
                    "edge_score": None,
                    "spatial_edge_mass": None,
                    "capacity_transport": None,
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            if record["status"] == "ok" and nominal.get("status") == "ok":
                component_differences = _repeat_component_differences(
                    nominal,
                    record,
                )
            else:
                component_differences = [
                    {
                        "path": "status",
                        "nominal": nominal.get("status"),
                        "repeated": record.get("status"),
                        "reason": "candidate_not_comparable",
                    }
                ]
            record["component_metrics_match_nominal"] = not component_differences
            record["component_metric_differences"] = component_differences
            records.append(record)

        records.sort(key=_v2_record_sort_key)
        for rank, record in enumerate(records, start=1):
            record["rank"] = rank
        ranked_ids = [str(record["candidate_id"]) for record in records]
        top2_ids = ranked_ids[:2]
        complete = (
            len(records) == len(domain_ids)
            and set(ranked_ids) == set(domain_ids)
            and all(record["status"] == "ok" for record in records)
        )
        successful = [record for record in records if record["status"] == "ok"]
        eligible_accepted = [
            record
            for record in successful
            if bool(record.get("eligible")) and str(record["candidate_id"]) in accepted
        ]
        coverage = [_multistart_coverage(record) for record in records]
        coverage_complete = bool(coverage) and all(
            item["complete"] for item in coverage
        )
        elapsed_seconds = time.perf_counter() - started
        log.write(
            f"v2 perturbation {variant_id} complete={complete} "
            f"top2={','.join(top2_ids)} elapsed={elapsed_seconds:.3f}s"
        )
        return {
            "variant_id": variant_id,
            "scope": {
                "kind": "complete_candidate_domain",
                "candidate_count": len(domain_ids),
                "candidate_ids": domain_ids,
                "fresh_alignment_per_candidate": True,
                "full_candidate_rerank": True,
                "fixed_nominal_transform_reused": False,
            },
            "change": dict(change),
            "seed": seed,
            "scorer_config": _jsonable(variant_config),
            "reference_envelope": _jsonable(variant_geometry.diagnostics),
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": elapsed_seconds,
            "candidate_results": records,
            "ranked_candidate_ids": ranked_ids,
            "top2_candidate_ids": top2_ids,
            "top2_contains_accepted": any(
                record["status"] == "ok"
                and bool(record.get("eligible"))
                and str(record["candidate_id"]) in accepted
                for record in records[:2]
            ),
            "oracle_best_rank": min(
                (int(record["rank"]) for record in eligible_accepted),
                default=None,
            ),
            "oracle_raw_best_rank": min(
                (
                    int(record["rank"])
                    for record in successful
                    if record["candidate_id"] in accepted
                ),
                default=None,
            ),
            "complete": complete,
            "multistart_coverage": coverage,
            "multistart_coverage_complete": coverage_complete,
            "same_full_ranking_as_nominal": ranked_ids == nominal_order,
            "same_eligibility_as_nominal": complete
            and all(
                bool(record["eligible"])
                == bool(nominal_by_id[str(record["candidate_id"])]["eligible"])
                for record in records
            ),
            "same_component_metrics_as_nominal": complete
            and all(
                bool(record["component_metrics_match_nominal"]) for record in records
            ),
            "component_metric_mismatch_candidate_ids": [
                str(record["candidate_id"])
                for record in records
                if not record["component_metrics_match_nominal"]
            ],
            "max_abs_score_delta": (
                max(
                    abs(float(record["score_delta_from_nominal"]))
                    for record in successful
                )
                if complete
                else None
            ),
            "max_abs_raw_score_delta": (
                max(
                    abs(float(record["raw_score_delta_from_nominal"]))
                    for record in successful
                )
                if complete
                else None
            ),
            "max_abs_transform_delta": (
                max(
                    abs(float(record["transform_delta_from_nominal"]))
                    for record in successful
                )
                if complete
                else None
            ),
        }

    nominal_seed = int(options["seed"])
    specs: list[dict[str, Any]] = [
        {
            "variant_id": "nominal-repeat",
            "seed": nominal_seed,
            "config": default_config,
            "geometry": reference_geometry,
            "threshold": None,
            "change": {"kind": "exact_repeat"},
        },
        {
            "variant_id": "seed-plus-one",
            "seed": nominal_seed + 1,
            "config": default_config,
            "geometry": reference_geometry,
            "threshold": None,
            "change": {
                "kind": "optimizer_seed",
                "nominal": nominal_seed,
                "value": nominal_seed + 1,
            },
        },
    ]
    selected_threshold = int(reference_geometry.diagnostics.selected_threshold)
    threshold_delta = int(default_config.threshold_perturbation)
    closing_radius = int(reference_geometry.diagnostics.closing_radius_px)
    for variant_id, threshold_value in (
        ("threshold-minus", max(0, selected_threshold - threshold_delta)),
        ("threshold-plus", min(255, selected_threshold + threshold_delta)),
    ):
        specs.append(
            {
                "variant_id": variant_id,
                "seed": nominal_seed,
                "config": default_config,
                "geometry": drawing_view.extract_envelope_geometry_v2(
                    reference_path,
                    roi=None,
                    threshold=threshold_value,
                    config=default_config,
                ),
                "threshold": threshold_value,
                "change": {
                    "kind": "threshold_override",
                    "nominal_selected_threshold": selected_threshold,
                    "value": threshold_value,
                },
            }
        )
    for radius in (1, 2, 3):
        variant_id = f"closing-{radius}px"
        scorer_config = dataclasses.replace(
            default_config,
            closing_radius_px=radius,
        )
        specs.append(
            {
                "variant_id": variant_id,
                "seed": nominal_seed,
                "config": scorer_config,
                "geometry": drawing_view.extract_envelope_geometry_v2(
                    reference_path,
                    roi=None,
                    config=scorer_config,
                ),
                "threshold": None,
                "change": {
                    "kind": "closing_radius_px",
                    "nominal": closing_radius,
                    "value": radius,
                },
            }
        )

    variants = [
        run_variant(
            str(spec["variant_id"]),
            seed=int(spec["seed"]),
            variant_config=spec["config"],
            variant_geometry=spec["geometry"],
            threshold=spec["threshold"],
            change=spec["change"],
        )
        for spec in specs
    ]
    by_id = {str(variant["variant_id"]): variant for variant in variants}
    repeat = by_id["nominal-repeat"]
    seed_variant = by_id["seed-plus-one"]
    morphology_variants = [
        by_id[variant_id]
        for variant_id in (
            "threshold-minus",
            "threshold-plus",
            "closing-1px",
            "closing-2px",
            "closing-3px",
        )
    ]
    repeat_exact = bool(
        repeat["complete"]
        and repeat["multistart_coverage_complete"]
        and repeat["same_full_ranking_as_nominal"]
        and repeat["same_eligibility_as_nominal"]
        and repeat["same_component_metrics_as_nominal"]
        and float(repeat["max_abs_score_delta"] or 0.0) <= 1e-12
        and float(repeat["max_abs_raw_score_delta"] or 0.0) <= 1e-12
        and float(repeat["max_abs_transform_delta"] or 0.0) <= 1e-9
    )
    seed_stable = bool(
        seed_variant["complete"]
        and seed_variant["multistart_coverage_complete"]
        and seed_variant["top2_contains_accepted"]
    )
    morphology_stable = bool(morphology_variants) and all(
        variant["complete"]
        and variant["multistart_coverage_complete"]
        and variant["top2_contains_accepted"]
        for variant in morphology_variants
    )
    all_variants_full_domain = all(
        variant["complete"]
        and variant["scope"]["candidate_count"] == len(domain_ids)
        and variant["scope"]["fresh_alignment_per_candidate"]
        and variant["scope"]["full_candidate_rerank"]
        and not variant["scope"]["fixed_nominal_transform_reused"]
        for variant in variants
    )
    return {
        "scope": {
            "kind": "complete_candidate_domain",
            "candidate_count": len(domain_ids),
            "candidate_ids": domain_ids,
            "variant_count": len(variants),
            "fresh_alignment_per_candidate_per_variant": True,
            "full_candidate_rerank_per_variant": True,
            "total_fresh_candidate_alignments": len(domain_ids) * len(variants),
        },
        "started_at": audit_started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.perf_counter() - audit_started,
        "nominal_top2_candidate_ids": nominal_order[:2],
        "accepted_candidate_ids": list(accepted_ids),
        "variants": variants,
        # Retain the early artifact grouping keys while fixing their semantics.
        "full_alignment_variants": [repeat, seed_variant],
        "morphology_variants": morphology_variants,
        "summary": {
            "all_variants_full_candidate_domain": all_variants_full_domain,
            "repeat_exact": repeat_exact,
            "seed_stable": seed_stable,
            "threshold_and_closing_stable": morphology_stable,
            "all_passed": (
                all_variants_full_domain
                and repeat_exact
                and seed_stable
                and morphology_stable
            ),
        },
    }


def _nominal_only_perturbation_record(
    *,
    nominal_records: Sequence[Mapping[str, Any]],
    accepted_ids: Sequence[str],
) -> dict[str, Any]:
    """Persist an explicit skip instead of pretending robustness was tested."""

    candidate_ids = [str(record["candidate_id"]) for record in nominal_records]
    return {
        "status": "skipped",
        "reason": "nominal_only",
        "scope": {
            "kind": "complete_candidate_domain",
            "candidate_count": len(candidate_ids),
            "candidate_ids": candidate_ids,
            "variant_count": 0,
            "fresh_alignment_per_candidate_per_variant": False,
            "full_candidate_rerank_per_variant": False,
            "total_fresh_candidate_alignments": 0,
        },
        "nominal_top2_candidate_ids": candidate_ids[:2],
        "accepted_candidate_ids": list(accepted_ids),
        "variants": [],
        "full_alignment_variants": [],
        "morphology_variants": [],
        "summary": {
            "nominal_only": True,
            "all_variants_full_candidate_domain": False,
            "repeat_exact": False,
            "seed_stable": False,
            "threshold_and_closing_stable": False,
            "all_passed": False,
        },
    }


def _finalize_v2_region(
    drawing_view: ModuleType,
    *,
    reference_path: Path,
    reference_geometry: Any,
    reference_overlay: Path,
    output_dir: Path,
    region: Mapping[str, Any],
    options: Mapping[str, Any],
    accepted_ids: Sequence[str],
    run_dir: Path,
    log: RunLog,
    selected_variant: str,
    scorer_config: Any,
    run_perturbations: bool,
    records: list[dict[str, Any]],
    projections: Mapping[str, Any],
    expected_candidate_ids: Sequence[str],
    render_all_candidates: bool = True,
    parallelism: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Sort deterministic records and emit one region's aggregate artifacts."""

    _expected_candidate_domain(expected_candidate_ids, records)
    records.sort(key=_v2_record_sort_key)
    for rank, record in enumerate(records, start=1):
        record["rank"] = rank
        candidate_dir = output_dir / "candidates" / str(record["candidate_id"])
        _write_json_atomic(candidate_dir / "result.json", record)
    successful = [record for record in records if record["status"] == "ok"]
    if not successful:
        raise RuntimeError(
            f"Every v2 candidate failed for region {region['region_id']}"
        )
    render_records = successful if render_all_candidates else successful[:2]
    for record in render_records:
        if _v2_render_artifacts_complete(record, run_dir=run_dir):
            continue
        candidate_id = str(record["candidate_id"])
        projection = projections.get(candidate_id)
        if projection is None:
            raise RuntimeError(f"Missing cached projection for {candidate_id}")
        rendered = _render_v2_record(
            drawing_view,
            record=record,
            projection=projection,
            reference_path=reference_path,
            output_dir=output_dir,
            run_dir=run_dir,
        )
        record.update(rendered)
    best_comparison = run_dir / successful[0]["aligned_overlay_png"]
    overlay = output_dir / "overlay.png"
    shutil.copy2(best_comparison, overlay)
    top2_contact = output_dir / "top2.png"
    _write_top2_contact_sheet(
        reference_path,
        [
            (
                (
                    f"#{record['rank']} {record['candidate_id']} "
                    f"score={record['score']:.4f} "
                    "(red=reference, blue=CAD)"
                ),
                run_dir / record["aligned_overlay_png"],
            )
            for record in successful[:2]
        ],
        top2_contact,
    )
    perturbations = (
        _run_v2_perturbations(
            drawing_view,
            reference_path=reference_path,
            reference_geometry=reference_geometry,
            projections=projections,
            nominal_records=records,
            accepted_ids=accepted_ids,
            options=options,
            log=log,
            scorer_config=scorer_config,
        )
        if run_perturbations
        else _nominal_only_perturbation_record(
            nominal_records=records,
            accepted_ids=accepted_ids,
        )
    )
    _write_json(output_dir / "perturbations.json", perturbations)
    payload = {
        "scorer": "v2",
        "scorer_variant": selected_variant,
        "region_id": region["region_id"],
        "scorer_inputs": {
            **dict(options),
            "scorer_variant": selected_variant,
            "scorer_config": _jsonable(scorer_config),
            "run_perturbations": bool(run_perturbations),
            "render_all_candidates": bool(render_all_candidates),
        },
        "parallelism": dict(parallelism or {}),
        "reference_envelope": _jsonable(reference_geometry.diagnostics),
        "reference_envelope_overlay_png": _artifact_path(reference_overlay, run_dir),
        "rankings": records,
        "top2": records[:2],
        "margin": float(records[0]["score"] - records[1]["score"]),
        "oracle": _oracle_summary(region, records, accepted_ids),
        "overlay_png": _artifact_path(overlay, run_dir),
        "top2_contact_sheet_png": _artifact_path(top2_contact, run_dir),
        "perturbations": perturbations,
    }
    _write_json(output_dir / "ranking.json", payload)
    _write_json(
        output_dir / "top2.json", {"top2": records[:2], "oracle": payload["oracle"]}
    )
    return payload


def _run_v2_region(
    drawing_view: ModuleType,
    *,
    model_path: Path,
    reference_path: Path,
    output_dir: Path,
    candidates: Sequence[Mapping[str, Any]],
    region: Mapping[str, Any],
    options: Mapping[str, Any],
    accepted_ids: Sequence[str],
    run_dir: Path,
    log: RunLog,
    scorer_variant: str = DEFAULT_SCORER_VARIANT,
    scorer_config: Any | None = None,
    run_perturbations: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_variant = _validate_scorer_variant(scorer_variant)
    resolved_scorer_config = (
        _v2_config_for_variant(drawing_view, selected_variant)
        if scorer_config is None
        else scorer_config
    )
    if selected_variant in {
        NO_ENVELOPE_MATCH_VARIANT,
        CAPACITY_TRANSPORT_VARIANT,
    } and bool(getattr(resolved_scorer_config, "use_envelope_matching", True)):
        raise ValueError(
            f"{selected_variant} requires ScorerV2Config.use_envelope_matching=False"
        )
    if selected_variant == CAPACITY_TRANSPORT_VARIANT and not bool(
        getattr(resolved_scorer_config, "use_capacity_transport", False)
    ):
        raise ValueError(
            "capacity-transport requires ScorerV2Config.use_capacity_transport=True"
        )
    # This call is intentionally against the complete saved crop.  The
    # audit_only_expected_part_bbox_xywh metadata never crosses this boundary.
    reference_geometry = drawing_view.extract_envelope_geometry_v2(
        reference_path,
        roi=None,
        config=resolved_scorer_config,
    )
    reference_overlay = output_dir / "reference_envelope_overlay.png"
    _write_reference_envelope_overlay(
        reference_path,
        reference_geometry,
        reference_overlay,
    )
    records: list[dict[str, Any]] = []
    projections: dict[str, Any] = {}
    model_sha256 = _sha256(model_path) if model_path.is_file() else "unavailable"
    reference_sha256 = _sha256(reference_path)
    module_path_value = getattr(drawing_view, "__file__", None)
    drawing_view_sha256 = (
        _sha256(Path(module_path_value))
        if module_path_value and Path(module_path_value).is_file()
        else "unavailable"
    )
    experiment_runner_sha256 = _sha256(Path(__file__).resolve())
    runtime_fingerprint = _cache_runtime_fingerprint()
    for index, candidate in enumerate(candidates, start=1):
        candidate_id = str(candidate["candidate_id"])
        spec = candidate["view_spec"]
        candidate_dir = output_dir / "candidates" / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        log.write(
            f"v2 {region['region_id']} candidate {index}/{len(candidates)}: {candidate_id}"
        )
        request_identity, request_identity_sha256 = _candidate_request_identity(
            model_sha256=model_sha256,
            reference_sha256=reference_sha256,
            drawing_view_sha256=drawing_view_sha256,
            experiment_runner_sha256=experiment_runner_sha256,
            runtime_fingerprint=runtime_fingerprint,
            candidate=candidate,
            options=options,
            selected_variant=selected_variant,
            scorer_config=resolved_scorer_config,
        )
        try:
            projection = drawing_view.project_step(
                model_path,
                view=candidate_id,
                direction=tuple(spec["direction"]),
                up=tuple(spec["up"]),
                section_offset_fraction=spec.get("section_offset_fraction"),
                section_keep=spec.get("section_keep"),
            )
            projections[candidate_id] = projection
            record = _evaluate_v2_candidate(
                drawing_view,
                projection=projection,
                candidate=candidate,
                reference_path=reference_path,
                reference_geometry=reference_geometry,
                output_dir=output_dir,
                run_dir=run_dir,
                options=options,
                selected_variant=selected_variant,
                scorer_config=resolved_scorer_config,
                request_identity=request_identity,
                request_identity_sha256=request_identity_sha256,
            )
        except Exception as error:  # noqa: BLE001 - keep other candidates diagnosable
            record = _v2_candidate_error_record(
                candidate,
                error,
                request_identity=request_identity,
                request_identity_sha256=request_identity_sha256,
            )
            _write_json_atomic(candidate_dir / "result.json", record)
            log.write(
                f"v2 candidate error {candidate_id}: {type(error).__name__}: {error}"
            )
        records.append(record)
    return _finalize_v2_region(
        drawing_view,
        reference_path=reference_path,
        reference_geometry=reference_geometry,
        reference_overlay=reference_overlay,
        output_dir=output_dir,
        region=region,
        options=options,
        accepted_ids=accepted_ids,
        run_dir=run_dir,
        log=log,
        selected_variant=selected_variant,
        scorer_config=resolved_scorer_config,
        run_perturbations=run_perturbations,
        records=records,
        projections=projections,
        expected_candidate_ids=[
            str(candidate["candidate_id"]) for candidate in candidates
        ],
        render_all_candidates=True,
    )


def _run_v2_regions_parallel(
    drawing_view: ModuleType,
    *,
    model_path: Path,
    fixture_root: Path,
    config: Mapping[str, Any],
    regions: Sequence[Mapping[str, Any]],
    defaults: Mapping[str, Any],
    max_iterations: int,
    run_dir: Path,
    log: RunLog,
    selected_variant: str,
    scorer_config: Any,
    run_perturbations: bool,
    requested_jobs: int,
    render_all_candidates: bool,
    resume: bool,
) -> tuple[list[dict[str, Any]], list[tuple[str, Path]], dict[str, Any]]:
    """Project and score every selected V2 candidate through one global pool."""

    options = _scorer_common_options(defaults, max_iterations=max_iterations)
    contexts: dict[str, dict[str, Any]] = {}
    entries_by_region: dict[str, dict[str, Any]] = {}
    overlays_by_region: dict[str, tuple[str, Path]] = {}
    for region in regions:
        region_id = str(region["region_id"])
        candidates = expand_region_candidates(config, region)
        accepted_ids = accepted_oracle_candidate_ids(config, region, candidates)
        reference_path = _fixture_path(fixture_root, region["reference"])
        output_dir = run_dir / "v2" / region_id
        output_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        log.write(f"begin scorer=v2 region={region_id} candidates={len(candidates)}")
        contexts[region_id] = {
            "region": region,
            "candidates": candidates,
            "accepted_ids": accepted_ids,
            "reference_path": reference_path,
            "reference_geometry": None,
            "reference_overlay": output_dir / "reference_envelope_overlay.png",
            "reference_error": None,
            "output_dir": output_dir,
            "started": started,
            "records": [],
            "projections": {},
        }

    model_sha256 = _sha256(model_path)
    drawing_view_sha256 = _sha256(Path(drawing_view.__file__).resolve())
    experiment_runner_sha256 = _sha256(Path(__file__).resolve())
    runtime_fingerprint = _cache_runtime_fingerprint()
    cache_hits = 0
    for context in contexts.values():
        reference_sha256 = _sha256(context["reference_path"])
        request_identities: dict[str, tuple[dict[str, Any], str]] = {}
        cached_ids: set[str] = set()
        for candidate in context["candidates"]:
            candidate_id = str(candidate["candidate_id"])
            identity = _candidate_request_identity(
                model_sha256=model_sha256,
                reference_sha256=reference_sha256,
                drawing_view_sha256=drawing_view_sha256,
                experiment_runner_sha256=experiment_runner_sha256,
                runtime_fingerprint=runtime_fingerprint,
                candidate=candidate,
                options=options,
                selected_variant=selected_variant,
                scorer_config=scorer_config,
            )
            request_identities[candidate_id] = identity
            if not resume:
                continue
            cached = _load_cached_v2_record(
                context["output_dir"] / "candidates" / candidate_id / "result.json",
                candidate=candidate,
                request_identity=identity[0],
                request_identity_sha256=identity[1],
                run_dir=run_dir,
            )
            if cached is not None:
                context["records"].append(cached)
                cached_ids.add(candidate_id)
                cache_hits += 1
        context["request_identities"] = request_identities
        context["cached_ids"] = cached_ids
        context["cached_payload"] = None
        if (
            resume
            and not run_perturbations
            and len(cached_ids) == len(context["candidates"])
        ):
            ranking_path = context["output_dir"] / "ranking.json"
            try:
                cached_payload = load_json_strict(ranking_path)
            except (OSError, TypeError, ValueError):
                cached_payload = None
            if (
                cached_payload is not None
                and cached_payload.get("scorer") == "v2"
                and cached_payload.get("scorer_variant") == selected_variant
                and cached_payload.get("region_id") == context["region"]["region_id"]
                and cached_payload.get("perturbations", {}).get("status") == "skipped"
            ):
                expected_ids = [
                    str(candidate["candidate_id"])
                    for candidate in context["candidates"]
                ]
                _expected_candidate_domain(expected_ids, context["records"])
                context["records"].sort(key=_v2_record_sort_key)
                for rank, record in enumerate(context["records"], start=1):
                    record["rank"] = rank
                top2 = context["records"][:2]
                aggregate_artifacts = (
                    cached_payload.get("overlay_png"),
                    cached_payload.get("top2_contact_sheet_png"),
                    cached_payload.get("reference_envelope_overlay_png"),
                )
                top2_artifacts_present = all(
                    _v2_render_artifacts_complete(record, run_dir=run_dir)
                    for record in top2
                )
                if top2_artifacts_present and all(
                    _safe_cached_artifact(path, run_dir=run_dir) is not None
                    for path in aggregate_artifacts
                ):
                    cached_payload["rankings"] = context["records"]
                    cached_payload["top2"] = top2
                    cached_payload["margin"] = float(
                        top2[0]["score"] - top2[1]["score"]
                    )
                    cached_payload["oracle"] = _oracle_summary(
                        context["region"],
                        context["records"],
                        context["accepted_ids"],
                    )
                    context["cached_payload"] = cached_payload

    projection_groups: dict[str, dict[str, Any]] = {}
    for region_id, context in contexts.items():
        if context["cached_payload"] is not None:
            continue
        for candidate in context["candidates"]:
            candidate_id = str(candidate["candidate_id"])
            if candidate_id in context["cached_ids"]:
                continue
            key = _projection_work_key(model_path, candidate["view_spec"])
            group = projection_groups.setdefault(
                key,
                {
                    "projection_key": key,
                    "view_name": candidate["candidate_id"],
                    "view_spec": candidate["view_spec"],
                    "members": [],
                },
            )
            group["members"].append((region_id, candidate))

    evaluation_count = sum(len(context["candidates"]) for context in contexts.values())
    planned_projection_keys = {
        _projection_work_key(model_path, candidate["view_spec"])
        for context in contexts.values()
        for candidate in context["candidates"]
    }
    active_contexts = {
        region_id: context
        for region_id, context in contexts.items()
        if context["cached_payload"] is None
    }
    scheduled_evaluations = evaluation_count - cache_hits
    scheduled_work_items = max(
        scheduled_evaluations,
        len(projection_groups) + len(active_contexts),
    )
    effective_jobs = _resolve_parallel_jobs(
        requested_jobs,
        task_count=scheduled_work_items,
    )
    parallelism = {
        "requested_jobs": requested_jobs,
        "resolved_jobs": effective_jobs,
        "logical_cpus_in_affinity": _available_cpu_count(),
        "physical_cpus_in_affinity": _available_physical_cpu_count(),
        "process_start_method": "spawn",
        "worker_native_threads": 1,
        "candidate_evaluations": evaluation_count,
        "scheduled_candidate_evaluations": scheduled_evaluations,
        "candidate_cache_hits": cache_hits,
        "candidate_cache_misses": evaluation_count - cache_hits,
        "unique_hlr_projections": len(planned_projection_keys),
        "planned_unique_hlr_projections": len(planned_projection_keys),
        "scheduled_unique_hlr_projections": len(projection_groups),
        "deduplicated_hlr_projections": evaluation_count - len(planned_projection_keys),
        "render_mode": "all" if render_all_candidates else "top2_only",
        "reference_preprocessing_tasks": sum(
            context["cached_payload"] is None for context in contexts.values()
        ),
        "nominal_parallelized": True,
        "perturbations_parallelized": False,
        "observed_worker_processes": 0,
        "on_demand_render_hlr_projections": 0,
    }
    log.write(
        "v2 global scheduler "
        f"jobs={effective_jobs} requested={requested_jobs} "
        f"unique_hlr={len(planned_projection_keys)} "
        f"scheduled_hlr={len(projection_groups)} evaluations={evaluation_count} "
        f"cache_hits={cache_hits} render={parallelism['render_mode']}"
    )

    def append_candidate_error(
        region_id: str,
        candidate: Mapping[str, Any],
        error: BaseException,
    ) -> None:
        context = contexts[region_id]
        candidate_id = str(candidate["candidate_id"])
        if any(
            str(record.get("candidate_id")) == candidate_id
            for record in context["records"]
        ):
            return
        identity, identity_sha256 = context["request_identities"][candidate_id]
        record = _v2_candidate_error_record(
            candidate,
            error,
            request_identity=identity,
            request_identity_sha256=identity_sha256,
        )
        context["records"].append(record)
        candidate_dir = context["output_dir"] / "candidates" / candidate_id
        _write_json_atomic(candidate_dir / "result.json", record)

    if projection_groups or active_contexts:
        worker_pids: set[int] = set()
        with (
            _single_thread_worker_environment(),
            _managed_v2_process_pool(max_workers=effective_jobs) as executor,
        ):
            reference_futures: dict[concurrent.futures.Future[dict[str, Any]], str] = {}
            for region_id, context in active_contexts.items():
                try:
                    future = executor.submit(
                        _extract_reference_v2_worker,
                        {
                            "region_id": region_id,
                            "reference_path": str(context["reference_path"]),
                            "reference_overlay": str(context["reference_overlay"]),
                            "scorer_config": scorer_config,
                        },
                    )
                    reference_futures[future] = region_id
                except Exception as error:  # noqa: BLE001 - record broken pool
                    context["reference_error"] = error

            projection_futures: dict[
                concurrent.futures.Future[dict[str, Any]], str
            ] = {}
            for key, group in projection_groups.items():
                try:
                    future = executor.submit(
                        _project_v2_worker,
                        {
                            "projection_key": key,
                            "model_path": str(model_path),
                            "view_name": group["view_name"],
                            "view_spec": group["view_spec"],
                        },
                    )
                    projection_futures[future] = key
                except Exception as error:  # noqa: BLE001 - fan out submit error
                    log.write(
                        "v2 projection submit error "
                        f"members={len(group['members'])} "
                        f"{type(error).__name__}: {error}"
                    )
                    for region_id, candidate in group["members"]:
                        append_candidate_error(region_id, candidate, error)

            for future in concurrent.futures.as_completed(reference_futures):
                region_id = reference_futures[future]
                context = contexts[region_id]
                try:
                    result = future.result()
                    context["reference_geometry"] = result["reference_geometry"]
                    worker_pids.add(int(result["worker_pid"]))
                except Exception as error:  # noqa: BLE001 - isolate one crop
                    context["reference_error"] = error
                    log.write(
                        f"v2 reference error region={region_id} "
                        f"{type(error).__name__}: {error}"
                    )

            evaluation_futures: dict[
                concurrent.futures.Future[dict[str, Any]],
                tuple[str, Mapping[str, Any]],
            ] = {}
            for future in concurrent.futures.as_completed(projection_futures):
                key = projection_futures[future]
                group = projection_groups[key]
                try:
                    result = future.result()
                    projection = result["projection"]
                    worker_pids.add(int(result["worker_pid"]))
                    for region_id, candidate in group["members"]:
                        context = contexts[region_id]
                        candidate_id = str(candidate["candidate_id"])
                        context["projections"][candidate_id] = projection
                        if (
                            context["reference_geometry"] is None
                            or candidate_id in context["cached_ids"]
                            or any(
                                str(record.get("candidate_id")) == candidate_id
                                for record in context["records"]
                            )
                        ):
                            continue
                        identity, identity_sha256 = context["request_identities"][
                            candidate_id
                        ]
                        try:
                            evaluation_future = executor.submit(
                                _evaluate_v2_worker,
                                {
                                    "projection": projection,
                                    "candidate": candidate,
                                    "reference_path": str(context["reference_path"]),
                                    "reference_geometry": context["reference_geometry"],
                                    "output_dir": str(context["output_dir"]),
                                    "run_dir": str(run_dir),
                                    "options": options,
                                    "selected_variant": selected_variant,
                                    "scorer_config": scorer_config,
                                    "render_artifacts": False,
                                    "request_identity": identity,
                                    "request_identity_sha256": identity_sha256,
                                },
                            )
                            evaluation_futures[evaluation_future] = (
                                region_id,
                                candidate,
                            )
                        except Exception as error:  # noqa: BLE001 - broken pool
                            append_candidate_error(region_id, candidate, error)
                except Exception as error:  # noqa: BLE001 - fan out one HLR error
                    log.write(
                        "v2 projection error "
                        f"members={len(group['members'])} "
                        f"{type(error).__name__}: {error}"
                    )
                    for region_id, candidate in group["members"]:
                        append_candidate_error(region_id, candidate, error)

            for region_id, context in contexts.items():
                reference_error = context["reference_error"]
                if reference_error is None:
                    continue
                for candidate in context["candidates"]:
                    append_candidate_error(region_id, candidate, reference_error)

            completed = sum(len(context["records"]) for context in contexts.values())
            for future in concurrent.futures.as_completed(evaluation_futures):
                region_id, candidate = evaluation_futures[future]
                try:
                    worker_result = future.result()
                    record = worker_result["record"]
                    worker_pids.add(int(worker_result["worker_pid"]))
                except Exception as error:  # noqa: BLE001 - survive worker failure
                    append_candidate_error(region_id, candidate, error)
                    record = next(
                        item
                        for item in contexts[region_id]["records"]
                        if item["candidate_id"] == candidate["candidate_id"]
                    )
                else:
                    contexts[region_id]["records"].append(record)
                completed += 1
                log.write(
                    f"v2 candidate complete {completed}/{evaluation_count} "
                    f"region={region_id} candidate={record['candidate_id']} "
                    f"status={record['status']}"
                )

            render_records_by_region: dict[str, list[dict[str, Any]]] = {}
            on_demand_projection_groups: dict[str, dict[str, Any]] = {}
            for region_id, context in active_contexts.items():
                if context["reference_geometry"] is None:
                    continue
                expected_ids = [
                    str(candidate["candidate_id"])
                    for candidate in context["candidates"]
                ]
                _expected_candidate_domain(expected_ids, context["records"])
                context["records"].sort(key=_v2_record_sort_key)
                for rank, record in enumerate(context["records"], start=1):
                    record["rank"] = rank
                successful = [
                    record for record in context["records"] if record["status"] == "ok"
                ]
                render_records = successful if render_all_candidates else successful[:2]
                render_records_by_region[region_id] = render_records
                for record in render_records:
                    if _v2_render_artifacts_complete(record, run_dir=run_dir):
                        continue
                    candidate_id = str(record["candidate_id"])
                    projection = context["projections"].get(candidate_id)
                    if projection is not None:
                        continue
                    candidate = next(
                        item
                        for item in context["candidates"]
                        if str(item["candidate_id"]) == candidate_id
                    )
                    key = _projection_work_key(model_path, candidate["view_spec"])
                    group = on_demand_projection_groups.setdefault(
                        key,
                        {
                            "view_name": candidate_id,
                            "view_spec": candidate["view_spec"],
                            "members": [],
                        },
                    )
                    group["members"].append((region_id, candidate_id))

            on_demand_futures: dict[concurrent.futures.Future[dict[str, Any]], str] = {}
            for key, group in on_demand_projection_groups.items():
                try:
                    future = executor.submit(
                        _project_v2_worker,
                        {
                            "projection_key": key,
                            "model_path": str(model_path),
                            "view_name": group["view_name"],
                            "view_spec": group["view_spec"],
                        },
                    )
                    on_demand_futures[future] = key
                except Exception as error:  # noqa: BLE001 - parent can report
                    log.write(
                        "v2 on-demand projection submit error "
                        f"{type(error).__name__}: {error}"
                    )
            for future in concurrent.futures.as_completed(on_demand_futures):
                key = on_demand_futures[future]
                try:
                    result = future.result()
                    projection = result["projection"]
                    worker_pids.add(int(result["worker_pid"]))
                except Exception as error:  # noqa: BLE001 - parent can report
                    log.write(
                        f"v2 on-demand projection error {type(error).__name__}: {error}"
                    )
                    continue
                for region_id, candidate_id in on_demand_projection_groups[key][
                    "members"
                ]:
                    contexts[region_id]["projections"][candidate_id] = projection
            parallelism["on_demand_render_hlr_projections"] = len(
                on_demand_projection_groups
            )

            render_futures: dict[
                concurrent.futures.Future[dict[str, Any]], tuple[str, str]
            ] = {}
            for region_id, render_records in render_records_by_region.items():
                context = contexts[region_id]
                for record in render_records:
                    if _v2_render_artifacts_complete(record, run_dir=run_dir):
                        continue
                    candidate_id = str(record["candidate_id"])
                    projection = context["projections"].get(candidate_id)
                    if projection is None:
                        continue
                    try:
                        future = executor.submit(
                            _render_v2_worker,
                            {
                                "record": record,
                                "projection": projection,
                                "reference_path": str(context["reference_path"]),
                                "output_dir": str(context["output_dir"]),
                                "run_dir": str(run_dir),
                            },
                        )
                        render_futures[future] = (region_id, candidate_id)
                    except Exception as error:  # noqa: BLE001 - parent can retry
                        log.write(
                            f"v2 render submit error region={region_id} "
                            f"candidate={candidate_id} {type(error).__name__}: {error}"
                        )

            for future in concurrent.futures.as_completed(render_futures):
                region_id, candidate_id = render_futures[future]
                context = contexts[region_id]
                try:
                    worker_result = future.result()
                    rendered = worker_result["record"]
                    worker_pids.add(int(worker_result["worker_pid"]))
                except Exception as error:  # noqa: BLE001 - parent can retry
                    log.write(
                        f"v2 render error region={region_id} candidate={candidate_id} "
                        f"{type(error).__name__}: {error}"
                    )
                    continue
                for index, record in enumerate(context["records"]):
                    if str(record["candidate_id"]) == candidate_id:
                        context["records"][index] = rendered
                        break
            parallelism["observed_worker_processes"] = len(worker_pids)

    for region in regions:
        region_id = str(region["region_id"])
        context = contexts.get(region_id)
        if context is None:
            continue
        cached_payload = context["cached_payload"]
        if cached_payload is not None:
            cached_payload["parallelism"] = parallelism
            elapsed = time.perf_counter() - context["started"]
            entries_by_region[region_id] = {
                "scorer": "v2",
                "region_id": region_id,
                "status": "ok",
                "elapsed_seconds": elapsed,
                "result": cached_payload,
            }
            overlays_by_region[region_id] = (
                f"v2 / {region_id}",
                run_dir / cached_payload["overlay_png"],
            )
            log.write(
                f"complete scorer=v2 region={region_id} cache_hit=full "
                f"winner={cached_payload['rankings'][0]['candidate_id']} "
                f"elapsed={elapsed:.3f}s"
            )
            continue
        try:
            payload = _finalize_v2_region(
                drawing_view,
                reference_path=context["reference_path"],
                reference_geometry=context["reference_geometry"],
                reference_overlay=context["reference_overlay"],
                output_dir=context["output_dir"],
                region=context["region"],
                options=options,
                accepted_ids=context["accepted_ids"],
                run_dir=run_dir,
                log=log,
                selected_variant=selected_variant,
                scorer_config=scorer_config,
                run_perturbations=run_perturbations,
                records=context["records"],
                projections=context["projections"],
                expected_candidate_ids=[
                    str(candidate["candidate_id"])
                    for candidate in context["candidates"]
                ],
                render_all_candidates=render_all_candidates,
                parallelism=parallelism,
            )
            elapsed = time.perf_counter() - context["started"]
            entries_by_region[region_id] = {
                "scorer": "v2",
                "region_id": region_id,
                "status": "ok",
                "elapsed_seconds": elapsed,
                "result": payload,
            }
            overlays_by_region[region_id] = (
                f"v2 / {region_id}",
                run_dir / payload["overlay_png"],
            )
            log.write(
                f"complete scorer=v2 region={region_id} "
                f"winner={payload['rankings'][0]['candidate_id']} "
                f"elapsed={elapsed:.3f}s"
            )
        except Exception as error:  # noqa: BLE001 - persist partial batch output
            elapsed = time.perf_counter() - context["started"]
            error_overlay = context["output_dir"] / "overlay.png"
            _write_error_overlay(
                error_overlay,
                f"v2 / {region_id} failed",
                f"{type(error).__name__}: {error}",
            )
            entries_by_region[region_id] = {
                "scorer": "v2",
                "region_id": region_id,
                "status": "error",
                "elapsed_seconds": elapsed,
                "error_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
                "overlay_png": _artifact_path(error_overlay, run_dir),
            }
            overlays_by_region[region_id] = (
                f"v2 / {region_id} ERROR",
                error_overlay,
            )
            log.write(
                f"error scorer=v2 region={region_id} {type(error).__name__}: {error}"
            )

    ordered_ids = [str(region["region_id"]) for region in regions]
    return (
        [entries_by_region[region_id] for region_id in ordered_ids],
        [overlays_by_region[region_id] for region_id in ordered_ids],
        parallelism,
    )


class RunLog:
    def __init__(self, path: Path, *, append: bool = False) -> None:
        self.path = path
        if not append:
            self.path.write_text("", encoding="utf-8")

    def write(self, message: str) -> None:
        timestamp = datetime.now(UTC).isoformat(timespec="milliseconds")
        line = f"[{timestamp}] {message}"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)


def _safe_run_id(value: str | None) -> str:
    run_id = value or datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run-id must start with an alphanumeric character and contain only "
            "letters, digits, dot, underscore, or hyphen (maximum 80 characters)"
        )
    return run_id


def _argv_without_resume(argv: Sequence[str]) -> list[str]:
    return [str(argument) for argument in argv if argument != "--resume"]


def _resume_identity_payload(
    *,
    argv: Sequence[str],
    resolved_config: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_argv = _argv_without_resume(argv)
    command = {
        "python_executable": str(Path(sys.executable).resolve()),
        "module": MODULE_NAME,
        "argv": normalized_argv,
    }
    drawing_view = importlib.import_module("cadgenbench.tools.drawing_view")
    implementation = {
        "experiment_runner_sha256": _sha256(Path(__file__).resolve()),
        "drawing_view_sha256": _sha256(Path(drawing_view.__file__).resolve()),
        "candidate_cache_schema_version": CANDIDATE_CACHE_SCHEMA_VERSION,
        "runtime_fingerprint": _cache_runtime_fingerprint(),
    }
    return {
        "schema_version": RESUME_IDENTITY_SCHEMA_VERSION,
        "command": command,
        "command_sha256": _canonical_json_sha256(command),
        "config_sha256": _canonical_json_sha256(resolved_config),
        "implementation": implementation,
        "implementation_sha256": _canonical_json_sha256(implementation),
    }


def _validate_resume_identity(
    run_dir: Path,
    *,
    expected_identity: Mapping[str, Any],
    expected_config: Mapping[str, Any],
) -> None:
    identity_path = run_dir / "resume_identity.json"
    config_path = run_dir / "config.json"
    if not identity_path.is_file() or not config_path.is_file():
        raise ValueError(
            "resume requires existing resume_identity.json and config.json"
        )
    saved_identity = load_json_strict(identity_path)
    saved_config = load_json_strict(config_path)
    if saved_identity != _jsonable(expected_identity):
        raise ValueError("resume command/config identity does not match existing run")
    if saved_config != _jsonable(expected_config):
        raise ValueError("resume resolved config does not match existing run")
    if saved_identity.get("config_sha256") != _canonical_json_sha256(saved_config):
        raise ValueError("existing resume config hash is invalid")
    command = saved_identity.get("command")
    if not isinstance(command, Mapping) or saved_identity.get(
        "command_sha256"
    ) != _canonical_json_sha256(command):
        raise ValueError("existing resume command hash is invalid")
    implementation = saved_identity.get("implementation")
    if not isinstance(implementation, Mapping) or saved_identity.get(
        "implementation_sha256"
    ) != _canonical_json_sha256(implementation):
        raise ValueError("existing resume implementation hash is invalid")


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _environment_payload(
    drawing_view: ModuleType,
    *,
    argv: Sequence[str],
    parallelism: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    module_path = Path(drawing_view.__file__).resolve()
    return {
        "captured_at": datetime.now(UTC).isoformat(),
        "cwd": str(Path.cwd().resolve()),
        "argv": list(argv),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "implementation": platform.python_implementation(),
        },
        "platform": platform.platform(),
        "parallelism": dict(parallelism or {}),
        "packages": {
            name: _package_version(name)
            for name in (
                "cadgenbench",
                "numpy",
                "scipy",
                "Pillow",
                "build123d",
                "cadquery-ocp",
            )
        },
        "drawing_view_module": {
            "path": str(module_path),
            "sha256": _sha256(module_path),
        },
        "native_import_compatibility": {
            "preloaded": "xml.parsers.expat",
            "reason": "Load the active Python environment's libexpat before OCCT/build123d.",
        },
    }


def _write_error_overlay(path: Path, title: str, message: str) -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1200, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.text((24, 24), title, fill=(150, 20, 20))
    draw.multiline_text((24, 72), message[:1000], fill=(30, 30, 30), spacing=6)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)


def _write_overlay_sheet(
    output: Path,
    overlays: Sequence[tuple[str, Path]],
) -> None:
    from PIL import Image, ImageDraw

    if not overlays:
        _write_error_overlay(output, "No overlays", "No scorer region completed.")
        return
    rows: list[tuple[str, Any]] = []
    for label, path in overlays:
        with Image.open(path) as source:
            panel = source.convert("RGB")
        panel.thumbnail((1400, 700))
        rows.append((label, panel))
    width = max(panel.width for _, panel in rows) + 32
    height = sum(panel.height + 52 for _, panel in rows) + 16
    sheet = Image.new("RGB", (width, height), (238, 241, 245))
    draw = ImageDraw.Draw(sheet)
    y = 16
    for label, panel in rows:
        draw.text((16, y), label, fill=(20, 25, 35))
        y += 32
        sheet.paste(panel, (16, y))
        y += panel.height + 20
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="PNG", optimize=True)


def _report_markdown(
    result: Mapping[str, Any],
    decision_report: Mapping[str, Any],
) -> str:
    selected_scorers = set(
        result.get("scope", {}).get(
            "selected_scorers",
            [entry["scorer"] for entry in result.get("entries", [])],
        )
    )
    if selected_scorers == {"baseline", "v2"}:
        scorer_scope_sentence = (
            "The baseline and v2 scorers received the complete pinned crop "
            "(`roi = null`)."
        )
    elif selected_scorers == {"v2"}:
        scorer_scope_sentence = (
            "Only scorer v2 was run on the complete pinned crop (`roi = null`). "
            "Baseline was not run, so this run cannot produce a fixture-level GO."
        )
    elif selected_scorers == {"baseline"}:
        scorer_scope_sentence = (
            "Only the baseline scorer was run on the complete pinned crop "
            "(`roi = null`); scorer v2 evidence is missing."
        )
    else:
        scorer_scope_sentence = "No complete baseline/v2 scorer selection was recorded."
    lines = [
        "# Fixture 145 drawing scorer experiment",
        "",
        f"Status: `{result['status']}`",
        f"Scorer-v2 decision: `{decision_report['decision']}`",
        f"Scope: `{decision_report['scope']['kind']}`",
        (
            "V2 variant: `{}`; nominal only: `{}`".format(
                result.get("scope", {}).get("scorer_variant", DEFAULT_SCORER_VARIANT),
                bool(result.get("scope", {}).get("nominal_only", False)),
            )
        ),
        f"Elapsed: `{float(result.get('elapsed_seconds', 0.0)):.3f}s`",
        "",
        scorer_scope_sentence,
        "Human-reviewed expected-part boxes were not used for alignment or scoring.",
        "",
        "| Scorer | Region | Top 1 | Score | Eligible | Best accepted rank | Canonical rank | Accepted direction in top 2 |",
        "|---|---|---|---:|:---:|---:|---:|:---:|",
    ]
    for entry in result.get("entries", []):
        if entry.get("status") != "ok":
            lines.append(
                f"| {entry['scorer']} | {entry['region_id']} | ERROR: "
                f"{entry.get('error_type')} | — | — | — | — | — |"
            )
            continue
        ranking = entry["result"]["rankings"]
        winner = ranking[0]
        oracle = entry["result"]["oracle"]
        lines.append(
            "| {scorer} | {region} | `{winner}` | {score:.6f} | {eligible} | "
            "{accepted} | {canonical} | {top2} |".format(
                scorer=entry["scorer"],
                region=entry["region_id"],
                winner=winner["candidate_id"],
                score=float(winner["score"]),
                eligible="yes" if winner.get("eligible") else "no",
                accepted=oracle.get("best_accepted_rank") or "—",
                canonical=oracle.get("canonical_rank") or "—",
                top2="yes" if oracle.get("top2_contains_accepted") else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Reproduction artifacts",
            "",
            "- `command.txt`: exact command captured by the process",
            "- `environment.json`: interpreter, package, platform, and scorer-module hash",
            "- `config.json`: pinned fixture plus fully expanded candidate/oracle manifest",
            "- `baseline_rankings.json` and `scorer_v2_rankings.json`: scorer-specific full rankings",
            "- `report.json`: per-region hard gates and overall go/no-go/partial decision",
            "- `top2.json`: compact top-two trace for every scorer and region",
            "- `overlay.png`: contact sheet of every winning comparison",
            "- `run.log`: timestamped execution trace",
            "",
        ]
    )
    return "\n".join(lines)


def _scorer_rankings_artifact(
    entries: Sequence[Mapping[str, Any]],
    scorer: str,
) -> dict[str, Any]:
    selected = [entry for entry in entries if entry["scorer"] == scorer]

    def completed_region(entry: Mapping[str, Any]) -> dict[str, Any]:
        payload = entry["result"]
        artifact = {
            "region_id": entry["region_id"],
            "status": "ok",
            "rankings": payload["rankings"],
            "oracle": payload["oracle"],
            "overlay_png": payload["overlay_png"],
            "top2_contact_sheet_png": payload["top2_contact_sheet_png"],
        }
        if scorer == "v2":
            artifact.update(
                {
                    "scorer_variant": payload.get(
                        "scorer_variant", DEFAULT_SCORER_VARIANT
                    ),
                    "reference_envelope": payload["reference_envelope"],
                    "reference_envelope_overlay_png": payload[
                        "reference_envelope_overlay_png"
                    ],
                    "perturbations": payload["perturbations"],
                }
            )
        return artifact

    return {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "scorer": scorer,
        "status": "not_run"
        if not selected
        else (
            "complete"
            if all(entry["status"] == "ok" for entry in selected)
            else "partial"
        ),
        "regions": [
            (
                completed_region(entry)
                if entry["status"] == "ok"
                else {
                    "region_id": entry["region_id"],
                    "status": "error",
                    "error_type": entry.get("error_type"),
                    "message": entry.get("message"),
                }
            )
            for entry in selected
        ],
    }


def _component_decision(
    region_reports: Mapping[str, Mapping[str, Any]],
    required_region_ids: Sequence[str],
    *,
    decision_field: str = "decision",
) -> str:
    selected = [
        region_reports[region_id]
        for region_id in required_region_ids
        if region_id in region_reports
    ]
    if any(report[decision_field] == "no-go" for report in selected):
        return "no-go"
    if len(selected) != len(required_region_ids) or any(
        report[decision_field] != "go" for report in selected
    ):
        return "partial"
    return "go"


def _historical_edge_on_evidence(
    region_id: str,
    rankings: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    base_ids = EDGE_ON_CANDIDATE_BASES.get(region_id)
    if base_ids is None:
        return None
    expected_ids = [
        f"{base_id}-r{roll:03d}" for base_id in base_ids for roll in (0, 90, 180, 270)
    ]
    by_id = {str(candidate["candidate_id"]): candidate for candidate in rankings}
    candidate_evidence = []
    for candidate_id in expected_ids:
        record = by_id.get(candidate_id)
        failed_gates = (
            []
            if record is None
            else [str(gate) for gate in record.get("failed_gates", [])]
        )
        shape_gates = sorted(EDGE_ON_REJECTION_GATES.intersection(failed_gates))
        candidate_evidence.append(
            {
                "candidate_id": candidate_id,
                "found": record is not None,
                "status": None if record is None else record.get("status"),
                "rank": None if record is None else int(record["rank"]),
                "in_top2": record is not None and int(record["rank"]) <= 2,
                "eligible": None if record is None else bool(record.get("eligible")),
                "failed_gates": failed_gates,
                "shape_rejection_gates": shape_gates,
                "rejected_by_aspect_or_short_axis_fill": bool(shape_gates),
                "gross": None if record is None else record.get("gross"),
            }
        )
    all_found = all(item["found"] for item in candidate_evidence)
    return {
        "historical_candidate_id": HISTORICAL_EDGE_ON_CANDIDATES[region_id],
        "base_candidate_ids": list(base_ids),
        "expected_candidate_ids": expected_ids,
        "all_found": all_found,
        "in_top2_candidate_ids": [
            str(item["candidate_id"]) for item in candidate_evidence if item["in_top2"]
        ],
        "not_shape_rejected_candidate_ids": [
            str(item["candidate_id"])
            for item in candidate_evidence
            if not item["rejected_by_aspect_or_short_axis_fill"]
        ],
        "all_outside_top2": all_found
        and not any(item["in_top2"] for item in candidate_evidence),
        "all_rejected_by_aspect_or_short_axis_fill": all_found
        and all(
            item["rejected_by_aspect_or_short_axis_fill"] for item in candidate_evidence
        ),
        "candidates": candidate_evidence,
    }


def _build_decision_report(
    *,
    entries: Sequence[Mapping[str, Any]],
    selected_region_ids: Sequence[str],
) -> dict[str, Any]:
    """Apply fixture-145's reviewed hard gates without overstating subsets."""

    selected_ids = list(dict.fromkeys(str(item) for item in selected_region_ids))
    selected_set = set(selected_ids)
    required_set = set(FULL_FIXTURE_REGION_IDS)
    full_fixture_scope = selected_set == required_set
    baseline_by_region = {
        str(entry["region_id"]): entry
        for entry in entries
        if entry["scorer"] == "baseline"
    }
    v2_by_region = {
        str(entry["region_id"]): entry for entry in entries if entry["scorer"] == "v2"
    }
    region_reports: list[dict[str, Any]] = []

    for region_id in selected_ids:
        v2_entry = v2_by_region.get(region_id)
        if v2_entry is None or v2_entry["status"] != "ok":
            v2_reason = (
                "scorer_v2_not_run"
                if v2_entry is None
                else f"scorer_v2_error:{v2_entry.get('error_type')}"
            )
            region_reports.append(
                {
                    "region_id": region_id,
                    "decision": (
                        "no-go"
                        if region_id in MAIN_ORTHOGRAPHIC_REGION_IDS
                        else "partial"
                    ),
                    "v2_decision": (
                        "no-go"
                        if region_id in MAIN_ORTHOGRAPHIC_REGION_IDS
                        else "partial"
                    ),
                    "reason": v2_reason,
                    "comparison_evidence": {
                        "status": "incomplete",
                        "reason": v2_reason,
                    },
                    "hard_gates": [],
                }
            )
            continue

        payload = v2_entry["result"]
        rankings = payload["rankings"]
        top2 = rankings[:2]
        top2_ids = [str(record["candidate_id"]) for record in top2]
        oracle = payload["oracle"]
        accepted = {str(item) for item in oracle["accepted_candidate_ids"]}
        candidate_errors = [record for record in rankings if record["status"] != "ok"]
        perturbations = payload["perturbations"]
        perturbation_summary = perturbations["summary"]
        robustness_not_evaluated = perturbations.get("status") == "skipped"
        reference = payload["reference_envelope"]
        historical = _historical_edge_on_evidence(region_id, rankings)

        baseline_entry = baseline_by_region.get(region_id)
        if baseline_entry is None:
            parity_passed = False
            parity_evidence: Any = "baseline_not_run"
            comparison_reason = "baseline_not_run"
        elif baseline_entry["status"] != "ok":
            parity_passed = False
            parity_evidence = f"baseline_error:{baseline_entry.get('error_type')}"
            comparison_reason = parity_evidence
        else:
            baseline_ids = {
                str(record["candidate_id"])
                for record in baseline_entry["result"]["rankings"]
            }
            v2_ids = {str(record["candidate_id"]) for record in rankings}
            parity_passed = baseline_ids == v2_ids
            parity_evidence = {
                "baseline_count": len(baseline_ids),
                "v2_count": len(v2_ids),
                "only_baseline": sorted(baseline_ids - v2_ids),
                "only_v2": sorted(v2_ids - baseline_ids),
            }
            comparison_reason = (
                None if parity_passed else "baseline_v2_candidate_set_mismatch"
            )

        gate_values: list[tuple[str, bool, Any]] = [
            (
                "all_candidates_evaluated",
                not candidate_errors,
                {
                    "candidate_count": len(rankings),
                    "error_candidate_ids": [
                        record["candidate_id"] for record in candidate_errors
                    ],
                },
            ),
            (
                "winner_eligible",
                bool(top2 and top2[0]["eligible"]),
                {
                    "winner": top2[0]["candidate_id"] if top2 else None,
                    "failed_gates": top2[0]["failed_gates"] if top2 else [],
                },
            ),
            (
                "oracle_direction_in_top2",
                any(
                    record["status"] == "ok"
                    and bool(record.get("eligible"))
                    and str(record["candidate_id"]) in accepted
                    for record in top2
                ),
                {
                    "actual_top2": top2_ids,
                    "accepted_candidate_ids": sorted(accepted),
                    "best_accepted_rank": oracle.get("best_accepted_rank"),
                    "best_eligible_accepted_rank": oracle.get(
                        "best_eligible_accepted_rank"
                    ),
                    "canonical_rank": oracle.get("canonical_rank"),
                },
            ),
        ]
        if not robustness_not_evaluated:
            gate_values.extend(
                [
                    (
                        "all_perturbations_realign_full_candidate_domain",
                        bool(
                            perturbation_summary.get(
                                "all_variants_full_candidate_domain"
                            )
                        ),
                        perturbation_summary,
                    ),
                    (
                        "repeat_reproducible",
                        bool(perturbation_summary.get("repeat_exact")),
                        perturbation_summary,
                    ),
                    (
                        "seed_perturbation_stable",
                        bool(perturbation_summary.get("seed_stable")),
                        perturbation_summary,
                    ),
                    (
                        "threshold_and_closing_perturbations_stable",
                        bool(perturbation_summary.get("threshold_and_closing_stable")),
                        perturbation_summary,
                    ),
                ]
            )
        if region_id in ISOMETRIC_REGION_IDS:
            gate_values.append(
                (
                    "isometric_winner_is_oracle_direction_and_roll",
                    bool(top2_ids and top2_ids[0] in accepted),
                    {
                        "winner": top2_ids[0] if top2_ids else None,
                        "accepted_candidate_ids": sorted(accepted),
                    },
                )
            )
        if historical is not None:
            gate_values.extend(
                [
                    (
                        "historical_edge_on_not_in_top2",
                        bool(historical["all_outside_top2"]),
                        historical,
                    ),
                    (
                        "historical_edge_on_rejected_by_aspect_or_short_axis_fill",
                        bool(historical["all_rejected_by_aspect_or_short_axis_fill"]),
                        historical,
                    ),
                ]
            )

        hard_gates = [
            {"name": name, "passed": bool(passed), "evidence": evidence}
            for name, passed, evidence in gate_values
        ]
        failed_gate_counts: dict[str, int] = {}
        for record in rankings:
            for gate in record.get("failed_gates", []):
                failed_gate_counts[str(gate)] = failed_gate_counts.get(str(gate), 0) + 1
        nominal_passed = all(gate["passed"] for gate in hard_gates)
        v2_passed = nominal_passed and not robustness_not_evaluated
        if not nominal_passed:
            region_decision = "no-go"
            region_reason = "scorer_v2_hard_gate_failed"
        elif robustness_not_evaluated:
            region_decision = "partial"
            region_reason = "robustness_not_evaluated"
        elif not parity_passed:
            region_decision = "partial"
            region_reason = comparison_reason
        else:
            region_decision = "go"
            region_reason = None
        report: dict[str, Any] = {
            "region_id": region_id,
            "decision": region_decision,
            "v2_decision": (
                "partial"
                if robustness_not_evaluated and nominal_passed
                else ("go" if v2_passed else "no-go")
            ),
            "reason": region_reason,
            "hard_gates": hard_gates,
            "robustness_evidence": {
                "status": (
                    "not_evaluated" if robustness_not_evaluated else "evaluated"
                ),
                "reason": perturbations.get("reason"),
            },
            "comparison_evidence": {
                "status": "complete" if parity_passed else "incomplete",
                "reason": comparison_reason,
                "candidate_set_parity": parity_passed,
                "details": parity_evidence,
            },
            "reference_envelope_diagnostic": {
                "stable": bool(reference["stable"]),
                "confidence": reference["confidence"],
                "failure_reason": reference["failure_reason"],
                "fallback_used": reference["fallback_used"],
            },
            "candidate_gate_summary": {
                "eligible_count": sum(
                    bool(record.get("eligible")) for record in rankings
                ),
                "ineligible_count": sum(
                    not bool(record.get("eligible")) for record in rankings
                ),
                "failed_gate_counts": failed_gate_counts,
            },
        }
        if region_id in SECTION_REGION_IDS and not v2_passed:
            report["section_failure_classification"] = (
                "envelope_not_applicable"
                if not bool(reference["stable"])
                else "ranking_or_robustness_failure"
            )
        region_reports.append(report)

    reports_by_id = {str(report["region_id"]): report for report in region_reports}
    selected_scope_decision = (
        "no-go"
        if any(report["decision"] == "no-go" for report in region_reports)
        else (
            "partial"
            if any(report["decision"] != "go" for report in region_reports)
            else "go"
        )
    )
    orthographic_decision = _component_decision(
        reports_by_id,
        MAIN_ORTHOGRAPHIC_REGION_IDS,
        decision_field="v2_decision",
    )
    isometric_decision = _component_decision(
        reports_by_id,
        ISOMETRIC_REGION_IDS,
        decision_field="v2_decision",
    )
    section_decision = _component_decision(
        reports_by_id,
        SECTION_REGION_IDS,
        decision_field="v2_decision",
    )
    comparison_incomplete_region_ids = [
        region_id
        for region_id in selected_ids
        if reports_by_id.get(region_id, {}).get("comparison_evidence", {}).get("status")
        != "complete"
    ]
    baseline_missing_region_ids = [
        region_id for region_id in selected_ids if region_id not in baseline_by_region
    ]
    baseline_error_regions = {
        region_id: str(entry.get("error_type") or "unknown_error")
        for region_id, entry in baseline_by_region.items()
        if region_id in selected_set and entry.get("status") != "ok"
    }
    v2_missing_region_ids = [
        region_id for region_id in selected_ids if region_id not in v2_by_region
    ]
    v2_error_regions = {
        region_id: str(entry.get("error_type") or "unknown_error")
        for region_id, entry in v2_by_region.items()
        if region_id in selected_set and entry.get("status") != "ok"
    }
    comparison_complete = not comparison_incomplete_region_ids

    # A failed main orthographic view is decisive.  Passing only a subset can
    # never claim fixture-level GO.  When both main views pass but ISO/section
    # fail, the useful result is explicitly orthographic-only and partial.
    if orthographic_decision == "no-go":
        decision = "no-go"
    elif not full_fixture_scope or orthographic_decision != "go":
        decision = "partial"
    elif (
        comparison_complete and isometric_decision == "go" and section_decision == "go"
    ):
        decision = "go"
    else:
        decision = "partial"

    if not full_fixture_scope:
        recommendation = "complete_remaining_scope_before_fixture_decision"
    elif orthographic_decision == "no-go":
        recommendation = "stop_and_fix_orthographic_scorer"
    elif not comparison_complete:
        recommendation = "complete_baseline_and_v2_comparison_evidence"
    elif decision == "go":
        recommendation = "proceed_with_all_view_types"
    elif decision == "partial" and orthographic_decision == "go":
        recommendation = "proceed_with_orthographic_only; keep_iso_and_section_no-go"
    else:
        recommendation = "stop_and_fix_orthographic_scorer"

    return {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "decision": decision,
        "scope": {
            "kind": "full_fixture" if full_fixture_scope else "subset",
            "required_region_ids": list(FULL_FIXTURE_REGION_IDS),
            "selected_region_ids": selected_ids,
            "missing_region_ids": sorted(required_set - selected_set),
            "unexpected_region_ids": sorted(selected_set - required_set),
            "full_fixture_complete": full_fixture_scope,
            "selected_scope_decision": selected_scope_decision,
        },
        "track_decisions": {
            "orthographic": orthographic_decision,
            "isometric": isometric_decision,
            "section": section_decision,
        },
        "comparison_evidence": {
            "status": "complete" if comparison_complete else "incomplete",
            "required_scorers": ["baseline", "v2"],
            "complete": comparison_complete,
            "incomplete_region_ids": comparison_incomplete_region_ids,
            "baseline": {
                "missing_region_ids": baseline_missing_region_ids,
                "error_regions": baseline_error_regions,
            },
            "v2": {
                "missing_region_ids": v2_missing_region_ids,
                "error_regions": v2_error_regions,
            },
        },
        "integration_recommendation": recommendation,
        "meaning": {
            "go": "The complete fixture passed orthographic, isometric, and section gates.",
            "no-go": (
                "At least one selected main orthographic v2 result is missing, "
                "errored, or failed a hard gate."
            ),
            "partial": (
                "The fixture scope or baseline/v2 comparison evidence is incomplete, "
                "or only the ordinary orthographic track is ready; this is not a "
                "fixture-level GO."
            ),
        }[decision],
        "hard_gate_policy": [
            "all candidates finish",
            "the winner is eligible",
            "at least one reviewed correct direction and page roll enters top two",
            "an isometric winner must itself be a reviewed direction and page roll",
            "historical main-view edge-on failures stay out of top two and fail an aspect or short-axis-fill gate",
            "baseline and v2 use the same candidate IDs when baseline is selected",
            "repeat, seed, threshold, and closing audits freshly align and re-rank the complete candidate domain",
            "same-input repeat is reproducible and every perturbation keeps an accepted direction in top two",
        ],
        "regions": region_reports,
    }


def _resolved_config(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    selected_regions: Sequence[Mapping[str, Any]],
    scorers: Sequence[str],
    max_iterations: int,
    scorer_variant: str = DEFAULT_SCORER_VARIANT,
    nominal_only: bool = False,
    requested_jobs: int = 0,
    resolved_jobs: int = 0,
    render_all_candidates: bool = False,
    unique_hlr_projections: int = 0,
    candidate_evaluations: int = 0,
) -> dict[str, Any]:
    selected_variant = _validate_scorer_variant(scorer_variant)
    defaults = config["defaults"]
    resolved_regions = []
    for region in selected_regions:
        candidates = expand_region_candidates(config, region)
        resolved_regions.append(
            {
                "region_id": region["region_id"],
                "reference": region["reference"],
                "candidates": candidates,
                "oracle_accepted_candidate_ids": list(
                    accepted_oracle_candidate_ids(config, region, candidates)
                ),
            }
        )
    variant_contract = {
        "name": selected_variant,
        "use_envelope_matching": (
            selected_variant
            not in {NO_ENVELOPE_MATCH_VARIANT, CAPACITY_TRANSPORT_VARIANT}
        ),
        "perturbations": "skipped" if nominal_only else "full",
    }
    if selected_variant == CAPACITY_TRANSPORT_VARIANT:
        variant_contract["use_capacity_transport"] = True
    return {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "experiment": "drawing-scorer-v2",
        "source_config": {
            "path": str(config_path),
            "sha256": _sha256(config_path),
        },
        "selection": {
            "scorers": list(scorers),
            "regions": [region["region_id"] for region in selected_regions],
            "max_iterations": max_iterations,
            "scorer_variant": selected_variant,
            "nominal_only": bool(nominal_only),
        },
        "scorer_contract": {
            "roi": None,
            "reference_scope": "complete_pinned_crop",
            "audit_only_expected_part_bbox_used": False,
            "page_rolls_deg": list(_configured_rolls(config)),
        },
        "v2_variant_contract": variant_contract,
        "parallel_execution": {
            "requested_jobs": requested_jobs,
            "resolved_jobs": resolved_jobs,
            "logical_cpus_in_affinity": _available_cpu_count(),
            "physical_cpus_in_affinity": _available_physical_cpu_count(),
            "process_start_method": "spawn",
            "worker_native_threads": 1,
            "candidate_evaluations": candidate_evaluations,
            "unique_hlr_projections": unique_hlr_projections,
            "deduplicated_hlr_projections": (
                candidate_evaluations - unique_hlr_projections
            ),
            "render_mode": "all" if render_all_candidates else "top2_only",
        },
        "fixture": config,
        "resolved_regions": resolved_regions,
        "resolved_scorer_options": {
            "baseline": _baseline_options(defaults, max_iterations=max_iterations),
            "v2": _scorer_common_options(defaults, max_iterations=max_iterations),
        },
    }


def run_experiment(
    *,
    config_path: Path,
    output_root: Path,
    run_id: str | None,
    scorer_selection: str,
    region_ids: Sequence[str],
    max_iterations_override: int | None,
    argv: Sequence[str],
    scorer_variant: str = DEFAULT_SCORER_VARIANT,
    nominal_only: bool = False,
    jobs: int = 0,
    render_all_candidates: bool = False,
    resume: bool = False,
) -> tuple[Path, dict[str, Any]]:
    experiment_started_at = datetime.now(UTC)
    experiment_started = time.perf_counter()
    config, fixture_root = load_and_validate_fixture(config_path)
    selected_run_id = _safe_run_id(run_id)
    all_regions = _regions(config)
    available_ids = {str(region["region_id"]) for region in all_regions}
    unknown = sorted(set(region_ids) - available_ids)
    if unknown:
        raise ValueError(f"Unknown region ID(s): {', '.join(unknown)}")
    selected_regions = (
        [region for region in all_regions if region["region_id"] in set(region_ids)]
        if region_ids
        else all_regions
    )
    scorers = ("baseline", "v2") if scorer_selection == "both" else (scorer_selection,)
    selected_variant = _validate_scorer_variant(scorer_variant)
    if scorer_selection not in {"baseline", "v2", "both"}:
        raise ValueError(f"Unknown scorer selection: {scorer_selection!r}")
    if selected_variant != DEFAULT_SCORER_VARIANT and "v2" not in scorers:
        raise ValueError(f"Scorer variant {selected_variant!r} requires scorer v2")
    if nominal_only and "v2" not in scorers:
        raise ValueError("nominal-only mode requires scorer v2")
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 0:
        raise ValueError("jobs must be zero (auto) or a positive integer")
    if resume and run_id is None:
        raise ValueError("resume requires an explicit run-id")
    defaults = config["defaults"]
    max_iterations = (
        int(max_iterations_override)
        if max_iterations_override is not None
        else int(defaults["max_iterations"])
    )
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    drawing_view = importlib.import_module("cadgenbench.tools.drawing_view")
    v2_scorer_config = (
        _v2_config_for_variant(drawing_view, selected_variant)
        if "v2" in scorers
        else None
    )
    model_path = _fixture_path(fixture_root, config["model"]["path"])
    planned_candidate_evaluations = (
        sum(
            len(expand_region_candidates(config, region)) for region in selected_regions
        )
        if "v2" in scorers
        else 0
    )
    planned_projection_keys = {
        _projection_work_key(model_path, candidate["view_spec"])
        for region in selected_regions
        for candidate in expand_region_candidates(config, region)
    }
    planned_unique_hlr = len(planned_projection_keys) if "v2" in scorers else 0
    planned_resolved_jobs = _resolve_parallel_jobs(
        jobs,
        task_count=planned_candidate_evaluations,
    )
    planned_parallelism = {
        "requested_jobs": jobs,
        "resolved_jobs": planned_resolved_jobs,
        "logical_cpus_in_affinity": _available_cpu_count(),
        "physical_cpus_in_affinity": _available_physical_cpu_count(),
        "process_start_method": "spawn",
        "worker_native_threads": 1,
        "candidate_evaluations": planned_candidate_evaluations,
        "unique_hlr_projections": planned_unique_hlr,
        "deduplicated_hlr_projections": (
            planned_candidate_evaluations - planned_unique_hlr
        ),
        "render_mode": "all" if render_all_candidates else "top2_only",
    }
    resolved_config = _resolved_config(
        config=config,
        config_path=Path(config_path).resolve(),
        selected_regions=selected_regions,
        scorers=scorers,
        max_iterations=max_iterations,
        scorer_variant=selected_variant,
        nominal_only=nominal_only,
        requested_jobs=jobs,
        resolved_jobs=planned_resolved_jobs,
        render_all_candidates=render_all_candidates,
        unique_hlr_projections=planned_unique_hlr,
        candidate_evaluations=planned_candidate_evaluations,
    )
    resume_identity = _resume_identity_payload(
        argv=argv,
        resolved_config=resolved_config,
    )
    run_dir = Path(output_root).resolve() / selected_run_id
    if resume:
        if not run_dir.is_dir():
            raise FileNotFoundError(f"resume run directory not found: {run_dir}")
        _validate_resume_identity(
            run_dir,
            expected_identity=resume_identity,
            expected_config=resolved_config,
        )
        log = RunLog(run_dir / "run.log", append=True)
        log.write("resume identity verified; reusing complete candidate checkpoints")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        log = RunLog(run_dir / "run.log")
        exact_command = shlex.join([sys.executable, "-m", MODULE_NAME, *argv])
        (run_dir / "command.txt").write_text(exact_command + "\n", encoding="utf-8")
        _write_json(run_dir / "config.json", resolved_config)
        _write_json(run_dir / "resume_identity.json", resume_identity)
        _write_json(
            run_dir / "environment.json",
            _environment_payload(
                drawing_view,
                argv=argv,
                parallelism=planned_parallelism,
            ),
        )
    log.write(
        f"start fixture={config['fixture_id']} scorers={','.join(scorers)} "
        f"variant={selected_variant} nominal_only={nominal_only} "
        f"regions={','.join(str(region['region_id']) for region in selected_regions)}"
    )
    entries: list[dict[str, Any]] = []
    overlays: list[tuple[str, Path]] = []
    actual_parallelism = planned_parallelism
    for scorer in scorers:
        if scorer == "v2":
            if v2_scorer_config is None:
                raise RuntimeError("V2 scorer configuration was not initialized")
            batch_entries, batch_overlays, actual_parallelism = (
                _run_v2_regions_parallel(
                    drawing_view,
                    model_path=model_path,
                    fixture_root=fixture_root,
                    config=config,
                    regions=selected_regions,
                    defaults=defaults,
                    max_iterations=max_iterations,
                    run_dir=run_dir,
                    log=log,
                    selected_variant=selected_variant,
                    scorer_config=v2_scorer_config,
                    run_perturbations=not nominal_only,
                    requested_jobs=jobs,
                    render_all_candidates=render_all_candidates,
                    resume=resume,
                )
            )
            entries.extend(batch_entries)
            overlays.extend(batch_overlays)
            continue
        for region in selected_regions:
            region_id = str(region["region_id"])
            candidates = expand_region_candidates(config, region)
            accepted_ids = accepted_oracle_candidate_ids(config, region, candidates)
            reference_path = _fixture_path(fixture_root, region["reference"])
            output_dir = run_dir / scorer / region_id
            output_dir.mkdir(parents=True, exist_ok=True)
            log.write(
                f"begin scorer={scorer} region={region_id} candidates={len(candidates)}"
            )
            started = datetime.now(UTC)
            try:
                payload = _run_baseline_region(
                    drawing_view,
                    model_path=model_path,
                    reference_path=reference_path,
                    output_dir=output_dir,
                    candidates=candidates,
                    region=region,
                    options=_baseline_options(defaults, max_iterations=max_iterations),
                    accepted_ids=accepted_ids,
                    run_dir=run_dir,
                )
                elapsed = (datetime.now(UTC) - started).total_seconds()
                entry = {
                    "scorer": scorer,
                    "region_id": region_id,
                    "status": "ok",
                    "elapsed_seconds": elapsed,
                    "result": payload,
                }
                overlays.append(
                    (f"{scorer} / {region_id}", run_dir / payload["overlay_png"])
                )
                log.write(
                    f"complete scorer={scorer} region={region_id} "
                    f"winner={payload['rankings'][0]['candidate_id']} elapsed={elapsed:.3f}s"
                )
            except Exception as error:  # noqa: BLE001 - persist a partial experiment
                elapsed = (datetime.now(UTC) - started).total_seconds()
                error_overlay = output_dir / "overlay.png"
                _write_error_overlay(
                    error_overlay,
                    f"{scorer} / {region_id} failed",
                    f"{type(error).__name__}: {error}",
                )
                overlays.append((f"{scorer} / {region_id} ERROR", error_overlay))
                entry = {
                    "scorer": scorer,
                    "region_id": region_id,
                    "status": "error",
                    "elapsed_seconds": elapsed,
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                    "overlay_png": _artifact_path(error_overlay, run_dir),
                }
                log.write(
                    f"error scorer={scorer} region={region_id} "
                    f"{type(error).__name__}: {error}"
                )
            entries.append(entry)

    status = (
        "complete" if all(entry["status"] == "ok" for entry in entries) else "partial"
    )
    experiment_finished_at = datetime.now(UTC)
    result = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "experiment": "drawing-scorer-v2",
        "fixture_id": config["fixture_id"],
        "run_id": selected_run_id,
        "status": status,
        "scope": {
            "kind": (
                "full_fixture"
                if {str(region["region_id"]) for region in selected_regions}
                == set(FULL_FIXTURE_REGION_IDS)
                else "subset"
            ),
            "selected_region_ids": [
                str(region["region_id"]) for region in selected_regions
            ],
            "selected_scorers": list(scorers),
            "scorer_variant": selected_variant,
            "nominal_only": bool(nominal_only),
            "resumed": bool(resume),
            "parallelism": actual_parallelism,
        },
        "started_at": experiment_started_at.isoformat(),
        "finished_at": experiment_finished_at.isoformat(),
        "elapsed_seconds": time.perf_counter() - experiment_started,
        "entries": entries,
    }
    compact_top2 = []
    for entry in entries:
        compact = {
            "scorer": entry["scorer"],
            "region_id": entry["region_id"],
            "status": entry["status"],
        }
        if entry["status"] == "ok":
            compact["top2"] = entry["result"]["top2"]
            compact["oracle"] = entry["result"]["oracle"]
        else:
            compact["error_type"] = entry.get("error_type")
            compact["message"] = entry.get("message")
        compact_top2.append(compact)
    _write_json(run_dir / "results.json", result)
    _write_json(run_dir / "top2.json", {"entries": compact_top2})
    _write_json(
        run_dir / "baseline_rankings.json",
        _scorer_rankings_artifact(entries, "baseline"),
    )
    _write_json(
        run_dir / "scorer_v2_rankings.json",
        _scorer_rankings_artifact(entries, "v2"),
    )
    decision_report = _build_decision_report(
        entries=entries,
        selected_region_ids=[str(region["region_id"]) for region in selected_regions],
    )
    _write_json(run_dir / "report.json", decision_report)
    _write_overlay_sheet(run_dir / "overlay.png", overlays)
    (run_dir / "report.md").write_text(
        _report_markdown(result, decision_report),
        encoding="utf-8",
    )
    _verify_top_level_artifacts(run_dir)
    log.write(f"finish status={status} decision={decision_report['decision']}")
    return run_dir, result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the pinned, offline fixture-145 drawing scorer experiment."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_FIXTURE_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument("--scorer", choices=("baseline", "v2", "both"), default="both")
    parser.add_argument(
        "--variant",
        choices=SCORER_VARIANTS,
        default=DEFAULT_SCORER_VARIANT,
        help=(
            "Select the V2 scorer contract. no-envelope-match requires the "
            "strict ScorerV2Config switch and never approximates the ablation."
        ),
    )
    parser.add_argument(
        "--nominal-only",
        action="store_true",
        help="Run the complete nominal candidate ranking without perturbation reruns.",
    )
    parser.add_argument(
        "--region",
        action="append",
        default=[],
        help="Run only this region ID; repeat to select more than one.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        help="Override the pinned optimisation iteration count for a smoke run.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help=(
            "Global V2 process workers. 0 uses all logical CPUs available "
            "inside this process's CPU affinity, capped by candidate count."
        ),
    )
    parser.add_argument(
        "--render-all-candidates",
        action="store_true",
        help=(
            "Render three PNG audit images for every candidate. By default only "
            "the final top two are rendered after numeric ranking."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an existing explicit run-id after strict command/config and "
            "per-candidate request-identity validation."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # The active environment's libexpat must be loaded before OCCT/build123d.
    # Re-exec first so the corrected library path applies to the preload.
    from cadgenbench.tools._cli import _reexec_with_environment_libraries

    _reexec_with_environment_libraries()
    import xml.parsers.expat  # noqa: F401

    args_list = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(args_list)
    try:
        run_dir, result = run_experiment(
            config_path=args.config,
            output_root=args.output_root,
            run_id=args.run_id,
            scorer_selection=args.scorer,
            region_ids=args.region,
            max_iterations_override=args.max_iterations,
            argv=args_list,
            scorer_variant=args.variant,
            nominal_only=args.nominal_only,
            jobs=args.jobs,
            render_all_candidates=args.render_all_candidates,
            resume=args.resume,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "run_dir": str(run_dir),
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
