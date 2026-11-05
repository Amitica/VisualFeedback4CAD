# Copyright 2026 Hugging Face
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Minimal CAD drawing projection, registration, and comparison.

The module deliberately separates three concerns:

* :func:`project_step` performs exact B-rep hidden-line removal for an
  explicitly selected view.  It never guesses the view.
* :func:`align_projection` estimates only a 2-D similarity transform
  (uniform scale, roll, translation, and optional reflection).
* :func:`compare_step_to_drawing` returns one score and one comparison image.
* :func:`compare_view_candidates` ranks LLM-proposed views internally, but
  exposes only the best score and comparison image.
* :func:`compare_step_to_drawings` combines several selected drawing views
  into one comparison sheet and one mean score.

No anisotropic scale, shear, or projective warp is allowed: those transforms
can make a geometrically wrong CAD model appear to match a drawing.
"""

from __future__ import annotations

import math
import re
import shutil
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from numbers import Integral, Real
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy import ndimage
from scipy.optimize import differential_evolution, linear_sum_assignment, minimize


@dataclass(frozen=True)
class ViewPreset:
    """World-space camera direction and screen-up vector.

    ``direction`` points from the look-at target towards the observer.
    """

    direction: tuple[float, float, float]
    up: tuple[float, float, float]


VIEW_PRESETS: dict[str, ViewPreset] = {
    "front": ViewPreset((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    "rear": ViewPreset((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "left": ViewPreset((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "right": ViewPreset((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "top": ViewPreset((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    # Preserve world +X left-to-right while switching to the underside.
    "bottom": ViewPreset((0.0, 0.0, -1.0), (0.0, -1.0, 0.0)),
    "iso": ViewPreset((1.0, -1.0, 1.0), (0.0, 0.0, 1.0)),
}


@dataclass(frozen=True)
class Projection2D:
    """Exact HLR shapes plus sampled polylines in projection coordinates."""

    view_name: str
    direction: tuple[float, float, float]
    up: tuple[float, float, float]
    visible_shape: object
    hidden_shape: object
    visible_polylines: tuple[np.ndarray, ...]
    hidden_polylines: tuple[np.ndarray, ...]
    bounds: tuple[float, float, float, float]
    section_origin: tuple[float, float, float] | None = None
    section_offset_fraction: float | None = None
    section_keep: str | None = None

    @property
    def width(self) -> float:
        return self.bounds[2] - self.bounds[0]

    @property
    def height(self) -> float:
        return self.bounds[3] - self.bounds[1]

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bounds
        return ((x0 + x1) * 0.5, (y0 + y1) * 0.5)


@dataclass(frozen=True)
class SimilarityTransform:
    """Projection-space to image-space similarity transform."""

    scale_px_per_unit: float
    rotation_deg: float
    center_x_px: float
    center_y_px: float
    model_center_x: float
    model_center_y: float
    reflected: bool = False

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Map ``(N,2)`` projection points to image pixel coordinates."""

        points = np.asarray(points, dtype=np.float64)
        centred = points - np.array(
            [self.model_center_x, self.model_center_y], dtype=np.float64
        )
        if self.reflected:
            centred = centred.copy()
            centred[:, 0] *= -1.0

        angle = math.radians(self.rotation_deg)
        cosine, sine = math.cos(angle), math.sin(angle)
        rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
        rotated = centred @ rotation.T
        # Projection coordinates are Y-up; raster coordinates are Y-down.
        pixels = rotated * self.scale_px_per_unit
        pixels[:, 1] *= -1.0
        pixels += np.array([self.center_x_px, self.center_y_px])
        return pixels


@dataclass(frozen=True)
class AlignmentMetrics:
    """Robust one-way projection-to-reference alignment diagnostics."""

    loss: float
    mean_distance_px: float
    median_distance_px: float
    p90_distance_px: float
    inlier_fraction_3px: float
    outside_fraction: float
    reference_threshold: int


@dataclass(frozen=True)
class AlignmentResult:
    transform: SimilarityTransform
    metrics: AlignmentMetrics


@dataclass(frozen=True)
class GeometryMetrics:
    """Candidate-comparison score after 2-D registration.

    ``score`` is in ``[0, 1]`` and higher is better.  It combines a robust
    CAD-to-reference distance with a lower-weight reference-to-CAD distance.
    The latter is deliberately conservative because the current reference
    layer still contains dimensions, text, and centre lines.
    """

    score: float
    bidirectional_loss: float
    cad_to_reference_loss: float
    reference_to_cad_loss: float


@dataclass(frozen=True)
class ScorerV2Config:
    """Central, versioned knobs for the deterministic envelope scorer.

    The legacy scorer above deliberately remains unchanged.  V2 first asks
    whether the projected part has the same gross silhouette as the drawing;
    only candidates that pass those checks receive an edge-match score.
    Keeping the values in one immutable object makes threshold sweeps and
    baseline comparisons reproducible instead of scattering magic constants
    through the optimiser.
    """

    work_max_dimension: int = 320
    canonical_size: int = 320
    relaxed_threshold_floor: int = 200
    background_margin: int = 24
    threshold_perturbation: int = 8
    closing_radius_fraction: float = 0.0025
    opening_radius_fraction: float = 0.0035
    closing_radius_px: int | None = None
    opening_radius_px: int | None = None
    coarse_opening_radius_fraction: float = 0.60
    coarse_opening_radius_perturbation: float = 0.05
    coarse_opening_min_radius_px: float = 3.0
    coarse_opening_min_area_retained: float = 0.75
    coarse_opening_min_bbox_reduction: float = 0.08
    coarse_opening_min_compactness_gain: float = 0.02
    coarse_opening_min_stability_iou: float = 0.95
    edge_band_fraction: float = 0.010
    min_envelope_area_fraction: float = 0.008
    min_bbox_fill_fraction: float = 0.08
    max_secondary_area_ratio: float = 0.70
    border_margin_fraction: float = 0.004
    min_threshold_stability_iou: float = 0.72
    aspect_log_gate: float = math.log(1.8)
    max_area_log_error: float = math.log(2.5)
    min_axis_fill: float = 0.30
    max_axis_fill: float = 2.50
    max_center_error: float = 0.30
    max_support_outside_fraction: float = 0.02
    min_envelope_dice: float = 0.10
    edge_tolerance_fraction: float = 0.004
    min_edge_tolerance_px: float = 2.0
    # Version 2 supplements tolerant pixel F1 with a capacity-aware spatial
    # edge-mass Dice and an unmatched overflow cell for all CAD pixels outside
    # the reference envelope.  Four phase origins make the 4x4 binning
    # insensitive to a feature landing on one particular cell boundary.
    # Changing this definition requires a new version so persisted score
    # traces remain replayable.
    spatial_edge_mass_version: int = 2
    spatial_edge_mass_grid_size: int = 4
    spatial_edge_mass_phase_offsets: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.5, 0.0),
        (0.0, 0.5),
        (0.5, 0.5),
    )
    # Experimental continuous, capacity-limited partial edge assignment.  The
    # reference and CAD masks are reduced to equal-mass spatial tokens, then a
    # globally optimal linear assignment decides which tokens match and which
    # remain unmatched.  It is deliberately independent of the coarse
    # envelope; enable it only through the dedicated experiment variant.
    use_capacity_transport: bool = False
    capacity_transport_version: int = 1
    capacity_transport_max_tokens: int = 256
    capacity_transport_work_max_tokens: int = 96
    capacity_transport_distance_fraction: float = 0.018
    capacity_transport_min_distance_px: float = 3.0
    capacity_transport_unmatched_cad_cost: float = 1.0
    capacity_transport_unmatched_reference_cost: float = 0.25
    # Turn this off only for the controlled no-envelope-match ablation.  The
    # default preserves the production V2 scorer exactly.  In the ablation,
    # the already-extracted reference edge evidence is held fixed while the
    # envelope is prevented from affecting registration, gates, or ranking.
    use_envelope_matching: bool = True
    envelope_weight: float = 0.50
    aspect_temperature: float = 0.35
    size_temperature: float = 0.45
    center_temperature: float = 0.18
    fallback_score_scale: float = 0.65
    translation_refine_fraction: float = 0.08
    refinement_scale_fraction: float = 0.08
    refinement_rotation_deg: float = 1.5
    refinement_translation_fraction: float = 0.03
    seed_translation_jitter_fraction: float = 0.01
    parameter_bound_tolerance_fraction: float = 0.01
    rotation_bound_tolerance_deg: float = 0.25


DEFAULT_SCORER_V2_CONFIG = ScorerV2Config()

# Production scorer used by the Pi view-search worker.  Keep this profile
# controller-owned: an Agent may propose cameras, but it cannot change how
# candidates are scored.
CAPACITY_TRANSPORT_SCORER_V1_CONFIG = replace(
    DEFAULT_SCORER_V2_CONFIG,
    use_capacity_transport=True,
    use_envelope_matching=False,
    capacity_transport_version=1,
    capacity_transport_max_tokens=256,
    capacity_transport_work_max_tokens=96,
    capacity_transport_distance_fraction=0.018,
    capacity_transport_min_distance_px=3.0,
    capacity_transport_unmatched_cad_cost=1.0,
    capacity_transport_unmatched_reference_cost=0.25,
)


@dataclass(frozen=True)
class EnvelopeDiagnosticsV2:
    """JSON-friendly facts about one deterministic envelope extraction."""

    stable: bool
    confidence: str
    failure_reason: str | None
    dark_threshold: int
    relaxed_threshold: int
    selected_threshold: int
    threshold_override: bool
    fallback_used: bool
    dark_relaxed_iou: float | None
    threshold_stability_iou: float | None
    bbox_px: tuple[int, int, int, int] | None
    centroid_px: tuple[float, float] | None
    area_px: int
    area_fraction: float
    component_count: int
    secondary_area_ratio: float
    touches_boundary: bool
    closing_radius_px: int
    opening_radius_px: int
    roi_px: tuple[int, int, int, int]
    coarse_cleanup_attempted: bool = False
    coarse_cleanup_used: bool = False
    coarse_cleanup_radius_px: float | None = None
    coarse_cleanup_area_retained: float | None = None
    coarse_cleanup_bbox_reduction: float | None = None
    coarse_cleanup_compactness_gain: float | None = None
    coarse_cleanup_stability_iou: float | None = None
    coarse_cleanup_failure_reason: str | None = None


@dataclass(frozen=True)
class EnvelopeGeometryV2:
    """Reference masks plus separately serialisable extraction diagnostics.

    Masks are local to ``diagnostics.roi_px``.  The diagnostics object contains
    no NumPy arrays and can therefore be persisted with ``dataclasses.asdict``.
    """

    ink_mask: np.ndarray
    envelope_mask: np.ndarray
    boundary_mask: np.ndarray
    edge_mask: np.ndarray
    diagnostics: EnvelopeDiagnosticsV2


@dataclass(frozen=True)
class EnvelopeMatchMetricsV2:
    """Gross silhouette agreement after applying one similarity transform."""

    dice: float
    iou: float
    aspect_error: float
    width_ratio: float
    height_ratio: float
    area_ratio: float
    area_log_error: float
    center_error: float
    support_outside_fraction: float


@dataclass(frozen=True)
class SpatialEdgeMassPhaseMetricsV2:
    """One phase of the capacity-aware spatial edge-mass comparison."""

    offset_x_fraction: float
    offset_y_fraction: float
    dice: float
    matched_edge_pixels: int
    reference_edge_pixels: int
    cad_edge_pixels: int
    cad_edge_pixels_inside_bbox: int
    cad_unmatched_overflow_pixels: int
    active_cells: int


@dataclass(frozen=True)
class SpatialEdgeMassMetricsV2:
    """Versioned, JSON-safe local edge-density diagnostics.

    ``bbox_px`` is local to the mask/ROI supplied to the scorer.  Each phase
    partitions that box into the configured grid after shifting its cell
    origin by a fraction of one cell.  Per-cell capacity is the smaller of
    the two edge counts, preventing multiple nearby CAD contours from all
    receiving credit for one reference contour.  CAD pixels outside the box
    remain in the score denominator through an unmatched overflow cell.
    """

    version: int
    grid_size: int
    bbox_px: tuple[int, int, int, int] | None
    score: float
    aggregation: str
    phases: tuple[SpatialEdgeMassPhaseMetricsV2, ...]


@dataclass(frozen=True)
class CapacityTransportMetricsV2:
    """Continuous partial one-to-one edge assignment diagnostics.

    Every representative token has one unit of capacity.  Matching a CAD
    token to a reference token consumes both capacities; leaving either token
    unmatched pays the corresponding asymmetric cost.
    """

    version: int
    score: float
    normalized_cost: float
    distance_scale_px: float
    unmatched_cad_cost: float
    unmatched_reference_cost: float
    reference_edge_pixels: int
    cad_edge_pixels: int
    reference_tokens: int
    cad_tokens: int
    matched_pairs: int
    unmatched_cad_tokens: int
    unmatched_reference_tokens: int
    total_cost: float
    matched_cad_fraction: float
    matched_reference_fraction: float
    mean_matched_distance_px: float
    p90_matched_distance_px: float


@dataclass(frozen=True)
class AlignmentTrialV2:
    """One replayable local optimisation start."""

    start_scale_px_per_unit: float
    start_rotation_deg: float
    start_center_x_px: float
    start_center_y_px: float
    start_offset_x_px: float
    start_offset_y_px: float
    coarse_scale_px_per_unit: float
    coarse_rotation_deg: float
    coarse_center_x_px: float
    coarse_center_y_px: float
    final_scale_px_per_unit: float
    final_rotation_deg: float
    final_center_x_px: float
    final_center_y_px: float
    final_offset_x_px: float
    final_offset_y_px: float
    scale_at_bound: bool
    rotation_at_limit: bool
    translation_at_limit: bool
    coarse_objective: float
    objective: float
    score: float
    raw_score: float
    eligible: bool
    failed_gates: tuple[str, ...]
    warnings: tuple[str, ...]
    metrics: EnvelopeMatchMetricsV2
    edge: EdgeMatchMetricsV2
    spatial_edge_mass: SpatialEdgeMassMetricsV2
    capacity_transport: CapacityTransportMetricsV2
    edge_score: float
    coarse_success: bool
    coarse_iterations: int | None
    coarse_evaluations: int
    coarse_message: str
    success: bool
    iterations: int | None
    evaluations: int
    message: str


@dataclass(frozen=True)
class AlignmentDiagnosticsV2:
    """Complete deterministic multi-start trace without image arrays."""

    work_image_size: tuple[int, int]
    work_scale: float
    seed: int
    seed_translation_jitter_fraction: float
    scale_seeds_px_per_unit: tuple[float, ...]
    rotation_seeds_deg: tuple[float, ...]
    selected_trial: int
    trials: tuple[AlignmentTrialV2, ...]
    mode: str
    confidence: str
    fallback_used: bool
    reference: EnvelopeDiagnosticsV2
    cad: EnvelopeDiagnosticsV2


@dataclass(frozen=True)
class AlignmentResultV2:
    transform: SimilarityTransform
    objective: float
    score: float
    raw_score: float
    eligible: bool
    failed_gates: tuple[str, ...]
    warnings: tuple[str, ...]
    mode: str
    confidence: str
    fallback_used: bool
    scale_at_bound: bool
    rotation_at_limit: bool
    translation_at_limit: bool
    metrics: EnvelopeMatchMetricsV2
    edge: EdgeMatchMetricsV2
    spatial_edge_mass: SpatialEdgeMassMetricsV2
    capacity_transport: CapacityTransportMetricsV2
    edge_score: float
    diagnostics: AlignmentDiagnosticsV2


@dataclass(frozen=True)
class EdgeMatchMetricsV2:
    """Symmetric tolerant edge precision/recall diagnostics."""

    tolerance_px: float
    precision: float
    recall: float
    f1: float
    cad_to_reference_mean_px: float
    reference_to_cad_mean_px: float
    cad_edge_pixels: int
    reference_edge_pixels: int


@dataclass(frozen=True)
class _ReferenceEdgeCacheV2:
    """Reference-only facts reused by every alignment objective evaluation."""

    shape: tuple[int, int]
    edge_pixels: int
    fallback_distance_px: float
    distance_to_reference: np.ndarray | None


@dataclass(frozen=True)
class _ReferenceSpatialEdgeMassCacheV2:
    """Reference-side spatial bins reused while only the CAD mask changes."""

    shape: tuple[int, int]
    version: int
    grid_size: int
    phase_offsets: tuple[tuple[float, float], ...]
    use_envelope_bbox: bool
    bbox_px: tuple[int, int, int, int] | None
    reference_edge_pixels: int
    reference_cells: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class _ReferenceCapacityTransportCacheV2:
    """Reference edge coordinates reused by transport objective evaluations."""

    shape: tuple[int, int]
    edge_pixels: int
    coordinates_xy: np.ndarray


@dataclass(frozen=True)
class GeometryMetricsV2:
    """Unified V2 score and every hard-gate decision."""

    score: float
    raw_score: float
    eligible: bool
    failed_gates: tuple[str, ...]
    warnings: tuple[str, ...]
    mode: str
    confidence: str
    fallback_used: bool
    envelope_score: float
    edge_score: float
    gross: EnvelopeMatchMetricsV2
    edge: EdgeMatchMetricsV2
    spatial_edge_mass: SpatialEdgeMassMetricsV2
    capacity_transport: CapacityTransportMetricsV2
    reference: EnvelopeDiagnosticsV2
    cad: EnvelopeDiagnosticsV2


@dataclass(frozen=True)
class ProjectionEvaluationV2:
    """High-level result consumed by candidate-ranking experiments."""

    transform: SimilarityTransform
    alignment: AlignmentResultV2
    metrics: GeometryMetricsV2


@dataclass(frozen=True)
class _EnvelopeComponentV2:
    mask: np.ndarray
    stable: bool
    failure_reason: str | None
    bbox: tuple[int, int, int, int] | None
    centroid: tuple[float, float] | None
    area: int
    component_count: int
    secondary_area_ratio: float
    touches_boundary: bool
    closing_radius: int
    opening_radius: int
    coarse_cleanup_attempted: bool = False
    coarse_cleanup_used: bool = False
    coarse_cleanup_radius: float | None = None
    coarse_cleanup_area_retained: float | None = None
    coarse_cleanup_bbox_reduction: float | None = None
    coarse_cleanup_compactness_gain: float | None = None
    coarse_cleanup_stability_iou: float | None = None
    coarse_cleanup_failure_reason: str | None = None


@dataclass(frozen=True)
class _CoarseCleanupV2:
    mask: np.ndarray
    used: bool
    failure_reason: str | None
    radius_px: float | None
    area_retained: float | None
    bbox_reduction: float | None
    compactness_gain: float | None
    stability_iou: float | None


@dataclass(frozen=True)
class _MaskScoreV2:
    score: float
    raw_score: float
    eligible: bool
    failed_gates: tuple[str, ...]
    warnings: tuple[str, ...]
    envelope_score: float
    edge_score: float
    gross: EnvelopeMatchMetricsV2
    edge: EdgeMatchMetricsV2
    spatial_edge_mass: SpatialEdgeMassMetricsV2
    capacity_transport: CapacityTransportMetricsV2


@dataclass(frozen=True)
class ComparisonResult:
    """The deliberately small Agent-facing return value."""

    score: float
    comparison_png: Path


@dataclass(frozen=True)
class ViewSpec:
    """A fully resolved, executable view manifest entry.

    Unlike :class:`ViewCandidate`, a ``ViewSpec`` always contains normalised
    camera vectors.  It is therefore safe to persist and replay after a view
    search.  A relative section offset is measured through the model's
    axis-aligned bounding box along ``direction``: ``0`` is the centre and
    values approach ``-1``/``1`` at the two supporting faces.
    """

    name: str
    direction: tuple[float, float, float]
    up: tuple[float, float, float]
    section_offset_fraction: float | None = None
    section_keep: str | None = None

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not name:
            raise ValueError("View spec name must not be empty")
        direction = _normalise3(self.direction)
        up = _normalise3(self.up)
        _validate_camera_basis(direction, up)
        offset = _validate_section_offset(self.section_offset_fraction)
        keep = _validate_section_keep(
            self.section_keep,
            has_section=offset is not None,
        )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "up", up)
        object.__setattr__(self, "section_offset_fraction", offset)
        object.__setattr__(self, "section_keep", keep)


@dataclass(frozen=True)
class ViewCandidate:
    """One LLM-proposed executable CAD view."""

    name: str
    direction: tuple[float, float, float] | None = None
    up: tuple[float, float, float] | None = None
    section_origin: tuple[float, float, float] | None = None
    section_offset_fraction: float | None = None
    section_keep: str | None = None


@dataclass(frozen=True)
class DrawingViewSpec:
    """One already-selected drawing view in a multi-view comparison."""

    name: str
    reference_image: str | Path
    view: str
    direction: tuple[float, float, float] | None = None
    up: tuple[float, float, float] | None = None
    section_origin: tuple[float, float, float] | None = None
    section_offset_fraction: float | None = None
    section_keep: str | None = None
    roi: tuple[int, int, int, int] | None = None
    allow_reflection: bool = False
    max_rotation_deg: float = 6.0
    min_roi_fill: float = 0.70


@dataclass(frozen=True)
class RankedViewCandidate:
    """One fully evaluated entry in a deterministic candidate ranking."""

    rank: int
    name: str
    score: float
    comparison_png: Path
    transform: SimilarityTransform
    spec: ViewSpec
    alignment_metrics: AlignmentMetrics
    geometry_metrics: GeometryMetrics

    @property
    def comparison_path(self) -> Path:
        """Alias used by JSON workers that call every artifact a path."""

        return self.comparison_png


@dataclass(frozen=True)
class CandidateRankingResult:
    """Complete candidate ranking plus convenient best/top-three fields."""

    rankings: tuple[RankedViewCandidate, ...]
    best: RankedViewCandidate
    top3: tuple[RankedViewCandidate, ...]
    margin: float
    score: float
    comparison_png: Path
    transform: SimilarityTransform
    spec: ViewSpec

    @property
    def comparison_path(self) -> Path:
        return self.comparison_png


def project_step(
    step_path: str | Path,
    *,
    view: str,
    direction: Sequence[float] | None = None,
    up: Sequence[float] | None = None,
    look_at: Sequence[float] | None = None,
    section_origin: Sequence[float] | None = None,
    section_offset_fraction: float | None = None,
    section_keep: str | None = None,
    sampling_deflection: float | None = None,
) -> Projection2D:
    """Project a STEP B-rep to exact visible/hidden edges.

    A named preset is normally supplied by the LLM.  ``direction`` and ``up``
    are escape hatches for a special view whose pose has already been refined
    by another tool.
    """

    step_path = Path(step_path)
    if not step_path.is_file():
        raise FileNotFoundError(f"STEP model not found: {step_path}")
    if not isinstance(view, str) or not view.strip():
        raise ValueError("View name must be a non-empty string")
    if (direction is None) != (up is None):
        raise ValueError("Custom camera overrides require both direction and up")
    if view not in VIEW_PRESETS and (direction is None or up is None):
        raise ValueError(
            f"Unknown view {view!r}; choose from {sorted(VIEW_PRESETS)} or "
            "supply both direction and up"
        )

    preset = VIEW_PRESETS.get(view)
    selected_direction = direction if direction is not None else preset.direction
    selected_up = up if up is not None else preset.up
    direction_tuple = _normalise3(selected_direction)
    up_tuple = _normalise3(selected_up)
    _validate_camera_basis(direction_tuple, up_tuple)
    look_at_tuple = _tuple3(look_at, name="look_at") if look_at is not None else None

    if section_origin is not None and section_offset_fraction is not None:
        raise ValueError(
            "Supply either section_origin or section_offset_fraction, not both"
        )
    relative_section_offset = _validate_section_offset(section_offset_fraction)
    has_section = section_origin is not None or relative_section_offset is not None
    resolved_section_keep = _validate_section_keep(
        section_keep,
        has_section=has_section,
    )

    # Import lazily so `cadgenbench --help` does not load OCCT.
    from build123d import Compound, Keep, Plane, import_step
    from build123d.exporters import Drawing

    part = import_step(str(step_path))
    try:
        wrapped = None if part is None else part.wrapped
    except AssertionError:
        # build123d 0.11 represents a failed STEP read as an empty Compound
        # whose `wrapped` property asserts. Convert that kernel detail into a
        # stable tool-level validation error for the calling agent.
        wrapped = None
    if wrapped is None:
        raise ValueError(f"STEP model contains no drawable shape: {step_path}")

    section_origin_tuple: tuple[float, float, float] | None = None
    if has_section:
        if section_origin is not None:
            section_origin_tuple = _tuple3(section_origin, name="section origin")
        else:
            section_origin_tuple = _relative_section_origin(
                part,
                direction_tuple,
                relative_section_offset,
            )
        direction_array = np.asarray(direction_tuple)
        up_array = np.asarray(up_tuple)
        plane_x = up_array - np.dot(up_array, direction_array) * direction_array
        plane_x /= np.linalg.norm(plane_x)
        cutting_plane = Plane(
            origin=section_origin_tuple,
            x_dir=tuple(float(value) for value in plane_x),
            z_dir=direction_tuple,
        )
        keep_mode = Keep.TOP if resolved_section_keep == "front" else Keep.BOTTOM
        sectioned = part.split(cutting_plane, keep=keep_mode)
        if sectioned is None:
            raise ValueError(
                f"Section plane at {section_origin_tuple} does not intersect the model"
            )
        if isinstance(sectioned, list):
            part = Compound(children=sectioned)
        else:
            part = sectioned

    drawing = Drawing(
        part,
        look_at=look_at_tuple,
        look_from=direction_tuple,
        look_up=up_tuple,
        with_hidden=True,
        focus=None,
    )
    visible_shape = drawing.visible_lines
    hidden_shape = drawing.hidden_lines
    if not visible_shape.edges():
        raise ValueError(f"HLR projection produced no visible edges: {step_path}")

    bbox = visible_shape.bounding_box()
    diagonal = max(math.hypot(bbox.size.X, bbox.size.Y), 1e-6)
    if sampling_deflection is None:
        # At most ~0.3 output pixels at a typical 1000 px fit, with a floor
        # that prevents pathological edge counts on tiny model-unit shapes.
        sampling_deflection = max(diagonal / 3000.0, 0.01)
    elif (
        isinstance(sampling_deflection, bool)
        or not isinstance(sampling_deflection, Real)
        or not math.isfinite(float(sampling_deflection))
        or sampling_deflection <= 0.0
    ):
        raise ValueError("sampling_deflection must be a finite positive number")

    visible_polylines = _sample_edges(
        visible_shape.edges(), deflection=sampling_deflection
    )
    hidden_polylines = _sample_edges(
        hidden_shape.edges(), deflection=sampling_deflection
    )
    all_points = np.concatenate(visible_polylines, axis=0)
    bounds = (
        float(all_points[:, 0].min()),
        float(all_points[:, 1].min()),
        float(all_points[:, 0].max()),
        float(all_points[:, 1].max()),
    )
    return Projection2D(
        view_name=view,
        direction=direction_tuple,
        up=up_tuple,
        visible_shape=visible_shape,
        hidden_shape=hidden_shape,
        visible_polylines=visible_polylines,
        hidden_polylines=hidden_polylines,
        bounds=bounds,
        section_origin=section_origin_tuple,
        section_offset_fraction=relative_section_offset,
        section_keep=resolved_section_keep,
    )


def export_projection_svg(projection: Projection2D, output: str | Path) -> Path:
    """Write exact visible/hidden OCCT curves as a layered SVG."""

    from build123d import ExportSVG, LineType

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    exporter = ExportSVG(scale=1.0, margin=5.0, line_weight=0.35)
    exporter.add_layer(
        "Visible",
        line_color=(0, 0, 0),
        line_weight=0.35,
    )
    exporter.add_layer(
        "Hidden",
        line_color=(125, 135, 150),
        line_weight=0.25,
        line_type=LineType.HIDDEN,
    )
    exporter.add_shape(projection.visible_shape, layer="Visible")
    if projection.section_origin is None and projection.hidden_shape.edges():
        exporter.add_shape(projection.hidden_shape, layer="Hidden")
    exporter.write(str(output))
    return output


def normalized_roi_to_pixels(
    roi: Sequence[float],
    image_size: Sequence[int],
) -> tuple[int, int, int, int]:
    """Convert a 0--1000 ``[left, top, right, bottom]`` ROI to pixels.

    The returned tuple follows Pillow's ``(left, top, right, bottom)`` crop
    convention.  The leading edges are rounded down and trailing edges up so
    a valid non-empty normalised region never disappears on a small image.
    """

    values = np.asarray(roi, dtype=np.float64)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError(
            "Normalized ROI must contain four finite values: [left, top, right, bottom]"
        )
    left, top, right, bottom = (float(value) for value in values)
    if not (0.0 <= left < right <= 1000.0 and 0.0 <= top < bottom <= 1000.0):
        raise ValueError(
            "Normalized ROI must satisfy "
            "0 <= left < right <= 1000 and 0 <= top < bottom <= 1000"
        )

    if len(image_size) != 2:
        raise ValueError(
            f"Image size must contain width and height, got {image_size!r}"
        )
    width, height = image_size
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, Integral)
        or not isinstance(height, Integral)
        or width <= 0
        or height <= 0
    ):
        raise ValueError(
            f"Image dimensions must be positive integers, got {image_size!r}"
        )

    pixel_left = math.floor(left * int(width) / 1000.0)
    pixel_top = math.floor(top * int(height) / 1000.0)
    pixel_right = math.ceil(right * int(width) / 1000.0)
    pixel_bottom = math.ceil(bottom * int(height) / 1000.0)
    return (
        max(0, min(pixel_left, int(width) - 1)),
        max(0, min(pixel_top, int(height) - 1)),
        max(1, min(pixel_right, int(width))),
        max(1, min(pixel_bottom, int(height))),
    )


def crop_normalized_region(
    reference_image: str | Path | Image.Image,
    roi: Sequence[float],
    output_path: str | Path | None = None,
) -> Image.Image | Path:
    """Crop one region from a merged drawing without changing the source.

    ``roi`` uses the image-model-friendly 0--1000 coordinate system and the
    ``[left, top, right, bottom]`` convention.  If ``output_path`` is omitted,
    an independent in-memory RGB image is returned; otherwise the crop is
    saved and the resulting path is returned.
    """

    reference = _open_reference(reference_image)
    pixel_box = normalized_roi_to_pixels(roi, reference.size)
    cropped = reference.crop(pixel_box)
    if output_path is None:
        return cropped
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(output, format="PNG", optimize=True)
    return output


# A readable alias for callers that use "ROI" rather than "region".
crop_normalized_roi = crop_normalized_region


def render_projection_png(
    projection: Projection2D,
    output_path: str | Path,
    *,
    width: int = 1200,
    height: int = 1200,
    margin: int = 48,
    show_hidden_lines: bool = True,
) -> Path:
    """Render an exact HLR projection as a clean, fitted white-background PNG."""

    width, height, margin = _validate_render_size(width, height, margin)
    transform = _fit_projection_transform(
        projection,
        image_size=(width, height),
        margin=margin,
    )
    image = _render_cad_view(
        (width, height),
        projection,
        transform,
        show_hidden_lines=show_hidden_lines,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True)
    return output


def render_step_view(
    step_path: str | Path,
    output_path: str | Path,
    *,
    view: str,
    direction: Sequence[float] | None = None,
    up: Sequence[float] | None = None,
    section_origin: Sequence[float] | None = None,
    section_offset_fraction: float | None = None,
    section_keep: str | None = None,
    width: int = 1200,
    height: int = 1200,
    margin: int = 48,
    show_hidden_lines: bool = True,
) -> Path:
    """Project a STEP model and write a standalone engineering-line PNG."""

    projection = project_step(
        step_path,
        view=view,
        direction=direction,
        up=up,
        section_origin=section_origin,
        section_offset_fraction=section_offset_fraction,
        section_keep=section_keep,
    )
    return render_projection_png(
        projection,
        output_path,
        width=width,
        height=height,
        margin=margin,
        show_hidden_lines=show_hidden_lines,
    )


def align_projection(
    projection: Projection2D,
    reference_image: str | Path | Image.Image,
    *,
    roi: tuple[int, int, int, int] | None = None,
    allow_reflection: bool = False,
    max_rotation_deg: float = 6.0,
    threshold: int | None = None,
    min_roi_fill: float = 0.70,
    seed: int = 0,
    max_iterations: int = 45,
) -> AlignmentResult:
    """Align visible CAD edges to a raster drawing with a similarity transform.

    The objective is a robust, directed CAD-to-reference edge distance.  Extra
    dimension/text pixels in the reference therefore do not directly count as
    missing CAD geometry.  ``roi`` is optional but strongly recommended when
    the supplied crop contains dense annotations far away from the part.
    """

    reference = _open_reference(reference_image)
    image_width, image_height = reference.size
    validated_roi = _validate_pixel_roi(roi, reference.size)
    x0, y0, roi_width, roi_height = validated_roi or (
        0,
        0,
        image_width,
        image_height,
    )
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, Integral)
        or max_iterations <= 0
    ):
        raise ValueError("max_iterations must be a positive integer")
    if not math.isfinite(float(max_rotation_deg)) or max_rotation_deg < 0.0:
        raise ValueError("max_rotation_deg must be a finite non-negative number")

    mask, used_threshold = _reference_mask(
        reference,
        threshold=threshold,
        roi=validated_roi,
    )
    if not np.any(mask):
        raise ValueError("Reference drawing contains no dark edge pixels")
    distance_map = ndimage.distance_transform_edt(~mask)
    points = _alignment_points(projection.visible_polylines, max_points=6000)

    fit_scale = min(
        roi_width / max(projection.width, 1e-9),
        roi_height / max(projection.height, 1e-9),
    )
    if not 0.0 < min_roi_fill <= 1.08:
        raise ValueError(f"min_roi_fill must be in (0, 1.08], got {min_roi_fill}")
    if validated_roi is None:
        # Whole-sheet mode is intentionally conservative. If the actual view
        # occupies less than roughly 40% of either image dimension, the
        # caller should provide an ROI; otherwise dimension/text clusters can
        # look like a plausible miniature projection.
        scale_bounds = (fit_scale * 0.44, fit_scale * 1.08)
    else:
        # The ROI contract is a rough *part* box, not merely a search window.
        # Requiring the projection to occupy most of it prevents a common
        # false minimum where a complete CAD outline shrinks onto one dense
        # internal feature or a cluster of dimension lines.
        scale_bounds = (fit_scale * min_roi_fill, fit_scale * 1.08)
    centre_x_bounds = (float(x0), float(x0 + roi_width))
    centre_y_bounds = (float(y0), float(y0 + roi_height))
    parameter_bounds = [
        (math.log(scale_bounds[0]), math.log(scale_bounds[1])),
        (-float(max_rotation_deg), float(max_rotation_deg)),
        centre_x_bounds,
        centre_y_bounds,
    ]

    model_center = projection.center
    cap_px = max(8.0, 0.025 * max(image_width, image_height))
    candidates: list[AlignmentResult] = []
    reflection_options = (False, True) if allow_reflection else (False,)
    for reflected in reflection_options:
        objective = _alignment_objective(
            points,
            model_center=model_center,
            reflected=reflected,
            distance_map=distance_map,
            cap_px=cap_px,
        )
        global_result = differential_evolution(
            objective,
            parameter_bounds,
            seed=seed,
            maxiter=max_iterations,
            popsize=8,
            tol=0.003,
            polish=False,
            updating="immediate",
            workers=1,
        )
        local_result = minimize(
            objective,
            global_result.x,
            method="Powell",
            bounds=parameter_bounds,
            options={"maxiter": 180, "xtol": 1e-4, "ftol": 1e-5},
        )
        params = (
            local_result.x if local_result.fun <= global_result.fun else global_result.x
        )
        transform = SimilarityTransform(
            scale_px_per_unit=float(math.exp(params[0])),
            rotation_deg=float(params[1]),
            center_x_px=float(params[2]),
            center_y_px=float(params[3]),
            model_center_x=float(model_center[0]),
            model_center_y=float(model_center[1]),
            reflected=reflected,
        )
        metrics = _alignment_metrics(
            transform.apply(points), distance_map, used_threshold, cap_px
        )
        candidates.append(AlignmentResult(transform=transform, metrics=metrics))

    return min(candidates, key=lambda candidate: candidate.metrics.loss)


def score_aligned_geometry(
    projection: Projection2D,
    transform: SimilarityTransform,
    reference_image: str | Path | Image.Image,
    *,
    roi: tuple[int, int, int, int] | None = None,
    threshold: int | None = None,
    extra_distance_px: float = 8.0,
    max_reference_points: int = 30_000,
) -> GeometryMetrics:
    """Score an already aligned candidate.

    Candidate CAD edges are sampled approximately uniformly by arc length, so
    their contribution approximates edge length. Reference ink is sampled only
    inside ``roi`` and receives a conservative 20% weight because it can still
    contain dimensions, text, and centre lines.
    """

    if extra_distance_px <= 0.0:
        raise ValueError("extra_distance_px must be positive")
    reference = _open_reference(reference_image)
    validated_roi = _validate_pixel_roi(roi, reference.size)
    reference_mask, _ = _reference_mask(
        reference,
        threshold=threshold,
        roi=validated_roi,
    )
    if not np.any(reference_mask):
        raise ValueError("Reference drawing contains no dark edge pixels")

    image_width, image_height = reference.size
    cap_px = max(extra_distance_px, 0.025 * max(image_width, image_height))
    reference_distance_map = ndimage.distance_transform_edt(~reference_mask)

    cad_points = _alignment_points(projection.visible_polylines, max_points=6000)
    cad_pixels = transform.apply(cad_points)
    valid = (
        (cad_pixels[:, 0] >= 0.0)
        & (cad_pixels[:, 0] <= image_width - 1.0)
        & (cad_pixels[:, 1] >= 0.0)
        & (cad_pixels[:, 1] <= image_height - 1.0)
    )
    cad_distances = np.full(len(cad_pixels), cap_px, dtype=np.float64)
    if np.any(valid):
        valid_pixels = cad_pixels[valid]
        cad_distances[valid] = ndimage.map_coordinates(
            reference_distance_map,
            [valid_pixels[:, 1], valid_pixels[:, 0]],
            order=1,
            mode="constant",
            cval=cap_px,
        )
    cad_distances = np.minimum(cad_distances, cap_px)

    cad_mask = _rasterize_visible_edges(
        reference.size,
        projection.visible_polylines,
        transform,
    )
    cad_distance_map = ndimage.distance_transform_edt(~cad_mask)
    reference_yx = np.argwhere(reference_mask)
    if len(reference_yx) > max_reference_points:
        indices = np.linspace(0, len(reference_yx) - 1, max_reference_points).astype(
            np.int64
        )
        reference_yx = reference_yx[indices]
    reference_distances = cad_distance_map[
        reference_yx[:, 0], reference_yx[:, 1]
    ].astype(np.float64)
    reference_distances = np.minimum(reference_distances, cap_px)

    cad_loss = float(np.mean(cad_distances) / cap_px)
    reference_loss = float(np.mean(reference_distances) / cap_px)
    bidirectional_loss = 0.80 * cad_loss + 0.20 * reference_loss
    return GeometryMetrics(
        score=float(np.clip(1.0 - bidirectional_loss, 0.0, 1.0)),
        bidirectional_loss=bidirectional_loss,
        cad_to_reference_loss=cad_loss,
        reference_to_cad_loss=reference_loss,
    )


def extract_envelope_geometry_v2(
    reference_image: str | Path | Image.Image,
    *,
    roi: tuple[int, int, int, int] | None = None,
    threshold: int | None = None,
    config: ScorerV2Config = DEFAULT_SCORER_V2_CONFIG,
) -> EnvelopeGeometryV2:
    """Extract a deterministic part envelope while retaining drawing ink.

    ``ink_mask`` and ``edge_mask`` still contain dimension text and leaders.
    The geometry-only signal is the filled envelope recovered from closed
    contours.  Automatic mode compares a dark Otsu mask with a relaxed mask,
    because real CAD drawings frequently render part geometry in light grey
    while annotations remain black.  Confidence and every fallback decision
    are explicit in :class:`EnvelopeDiagnosticsV2`.
    """

    _validate_scorer_v2_config(config)
    reference = _open_reference(reference_image)
    validated_roi = _validate_pixel_roi(roi, reference.size)
    roi_x, roi_y, roi_width, roi_height = validated_roi or (
        0,
        0,
        reference.width,
        reference.height,
    )
    gray_full = np.asarray(ImageOps.grayscale(reference), dtype=np.uint8)
    gray = gray_full[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]

    threshold_override = threshold is not None
    if threshold_override:
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, Integral)
            or not 0 <= int(threshold) <= 255
        ):
            raise ValueError(
                f"Reference threshold must be an integer in [0, 255], got {threshold}"
            )
        dark_threshold = relaxed_threshold = int(threshold)
    else:
        dark_threshold = max(_otsu_threshold(gray), 96)
        background = round(float(np.percentile(gray, 95.0)))
        relaxed_target = max(dark_threshold, config.relaxed_threshold_floor)
        relaxed_cap = max(dark_threshold, background - config.background_margin)
        relaxed_threshold = int(
            np.clip(min(relaxed_target, relaxed_cap), dark_threshold, 255)
        )

    dark_ink = gray <= dark_threshold
    relaxed_ink = gray <= relaxed_threshold
    dark_component = _extract_envelope_component_v2(dark_ink, config=config)
    relaxed_component = _extract_envelope_component_v2(relaxed_ink, config=config)
    dark_relaxed_iou = _binary_iou(
        dark_component.mask,
        relaxed_component.mask,
    )

    if threshold_override:
        selected = relaxed_component
        selected_ink = relaxed_ink
        selected_threshold = relaxed_threshold
        fallback_used = False
        threshold_stability_iou = 1.0 if selected.stable else None
        confidence = "explicit" if selected.stable else "none"
    elif relaxed_component.stable:
        selected = relaxed_component
        selected_ink = relaxed_ink
        selected_threshold = relaxed_threshold
        fallback_used = (
            not dark_component.stable
            or dark_relaxed_iou is None
            or dark_relaxed_iou < config.min_threshold_stability_iou
        )
        perturbation_ious: list[float] = []
        delta = config.threshold_perturbation
        for perturbed_threshold in {
            max(dark_threshold, relaxed_threshold - delta),
            min(255, relaxed_threshold + delta),
        }:
            perturbed = _extract_envelope_component_v2(
                gray <= perturbed_threshold,
                config=config,
            )
            similarity = _binary_iou(selected.mask, perturbed.mask)
            perturbation_ious.append(similarity if similarity is not None else 0.0)
        threshold_stability_iou = min(perturbation_ious, default=1.0)
        perturbation_stable = (
            threshold_stability_iou >= config.min_threshold_stability_iou
        )
        agrees_with_dark = (
            dark_component.stable
            and dark_relaxed_iou is not None
            and dark_relaxed_iou >= config.min_threshold_stability_iou
        )
        confidence = (
            "high"
            if perturbation_stable and agrees_with_dark
            else "medium"
            if perturbation_stable
            else "low"
        )
    elif dark_component.stable:
        selected = dark_component
        selected_ink = dark_ink
        selected_threshold = dark_threshold
        fallback_used = False
        threshold_stability_iou = dark_relaxed_iou
        confidence = "low"
    else:
        selected = (
            relaxed_component
            if relaxed_component.area >= dark_component.area
            else dark_component
        )
        selected_ink = relaxed_ink if selected is relaxed_component else dark_ink
        selected_threshold = (
            relaxed_threshold if selected is relaxed_component else dark_threshold
        )
        fallback_used = selected is relaxed_component
        threshold_stability_iou = dark_relaxed_iou
        confidence = "none"

    envelope = selected.mask.astype(bool, copy=False)
    if np.any(envelope):
        boundary = envelope & ~ndimage.binary_erosion(
            envelope,
            structure=_disk_structure(1),
            border_value=0,
        )
        edge_radius = max(
            1,
            round(config.edge_band_fraction * max(gray.shape)),
        )
        edge_support = ndimage.binary_dilation(
            envelope,
            structure=_disk_structure(edge_radius),
        )
        edge_mask = selected_ink & edge_support if selected.stable else selected_ink
    else:
        boundary = np.zeros_like(envelope)
        # A failed closed-envelope extraction still has a useful low-confidence
        # signal: symmetric tolerant distances over the original ink.  Keeping
        # it here lets the caller return a ranked fallback instead of aborting.
        edge_mask = selected_ink

    bbox = selected.bbox
    centroid = selected.centroid
    absolute_bbox = (
        None
        if bbox is None
        else (
            bbox[0] + roi_x,
            bbox[1] + roi_y,
            bbox[2] + roi_x,
            bbox[3] + roi_y,
        )
    )
    absolute_centroid = (
        None if centroid is None else (centroid[0] + roi_x, centroid[1] + roi_y)
    )
    diagnostics = EnvelopeDiagnosticsV2(
        stable=selected.stable,
        confidence=confidence,
        failure_reason=selected.failure_reason if not selected.stable else None,
        dark_threshold=dark_threshold,
        relaxed_threshold=relaxed_threshold,
        selected_threshold=selected_threshold,
        threshold_override=threshold_override,
        fallback_used=fallback_used,
        dark_relaxed_iou=dark_relaxed_iou,
        threshold_stability_iou=threshold_stability_iou,
        bbox_px=absolute_bbox,
        centroid_px=absolute_centroid,
        area_px=selected.area,
        area_fraction=float(selected.area / max(gray.size, 1)),
        component_count=selected.component_count,
        secondary_area_ratio=selected.secondary_area_ratio,
        touches_boundary=selected.touches_boundary,
        closing_radius_px=selected.closing_radius,
        opening_radius_px=selected.opening_radius,
        roi_px=(roi_x, roi_y, roi_width, roi_height),
        coarse_cleanup_attempted=selected.coarse_cleanup_attempted,
        coarse_cleanup_used=selected.coarse_cleanup_used,
        coarse_cleanup_radius_px=selected.coarse_cleanup_radius,
        coarse_cleanup_area_retained=selected.coarse_cleanup_area_retained,
        coarse_cleanup_bbox_reduction=selected.coarse_cleanup_bbox_reduction,
        coarse_cleanup_compactness_gain=(selected.coarse_cleanup_compactness_gain),
        coarse_cleanup_stability_iou=selected.coarse_cleanup_stability_iou,
        coarse_cleanup_failure_reason=selected.coarse_cleanup_failure_reason,
    )
    return EnvelopeGeometryV2(
        ink_mask=selected_ink.astype(bool, copy=False),
        envelope_mask=envelope,
        boundary_mask=boundary,
        edge_mask=edge_mask.astype(bool, copy=False),
        diagnostics=diagnostics,
    )


def align_projection_v2(
    projection: Projection2D,
    reference_image: str | Path | Image.Image,
    *,
    roi: tuple[int, int, int, int] | None = None,
    threshold: int | None = None,
    reference_geometry: EnvelopeGeometryV2 | None = None,
    max_rotation_deg: float = 6.0,
    seed: int = 0,
    max_iterations: int = 45,
    config: ScorerV2Config = DEFAULT_SCORER_V2_CONFIG,
) -> AlignmentResultV2:
    """Register a projection to the deterministic reference envelope.

    Centroid matching supplies the initial translation.  In production mode,
    complete-support canvas placements supply additional feasible starts.
    A deterministic
    multi-start optimisation then refines uniform scale, small scan roll, and
    a bounded local translation with the same raw envelope-plus-edge score
    used for final ranking.  Hard gates decide eligibility after refinement.
    If either closed envelope is unreliable, the function returns a labelled
    low-confidence symmetric-edge fallback instead of aborting the candidate.
    """

    _validate_scorer_v2_config(config)
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, Integral)
        or max_iterations <= 0
    ):
        raise ValueError("max_iterations must be a positive integer")
    if not math.isfinite(float(max_rotation_deg)) or max_rotation_deg < 0.0:
        raise ValueError("max_rotation_deg must be a finite non-negative number")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer")
    seed = int(seed)

    reference = _open_reference(reference_image)
    geometry = reference_geometry or extract_envelope_geometry_v2(
        reference,
        roi=roi,
        threshold=threshold,
        config=config,
    )
    _validate_reference_geometry_v2(geometry, reference.size, roi=roi)
    canonical_transform, canonical_geometry = _canonical_cad_geometry_v2(
        projection,
        config=config,
    )
    support_points = _alignment_points(
        projection.visible_polylines,
        max_points=6000,
    )
    if config.use_capacity_transport:
        mode = "capacity_transport"
    elif not config.use_envelope_matching:
        mode = "no_envelope_match"
    else:
        mode = (
            "envelope"
            if geometry.diagnostics.stable and canonical_geometry.diagnostics.stable
            else "edge_fallback"
        )
    confidence = (
        "not_applicable"
        if mode in {"no_envelope_match", "capacity_transport"}
        else geometry.diagnostics.confidence
        if mode == "envelope"
        else "low"
    )
    fallback_used = (
        mode == "edge_fallback" or geometry.diagnostics.fallback_used
        if mode not in {"no_envelope_match", "capacity_transport"}
        else False
    )

    roi_x, roi_y, roi_width, roi_height = geometry.diagnostics.roi_px
    work_scale = min(1.0, config.work_max_dimension / max(roi_width, roi_height))
    work_size = (
        max(8, round(roi_width * work_scale)),
        max(8, round(roi_height * work_scale)),
    )
    reference_work_envelope = _resize_binary_mask(
        geometry.envelope_mask,
        work_size,
    )
    reference_edge_mask = (
        geometry.ink_mask
        if mode == "capacity_transport"
        else geometry.edge_mask
        if mode == "no_envelope_match" or geometry.diagnostics.stable
        else geometry.ink_mask
    )
    reference_work_edges = _resize_binary_mask(reference_edge_mask, work_size)
    cad_work_envelope = canonical_geometry.envelope_mask
    cad_work_edges = canonical_geometry.edge_mask
    reference_work_edge_cache = _prepare_reference_edge_cache_v2(reference_work_edges)
    reference_work_spatial_cache = (
        None
        if mode == "capacity_transport"
        else _prepare_reference_spatial_edge_mass_cache_v2(
            reference_work_envelope,
            reference_work_edges,
            use_envelope_bbox=mode != "no_envelope_match",
            config=config,
        )
    )
    reference_work_transport_cache = _prepare_reference_capacity_transport_cache_v2(
        reference_work_edges
    )
    reference_seed_mask = (
        reference_work_edges
        if mode in {"no_envelope_match", "capacity_transport"}
        else reference_work_envelope
        if geometry.diagnostics.stable
        else reference_work_edges
    )
    # Always seed from the complete visible-edge support.  A closed local hole
    # is useful for Dice but must never define the projection's gross extent.
    cad_seed_mask = cad_work_edges
    reference_stats = (
        _robust_edge_stats_v2(reference_seed_mask)
        if mode == "capacity_transport"
        else _mask_stats(reference_seed_mask)
    )
    cad_stats = _mask_stats(cad_seed_mask)
    if reference_stats is None:
        raise ValueError("Reference drawing contains no usable ink for V2 alignment")
    if cad_stats is None:
        raise ValueError("Projection contains no usable edges for V2 alignment")

    canonical_anchor_model = _inverse_transform_point(
        canonical_transform,
        cad_stats["centroid"],
    )
    canonical_scale = canonical_transform.scale_px_per_unit
    width_scale = canonical_scale * reference_stats["width"] / cad_stats["width"]
    height_scale = canonical_scale * reference_stats["height"] / cad_stats["height"]
    scale_seeds_work = _unique_positive_values(
        (math.sqrt(width_scale * height_scale), width_scale, height_scale)
    )
    lower_scale = min(scale_seeds_work) * 0.72
    upper_scale = max(scale_seeds_work) * 1.28
    if upper_scale <= lower_scale:
        upper_scale = lower_scale * 1.01
    rotation_seeds = (
        (0.0,)
        if max_rotation_deg == 0.0
        else _unique_finite_values((-max_rotation_deg, 0.0, max_rotation_deg))
    )
    parameter_bounds = [
        (math.log(lower_scale), math.log(upper_scale)),
        (-float(max_rotation_deg), float(max_rotation_deg)),
        (
            -config.translation_refine_fraction * reference_stats["width"],
            config.translation_refine_fraction * reference_stats["width"],
        ),
        (
            -config.translation_refine_fraction * reference_stats["height"],
            config.translation_refine_fraction * reference_stats["height"],
        ),
    ]
    tolerance_work = max(
        1.0,
        config.min_edge_tolerance_px * work_scale,
        config.edge_tolerance_fraction * max(work_size),
    )
    translation_origin = reference_stats["centroid"]

    def evaluate_parameters(
        parameters: np.ndarray,
    ) -> tuple[_MaskScoreV2, SimilarityTransform]:
        target_centroid = (
            float(translation_origin[0] + parameters[2]),
            float(translation_origin[1] + parameters[3]),
        )
        transform = _transform_centered_on_anchor(
            projection,
            scale=math.exp(float(parameters[0])),
            rotation_deg=float(parameters[1]),
            anchor_model=canonical_anchor_model,
            target_centroid=target_centroid,
        )
        candidate_envelope = _warp_binary_mask(
            cad_work_envelope,
            source_transform=canonical_transform,
            target_transform=transform,
            output_size=work_size,
        )
        candidate_edges = _warp_binary_mask(
            cad_work_edges,
            source_transform=canonical_transform,
            target_transform=transform,
            output_size=work_size,
        )
        mask_score = _score_mask_pair_v2(
            reference_work_envelope,
            reference_work_edges,
            candidate_envelope,
            candidate_edges,
            cad_support_stats=_transformed_support_stats_v2(
                support_points,
                transform,
                image_shape=(work_size[1], work_size[0]),
            ),
            tolerance_px=tolerance_work,
            reference_stable=geometry.diagnostics.stable,
            cad_stable=canonical_geometry.diagnostics.stable,
            mode=mode,
            config=config,
            reference_edge_cache=reference_work_edge_cache,
            reference_spatial_cache=reference_work_spatial_cache,
            reference_transport_cache=reference_work_transport_cache,
        )
        return mask_score, transform

    trial_records: list[AlignmentTrialV2] = []
    trial_transforms: list[SimilarityTransform] = []
    tolerance_full = max(
        config.min_edge_tolerance_px,
        config.edge_tolerance_fraction * max(roi_width, roi_height),
    )
    reference_full_edge_cache = _prepare_reference_edge_cache_v2(reference_edge_mask)
    reference_full_spatial_cache = (
        None
        if mode == "capacity_transport"
        else _prepare_reference_spatial_edge_mass_cache_v2(
            geometry.envelope_mask,
            reference_edge_mask,
            use_envelope_bbox=mode != "no_envelope_match",
            config=config,
        )
    )
    reference_full_transport_cache = _prepare_reference_capacity_transport_cache_v2(
        reference_edge_mask
    )
    start_pairs = [
        (scale_seed, rotation_seed)
        for scale_seed in scale_seeds_work
        for rotation_seed in rotation_seeds
    ]
    start_origins = [reference_stats["centroid"] for _ in start_pairs]
    support_starts: dict[int, str] = {}
    if mode == "capacity_transport":
        for scale_seed in scale_seeds_work:
            provisional = _transform_centered_on_anchor(
                projection,
                scale=scale_seed,
                rotation_deg=0.0,
                anchor_model=canonical_anchor_model,
                target_centroid=(0.0, 0.0),
            )
            for preferred, label in (
                (reference_stats["centroid"], "nearest-support seed"),
                (None, "canvas-support seed"),
            ):
                target = _support_translation_origin_v2(
                    support_points,
                    provisional,
                    image_shape=(work_size[1], work_size[0]),
                    preferred_target=preferred,
                )
                if target is None or any(
                    pair == (scale_seed, 0.0) and np.allclose(origin, target)
                    for pair, origin in zip(start_pairs, start_origins, strict=True)
                ):
                    continue
                support_starts[len(start_pairs)] = label
                start_pairs.append((scale_seed, 0.0))
                start_origins.append(target)
    # Preserve the established full-grid order for a given seed.  A separate
    # deterministic stream perturbs only the non-anchor translations, so the
    # seed test changes actual starting states without changing grid coverage.
    start_order = np.random.default_rng(seed).permutation(len(start_pairs))
    jitter_random = np.random.default_rng([seed & 0xFFFFFFFF, 0xC0A25])
    anchor_index = min(
        range(len(start_pairs)),
        key=lambda index: (
            abs(math.log(start_pairs[index][0] / scale_seeds_work[0])),
            abs(start_pairs[index][1]),
        ),
    )
    bounded_jitter_fraction = min(
        config.seed_translation_jitter_fraction,
        0.5 * config.translation_refine_fraction,
    )
    jitter_x = bounded_jitter_fraction * float(reference_stats["width"])
    jitter_y = bounded_jitter_fraction * float(reference_stats["height"])
    start_offsets = [
        (0.0, 0.0)
        if index == anchor_index or index in support_starts
        else (
            float(jitter_random.uniform(-jitter_x, jitter_x)),
            float(jitter_random.uniform(-jitter_y, jitter_y)),
        )
        for index in range(len(start_pairs))
    ]
    for start_index in start_order:
        scale_seed, rotation_seed = start_pairs[int(start_index)]
        translation_origin = start_origins[int(start_index)]
        start_kind = support_starts.get(int(start_index), "")
        if start_kind:
            start_kind += "; "
        start_offset_x, start_offset_y = start_offsets[int(start_index)]
        initial = np.asarray(
            (
                math.log(scale_seed),
                rotation_seed,
                start_offset_x,
                start_offset_y,
            ),
            dtype=np.float64,
        )
        initial_score, initial_work_transform = evaluate_parameters(initial)

        def coarse_objective(parameters: np.ndarray) -> float:
            score, _ = evaluate_parameters(parameters)
            # Stage one cannot be pulled toward dimension text or internal
            # drawing ink.  With a stable envelope it uses only gross shape;
            # an unstable-reference fallback necessarily uses edge F1.
            signal = score.envelope_score if mode == "envelope" else score.raw_score
            return 1.0 - signal

        coarse_budget = max(1, math.ceil(max_iterations * 0.55))
        fine_budget = max(1, max_iterations - coarse_budget)
        coarse_optimised = minimize(
            coarse_objective,
            initial,
            method="Powell",
            bounds=parameter_bounds,
            options={
                "maxiter": int(coarse_budget),
                "maxfev": int(coarse_budget) * 24,
                "xtol": 2e-4,
                "ftol": 2e-5,
            },
        )
        coarse_score, coarse_work_transform = evaluate_parameters(coarse_optimised.x)
        initial_coarse_signal = (
            initial_score.envelope_score
            if mode == "envelope"
            else initial_score.raw_score
        )
        coarse_signal = (
            coarse_score.envelope_score
            if mode == "envelope"
            else coarse_score.raw_score
        )
        if initial_coarse_signal > coarse_signal:
            coarse_parameters = initial.copy()
            coarse_score = initial_score
            coarse_work_transform = initial_work_transform
        else:
            coarse_parameters = np.asarray(coarse_optimised.x, dtype=np.float64)

        log_scale_radius = math.log1p(config.refinement_scale_fraction)
        local_half_widths = (
            log_scale_radius,
            config.refinement_rotation_deg,
            config.refinement_translation_fraction * reference_stats["width"],
            config.refinement_translation_fraction * reference_stats["height"],
        )
        refinement_bounds = [
            (
                max(global_lower, float(center - half_width)),
                min(global_upper, float(center + half_width)),
            )
            for center, half_width, (global_lower, global_upper) in zip(
                coarse_parameters,
                local_half_widths,
                parameter_bounds,
                strict=True,
            )
        ]

        def final_objective(parameters: np.ndarray) -> float:
            score, _ = evaluate_parameters(parameters)
            # This is exactly the continuous objective used by final ranking;
            # hard gates remain a separate eligibility decision.
            return 1.0 - score.raw_score

        refined = minimize(
            final_objective,
            coarse_parameters,
            method="Powell",
            bounds=refinement_bounds,
            options={
                "maxiter": int(fine_budget),
                "maxfev": int(fine_budget) * 24,
                "xtol": 2e-4,
                "ftol": 2e-5,
            },
        )
        refined_score, refined_work_transform = evaluate_parameters(refined.x)
        work_candidates = [
            (initial_score, initial_work_transform, initial),
            (coarse_score, coarse_work_transform, coarse_parameters),
            (
                refined_score,
                refined_work_transform,
                np.asarray(refined.x, dtype=np.float64),
            ),
        ]
        if mode == "capacity_transport":
            # Text-heavy narrow crops can pull every raw-score optimum partly
            # outside the image.  Retain a deterministic feasible placement
            # for each existing start/optimum rather than discarding the whole
            # camera when a legal in-frame translation exists.  Camera, scale,
            # roll, parameter bounds, scoring and final gates remain unchanged.
            for _, transform, parameters in tuple(work_candidates):
                contained = _contained_translation_parameters_v2(
                    support_points,
                    transform,
                    parameters,
                    parameter_bounds,
                    image_shape=(work_size[1], work_size[0]),
                    bound_margin_fraction=config.parameter_bound_tolerance_fraction,
                )
                if contained is None or np.array_equal(contained, parameters):
                    continue
                contained_score, contained_transform = evaluate_parameters(contained)
                work_candidates.append(
                    (contained_score, contained_transform, contained)
                )
        _, selected_work_transform, selected_parameters = max(
            work_candidates,
            key=lambda item: (item[0].eligible, item[0].raw_score),
        )

        full_transform = SimilarityTransform(
            scale_px_per_unit=float(
                selected_work_transform.scale_px_per_unit / work_scale
            ),
            rotation_deg=selected_work_transform.rotation_deg,
            center_x_px=float(roi_x + selected_work_transform.center_x_px / work_scale),
            center_y_px=float(roi_y + selected_work_transform.center_y_px / work_scale),
            model_center_x=selected_work_transform.model_center_x,
            model_center_y=selected_work_transform.model_center_y,
            reflected=False,
        )
        full_cad_geometry = _cad_geometry_for_transform_v2(
            projection,
            full_transform,
            geometry.diagnostics.roi_px,
            config=config,
        )
        full_score = _score_mask_pair_v2(
            geometry.envelope_mask,
            reference_edge_mask,
            full_cad_geometry.envelope_mask,
            full_cad_geometry.edge_mask,
            cad_support_stats=_transformed_support_stats_v2(
                support_points,
                full_transform,
                image_shape=(roi_height, roi_width),
                origin_px=(roi_x, roi_y),
            ),
            tolerance_px=tolerance_full,
            reference_stable=geometry.diagnostics.stable,
            cad_stable=(
                canonical_geometry.diagnostics.stable
                and full_cad_geometry.diagnostics.stable
            ),
            mode=mode,
            config=config,
            reference_edge_cache=reference_full_edge_cache,
            reference_spatial_cache=reference_full_spatial_cache,
            reference_transport_cache=reference_full_transport_cache,
        )
        scale_tolerance = config.parameter_bound_tolerance_fraction * (
            upper_scale - lower_scale
        )
        scale_at_bound = bool(
            selected_work_transform.scale_px_per_unit <= lower_scale + scale_tolerance
            or selected_work_transform.scale_px_per_unit
            >= upper_scale - scale_tolerance
        )
        rotation_tolerance = max(
            config.rotation_bound_tolerance_deg,
            config.parameter_bound_tolerance_fraction
            * max(float(max_rotation_deg), 1.0),
        )
        rotation_at_limit = bool(
            max_rotation_deg > 0.0
            and abs(abs(selected_work_transform.rotation_deg) - max_rotation_deg)
            <= rotation_tolerance
        )
        translation_bound_x = config.translation_refine_fraction * float(
            reference_stats["width"]
        )
        translation_bound_y = config.translation_refine_fraction * float(
            reference_stats["height"]
        )
        translation_tolerance_x = config.parameter_bound_tolerance_fraction * (
            2.0 * translation_bound_x
        )
        translation_tolerance_y = config.parameter_bound_tolerance_fraction * (
            2.0 * translation_bound_y
        )
        translation_at_limit = bool(
            abs(float(selected_parameters[2]))
            >= translation_bound_x - translation_tolerance_x
            or abs(float(selected_parameters[3]))
            >= translation_bound_y - translation_tolerance_y
        )
        trial_failed_gates = list(full_score.failed_gates)
        if scale_at_bound:
            trial_failed_gates.append("scale_at_bound")
        if rotation_at_limit:
            trial_failed_gates.append("rotation_at_limit")
        if translation_at_limit:
            trial_failed_gates.append("translation_at_limit")
        trial_eligible = full_score.eligible and not (
            scale_at_bound or rotation_at_limit or translation_at_limit
        )
        trial_score = full_score.score if trial_eligible else 0.0
        initial_center_x = float(
            roi_x + initial_work_transform.center_x_px / work_scale
        )
        initial_center_y = float(
            roi_y + initial_work_transform.center_y_px / work_scale
        )
        coarse_center_x = float(roi_x + coarse_work_transform.center_x_px / work_scale)
        coarse_center_y = float(roi_y + coarse_work_transform.center_y_px / work_scale)
        coarse_signal = (
            coarse_score.envelope_score
            if mode == "envelope"
            else coarse_score.raw_score
        )
        trial_records.append(
            AlignmentTrialV2(
                start_scale_px_per_unit=float(scale_seed / work_scale),
                start_rotation_deg=float(rotation_seed),
                start_center_x_px=initial_center_x,
                start_center_y_px=initial_center_y,
                start_offset_x_px=float(initial[2] / work_scale),
                start_offset_y_px=float(initial[3] / work_scale),
                coarse_scale_px_per_unit=float(
                    coarse_work_transform.scale_px_per_unit / work_scale
                ),
                coarse_rotation_deg=coarse_work_transform.rotation_deg,
                coarse_center_x_px=coarse_center_x,
                coarse_center_y_px=coarse_center_y,
                final_scale_px_per_unit=full_transform.scale_px_per_unit,
                final_rotation_deg=full_transform.rotation_deg,
                final_center_x_px=full_transform.center_x_px,
                final_center_y_px=full_transform.center_y_px,
                final_offset_x_px=float(selected_parameters[2] / work_scale),
                final_offset_y_px=float(selected_parameters[3] / work_scale),
                scale_at_bound=scale_at_bound,
                rotation_at_limit=rotation_at_limit,
                translation_at_limit=translation_at_limit,
                coarse_objective=float(1.0 - coarse_signal),
                objective=float(1.0 - full_score.raw_score),
                score=trial_score,
                raw_score=full_score.raw_score,
                eligible=trial_eligible,
                failed_gates=tuple(trial_failed_gates),
                warnings=full_score.warnings,
                metrics=full_score.gross,
                edge=full_score.edge,
                spatial_edge_mass=full_score.spatial_edge_mass,
                capacity_transport=full_score.capacity_transport,
                edge_score=full_score.edge_score,
                coarse_success=bool(coarse_optimised.success),
                coarse_iterations=(
                    int(coarse_optimised.nit)
                    if getattr(coarse_optimised, "nit", None) is not None
                    else None
                ),
                coarse_evaluations=int(getattr(coarse_optimised, "nfev", 0)),
                coarse_message=start_kind + str(coarse_optimised.message),
                success=bool(refined.success),
                iterations=(
                    int(refined.nit)
                    if getattr(refined, "nit", None) is not None
                    else None
                ),
                evaluations=int(getattr(refined, "nfev", 0)),
                message=start_kind + str(refined.message),
            )
        )
        trial_transforms.append(full_transform)

    selected_trial = max(
        range(len(trial_records)),
        key=lambda index: (
            trial_records[index].eligible,
            trial_records[index].score,
            trial_records[index].raw_score,
        ),
    )
    selected = trial_records[selected_trial]
    selected_transform = trial_transforms[selected_trial]
    diagnostics = AlignmentDiagnosticsV2(
        work_image_size=work_size,
        work_scale=float(work_scale),
        seed=seed,
        seed_translation_jitter_fraction=(bounded_jitter_fraction),
        scale_seeds_px_per_unit=tuple(
            float(value / work_scale) for value in scale_seeds_work
        ),
        rotation_seeds_deg=tuple(float(value) for value in rotation_seeds),
        selected_trial=selected_trial,
        trials=tuple(trial_records),
        mode=mode,
        confidence=confidence,
        fallback_used=fallback_used,
        reference=geometry.diagnostics,
        cad=canonical_geometry.diagnostics,
    )
    return AlignmentResultV2(
        transform=selected_transform,
        objective=selected.objective,
        score=selected.score,
        raw_score=selected.raw_score,
        eligible=selected.eligible,
        failed_gates=selected.failed_gates,
        warnings=selected.warnings,
        mode=mode,
        confidence=confidence,
        fallback_used=fallback_used,
        scale_at_bound=selected.scale_at_bound,
        rotation_at_limit=selected.rotation_at_limit,
        translation_at_limit=selected.translation_at_limit,
        metrics=selected.metrics,
        edge=selected.edge,
        spatial_edge_mass=selected.spatial_edge_mass,
        capacity_transport=selected.capacity_transport,
        edge_score=selected.edge_score,
        diagnostics=diagnostics,
    )


def score_aligned_geometry_v2(
    projection: Projection2D,
    transform: SimilarityTransform,
    reference_image: str | Path | Image.Image,
    *,
    roi: tuple[int, int, int, int] | None = None,
    threshold: int | None = None,
    reference_geometry: EnvelopeGeometryV2 | None = None,
    scoring_mode: str | None = None,
    config: ScorerV2Config = DEFAULT_SCORER_V2_CONFIG,
) -> GeometryMetricsV2:
    """Apply gross-shape gates, then score tolerant symmetric edge overlap."""

    _validate_scorer_v2_config(config)
    reference = _open_reference(reference_image)
    geometry = reference_geometry or extract_envelope_geometry_v2(
        reference,
        roi=roi,
        threshold=threshold,
        config=config,
    )
    _validate_reference_geometry_v2(geometry, reference.size, roi=roi)
    cad_geometry = _cad_geometry_for_transform_v2(
        projection,
        transform,
        geometry.diagnostics.roi_px,
        config=config,
    )
    roi_x, roi_y, roi_width, roi_height = geometry.diagnostics.roi_px
    support_points = _alignment_points(
        projection.visible_polylines,
        max_points=6000,
    )
    automatic_mode = (
        "capacity_transport"
        if config.use_capacity_transport
        else "no_envelope_match"
        if not config.use_envelope_matching
        else "envelope"
        if geometry.diagnostics.stable and cad_geometry.diagnostics.stable
        else "edge_fallback"
    )
    mode = automatic_mode if scoring_mode is None else scoring_mode
    if mode not in {
        "envelope",
        "edge_fallback",
        "no_envelope_match",
        "capacity_transport",
    }:
        raise ValueError(
            "scoring_mode must be 'envelope', 'edge_fallback', "
            "'no_envelope_match', or 'capacity_transport'"
        )
    if config.use_envelope_matching and mode == "no_envelope_match":
        raise ValueError(
            "scoring_mode='no_envelope_match' requires "
            "config.use_envelope_matching=False"
        )
    if (
        not config.use_envelope_matching
        and mode != "no_envelope_match"
        and not (config.use_capacity_transport and mode == "capacity_transport")
    ):
        raise ValueError(
            "config.use_envelope_matching=False requires "
            "scoring_mode='no_envelope_match' or enabled capacity transport"
        )
    if mode == "capacity_transport" and not config.use_capacity_transport:
        raise ValueError(
            "scoring_mode='capacity_transport' requires "
            "config.use_capacity_transport=True"
        )
    confidence = (
        "not_applicable"
        if mode in {"no_envelope_match", "capacity_transport"}
        else geometry.diagnostics.confidence
        if mode == "envelope"
        else "low"
    )
    fallback_used = (
        mode == "edge_fallback" or geometry.diagnostics.fallback_used
        if mode not in {"no_envelope_match", "capacity_transport"}
        else False
    )
    tolerance_px = max(
        config.min_edge_tolerance_px,
        config.edge_tolerance_fraction * max(geometry.envelope_mask.shape),
    )
    mask_score = _score_mask_pair_v2(
        geometry.envelope_mask,
        (
            geometry.ink_mask
            if mode == "capacity_transport"
            else geometry.edge_mask
            if mode == "no_envelope_match" or geometry.diagnostics.stable
            else geometry.ink_mask
        ),
        cad_geometry.envelope_mask,
        cad_geometry.edge_mask,
        cad_support_stats=_transformed_support_stats_v2(
            support_points,
            transform,
            image_shape=(roi_height, roi_width),
            origin_px=(roi_x, roi_y),
        ),
        tolerance_px=tolerance_px,
        reference_stable=geometry.diagnostics.stable,
        cad_stable=cad_geometry.diagnostics.stable,
        mode=mode,
        config=config,
    )
    return GeometryMetricsV2(
        score=mask_score.score,
        raw_score=mask_score.raw_score,
        eligible=mask_score.eligible,
        failed_gates=mask_score.failed_gates,
        warnings=mask_score.warnings,
        mode=mode,
        confidence=confidence,
        fallback_used=fallback_used,
        envelope_score=mask_score.envelope_score,
        edge_score=mask_score.edge_score,
        gross=mask_score.gross,
        edge=mask_score.edge,
        spatial_edge_mass=mask_score.spatial_edge_mass,
        capacity_transport=mask_score.capacity_transport,
        reference=geometry.diagnostics,
        cad=cad_geometry.diagnostics,
    )


def evaluate_projection_v2(
    projection: Projection2D,
    reference_image: str | Path | Image.Image,
    *,
    roi: tuple[int, int, int, int] | None = None,
    threshold: int | None = None,
    reference_geometry: EnvelopeGeometryV2 | None = None,
    max_rotation_deg: float = 6.0,
    seed: int = 0,
    max_iterations: int = 45,
    config: ScorerV2Config = DEFAULT_SCORER_V2_CONFIG,
) -> ProjectionEvaluationV2:
    """Run the complete V2 envelope alignment and scoring pipeline."""

    reference = _open_reference(reference_image)
    geometry = reference_geometry or extract_envelope_geometry_v2(
        reference,
        roi=roi,
        threshold=threshold,
        config=config,
    )
    alignment = align_projection_v2(
        projection,
        reference,
        roi=roi,
        threshold=threshold,
        reference_geometry=geometry,
        max_rotation_deg=max_rotation_deg,
        seed=seed,
        max_iterations=max_iterations,
        config=config,
    )
    metrics = score_aligned_geometry_v2(
        projection,
        alignment.transform,
        reference,
        roi=roi,
        threshold=threshold,
        reference_geometry=geometry,
        scoring_mode=alignment.mode,
        config=config,
    )
    boundary_gates = tuple(
        gate
        for gate, active in (
            ("scale_at_bound", alignment.scale_at_bound),
            ("rotation_at_limit", alignment.rotation_at_limit),
            ("translation_at_limit", alignment.translation_at_limit),
        )
        if active
    )
    if boundary_gates:
        merged_gates = tuple(dict.fromkeys((*metrics.failed_gates, *boundary_gates)))
        metrics = replace(
            metrics,
            score=0.0,
            eligible=False,
            failed_gates=merged_gates,
        )
    if alignment.warnings:
        metrics = replace(
            metrics,
            warnings=tuple(dict.fromkeys((*metrics.warnings, *alignment.warnings))),
        )
    return ProjectionEvaluationV2(
        transform=alignment.transform,
        alignment=alignment,
        metrics=metrics,
    )


def compare_step_to_drawing(
    step_path: str | Path,
    reference_image: str | Path,
    output_dir: str | Path,
    *,
    view: str,
    direction: Sequence[float] | None = None,
    up: Sequence[float] | None = None,
    section_origin: Sequence[float] | None = None,
    section_offset_fraction: float | None = None,
    section_keep: str | None = None,
    roi: tuple[int, int, int, int] | None = None,
    allow_reflection: bool = False,
    max_rotation_deg: float = 6.0,
    threshold: int | None = None,
    min_roi_fill: float = 0.70,
    seed: int = 0,
    max_iterations: int = 45,
    label: str | None = None,
    show_score: bool = True,
) -> ComparisonResult:
    """Return one geometric score and one input-versus-CAD comparison image."""

    reference_image = Path(reference_image)
    if not reference_image.is_file():
        raise FileNotFoundError(f"Reference drawing not found: {reference_image}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    projection = project_step(
        step_path,
        view=view,
        direction=direction,
        up=up,
        section_origin=section_origin,
        section_offset_fraction=section_offset_fraction,
        section_keep=section_keep,
    )
    alignment = align_projection(
        projection,
        reference_image,
        roi=roi,
        allow_reflection=allow_reflection,
        max_rotation_deg=max_rotation_deg,
        threshold=threshold,
        min_roi_fill=min_roi_fill,
        seed=seed,
        max_iterations=max_iterations,
    )
    geometry = score_aligned_geometry(
        projection,
        alignment.transform,
        reference_image,
        roi=roi,
        threshold=alignment.metrics.reference_threshold,
    )

    reference = _open_reference(reference_image)
    comparison_png = output_dir / "comparison.png"
    cad_view_image = _render_cad_view(
        reference.size,
        projection,
        alignment.transform,
    )
    selected_view = (
        f"section {projection.view_name}"
        if projection.section_origin is not None
        else projection.view_name
    )
    if label:
        selected_view = f"{label} · {selected_view}"
    _write_comparison(
        reference,
        cad_view_image,
        comparison_png,
        view_name=selected_view,
        score=geometry.score if show_score else None,
    )
    return ComparisonResult(score=geometry.score, comparison_png=comparison_png)


def compare_view_candidates(
    step_path: str | Path,
    reference_image: str | Path,
    output_dir: str | Path,
    *,
    candidates: Sequence[str | ViewCandidate],
    roi: tuple[int, int, int, int] | None = None,
    allow_reflection: bool = False,
    max_rotation_deg: float = 6.0,
    threshold: int | None = None,
    min_roi_fill: float = 0.70,
    seed: int = 0,
    max_iterations: int = 45,
) -> ComparisonResult:
    """Return only the score and comparison for the best proposed view.

    This is a generate-and-verify tool: it does not ask an optimiser to invent
    an unconstrained 3-D camera.  Custom candidates can carry explicit
    ``direction`` and ``up`` vectors. Candidate rankings remain internal.
    """

    resolved_candidates = tuple(_coerce_view_candidate(item) for item in candidates)
    if len(resolved_candidates) < 2:
        raise ValueError("View search requires at least two candidates")
    names = [candidate.name for candidate in resolved_candidates]
    if len(set(names)) != len(names):
        raise ValueError(f"Candidate names must be unique, got {names}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluated: list[tuple[ViewCandidate, ComparisonResult]] = []
    with tempfile.TemporaryDirectory(prefix="cadgenbench-view-search-") as temporary:
        temporary_root = Path(temporary)
        for index, candidate in enumerate(resolved_candidates):
            comparison = compare_step_to_drawing(
                step_path,
                reference_image,
                temporary_root / f"candidate_{index:03d}",
                view=candidate.name,
                direction=candidate.direction,
                up=candidate.up,
                section_origin=candidate.section_origin,
                section_offset_fraction=candidate.section_offset_fraction,
                section_keep=candidate.section_keep,
                roi=roi,
                allow_reflection=allow_reflection,
                max_rotation_deg=max_rotation_deg,
                threshold=threshold,
                min_roi_fill=min_roi_fill,
                seed=seed,
                max_iterations=max_iterations,
            )
            evaluated.append((candidate, comparison))

        evaluated.sort(key=lambda item: item[1].score, reverse=True)
        return _materialize_comparison(evaluated[0][1], output_dir)


def rank_view_candidates_detailed(
    step_path: str | Path,
    reference_image: str | Path,
    output_dir: str | Path,
    *,
    candidates: Sequence[str | ViewCandidate],
    roi: tuple[int, int, int, int] | None = None,
    allow_reflection: bool = False,
    max_rotation_deg: float = 6.0,
    threshold: int | None = None,
    min_roi_fill: float = 0.70,
    seed: int = 0,
    max_iterations: int = 45,
) -> CandidateRankingResult:
    """Evaluate all proposed views and persist a complete, replayable ranking.

    This is the worker-facing counterpart to :func:`compare_view_candidates`.
    The old function intentionally exposes only the winner; this function
    keeps every aligned comparison and returns the top-three, winner-versus-
    runner-up margin, fitted transform, and resolved view specification.
    """

    resolved_candidates = tuple(_coerce_view_candidate(item) for item in candidates)
    if len(resolved_candidates) < 2:
        raise ValueError("View ranking requires at least two candidates")
    names = [candidate.name for candidate in resolved_candidates]
    if len(set(names)) != len(names):
        raise ValueError(f"Candidate names must be unique, got {names}")

    reference_path = Path(reference_image)
    if not reference_path.is_file():
        raise FileNotFoundError(f"Reference drawing not found: {reference_path}")
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    candidate_root = output_root / "candidates"
    candidate_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="cadgenbench-view-ranking-") as temporary:
        temporary_root = Path(temporary)
        evaluated = [
            _evaluate_view_candidate(
                step_path,
                reference_path,
                temporary_root / f"candidate_{index:03d}.png",
                candidate=candidate,
                roi=roi,
                allow_reflection=allow_reflection,
                max_rotation_deg=max_rotation_deg,
                threshold=threshold,
                min_roi_fill=min_roi_fill,
                seed=seed,
                max_iterations=max_iterations,
            )
            for index, candidate in enumerate(resolved_candidates)
        ]
        evaluated.sort(key=lambda item: item.score, reverse=True)

        rankings: list[RankedViewCandidate] = []
        for rank, item in enumerate(evaluated, start=1):
            destination = candidate_root / (
                f"{rank:03d}_{_safe_artifact_stem(item.name)}.png"
            )
            shutil.copy2(item.comparison_png, destination)
            rankings.append(
                RankedViewCandidate(
                    rank=rank,
                    name=item.name,
                    score=item.score,
                    comparison_png=destination,
                    transform=item.transform,
                    spec=item.spec,
                    alignment_metrics=item.alignment_metrics,
                    geometry_metrics=item.geometry_metrics,
                )
            )

    ranked = tuple(rankings)
    best = ranked[0]
    comparison_png = output_root / "comparison.png"
    shutil.copy2(best.comparison_png, comparison_png)
    margin = float(best.score - ranked[1].score)
    return CandidateRankingResult(
        rankings=ranked,
        best=best,
        top3=ranked[:3],
        margin=margin,
        score=best.score,
        comparison_png=comparison_png,
        transform=best.transform,
        spec=best.spec,
    )


def _evaluate_view_candidate(
    step_path: str | Path,
    reference_image: Path,
    comparison_png: Path,
    *,
    candidate: ViewCandidate,
    roi: tuple[int, int, int, int] | None,
    allow_reflection: bool,
    max_rotation_deg: float,
    threshold: int | None,
    min_roi_fill: float,
    seed: int,
    max_iterations: int,
) -> RankedViewCandidate:
    projection = project_step(
        step_path,
        view=candidate.name,
        direction=candidate.direction,
        up=candidate.up,
        section_origin=candidate.section_origin,
        section_offset_fraction=candidate.section_offset_fraction,
        section_keep=candidate.section_keep,
    )
    alignment = align_projection(
        projection,
        reference_image,
        roi=roi,
        allow_reflection=allow_reflection,
        max_rotation_deg=max_rotation_deg,
        threshold=threshold,
        min_roi_fill=min_roi_fill,
        seed=seed,
        max_iterations=max_iterations,
    )
    geometry = score_aligned_geometry(
        projection,
        alignment.transform,
        reference_image,
        roi=roi,
        threshold=alignment.metrics.reference_threshold,
    )
    reference = _open_reference(reference_image)
    cad_view = _render_cad_view(
        reference.size,
        projection,
        alignment.transform,
    )
    selected_view = (
        f"section {projection.view_name}"
        if projection.section_origin is not None
        else projection.view_name
    )
    _write_comparison(
        reference,
        cad_view,
        comparison_png,
        view_name=selected_view,
        score=geometry.score,
    )
    spec = ViewSpec(
        name=candidate.name,
        direction=projection.direction,
        up=projection.up,
        section_offset_fraction=projection.section_offset_fraction,
        section_keep=(
            projection.section_keep
            if projection.section_offset_fraction is not None
            else None
        ),
    )
    return RankedViewCandidate(
        rank=0,
        name=candidate.name,
        score=geometry.score,
        comparison_png=comparison_png,
        transform=alignment.transform,
        spec=spec,
        alignment_metrics=alignment.metrics,
        geometry_metrics=geometry,
    )


def _safe_artifact_stem(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", name.strip()).strip(".-")
    return (stem or "view")[:80]


def compare_step_to_drawings(
    step_path: str | Path,
    views: Sequence[DrawingViewSpec],
    output_dir: str | Path,
    *,
    threshold: int | None = None,
    seed: int = 0,
    max_iterations: int = 45,
) -> ComparisonResult:
    """Return one mean score and one comparison sheet for multiple views."""

    if not views:
        raise ValueError("Multi-view comparison requires at least one view")
    names = [view.name.strip() for view in views]
    if any(not name for name in names):
        raise ValueError("Every drawing view requires a non-empty name")
    if len(set(names)) != len(names):
        raise ValueError(f"Drawing view names must be unique, got {names}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cadgenbench-drawing-sheet-") as temporary:
        temporary_root = Path(temporary)
        results = []
        for index, spec in enumerate(views):
            results.append(
                compare_step_to_drawing(
                    step_path,
                    spec.reference_image,
                    temporary_root / f"view_{index:03d}",
                    view=spec.view,
                    direction=spec.direction,
                    up=spec.up,
                    section_origin=spec.section_origin,
                    section_offset_fraction=spec.section_offset_fraction,
                    section_keep=spec.section_keep,
                    roi=spec.roi,
                    allow_reflection=spec.allow_reflection,
                    max_rotation_deg=spec.max_rotation_deg,
                    threshold=threshold,
                    min_roi_fill=spec.min_roi_fill,
                    seed=seed,
                    max_iterations=max_iterations,
                    label=spec.name,
                    show_score=False,
                )
            )

        score = float(np.mean([result.score for result in results]))
        comparison_png = output_dir / "comparison.png"
        _write_comparison_sheet(
            [result.comparison_png for result in results],
            comparison_png,
            score=score,
        )
    return ComparisonResult(score=score, comparison_png=comparison_png)


def _materialize_comparison(
    comparison: ComparisonResult, output_dir: Path
) -> ComparisonResult:
    """Copy only the selected candidate's one public image."""

    destination = output_dir / "comparison.png"
    shutil.copy2(comparison.comparison_png, destination)
    return ComparisonResult(score=comparison.score, comparison_png=destination)


def _coerce_view_candidate(candidate: str | ViewCandidate) -> ViewCandidate:
    if isinstance(candidate, ViewCandidate):
        name = candidate.name.strip()
        if not name:
            raise ValueError("Candidate name must not be empty")
        if (candidate.direction is None) != (candidate.up is None):
            raise ValueError(
                f"Candidate {candidate.name!r} must supply both direction and up"
            )
        if candidate.direction is None and name not in VIEW_PRESETS:
            raise ValueError(
                f"Unknown candidate {name!r}; custom candidates must supply "
                "both direction and up"
            )
        if candidate.section_origin is not None and (
            candidate.section_offset_fraction is not None
        ):
            raise ValueError(
                f"Candidate {name!r} supplies both an absolute and relative section"
            )
        direction = (
            _normalise3(candidate.direction)
            if candidate.direction is not None
            else None
        )
        up = _normalise3(candidate.up) if candidate.up is not None else None
        if direction is not None and up is not None:
            _validate_camera_basis(direction, up)
        origin = (
            _tuple3(candidate.section_origin, name="section origin")
            if candidate.section_origin is not None
            else None
        )
        offset = _validate_section_offset(candidate.section_offset_fraction)
        keep = _validate_section_keep(
            candidate.section_keep,
            has_section=origin is not None or offset is not None,
        )
        return ViewCandidate(
            name=name,
            direction=direction,
            up=up,
            section_origin=origin,
            section_offset_fraction=offset,
            section_keep=keep,
        )
    if candidate not in VIEW_PRESETS:
        raise ValueError(
            f"Unknown candidate {candidate!r}; choose from {sorted(VIEW_PRESETS)} "
            "or pass ViewCandidate with direction and up"
        )
    return ViewCandidate(name=candidate)


def _normalise3(values: Sequence[float]) -> tuple[float, float, float]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"Expected a finite 3-vector, got {values!r}")
    length = float(np.linalg.norm(vector))
    if length <= 1e-12:
        raise ValueError("View vectors must be non-zero")
    vector /= length
    return (float(vector[0]), float(vector[1]), float(vector[2]))


def _validate_camera_basis(
    direction: Sequence[float],
    up: Sequence[float],
) -> None:
    dot = float(np.dot(direction, up))
    if not math.isfinite(dot) or abs(dot) > 1.0 - 1e-8:
        raise ValueError("View direction and up vector must not be parallel")


def _validate_section_offset(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(
            f"section_offset_fraction must be a finite number, got {value!r}"
        )
    offset = float(value)
    if not math.isfinite(offset) or not -1.0 < offset < 1.0:
        raise ValueError(
            "section_offset_fraction must be finite and strictly between -1 and 1"
        )
    return offset


def _validate_section_keep(value: str | None, *, has_section: bool) -> str | None:
    if not has_section:
        if value is not None:
            raise ValueError("section_keep requires a section plane")
        return None
    if value is None:
        # Keep.BOTTOM was the original section behaviour. Retaining it as the
        # default preserves old callers while new manifests persist it
        # explicitly as ``back``.
        return "back"
    if value not in {"front", "back"}:
        raise ValueError("section_keep must be either 'front' or 'back'")
    return value


def _relative_section_origin(
    part: object,
    direction: tuple[float, float, float],
    offset_fraction: float | None,
) -> tuple[float, float, float]:
    """Return a plane origin at a relative support interval position."""

    if offset_fraction is None:  # pragma: no cover - guarded by caller
        raise ValueError("A relative section requires section_offset_fraction")
    bbox = part.bounding_box()
    minimum = np.asarray(
        (bbox.min.X, bbox.min.Y, bbox.min.Z),
        dtype=np.float64,
    )
    maximum = np.asarray(
        (bbox.max.X, bbox.max.Y, bbox.max.Z),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
        raise ValueError("Cannot section a model with non-finite bounds")
    centre = 0.5 * (minimum + maximum)
    half_size = 0.5 * (maximum - minimum)
    direction_array = np.asarray(direction, dtype=np.float64)
    half_span = float(np.dot(np.abs(direction_array), half_size))
    if half_span <= 1e-12:
        raise ValueError("Model has no extent along the section direction")
    origin = centre + float(offset_fraction) * half_span * direction_array
    return (float(origin[0]), float(origin[1]), float(origin[2]))


def _tuple3(values: Sequence[float], *, name: str) -> tuple[float, float, float]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"Expected a finite 3-vector for {name}, got {values!r}")
    return (float(vector[0]), float(vector[1]), float(vector[2]))


def _sample_edges(
    edges: Iterable[object], *, deflection: float
) -> tuple[np.ndarray, ...]:
    from build123d import PositionMode

    polylines: list[np.ndarray] = []
    for edge in edges:
        length = float(edge.length)
        if math.isfinite(length) and length > 0.0:
            # build123d 0.10 ``Edge.positions`` accepts a sequence of distances,
            # not a deflection keyword. Sample approximately uniformly by arc
            # length and cap pathological single-edge output sizes.
            point_count = min(4097, max(2, math.ceil(length / deflection) + 1))
            distances = np.linspace(0.0, length, point_count).tolist()
            positions = edge.positions(distances, PositionMode.LENGTH)
        else:
            positions = []
        if len(positions) < 2:
            positions = [edge.position_at(0.0), edge.position_at(1.0)]
        points = np.asarray(
            [(point.X, point.Y) for point in positions], dtype=np.float64
        )
        if len(points) < 2:
            continue
        keep = np.ones(len(points), dtype=bool)
        keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-10
        points = points[keep]
        if len(points) >= 2:
            polylines.append(points)
    if not polylines:
        raise ValueError("Projection produced no sampleable edges")
    return tuple(polylines)


def _open_reference(reference: str | Path | Image.Image) -> Image.Image:
    if isinstance(reference, Image.Image):
        return reference.convert("RGB")
    with Image.open(reference) as image:
        return image.convert("RGB")


def _otsu_threshold(gray: np.ndarray) -> int:
    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = float(gray.size)
    probabilities = histogram / max(total, 1.0)
    cumulative_probability = np.cumsum(probabilities)
    cumulative_mean = np.cumsum(probabilities * np.arange(256))
    global_mean = cumulative_mean[-1]
    denominator = cumulative_probability * (1.0 - cumulative_probability)
    variance = np.zeros_like(denominator)
    valid = denominator > 1e-12
    variance[valid] = (
        global_mean * cumulative_probability[valid] - cumulative_mean[valid]
    ) ** 2 / denominator[valid]
    return int(np.argmax(variance))


def _validate_pixel_roi(
    roi: tuple[int, int, int, int] | None,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """Validate legacy pixel ``(x, y, width, height)`` regions."""

    if roi is None:
        return None
    if len(roi) != 4 or any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in roi
    ):
        raise ValueError(
            f"Pixel ROI must contain integer (x, y, width, height), got {roi!r}"
        )
    x, y, width, height = (int(value) for value in roi)
    if width <= 0 or height <= 0:
        raise ValueError(f"ROI must have positive width and height, got {roi}")
    image_width, image_height = image_size
    if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
        raise ValueError(f"ROI {roi} lies outside {image_size}")
    return x, y, width, height


def _reference_mask(
    reference: Image.Image,
    *,
    threshold: int | None,
    roi: tuple[int, int, int, int] | None,
) -> tuple[np.ndarray, int]:
    gray = np.asarray(ImageOps.grayscale(reference), dtype=np.uint8)
    if threshold is not None and not 0 <= threshold <= 255:
        raise ValueError(f"Reference threshold must be in [0, 255], got {threshold}")
    used_threshold = int(threshold) if threshold is not None else _otsu_threshold(gray)
    # Otsu can return zero for nearly binary drawings; include antialiased dark
    # pixels without swallowing a lightly shaded background.
    used_threshold = max(used_threshold, 96)
    mask = gray <= used_threshold
    if roi is not None:
        x, y, width, height = roi
        roi_mask = np.zeros_like(mask)
        roi_mask[y : y + height, x : x + width] = True
        mask &= roi_mask
    return mask, used_threshold


def _validate_scorer_v2_config(config: ScorerV2Config) -> None:
    if not isinstance(config, ScorerV2Config):
        raise TypeError("config must be a ScorerV2Config")
    if not isinstance(config.use_envelope_matching, bool):
        raise TypeError("use_envelope_matching must be a boolean")
    if not isinstance(config.use_capacity_transport, bool):
        raise TypeError("use_capacity_transport must be a boolean")
    if config.use_capacity_transport and config.use_envelope_matching:
        raise ValueError("use_capacity_transport requires use_envelope_matching=False")
    if (
        isinstance(config.capacity_transport_version, bool)
        or not isinstance(config.capacity_transport_version, Integral)
        or config.capacity_transport_version != 1
    ):
        raise ValueError("Only capacity_transport_version=1 is supported")
    if (
        isinstance(config.capacity_transport_max_tokens, bool)
        or not isinstance(config.capacity_transport_max_tokens, Integral)
        or config.capacity_transport_max_tokens < 8
    ):
        raise ValueError("capacity_transport_max_tokens must be an integer >= 8")
    if (
        isinstance(config.capacity_transport_work_max_tokens, bool)
        or not isinstance(config.capacity_transport_work_max_tokens, Integral)
        or config.capacity_transport_work_max_tokens < 8
        or config.capacity_transport_work_max_tokens
        > config.capacity_transport_max_tokens
    ):
        raise ValueError(
            "capacity_transport_work_max_tokens must be in "
            "[8, capacity_transport_max_tokens]"
        )
    if config.work_max_dimension < 32 or config.canonical_size < 32:
        raise ValueError("V2 work and canonical dimensions must be at least 32")
    if (
        isinstance(config.spatial_edge_mass_version, bool)
        or not isinstance(config.spatial_edge_mass_version, Integral)
        or config.spatial_edge_mass_version != 2
    ):
        raise ValueError("Only spatial_edge_mass_version=2 is supported")
    if (
        isinstance(config.spatial_edge_mass_grid_size, bool)
        or not isinstance(config.spatial_edge_mass_grid_size, Integral)
        or config.spatial_edge_mass_grid_size != 4
    ):
        raise ValueError("spatial_edge_mass_version=2 requires a 4x4 grid")
    phases = config.spatial_edge_mass_phase_offsets
    if not isinstance(phases, tuple) or len(phases) < 2:
        raise ValueError("spatial_edge_mass_phase_offsets must contain 2+ phases")
    normalised_phases: list[tuple[float, float]] = []
    for phase in phases:
        if not isinstance(phase, tuple) or len(phase) != 2:
            raise ValueError("Every spatial edge-mass phase must be an (x, y) pair")
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) < 1.0
            for value in phase
        ):
            raise ValueError("Spatial edge-mass phase offsets must be in [0, 1)")
        resolved_phase = (float(phase[0]), float(phase[1]))
        if resolved_phase in normalised_phases:
            raise ValueError("Spatial edge-mass phase offsets must be unique")
        normalised_phases.append(resolved_phase)
    if not 0 <= config.relaxed_threshold_floor <= 255:
        raise ValueError("relaxed_threshold_floor must be in [0, 255]")
    if config.background_margin < 0 or config.threshold_perturbation < 0:
        raise ValueError("Threshold margins must be non-negative")
    for name in ("closing_radius_px", "opening_radius_px"):
        value = getattr(config, name)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, Integral) or value < 0
        ):
            raise ValueError(f"{name} must be a non-negative integer or None")
    positive = (
        "closing_radius_fraction",
        "opening_radius_fraction",
        "coarse_opening_radius_fraction",
        "coarse_opening_radius_perturbation",
        "coarse_opening_min_radius_px",
        "coarse_opening_min_area_retained",
        "coarse_opening_min_bbox_reduction",
        "coarse_opening_min_compactness_gain",
        "coarse_opening_min_stability_iou",
        "edge_band_fraction",
        "min_envelope_area_fraction",
        "min_bbox_fill_fraction",
        "max_secondary_area_ratio",
        "border_margin_fraction",
        "min_threshold_stability_iou",
        "aspect_log_gate",
        "max_area_log_error",
        "min_axis_fill",
        "max_axis_fill",
        "max_center_error",
        "max_support_outside_fraction",
        "min_envelope_dice",
        "edge_tolerance_fraction",
        "min_edge_tolerance_px",
        "capacity_transport_distance_fraction",
        "capacity_transport_min_distance_px",
        "capacity_transport_unmatched_cad_cost",
        "capacity_transport_unmatched_reference_cost",
        "aspect_temperature",
        "size_temperature",
        "center_temperature",
        "fallback_score_scale",
        "translation_refine_fraction",
        "refinement_scale_fraction",
        "refinement_rotation_deg",
        "refinement_translation_fraction",
        "seed_translation_jitter_fraction",
        "parameter_bound_tolerance_fraction",
        "rotation_bound_tolerance_deg",
    )
    for name in positive:
        value = float(getattr(config, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if config.max_axis_fill <= config.min_axis_fill:
        raise ValueError("max_axis_fill must exceed min_axis_fill")
    unit_interval = (
        "coarse_opening_radius_fraction",
        "coarse_opening_radius_perturbation",
        "coarse_opening_min_area_retained",
        "coarse_opening_min_bbox_reduction",
        "coarse_opening_min_compactness_gain",
        "coarse_opening_min_stability_iou",
        "refinement_scale_fraction",
        "refinement_translation_fraction",
        "seed_translation_jitter_fraction",
    )
    for name in unit_interval:
        value = float(getattr(config, name))
        if value >= 1.0:
            raise ValueError(f"{name} must be less than 1")
    if (
        config.coarse_opening_radius_perturbation
        >= config.coarse_opening_radius_fraction
    ):
        raise ValueError(
            "coarse_opening_radius_perturbation must be smaller than "
            "coarse_opening_radius_fraction"
        )
    for name in ("envelope_weight",):
        value = float(getattr(config, name))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if config.fallback_score_scale > 1.0:
        raise ValueError("fallback_score_scale must be in (0, 1]")


def _disk_structure(radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y) <= radius * radius


def _morphology_radius_v2(
    shape: tuple[int, int],
    *,
    override: int | None,
    fraction: float,
) -> int:
    if override is not None:
        return int(override)
    return max(1, round(float(fraction) * max(shape)))


def _coarse_opening_mask_v2(
    mask: np.ndarray,
    *,
    radius_fraction: float,
) -> tuple[np.ndarray, float]:
    """Open a filled mask with an O(n) Euclidean distance reconstruction.

    A conventional disk structuring element becomes prohibitively expensive
    at drawing resolution.  Eroding and reconstructing from distance maps is
    equivalent for the gross-shape purpose here and, crucially, removes a
    filled dimension box whose local inradius is much smaller than the part's
    main body.
    """

    source = np.asarray(mask, dtype=bool)
    distance_inside = ndimage.distance_transform_edt(source)
    radius_px = float(radius_fraction * float(np.max(distance_inside)))
    if radius_px <= 0.0:
        return source.copy(), radius_px
    core = distance_inside > radius_px
    if not np.any(core):
        return np.zeros_like(source), radius_px
    distance_to_core = ndimage.distance_transform_edt(~core)
    return source & (distance_to_core <= radius_px), radius_px


def _mask_bbox_compactness_v2(
    mask: np.ndarray,
) -> tuple[tuple[int, int, int, int], int, float] | None:
    y, x = np.nonzero(np.asarray(mask, dtype=bool))
    if len(x) == 0:
        return None
    bbox = (int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1))
    area = len(x)
    bbox_area = max((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 1)
    return bbox, area, float(area / bbox_area)


def _coarse_cleanup_envelope_v2(
    mask: np.ndarray,
    *,
    config: ScorerV2Config,
) -> _CoarseCleanupV2:
    """Choose between the raw envelope and one stable large-scale opening.

    The crop is never altered.  This only constructs a geometry-only mask.
    The opened candidate is accepted when it removes a substantial, narrow
    protrusion while retaining the main body, improving compactness, staying
    connected, and agreeing under neighbouring opening radii.  A material but
    unstable cleanup is an abstention instead of silently trusting the raw
    annotation-contaminated envelope.
    """

    source = np.asarray(mask, dtype=bool)
    source_stats = _mask_bbox_compactness_v2(source)
    if source_stats is None:
        return _CoarseCleanupV2(
            mask=source,
            used=False,
            failure_reason=None,
            radius_px=None,
            area_retained=None,
            bbox_reduction=None,
            compactness_gain=None,
            stability_iou=None,
        )

    candidate, radius_px = _coarse_opening_mask_v2(
        source,
        radius_fraction=config.coarse_opening_radius_fraction,
    )
    if radius_px < config.coarse_opening_min_radius_px:
        return _CoarseCleanupV2(
            mask=source,
            used=False,
            failure_reason=None,
            radius_px=radius_px,
            area_retained=1.0,
            bbox_reduction=0.0,
            compactness_gain=0.0,
            stability_iou=1.0,
        )

    candidate_stats = _mask_bbox_compactness_v2(candidate)
    if candidate_stats is None:
        return _CoarseCleanupV2(
            mask=source,
            used=False,
            failure_reason="coarse_cleanup_removed_all_geometry",
            radius_px=radius_px,
            area_retained=0.0,
            bbox_reduction=1.0,
            compactness_gain=None,
            stability_iou=0.0,
        )

    source_bbox, source_area, source_compactness = source_stats
    candidate_bbox, candidate_area, candidate_compactness = candidate_stats
    source_bbox_area = max(
        (source_bbox[2] - source_bbox[0]) * (source_bbox[3] - source_bbox[1]),
        1,
    )
    candidate_bbox_area = max(
        (candidate_bbox[2] - candidate_bbox[0])
        * (candidate_bbox[3] - candidate_bbox[1]),
        1,
    )
    area_retained = float(candidate_area / source_area)
    bbox_reduction = float(1.0 - candidate_bbox_area / source_bbox_area)
    compactness_gain = float(candidate_compactness - source_compactness)
    neighbour_ious: list[float] = []
    for fraction in (
        config.coarse_opening_radius_fraction
        - config.coarse_opening_radius_perturbation,
        config.coarse_opening_radius_fraction
        + config.coarse_opening_radius_perturbation,
    ):
        neighbour, _ = _coarse_opening_mask_v2(
            source,
            radius_fraction=fraction,
        )
        neighbour_iou = _binary_iou(candidate, neighbour)
        neighbour_ious.append(0.0 if neighbour_iou is None else neighbour_iou)
    stability_iou = float(min(neighbour_ious, default=1.0))
    _, cleaned_component_count = ndimage.label(
        candidate,
        structure=np.ones((3, 3), dtype=bool),
    )

    material_cleanup = (
        bbox_reduction >= config.coarse_opening_min_bbox_reduction
        and compactness_gain >= config.coarse_opening_min_compactness_gain
    )
    if not material_cleanup:
        return _CoarseCleanupV2(
            mask=source,
            used=False,
            failure_reason=None,
            radius_px=radius_px,
            area_retained=area_retained,
            bbox_reduction=bbox_reduction,
            compactness_gain=compactness_gain,
            stability_iou=stability_iou,
        )

    failure_reason = None
    if area_retained < config.coarse_opening_min_area_retained:
        failure_reason = "coarse_cleanup_removed_too_much_geometry"
    elif cleaned_component_count != 1:
        failure_reason = "coarse_cleanup_split_geometry"
    elif stability_iou < config.coarse_opening_min_stability_iou:
        failure_reason = "coarse_cleanup_radius_unstable"
    return _CoarseCleanupV2(
        mask=candidate if failure_reason is None else source,
        used=failure_reason is None,
        failure_reason=failure_reason,
        radius_px=radius_px,
        area_retained=area_retained,
        bbox_reduction=bbox_reduction,
        compactness_gain=compactness_gain,
        stability_iou=stability_iou,
    )


def _extract_envelope_component_v2(
    ink_mask: np.ndarray,
    *,
    config: ScorerV2Config,
    apply_coarse_cleanup: bool = True,
) -> _EnvelopeComponentV2:
    ink = np.asarray(ink_mask, dtype=bool)
    if ink.ndim != 2:
        raise ValueError(f"Envelope ink mask must be 2-D, got {ink.shape}")
    closing_radius = _morphology_radius_v2(
        ink.shape,
        override=config.closing_radius_px,
        fraction=config.closing_radius_fraction,
    )
    opening_radius = _morphology_radius_v2(
        ink.shape,
        override=config.opening_radius_px,
        fraction=config.opening_radius_fraction,
    )
    empty = np.zeros_like(ink)
    if not np.any(ink):
        return _EnvelopeComponentV2(
            mask=empty,
            stable=False,
            failure_reason="no_ink",
            bbox=None,
            centroid=None,
            area=0,
            component_count=0,
            secondary_area_ratio=0.0,
            touches_boundary=False,
            closing_radius=closing_radius,
            opening_radius=opening_radius,
        )

    closed = ndimage.binary_closing(
        ink,
        structure=_disk_structure(closing_radius),
    )
    solid = ndimage.binary_fill_holes(closed)
    if opening_radius > 0:
        solid = ndimage.binary_opening(
            solid,
            structure=_disk_structure(opening_radius),
        )
    # Restore small convex corners removed by opening without reconnecting the
    # one-pixel dimension/leader tendrils that opening intentionally removed.
    solid = ndimage.binary_closing(solid, structure=_disk_structure(1))

    labels, count = ndimage.label(solid, structure=np.ones((3, 3), dtype=bool))
    minimum_area = max(
        32,
        math.ceil(config.min_envelope_area_fraction * ink.size),
    )
    components: list[tuple[int, int]] = []
    for label in range(1, count + 1):
        area = int(np.count_nonzero(labels == label))
        if area >= minimum_area:
            components.append((area, label))
    components.sort(reverse=True)
    if not components:
        return _EnvelopeComponentV2(
            mask=empty,
            stable=False,
            failure_reason="no_closed_envelope",
            bbox=None,
            centroid=None,
            area=0,
            component_count=0,
            secondary_area_ratio=0.0,
            touches_boundary=False,
            closing_radius=closing_radius,
            opening_radius=opening_radius,
        )

    _, selected_label = components[0]
    raw_selected = labels == selected_label
    coarse_cleanup = (
        _coarse_cleanup_envelope_v2(
            raw_selected,
            config=config,
        )
        if apply_coarse_cleanup
        else _CoarseCleanupV2(
            mask=raw_selected,
            used=False,
            failure_reason=None,
            radius_px=None,
            area_retained=None,
            bbox_reduction=None,
            compactness_gain=None,
            stability_iou=None,
        )
    )
    selected = coarse_cleanup.mask
    y, x = np.nonzero(selected)
    bbox = (int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1))
    centroid = (float(np.mean(x)), float(np.mean(y)))
    area = len(x)
    bbox_area = max((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 1)
    bbox_fill = area / bbox_area
    secondary_area_ratio = (
        float(components[1][0] / area) if len(components) > 1 else 0.0
    )
    border_margin = max(
        1,
        round(config.border_margin_fraction * max(ink.shape)),
    )
    touches_boundary = bool(
        bbox[0] <= border_margin
        or bbox[1] <= border_margin
        or bbox[2] >= ink.shape[1] - border_margin
        or bbox[3] >= ink.shape[0] - border_margin
    )
    failure_reason = coarse_cleanup.failure_reason
    if failure_reason is not None:
        pass
    elif area < minimum_area:
        failure_reason = "coarse_cleanup_below_minimum_area"
    elif touches_boundary:
        failure_reason = "envelope_touches_crop_boundary"
    elif bbox_fill < config.min_bbox_fill_fraction:
        failure_reason = "envelope_too_sparse"
    elif secondary_area_ratio > config.max_secondary_area_ratio:
        failure_reason = "multiple_competing_envelopes"
    return _EnvelopeComponentV2(
        mask=selected,
        stable=failure_reason is None,
        failure_reason=failure_reason,
        bbox=bbox,
        centroid=centroid,
        area=area,
        component_count=len(components),
        secondary_area_ratio=secondary_area_ratio,
        touches_boundary=touches_boundary,
        closing_radius=closing_radius,
        opening_radius=opening_radius,
        coarse_cleanup_attempted=apply_coarse_cleanup,
        coarse_cleanup_used=coarse_cleanup.used,
        coarse_cleanup_radius=coarse_cleanup.radius_px,
        coarse_cleanup_area_retained=coarse_cleanup.area_retained,
        coarse_cleanup_bbox_reduction=coarse_cleanup.bbox_reduction,
        coarse_cleanup_compactness_gain=coarse_cleanup.compactness_gain,
        coarse_cleanup_stability_iou=coarse_cleanup.stability_iou,
        coarse_cleanup_failure_reason=coarse_cleanup.failure_reason,
    )


def _binary_iou(first: np.ndarray, second: np.ndarray) -> float | None:
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    if first.shape != second.shape:
        raise ValueError("IoU masks must have identical shapes")
    union = int(np.count_nonzero(first | second))
    if union == 0:
        return None
    return float(np.count_nonzero(first & second) / union)


def _resize_binary_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L")
    resized = image.resize(size, resample=Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.uint8) > 0


def _mask_stats(mask: np.ndarray) -> dict[str, float | tuple[float, float]] | None:
    y, x = np.nonzero(np.asarray(mask, dtype=bool))
    if len(x) == 0:
        return None
    x0, x1 = int(x.min()), int(x.max() + 1)
    y0, y1 = int(y.min()), int(y.max() + 1)
    return {
        "area": float(len(x)),
        "width": float(x1 - x0),
        "height": float(y1 - y0),
        "centroid": (float(np.mean(x)), float(np.mean(y))),
    }


def _robust_edge_stats_v2(
    mask: np.ndarray,
    *,
    trim_fraction: float = 0.025,
) -> dict[str, float | tuple[float, float]] | None:
    """Return an annotation-tolerant extent for transport initialisation.

    This is not a part mask.  It only trims the outer coordinate quantiles so
    one distant label cannot determine every scale seed and translation bound.
    The transport objective still receives every reference ink pixel.
    """

    y, x = np.nonzero(np.asarray(mask, dtype=bool))
    if len(x) == 0:
        return None
    if not 0.0 <= trim_fraction < 0.5:
        raise ValueError("trim_fraction must be in [0, 0.5)")
    lower = float(trim_fraction)
    upper = 1.0 - lower
    x0, x1 = np.quantile(x, (lower, upper))
    y0, y1 = np.quantile(y, (lower, upper))
    return {
        "area": float(len(x)),
        "width": max(float(x1 - x0 + 1.0), 1.0),
        "height": max(float(y1 - y0 + 1.0), 1.0),
        "centroid": (float(np.median(x)), float(np.median(y))),
    }


def _geometry_from_binary_ink_v2(
    ink_mask: np.ndarray,
    *,
    roi_px: tuple[int, int, int, int],
    config: ScorerV2Config,
) -> EnvelopeGeometryV2:
    ink = np.asarray(ink_mask, dtype=bool)
    # CAD HLR contains no dimensions, leaders, or title-block ink.  The large
    # EDT opening exists specifically to clean those reference-only artifacts;
    # forcing it on a valid perforated CAD silhouette can remove one third of
    # the true geometry and incorrectly downgrade an otherwise closed envelope.
    component = _extract_envelope_component_v2(
        ink,
        config=config,
        apply_coarse_cleanup=False,
    )
    envelope = component.mask
    if np.any(envelope):
        boundary = envelope & ~ndimage.binary_erosion(
            envelope,
            structure=_disk_structure(1),
            border_value=0,
        )
    else:
        boundary = np.zeros_like(ink)
    roi_x, roi_y, roi_width, roi_height = roi_px
    bbox = component.bbox
    centroid = component.centroid
    diagnostics = EnvelopeDiagnosticsV2(
        stable=component.stable,
        confidence="high" if component.stable else "none",
        failure_reason=component.failure_reason,
        dark_threshold=255,
        relaxed_threshold=255,
        selected_threshold=255,
        threshold_override=True,
        fallback_used=False,
        dark_relaxed_iou=1.0 if np.any(envelope) else None,
        threshold_stability_iou=1.0 if component.stable else None,
        bbox_px=(
            None
            if bbox is None
            else (
                bbox[0] + roi_x,
                bbox[1] + roi_y,
                bbox[2] + roi_x,
                bbox[3] + roi_y,
            )
        ),
        centroid_px=(
            None if centroid is None else (centroid[0] + roi_x, centroid[1] + roi_y)
        ),
        area_px=component.area,
        area_fraction=float(component.area / max(ink.size, 1)),
        component_count=component.component_count,
        secondary_area_ratio=component.secondary_area_ratio,
        touches_boundary=component.touches_boundary,
        closing_radius_px=component.closing_radius,
        opening_radius_px=component.opening_radius,
        roi_px=(roi_x, roi_y, roi_width, roi_height),
        coarse_cleanup_attempted=component.coarse_cleanup_attempted,
        coarse_cleanup_used=component.coarse_cleanup_used,
        coarse_cleanup_radius_px=component.coarse_cleanup_radius,
        coarse_cleanup_area_retained=component.coarse_cleanup_area_retained,
        coarse_cleanup_bbox_reduction=component.coarse_cleanup_bbox_reduction,
        coarse_cleanup_compactness_gain=(component.coarse_cleanup_compactness_gain),
        coarse_cleanup_stability_iou=component.coarse_cleanup_stability_iou,
        coarse_cleanup_failure_reason=component.coarse_cleanup_failure_reason,
    )
    return EnvelopeGeometryV2(
        ink_mask=ink,
        envelope_mask=envelope,
        boundary_mask=boundary,
        edge_mask=ink,
        diagnostics=diagnostics,
    )


def _canonical_cad_geometry_v2(
    projection: Projection2D,
    *,
    config: ScorerV2Config,
) -> tuple[SimilarityTransform, EnvelopeGeometryV2]:
    size = int(config.canonical_size)
    margin = max(8, size // 16)
    transform = _fit_projection_transform(
        projection,
        image_size=(size, size),
        margin=margin,
    )
    ink = _rasterize_visible_edges(
        (size, size),
        projection.visible_polylines,
        transform,
    )
    geometry = _geometry_from_binary_ink_v2(
        ink,
        roi_px=(0, 0, size, size),
        config=config,
    )
    return transform, geometry


def _transform_matrix_v2(transform: SimilarityTransform) -> np.ndarray:
    angle = math.radians(transform.rotation_deg)
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = np.asarray(
        ((cosine, -sine), (sine, cosine)),
        dtype=np.float64,
    )
    reflection = np.diag((-1.0, 1.0) if transform.reflected else (1.0, 1.0))
    raster_flip = np.diag((1.0, -1.0))
    linear = transform.scale_px_per_unit * raster_flip @ rotation @ reflection
    model_center = np.asarray(
        (transform.model_center_x, transform.model_center_y),
        dtype=np.float64,
    )
    image_center = np.asarray(
        (transform.center_x_px, transform.center_y_px),
        dtype=np.float64,
    )
    translation = image_center - linear @ model_center
    matrix = np.eye(3, dtype=np.float64)
    matrix[:2, :2] = linear
    matrix[:2, 2] = translation
    return matrix


def _inverse_transform_point(
    transform: SimilarityTransform,
    point: tuple[float, float],
) -> tuple[float, float]:
    inverse = np.linalg.inv(_transform_matrix_v2(transform))
    mapped = inverse @ np.asarray((point[0], point[1], 1.0), dtype=np.float64)
    return float(mapped[0]), float(mapped[1])


def _transform_centered_on_anchor(
    projection: Projection2D,
    *,
    scale: float,
    rotation_deg: float,
    anchor_model: tuple[float, float],
    target_centroid: tuple[float, float],
) -> SimilarityTransform:
    provisional = SimilarityTransform(
        scale_px_per_unit=float(scale),
        rotation_deg=float(rotation_deg),
        center_x_px=0.0,
        center_y_px=0.0,
        model_center_x=projection.center[0],
        model_center_y=projection.center[1],
        reflected=False,
    )
    offset = provisional.apply(np.asarray([anchor_model], dtype=np.float64))[0]
    return SimilarityTransform(
        scale_px_per_unit=float(scale),
        rotation_deg=float(rotation_deg),
        center_x_px=float(target_centroid[0] - offset[0]),
        center_y_px=float(target_centroid[1] - offset[1]),
        model_center_x=projection.center[0],
        model_center_y=projection.center[1],
        reflected=False,
    )


def _support_translation_origin_v2(
    points: np.ndarray,
    transform: SimilarityTransform,
    *,
    image_shape: tuple[int, int],
    preferred_target: tuple[float, float] | None = None,
) -> tuple[float, float] | None:
    """Find a feasible translation origin for a complete projected support.

    With a preferred annotation-derived target, choose its closest feasible
    point; otherwise use the centre of the feasible interval.  Each origin
    supplies a separate local-search start, not a new camera or a relaxed gate.
    The existing scale range, local translation radius and gates still apply.
    """

    pixels = transform.apply(points)
    if len(pixels) == 0:
        return None
    height, width = image_shape
    minimum = pixels.min(axis=0)
    maximum = pixels.max(axis=0)
    if np.any(maximum - minimum > np.asarray([width - 2.0, height - 2.0])):
        return None
    lower = 0.5 - minimum
    upper = np.asarray([width - 1.5, height - 1.5]) - maximum
    shift = (
        (lower + upper) / 2.0
        if preferred_target is None
        else np.clip(np.asarray(preferred_target, dtype=np.float64), lower, upper)
    )
    return float(shift[0]), float(shift[1])


def _contained_translation_parameters_v2(
    points: np.ndarray,
    transform: SimilarityTransform,
    parameters: np.ndarray,
    bounds: Sequence[tuple[float, float]],
    *,
    image_shape: tuple[int, int],
    bound_margin_fraction: float = 0.0,
) -> np.ndarray | None:
    """Return the nearest legal in-frame translation, with no geometry warp.

    A half-pixel image inset avoids numerical clipping at a boundary.  The
    feasibility interval is intersected with the existing translation search
    bounds, including the same parameter-bound guard used by final acceptance.
    If the projection cannot fit at its current scale/roll, or its legal
    translation interval is empty, no alternative is proposed.
    """

    pixels = transform.apply(points)
    if len(pixels) == 0:
        return None
    height, width = image_shape
    minimum = pixels.min(axis=0)
    maximum = pixels.max(axis=0)
    contained = np.array(parameters, dtype=np.float64, copy=True)
    for axis, extent in enumerate((width, height)):
        parameter_index = axis + 2
        lower, upper = bounds[parameter_index]
        # Strictly inside the unchanged final translation-at-limit guard.
        guard = bound_margin_fraction * (upper - lower) + 1e-9
        allowed_lower = max(
            0.5 - minimum[axis],
            lower + guard - parameters[parameter_index],
        )
        allowed_upper = min(
            extent - 1.5 - maximum[axis],
            upper - guard - parameters[parameter_index],
        )
        if allowed_lower > allowed_upper:
            return None
        contained[parameter_index] += float(
            np.clip(0.0, allowed_lower, allowed_upper)
        )
    return contained


def _transformed_support_stats_v2(
    points: np.ndarray,
    transform: SimilarityTransform,
    *,
    image_shape: tuple[int, int],
    origin_px: tuple[float, float] = (0.0, 0.0),
) -> dict[str, float | tuple[float, float]]:
    pixels = transform.apply(points)
    pixels = pixels - np.asarray(origin_px, dtype=np.float64)
    if len(pixels) == 0:  # pragma: no cover - projection validation prevents this
        return {
            "width": 0.0,
            "height": 0.0,
            "centroid": (0.0, 0.0),
            "outside_fraction": 1.0,
        }
    height, width = image_shape
    valid = (
        (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] <= width - 1.0)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] <= height - 1.0)
    )
    return {
        "width": float(max(pixels[:, 0].max() - pixels[:, 0].min(), 1e-9)),
        "height": float(max(pixels[:, 1].max() - pixels[:, 1].min(), 1e-9)),
        "centroid": (
            float(np.mean(pixels[:, 0])),
            float(np.mean(pixels[:, 1])),
        ),
        "outside_fraction": float(1.0 - np.mean(valid)),
    }


def _warp_binary_mask(
    mask: np.ndarray,
    *,
    source_transform: SimilarityTransform,
    target_transform: SimilarityTransform,
    output_size: tuple[int, int],
) -> np.ndarray:
    source_matrix = _transform_matrix_v2(source_transform)
    target_matrix = _transform_matrix_v2(target_transform)
    output_to_source = source_matrix @ np.linalg.inv(target_matrix)
    coefficients = tuple(float(value) for value in output_to_source[:2].ravel())
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L")
    warped = image.transform(
        output_size,
        Image.Transform.AFFINE,
        coefficients,
        resample=Image.Resampling.NEAREST,
        fillcolor=0,
    )
    return np.asarray(warped, dtype=np.uint8) > 0


def _unique_positive_values(values: Sequence[float]) -> tuple[float, ...]:
    result: list[float] = []
    for value in values:
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            continue
        if not any(math.isclose(value, old, rel_tol=1e-6) for old in result):
            result.append(value)
    if not result:
        raise ValueError("Alignment produced no finite positive scale seeds")
    return tuple(result)


def _unique_finite_values(values: Sequence[float]) -> tuple[float, ...]:
    result: list[float] = []
    for value in values:
        value = float(value)
        if math.isfinite(value) and not any(
            math.isclose(value, old, abs_tol=1e-10) for old in result
        ):
            result.append(value)
    return tuple(result)


def _envelope_match_metrics_v2(
    reference_mask: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    candidate_support_mask: np.ndarray | None = None,
    candidate_support_stats: dict[str, float | tuple[float, float]] | None = None,
) -> EnvelopeMatchMetricsV2:
    reference = np.asarray(reference_mask, dtype=bool)
    candidate = np.asarray(candidate_mask, dtype=bool)
    if reference.shape != candidate.shape:
        raise ValueError("Envelope masks must have identical shapes")
    reference_stats = _mask_stats(reference)
    candidate_stats = _mask_stats(candidate)
    if candidate_support_stats is None:
        candidate_support_stats = _mask_stats(
            candidate
            if candidate_support_mask is None
            else np.asarray(candidate_support_mask, dtype=bool)
        )
    if reference_stats is None or candidate_support_stats is None:
        return EnvelopeMatchMetricsV2(
            dice=0.0,
            iou=0.0,
            aspect_error=math.log(1e12),
            width_ratio=0.0,
            height_ratio=0.0,
            area_ratio=0.0,
            area_log_error=math.log(1e12),
            center_error=1.0,
            support_outside_fraction=1.0,
        )
    intersection = (
        float(np.count_nonzero(reference & candidate))
        if candidate_stats is not None
        else 0.0
    )
    reference_area = float(reference_stats["area"])
    candidate_area = (
        float(candidate_stats["area"]) if candidate_stats is not None else 0.0
    )
    union = reference_area + candidate_area - intersection
    dice = 2.0 * intersection / max(reference_area + candidate_area, 1.0)
    iou = intersection / max(union, 1.0)
    # Gross extent must come from *all* visible CAD edges, not the largest
    # closed cavity.  Otherwise an edge-on 11:1 projection can expose a small
    # circular hole that masquerades as a square candidate and bypasses the
    # aspect/minor-axis gates.
    width_ratio = float(candidate_support_stats["width"] / reference_stats["width"])
    height_ratio = float(candidate_support_stats["height"] / reference_stats["height"])
    area_ratio = candidate_area / reference_area
    reference_aspect = float(reference_stats["width"] / reference_stats["height"])
    candidate_aspect = float(
        candidate_support_stats["width"] / candidate_support_stats["height"]
    )
    aspect_error = abs(math.log(max(candidate_aspect / reference_aspect, 1e-12)))
    area_log_error = abs(math.log(max(area_ratio, 1e-12)))
    reference_centroid = np.asarray(reference_stats["centroid"], dtype=np.float64)
    candidate_centroid = np.asarray(
        candidate_support_stats["centroid"],
        dtype=np.float64,
    )
    reference_diagonal = math.hypot(
        float(reference_stats["width"]),
        float(reference_stats["height"]),
    )
    center_error = float(
        np.linalg.norm(candidate_centroid - reference_centroid)
        / max(reference_diagonal, 1.0)
    )
    return EnvelopeMatchMetricsV2(
        dice=float(np.clip(dice, 0.0, 1.0)),
        iou=float(np.clip(iou, 0.0, 1.0)),
        aspect_error=float(aspect_error),
        width_ratio=width_ratio,
        height_ratio=height_ratio,
        area_ratio=area_ratio,
        area_log_error=float(area_log_error),
        center_error=center_error,
        support_outside_fraction=float(
            candidate_support_stats.get("outside_fraction", 0.0)
        ),
    )


def _tolerant_edge_metrics_v2(
    reference_edge_mask: np.ndarray,
    cad_edge_mask: np.ndarray,
    *,
    tolerance_px: float,
    reference_cache: _ReferenceEdgeCacheV2 | None = None,
) -> EdgeMatchMetricsV2:
    reference = np.asarray(reference_edge_mask, dtype=bool)
    cad = np.asarray(cad_edge_mask, dtype=bool)
    if reference.shape != cad.shape:
        raise ValueError("Edge masks must have identical shapes")
    cache = reference_cache or _prepare_reference_edge_cache_v2(reference)
    if cache.shape != reference.shape:
        raise ValueError("Reference edge cache shape does not match edge masks")
    reference_count = cache.edge_pixels
    cad_count = int(np.count_nonzero(cad))
    fallback_distance = cache.fallback_distance_px
    if reference_count:
        if cache.distance_to_reference is None:  # pragma: no cover - invariant guard
            raise ValueError("Non-empty reference edge cache has no distance map")
        cad_distances = (
            cache.distance_to_reference[cad] if cad_count else np.asarray([])
        )
    else:
        cad_distances = np.full(cad_count, fallback_distance, dtype=np.float64)
    if cad_count:
        distance_to_cad = ndimage.distance_transform_edt(~cad)
        reference_distances = (
            distance_to_cad[reference] if reference_count else np.asarray([])
        )
    else:
        reference_distances = np.full(
            reference_count,
            fallback_distance,
            dtype=np.float64,
        )
    precision = float(np.mean(cad_distances <= tolerance_px)) if cad_count else 0.0
    recall = (
        float(np.mean(reference_distances <= tolerance_px)) if reference_count else 0.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    return EdgeMatchMetricsV2(
        tolerance_px=float(tolerance_px),
        precision=precision,
        recall=recall,
        f1=f1,
        cad_to_reference_mean_px=(
            float(np.mean(cad_distances)) if cad_count else fallback_distance
        ),
        reference_to_cad_mean_px=(
            float(np.mean(reference_distances))
            if reference_count
            else fallback_distance
        ),
        cad_edge_pixels=cad_count,
        reference_edge_pixels=reference_count,
    )


def _prepare_reference_edge_cache_v2(
    reference_edge_mask: np.ndarray,
) -> _ReferenceEdgeCacheV2:
    """Compute the invariant half of tolerant bidirectional edge matching."""

    reference = np.asarray(reference_edge_mask, dtype=bool)
    if reference.ndim != 2:
        raise ValueError("Reference edge mask must be 2-D")
    reference_count = int(np.count_nonzero(reference))
    distance_to_reference = (
        ndimage.distance_transform_edt(~reference) if reference_count else None
    )
    return _ReferenceEdgeCacheV2(
        shape=tuple(int(value) for value in reference.shape),
        edge_pixels=reference_count,
        fallback_distance_px=float(math.hypot(*reference.shape)),
        distance_to_reference=distance_to_reference,
    )


def _edge_coordinates_xy_v2(mask: np.ndarray) -> np.ndarray:
    """Return edge coordinates in deterministic raster order as ``(x, y)``."""

    y, x = np.nonzero(np.asarray(mask, dtype=bool))
    if len(x) == 0:
        return np.empty((0, 2), dtype=np.float64)
    return np.column_stack((x, y)).astype(np.float64, copy=False)


def _equal_mass_token_indices_v2(total_count: int, token_count: int) -> np.ndarray:
    if token_count < 0 or token_count > total_count:
        raise ValueError("token_count must be between zero and the edge count")
    if token_count == 0:
        return np.empty(0, dtype=np.int64)
    if token_count == total_count:
        return np.arange(total_count, dtype=np.int64)
    return np.clip(
        np.floor(
            (np.arange(token_count, dtype=np.float64) + 0.5) * total_count / token_count
        ).astype(np.int64),
        0,
        total_count - 1,
    )


def _sample_equal_mass_tokens_v2(
    coordinates_xy: np.ndarray,
    token_count: int,
) -> np.ndarray:
    """Deterministically quantise a raster measure into equal-mass tokens."""

    coordinates = np.asarray(coordinates_xy, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1:] != (2,):
        raise ValueError("Edge coordinates must have shape (N, 2)")
    return coordinates[_equal_mass_token_indices_v2(len(coordinates), token_count)]


def _prepare_reference_capacity_transport_cache_v2(
    reference_edge_mask: np.ndarray,
) -> _ReferenceCapacityTransportCacheV2:
    reference = np.asarray(reference_edge_mask, dtype=bool)
    if reference.ndim != 2:
        raise ValueError("Reference edge mask must be 2-D")
    coordinates = _edge_coordinates_xy_v2(reference)
    return _ReferenceCapacityTransportCacheV2(
        shape=tuple(int(value) for value in reference.shape),
        edge_pixels=len(coordinates),
        coordinates_xy=coordinates,
    )


def _capacity_transport_metrics_v2(
    reference_edge_mask: np.ndarray,
    cad_edge_mask: np.ndarray,
    *,
    config: ScorerV2Config,
    reference_cache: _ReferenceCapacityTransportCacheV2 | None = None,
) -> CapacityTransportMetricsV2:
    """Solve a continuous, capacity-one partial edge assignment.

    The larger raster measure is represented by at most ``max_tokens``;
    the smaller measure receives proportionally fewer equal-mass tokens.  A
    rectangular linear assignment then chooses globally optimal real matches
    or CAD dustbin columns.  Subtracting the reference-unmatched cost from
    every real pair makes the rectangular problem exactly equivalent to
    charging both unmatched sides, without constructing a quadratic dummy
    block.
    """

    reference = np.asarray(reference_edge_mask, dtype=bool)
    cad = np.asarray(cad_edge_mask, dtype=bool)
    if reference.shape != cad.shape:
        raise ValueError("Capacity transport masks must have identical shapes")
    cache = reference_cache or _prepare_reference_capacity_transport_cache_v2(reference)
    if cache.shape != reference.shape:
        raise ValueError("Reference transport cache shape does not match edge masks")

    reference_count = int(cache.edge_pixels)
    cad_coordinates = _edge_coordinates_xy_v2(cad)
    cad_count = len(cad_coordinates)
    distance_scale = max(
        float(config.capacity_transport_min_distance_px),
        float(config.capacity_transport_distance_fraction * max(reference.shape)),
    )
    fallback_distance = float(math.hypot(*reference.shape))
    max_tokens = int(config.capacity_transport_max_tokens)
    if max(reference.shape) <= config.work_max_dimension:
        max_tokens = min(max_tokens, int(config.capacity_transport_work_max_tokens))
    mass_quantum = max(reference_count, cad_count) / max_tokens
    reference_token_count = (
        min(
            reference_count,
            max(1, round(reference_count / max(mass_quantum, 1.0))),
        )
        if reference_count
        else 0
    )
    cad_token_count = (
        min(
            cad_count,
            max(1, round(cad_count / max(mass_quantum, 1.0))),
        )
        if cad_count
        else 0
    )
    cad_unmatched = float(config.capacity_transport_unmatched_cad_cost)
    reference_unmatched = float(config.capacity_transport_unmatched_reference_cost)
    if reference_count == 0 or cad_count == 0:
        total_cost = (
            cad_unmatched * cad_token_count
            + reference_unmatched * reference_token_count
        )
        return CapacityTransportMetricsV2(
            version=config.capacity_transport_version,
            score=0.0,
            normalized_cost=1.0,
            distance_scale_px=distance_scale,
            unmatched_cad_cost=cad_unmatched,
            unmatched_reference_cost=reference_unmatched,
            reference_edge_pixels=reference_count,
            cad_edge_pixels=cad_count,
            reference_tokens=reference_token_count,
            cad_tokens=cad_token_count,
            matched_pairs=0,
            unmatched_cad_tokens=cad_token_count,
            unmatched_reference_tokens=reference_token_count,
            total_cost=total_cost,
            matched_cad_fraction=0.0,
            matched_reference_fraction=0.0,
            mean_matched_distance_px=fallback_distance,
            p90_matched_distance_px=fallback_distance,
        )

    reference_indices = _equal_mass_token_indices_v2(
        reference_count,
        reference_token_count,
    )
    cad_indices = _equal_mass_token_indices_v2(cad_count, cad_token_count)
    reference_tokens = cache.coordinates_xy[reference_indices]
    cad_tokens = cad_coordinates[cad_indices]

    deltas = cad_tokens[:, None, :] - reference_tokens[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    real_cost = np.square(distances / distance_scale)
    assignment_cost = np.full(
        (cad_token_count, reference_token_count + cad_token_count),
        cad_unmatched,
        dtype=np.float64,
    )
    assignment_cost[:, :reference_token_count] = real_cost - reference_unmatched
    row_indices, column_indices = linear_sum_assignment(assignment_cost)
    selected_cost = assignment_cost[row_indices, column_indices]
    matched_mask = column_indices < reference_token_count
    matched_rows = row_indices[matched_mask]
    matched_columns = column_indices[matched_mask]
    matched_distances = distances[matched_rows, matched_columns]
    matched_pairs = len(matched_distances)

    total_cost = reference_unmatched * reference_token_count + float(
        np.sum(selected_cost)
    )
    unmatched_cost = (
        cad_unmatched * cad_token_count + reference_unmatched * reference_token_count
    )
    normalized_cost = float(np.clip(total_cost / max(unmatched_cost, 1e-12), 0.0, 1.0))
    return CapacityTransportMetricsV2(
        version=config.capacity_transport_version,
        score=float(1.0 - normalized_cost),
        normalized_cost=normalized_cost,
        distance_scale_px=distance_scale,
        unmatched_cad_cost=cad_unmatched,
        unmatched_reference_cost=reference_unmatched,
        reference_edge_pixels=reference_count,
        cad_edge_pixels=cad_count,
        reference_tokens=reference_token_count,
        cad_tokens=cad_token_count,
        matched_pairs=matched_pairs,
        unmatched_cad_tokens=cad_token_count - matched_pairs,
        unmatched_reference_tokens=reference_token_count - matched_pairs,
        total_cost=total_cost,
        matched_cad_fraction=float(matched_pairs / cad_token_count),
        matched_reference_fraction=float(matched_pairs / reference_token_count),
        mean_matched_distance_px=(
            float(np.mean(matched_distances)) if matched_pairs else fallback_distance
        ),
        p90_matched_distance_px=(
            float(np.percentile(matched_distances, 90))
            if matched_pairs
            else fallback_distance
        ),
    )


def _prepare_reference_spatial_edge_mass_cache_v2(
    reference_envelope_mask: np.ndarray,
    reference_edge_mask: np.ndarray,
    *,
    use_envelope_bbox: bool,
    config: ScorerV2Config,
) -> _ReferenceSpatialEdgeMassCacheV2:
    """Pre-bin the invariant reference edge capacity for every grid phase."""

    envelope = np.asarray(reference_envelope_mask, dtype=bool)
    reference = np.asarray(reference_edge_mask, dtype=bool)
    if envelope.shape != reference.shape:
        raise ValueError("Spatial reference masks must have identical shapes")
    bbox_source = envelope if use_envelope_bbox and np.any(envelope) else reference
    source_y, source_x = np.nonzero(bbox_source)
    bbox = (
        None
        if len(source_x) == 0
        else (
            int(source_x.min()),
            int(source_y.min()),
            int(source_x.max() + 1),
            int(source_y.max() + 1),
        )
    )
    phase_offsets = tuple(
        (float(offset_x), float(offset_y))
        for offset_x, offset_y in config.spatial_edge_mass_phase_offsets
    )
    if bbox is None:
        return _ReferenceSpatialEdgeMassCacheV2(
            shape=tuple(int(value) for value in reference.shape),
            version=config.spatial_edge_mass_version,
            grid_size=config.spatial_edge_mass_grid_size,
            phase_offsets=phase_offsets,
            use_envelope_bbox=use_envelope_bbox,
            bbox_px=None,
            reference_edge_pixels=0,
            reference_cells=(),
        )

    x0, y0, x1, y1 = bbox
    reference_y, reference_x = np.nonzero(reference[y0:y1, x0:x1])
    grid_size = config.spatial_edge_mass_grid_size
    bins_per_axis = grid_size + 1
    cell_count = bins_per_axis * bins_per_axis
    reference_scaled_x = reference_x.astype(np.float64) * grid_size / (x1 - x0)
    reference_scaled_y = reference_y.astype(np.float64) * grid_size / (y1 - y0)
    reference_cells: list[np.ndarray] = []
    for offset_x, offset_y in phase_offsets:
        reference_ix = np.floor(reference_scaled_x + offset_x).astype(np.int64)
        reference_iy = np.floor(reference_scaled_y + offset_y).astype(np.int64)
        cells = np.bincount(
            reference_iy * bins_per_axis + reference_ix,
            minlength=cell_count,
        )
        reference_cells.append(np.append(cells, 0))
    return _ReferenceSpatialEdgeMassCacheV2(
        shape=tuple(int(value) for value in reference.shape),
        version=config.spatial_edge_mass_version,
        grid_size=grid_size,
        phase_offsets=phase_offsets,
        use_envelope_bbox=use_envelope_bbox,
        bbox_px=bbox,
        reference_edge_pixels=len(reference_x),
        reference_cells=tuple(reference_cells),
    )


def _spatial_edge_mass_metrics_v2(
    reference_envelope_mask: np.ndarray,
    reference_edge_mask: np.ndarray,
    cad_edge_mask: np.ndarray,
    *,
    use_envelope_bbox: bool = True,
    config: ScorerV2Config,
    reference_cache: _ReferenceSpatialEdgeMassCacheV2 | None = None,
) -> SpatialEdgeMassMetricsV2:
    """Compare local edge capacity without allowing many-to-one credit.

    Tolerant precision/recall deliberately answers whether an edge has a
    nearby counterpart.  That neighbourhood test alone cannot distinguish
    one contour from two nearby contours: both CAD contours may be within the
    tolerance of the same reference contour.  This complementary score bins
    registered visible-edge pixels spatially and caps matched mass in every
    cell at ``min(reference_count, cad_count)``.

    The reference envelope normally supplies the comparison box.  An
    unstable/empty envelope falls back to the reference-edge box so the
    labelled V2 fallback path still returns complete diagnostics.  The strict
    no-envelope-match ablation explicitly selects the reference-edge box even
    when an envelope exists.  Multiple deterministic phase origins reduce
    cell-boundary sensitivity; their Dice values are averaged.  All CAD edge
    pixels are counted and therefore cannot disappear by moving beyond the
    comparison box.
    """

    envelope = np.asarray(reference_envelope_mask, dtype=bool)
    reference = np.asarray(reference_edge_mask, dtype=bool)
    cad = np.asarray(cad_edge_mask, dtype=bool)
    if envelope.shape != reference.shape or reference.shape != cad.shape:
        raise ValueError("Spatial edge-mass masks must have identical shapes")

    cache = reference_cache or _prepare_reference_spatial_edge_mass_cache_v2(
        envelope,
        reference,
        use_envelope_bbox=use_envelope_bbox,
        config=config,
    )
    expected_phase_offsets = tuple(
        (float(offset_x), float(offset_y))
        for offset_x, offset_y in config.spatial_edge_mass_phase_offsets
    )
    if (
        cache.shape != reference.shape
        or cache.version != config.spatial_edge_mass_version
        or cache.grid_size != config.spatial_edge_mass_grid_size
        or cache.phase_offsets != expected_phase_offsets
        or cache.use_envelope_bbox != use_envelope_bbox
    ):
        raise ValueError(
            "Reference spatial edge-mass cache does not match scoring inputs"
        )
    cad_total_count = int(np.count_nonzero(cad))
    bbox = cache.bbox_px
    phase_records: list[SpatialEdgeMassPhaseMetricsV2] = []
    if bbox is None:
        for offset_x, offset_y in config.spatial_edge_mass_phase_offsets:
            phase_records.append(
                SpatialEdgeMassPhaseMetricsV2(
                    offset_x_fraction=float(offset_x),
                    offset_y_fraction=float(offset_y),
                    dice=0.0,
                    matched_edge_pixels=0,
                    reference_edge_pixels=0,
                    cad_edge_pixels=cad_total_count,
                    cad_edge_pixels_inside_bbox=0,
                    cad_unmatched_overflow_pixels=cad_total_count,
                    active_cells=int(cad_total_count > 0),
                )
            )
        return SpatialEdgeMassMetricsV2(
            version=config.spatial_edge_mass_version,
            grid_size=config.spatial_edge_mass_grid_size,
            bbox_px=None,
            score=0.0,
            aggregation="arithmetic_mean",
            phases=tuple(phase_records),
        )

    x0, y0, x1, y1 = bbox
    cad_y, cad_x = np.nonzero(cad[y0:y1, x0:x1])
    reference_count = cache.reference_edge_pixels
    cad_inside_count = len(cad_x)
    cad_overflow_count = cad_total_count - cad_inside_count
    grid_size = config.spatial_edge_mass_grid_size
    # A positive phase can create one trailing cell beyond the nominal 4x4
    # grid.  It is intentional and applies identically to both masks.
    bins_per_axis = grid_size + 1
    cad_scaled_x = cad_x.astype(np.float64) * grid_size / (x1 - x0)
    cad_scaled_y = cad_y.astype(np.float64) * grid_size / (y1 - y0)
    cell_count = bins_per_axis * bins_per_axis

    for phase_index, (offset_x, offset_y) in enumerate(
        config.spatial_edge_mass_phase_offsets
    ):
        cad_ix = np.floor(cad_scaled_x + offset_x).astype(np.int64)
        cad_iy = np.floor(cad_scaled_y + offset_y).astype(np.int64)
        cad_inside_cells = np.bincount(
            cad_iy * bins_per_axis + cad_ix,
            minlength=cell_count,
        )
        # The last cell is a phase-invariant overflow bucket.  It has zero
        # reference capacity, so bbox-external CAD edges increase the Dice
        # denominator but can never be counted as a spatial match.
        reference_cells = cache.reference_cells[phase_index]
        cad_cells = np.append(cad_inside_cells, cad_overflow_count)
        matched = int(np.minimum(reference_cells, cad_cells).sum())
        denominator = reference_count + cad_total_count
        dice = 2.0 * matched / denominator if denominator else 0.0
        phase_records.append(
            SpatialEdgeMassPhaseMetricsV2(
                offset_x_fraction=float(offset_x),
                offset_y_fraction=float(offset_y),
                dice=float(np.clip(dice, 0.0, 1.0)),
                matched_edge_pixels=matched,
                reference_edge_pixels=reference_count,
                cad_edge_pixels=cad_total_count,
                cad_edge_pixels_inside_bbox=cad_inside_count,
                cad_unmatched_overflow_pixels=cad_overflow_count,
                active_cells=int(np.count_nonzero(reference_cells + cad_cells)),
            )
        )

    return SpatialEdgeMassMetricsV2(
        version=config.spatial_edge_mass_version,
        grid_size=grid_size,
        bbox_px=bbox,
        score=float(np.mean([phase.dice for phase in phase_records])),
        aggregation="arithmetic_mean",
        phases=tuple(phase_records),
    )


def _score_mask_pair_v2(
    reference_envelope: np.ndarray,
    reference_edges: np.ndarray,
    cad_envelope: np.ndarray,
    cad_edges: np.ndarray,
    *,
    cad_support_stats: dict[str, float | tuple[float, float]] | None = None,
    tolerance_px: float,
    reference_stable: bool,
    cad_stable: bool,
    mode: str,
    config: ScorerV2Config,
    reference_edge_cache: _ReferenceEdgeCacheV2 | None = None,
    reference_spatial_cache: _ReferenceSpatialEdgeMassCacheV2 | None = None,
    reference_transport_cache: _ReferenceCapacityTransportCacheV2 | None = None,
) -> _MaskScoreV2:
    gross = _envelope_match_metrics_v2(
        reference_envelope,
        cad_envelope,
        candidate_support_mask=cad_edges,
        candidate_support_stats=cad_support_stats,
    )
    if cad_support_stats is not None:
        gross = replace(
            gross,
            support_outside_fraction=float(
                cad_support_stats.get("outside_fraction", 0.0)
            ),
        )
    edge = _tolerant_edge_metrics_v2(
        reference_edges,
        cad_edges,
        tolerance_px=tolerance_px,
        reference_cache=reference_edge_cache,
    )
    capacity_transport = _capacity_transport_metrics_v2(
        reference_edges,
        cad_edges,
        config=config,
        reference_cache=reference_transport_cache,
    )
    if mode == "capacity_transport":
        # Capacity transport is the complete production score.  Deliberately
        # do not compute the legacy spatial grid here: it is neither a hidden
        # feature nor a diagnostic gate in this mode.
        spatial_edge_mass = SpatialEdgeMassMetricsV2(
            version=0,
            grid_size=0,
            bbox_px=None,
            score=0.0,
            aggregation="not_computed",
            phases=(),
        )
        failed = []
        if edge.reference_edge_pixels == 0:
            failed.append("reference_edges_empty")
        if edge.cad_edge_pixels == 0:
            failed.append("cad_edges_empty")
        if gross.support_outside_fraction > config.max_support_outside_fraction:
            failed.append("projection_outside_crop")
        edge_score = capacity_transport.score
        raw_score = capacity_transport.score
        eligible = not failed
        return _MaskScoreV2(
            score=raw_score if eligible else 0.0,
            raw_score=raw_score,
            eligible=eligible,
            failed_gates=tuple(failed),
            warnings=(),
            envelope_score=0.0,
            edge_score=edge_score,
            gross=gross,
            edge=edge,
            spatial_edge_mass=spatial_edge_mass,
            capacity_transport=capacity_transport,
        )
    spatial_edge_mass = _spatial_edge_mass_metrics_v2(
        reference_envelope,
        reference_edges,
        cad_edges,
        use_envelope_bbox=mode != "no_envelope_match",
        config=config,
        reference_cache=reference_spatial_cache,
    )
    edge_score = float(math.sqrt(edge.f1 * spatial_edge_mass.score))
    if mode == "no_envelope_match":
        failed = []
        if edge.reference_edge_pixels == 0:
            failed.append("reference_edges_empty")
        if edge.cad_edge_pixels == 0:
            failed.append("cad_edges_empty")
        if gross.support_outside_fraction > config.max_support_outside_fraction:
            failed.append("projection_outside_crop")
        raw_score = edge_score
        eligible = not failed
        return _MaskScoreV2(
            score=raw_score if eligible else 0.0,
            raw_score=raw_score,
            eligible=eligible,
            failed_gates=tuple(failed),
            warnings=(),
            envelope_score=0.0,
            edge_score=edge_score,
            gross=gross,
            edge=edge,
            spatial_edge_mass=spatial_edge_mass,
            capacity_transport=capacity_transport,
        )
    if mode == "edge_fallback":
        failed = []
        if edge.reference_edge_pixels == 0:
            failed.append("reference_edges_empty")
        if edge.cad_edge_pixels == 0:
            failed.append("cad_edges_empty")
        # A failed CAD closed contour must not erase information that is still
        # exact in the full visible HLR support.  Whenever the reference gross
        # envelope is stable, retain the aspect, two-axis fill, and centroid
        # gates even though Dice/area are unavailable in this fallback path.
        if reference_stable:
            if gross.aspect_error > config.aspect_log_gate:
                failed.append("aspect_ratio")
            if gross.width_ratio < config.min_axis_fill:
                failed.append("width_fill_low")
            if gross.height_ratio < config.min_axis_fill:
                failed.append("height_fill_low")
            if gross.width_ratio > config.max_axis_fill:
                failed.append("width_fill_high")
            if gross.height_ratio > config.max_axis_fill:
                failed.append("height_fill_high")
            if gross.center_error > config.max_center_error:
                failed.append("center_offset")
        if gross.support_outside_fraction > config.max_support_outside_fraction:
            failed.append("projection_outside_crop")
        warnings = []
        if not reference_stable:
            warnings.append("reference_envelope_unstable")
        if not cad_stable:
            warnings.append("cad_envelope_unstable")
        raw_score = float(edge_score * config.fallback_score_scale)
        eligible = not failed
        return _MaskScoreV2(
            score=raw_score if eligible else 0.0,
            raw_score=raw_score,
            eligible=eligible,
            failed_gates=tuple(failed),
            warnings=tuple(warnings),
            envelope_score=0.0,
            edge_score=edge_score,
            gross=gross,
            edge=edge,
            spatial_edge_mass=spatial_edge_mass,
            capacity_transport=capacity_transport,
        )
    if mode != "envelope":
        raise ValueError(f"Unknown V2 scoring mode: {mode}")

    failed_gates: list[str] = []
    if not reference_stable:
        failed_gates.append("reference_envelope_unstable")
    if not cad_stable:
        failed_gates.append("cad_envelope_unstable")
    if gross.aspect_error > config.aspect_log_gate:
        failed_gates.append("aspect_ratio")
    if gross.area_log_error > config.max_area_log_error:
        failed_gates.append("area_ratio")
    if gross.width_ratio < config.min_axis_fill:
        failed_gates.append("width_fill_low")
    if gross.height_ratio < config.min_axis_fill:
        failed_gates.append("height_fill_low")
    if gross.width_ratio > config.max_axis_fill:
        failed_gates.append("width_fill_high")
    if gross.height_ratio > config.max_axis_fill:
        failed_gates.append("height_fill_high")
    if gross.center_error > config.max_center_error:
        failed_gates.append("center_offset")
    if gross.support_outside_fraction > config.max_support_outside_fraction:
        failed_gates.append("projection_outside_crop")
    if gross.dice < config.min_envelope_dice:
        failed_gates.append("envelope_overlap")

    aspect_score = math.exp(-gross.aspect_error / config.aspect_temperature)
    size_error = 0.5 * (
        abs(math.log(max(gross.width_ratio, 1e-12)))
        + abs(math.log(max(gross.height_ratio, 1e-12)))
    )
    size_score = math.exp(-size_error / config.size_temperature)
    center_score = math.exp(-gross.center_error / config.center_temperature)
    envelope_score = float(
        math.exp(
            0.50 * math.log(max(gross.dice, 1e-12))
            + 0.20 * math.log(max(aspect_score, 1e-12))
            + 0.20 * math.log(max(size_score, 1e-12))
            + 0.10 * math.log(max(center_score, 1e-12))
        )
    )
    raw_score = float(
        max(envelope_score, 1e-12) ** config.envelope_weight
        * max(edge_score, 1e-12) ** (1.0 - config.envelope_weight)
    )
    eligible = not failed_gates
    return _MaskScoreV2(
        score=raw_score if eligible else 0.0,
        raw_score=raw_score,
        eligible=eligible,
        failed_gates=tuple(failed_gates),
        warnings=(),
        envelope_score=envelope_score,
        edge_score=edge_score,
        gross=gross,
        edge=edge,
        spatial_edge_mass=spatial_edge_mass,
        capacity_transport=capacity_transport,
    )


def _cad_geometry_for_transform_v2(
    projection: Projection2D,
    transform: SimilarityTransform,
    roi_px: tuple[int, int, int, int],
    *,
    config: ScorerV2Config,
) -> EnvelopeGeometryV2:
    roi_x, roi_y, roi_width, roi_height = roi_px
    local_transform = SimilarityTransform(
        scale_px_per_unit=transform.scale_px_per_unit,
        rotation_deg=transform.rotation_deg,
        center_x_px=transform.center_x_px - roi_x,
        center_y_px=transform.center_y_px - roi_y,
        model_center_x=transform.model_center_x,
        model_center_y=transform.model_center_y,
        reflected=transform.reflected,
    )
    ink = _rasterize_visible_edges(
        (roi_width, roi_height),
        projection.visible_polylines,
        local_transform,
    )
    return _geometry_from_binary_ink_v2(
        ink,
        roi_px=roi_px,
        config=config,
    )


def _validate_reference_geometry_v2(
    geometry: EnvelopeGeometryV2,
    image_size: tuple[int, int],
    *,
    roi: tuple[int, int, int, int] | None,
) -> None:
    expected_roi = _validate_pixel_roi(roi, image_size) or (
        0,
        0,
        image_size[0],
        image_size[1],
    )
    if geometry.diagnostics.roi_px != expected_roi:
        raise ValueError(
            "Cached V2 reference geometry ROI does not match this evaluation"
        )
    expected_shape = (expected_roi[3], expected_roi[2])
    masks = (
        geometry.ink_mask,
        geometry.envelope_mask,
        geometry.boundary_mask,
        geometry.edge_mask,
    )
    if any(np.asarray(mask).shape != expected_shape for mask in masks):
        raise ValueError("Cached V2 reference mask dimensions do not match its ROI")


def _alignment_points(
    polylines: tuple[np.ndarray, ...], *, max_points: int
) -> np.ndarray:
    """Sample all edges at near-uniform arc-length spacing.

    Tessellation vertices are an implementation detail: a circle may contain
    hundreds while a long straight edge contains only its two end points.
    Optimising those raw vertices badly overweights curved internal features.
    Arc-length resampling makes every millimetre of projected edge contribute
    approximately the same amount to registration.
    """

    lengths = np.asarray(
        [
            float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())
            for line in polylines
        ]
    )
    total_length = float(lengths.sum())
    if total_length <= 1e-12:
        return np.concatenate(polylines, axis=0)

    spacing = total_length / max(max_points - len(polylines), 1)
    sampled: list[np.ndarray] = []
    for line, length in zip(polylines, lengths, strict=True):
        segment_lengths = np.linalg.norm(np.diff(line, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        count = max(2, math.ceil(length / spacing) + 1)
        distances = np.linspace(0.0, length, count)
        sampled.append(
            np.column_stack(
                (
                    np.interp(distances, cumulative, line[:, 0]),
                    np.interp(distances, cumulative, line[:, 1]),
                )
            )
        )

    points = np.concatenate(sampled, axis=0)
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points).astype(np.int64)
        points = points[indices]
    return points


def _alignment_objective(
    points: np.ndarray,
    *,
    model_center: tuple[float, float],
    reflected: bool,
    distance_map: np.ndarray,
    cap_px: float,
):
    image_height, image_width = distance_map.shape

    def objective(parameters: np.ndarray) -> float:
        transform = SimilarityTransform(
            scale_px_per_unit=math.exp(float(parameters[0])),
            rotation_deg=float(parameters[1]),
            center_x_px=float(parameters[2]),
            center_y_px=float(parameters[3]),
            model_center_x=float(model_center[0]),
            model_center_y=float(model_center[1]),
            reflected=reflected,
        )
        pixels = transform.apply(points)
        valid = (
            (pixels[:, 0] >= 0.0)
            & (pixels[:, 0] <= image_width - 1.0)
            & (pixels[:, 1] >= 0.0)
            & (pixels[:, 1] <= image_height - 1.0)
        )
        outside_fraction = 1.0 - float(np.mean(valid))
        if not np.any(valid):
            return 5.0
        valid_pixels = pixels[valid]
        distances = ndimage.map_coordinates(
            distance_map,
            [valid_pixels[:, 1], valid_pixels[:, 0]],
            order=1,
            mode="constant",
            cval=cap_px,
        )
        # Cap isolated mismatches so an imperfect candidate can still register
        # on its shared structure, but keep all points in the mean so matching
        # only a tiny subset is never enough.
        robust_distance = float(np.mean(np.minimum(distances, cap_px))) / cap_px
        return robust_distance + 2.0 * outside_fraction

    return objective


def _alignment_metrics(
    pixels: np.ndarray,
    distance_map: np.ndarray,
    reference_threshold: int,
    cap_px: float,
) -> AlignmentMetrics:
    image_height, image_width = distance_map.shape
    valid = (
        (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] <= image_width - 1.0)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] <= image_height - 1.0)
    )
    outside_fraction = 1.0 - float(np.mean(valid))
    if np.any(valid):
        valid_pixels = pixels[valid]
        distances = ndimage.map_coordinates(
            distance_map,
            [valid_pixels[:, 1], valid_pixels[:, 0]],
            order=1,
            mode="constant",
            cval=cap_px,
        )
    else:
        distances = np.asarray([cap_px], dtype=np.float64)
    loss = float(
        np.mean(np.minimum(distances, cap_px)) / cap_px + 2.0 * outside_fraction
    )
    return AlignmentMetrics(
        loss=loss,
        mean_distance_px=float(np.mean(distances)),
        median_distance_px=float(np.median(distances)),
        p90_distance_px=float(np.percentile(distances, 90)),
        inlier_fraction_3px=float(np.mean(distances <= 3.0)),
        outside_fraction=outside_fraction,
        reference_threshold=reference_threshold,
    )


def _draw_polylines(
    draw: ImageDraw.ImageDraw,
    polylines: tuple[np.ndarray, ...],
    transform: SimilarityTransform,
    *,
    fill: tuple[int, int, int, int],
    width: int,
) -> None:
    for polyline in polylines:
        pixels = transform.apply(polyline)
        if len(pixels) < 2:
            continue
        draw.line(
            [tuple(map(float, point)) for point in pixels],
            fill=fill,
            width=width,
            joint="curve",
        )


def _rasterize_visible_edges(
    image_size: tuple[int, int],
    polylines: tuple[np.ndarray, ...],
    transform: SimilarityTransform,
) -> np.ndarray:
    """Rasterize aligned visible edges for reference-to-CAD distances."""

    image = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(image)
    for polyline in polylines:
        pixels = transform.apply(polyline)
        if len(pixels) >= 2:
            draw.line(
                [tuple(map(float, point)) for point in pixels],
                fill=255,
                width=2,
                joint="curve",
            )
    return np.asarray(image, dtype=np.uint8) > 0


def _render_cad_view(
    image_size: tuple[int, int],
    projection: Projection2D,
    transform: SimilarityTransform,
    *,
    show_hidden_lines: bool = True,
) -> Image.Image:
    """Render the current CAD as a clean, aligned in-memory view."""

    image = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(image)
    if show_hidden_lines and projection.section_origin is None:
        _draw_dashed_polylines(
            draw,
            projection.hidden_polylines,
            transform,
            fill=(135, 145, 155),
            width=1,
            dash_px=8.0,
            gap_px=5.0,
        )
    _draw_polylines(
        draw,
        projection.visible_polylines,
        transform,
        fill=(20, 25, 30, 255),
        width=2,
    )
    return image


def _validate_render_size(
    width: int,
    height: int,
    margin: int,
) -> tuple[int, int, int]:
    values = (width, height, margin)
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in values
    ):
        raise ValueError("Render width, height, and margin must be integers")
    width, height, margin = (int(value) for value in values)
    if width <= 0 or height <= 0:
        raise ValueError("Render width and height must be positive")
    if margin < 0 or 2 * margin >= width or 2 * margin >= height:
        raise ValueError("Render margin must leave a positive drawing area")
    return width, height, margin


def _fit_projection_transform(
    projection: Projection2D,
    *,
    image_size: tuple[int, int],
    margin: int,
) -> SimilarityTransform:
    bounds = np.asarray(projection.bounds, dtype=np.float64)
    if bounds.shape != (4,) or not np.all(np.isfinite(bounds)):
        raise ValueError(f"Projection has invalid bounds: {projection.bounds!r}")
    projection_width = float(bounds[2] - bounds[0])
    projection_height = float(bounds[3] - bounds[1])
    if projection_width < 0.0 or projection_height < 0.0:
        raise ValueError(f"Projection has inverted bounds: {projection.bounds!r}")
    if projection_width <= 1e-12 and projection_height <= 1e-12:
        raise ValueError("Projection has no drawable extent")

    available_width = image_size[0] - 2 * margin
    available_height = image_size[1] - 2 * margin
    scale_options = []
    if projection_width > 1e-12:
        scale_options.append(available_width / projection_width)
    if projection_height > 1e-12:
        scale_options.append(available_height / projection_height)
    scale = min(scale_options)
    return SimilarityTransform(
        scale_px_per_unit=float(scale),
        rotation_deg=0.0,
        center_x_px=image_size[0] * 0.5,
        center_y_px=image_size[1] * 0.5,
        model_center_x=projection.center[0],
        model_center_y=projection.center[1],
    )


def _draw_dashed_polylines(
    draw: ImageDraw.ImageDraw,
    polylines: tuple[np.ndarray, ...],
    transform: SimilarityTransform,
    *,
    fill: tuple[int, int, int],
    width: int,
    dash_px: float,
    gap_px: float,
) -> None:
    for polyline in polylines:
        pixels = transform.apply(polyline)
        if len(pixels) < 2:
            continue
        segment_lengths = np.linalg.norm(np.diff(pixels, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        total_length = float(cumulative[-1])
        for dash_start in np.arange(0.0, total_length, dash_px + gap_px):
            dash_end = min(float(dash_start + dash_px), total_length)
            internal = cumulative[(cumulative > dash_start) & (cumulative < dash_end)]
            distances = np.concatenate(([dash_start], internal, [dash_end]))
            dash_points = np.column_stack(
                (
                    np.interp(distances, cumulative, pixels[:, 0]),
                    np.interp(distances, cumulative, pixels[:, 1]),
                )
            )
            draw.line(
                [tuple(map(float, point)) for point in dash_points],
                fill=fill,
                width=width,
                joint="curve",
            )


def _write_comparison(
    reference: Image.Image,
    cad_view: Image.Image,
    output_path: Path,
    *,
    view_name: str,
    score: float | None,
) -> Path:
    """Compose the only public image: original input beside current CAD."""

    if cad_view.size != reference.size:
        raise ValueError("Both comparison panels must have the same image size")

    panel_width, panel_height = reference.size
    margin, gap, header_height = 16, 16, 64
    canvas_width = 2 * margin + 2 * panel_width + gap
    canvas_height = 2 * margin + header_height + panel_height
    canvas = Image.new("RGB", (canvas_width, canvas_height), (244, 246, 248))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
        score_font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
        score_font = ImageFont.load_default()

    panels = (
        (reference.convert("RGB"), "1  Original input"),
        (cad_view.convert("RGB"), f"2  Current CAD  ({view_name})"),
    )
    image_y = margin + header_height
    for index, (panel, title) in enumerate(panels):
        image_x = margin + index * (panel_width + gap)
        draw.text((image_x, margin + 6), title, fill=(25, 32, 42), font=font)
        canvas.paste(panel, (image_x, image_y))
        draw.rectangle(
            (
                image_x - 1,
                image_y - 1,
                image_x + panel_width,
                image_y + panel_height,
            ),
            outline=(190, 197, 205),
            width=1,
        )

    if score is not None:
        score_text = f"score = {score:.4f}"
        draw.text(
            (margin + panel_width + gap, margin + 35),
            score_text,
            fill=(48, 83, 125),
            font=score_font,
        )

    canvas.save(output_path, optimize=True)
    return output_path


def _write_comparison_sheet(
    comparison_images: Sequence[str | Path],
    output_path: str | Path,
    *,
    score: float | None = None,
) -> Path:
    """Stack comparisons into equal-size rows without cropping or distortion."""

    if not comparison_images:
        raise ValueError("Comparison sheet requires at least one image")
    opened = [Image.open(path).convert("RGB") for path in comparison_images]
    try:
        target_width = max(image.width for image in opened)
        scaled = []
        for image in opened:
            target_height = max(1, round(image.height * target_width / image.width))
            if image.width == target_width:
                scaled.append(image.copy())
            else:
                scaled.append(
                    image.resize(
                        (target_width, target_height), Image.Resampling.LANCZOS
                    )
                )

        row_height = max(image.height for image in scaled)
        gap = 16
        header_height = 52 if score is not None else 0
        canvas_height = (
            header_height + len(scaled) * row_height + (len(scaled) - 1) * gap
        )
        canvas = Image.new("RGB", (target_width, canvas_height), (238, 241, 244))
        if score is not None:
            draw = ImageDraw.Draw(canvas)
            try:
                font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
            except OSError:
                font = ImageFont.load_default()
            draw.text(
                (16, 12),
                f"Overall geometry score = {score:.4f}",
                fill=(35, 72, 115),
                font=font,
            )
        for index, image in enumerate(scaled):
            row_y = header_height + index * (row_height + gap)
            image_y = row_y + (row_height - image.height) // 2
            canvas.paste(image, (0, image_y))
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output, optimize=True)
        return output
    finally:
        for image in opened:
            image.close()


def write_comparison_sheet(
    comparison_images: Sequence[str | Path],
    output_path: str | Path,
    *,
    score: float | None = None,
) -> Path:
    """Compose aligned per-region comparisons into one stable sheet artifact."""

    if score is not None and (
        not math.isfinite(float(score)) or not 0.0 <= float(score) <= 1.0
    ):
        raise ValueError("Comparison-sheet score must be finite and in [0, 1]")
    return _write_comparison_sheet(
        comparison_images,
        output_path,
        score=None if score is None else float(score),
    )


__all__ = [
    "CAPACITY_TRANSPORT_SCORER_V1_CONFIG",
    "DEFAULT_SCORER_V2_CONFIG",
    "VIEW_PRESETS",
    "AlignmentDiagnosticsV2",
    "AlignmentMetrics",
    "AlignmentResult",
    "AlignmentResultV2",
    "AlignmentTrialV2",
    "CandidateRankingResult",
    "ComparisonResult",
    "DrawingViewSpec",
    "EdgeMatchMetricsV2",
    "EnvelopeDiagnosticsV2",
    "EnvelopeGeometryV2",
    "EnvelopeMatchMetricsV2",
    "GeometryMetrics",
    "GeometryMetricsV2",
    "Projection2D",
    "ProjectionEvaluationV2",
    "RankedViewCandidate",
    "ScorerV2Config",
    "SimilarityTransform",
    "SpatialEdgeMassMetricsV2",
    "SpatialEdgeMassPhaseMetricsV2",
    "ViewCandidate",
    "ViewPreset",
    "ViewSpec",
    "align_projection",
    "align_projection_v2",
    "compare_step_to_drawing",
    "compare_step_to_drawings",
    "compare_view_candidates",
    "crop_normalized_region",
    "crop_normalized_roi",
    "evaluate_projection_v2",
    "export_projection_svg",
    "extract_envelope_geometry_v2",
    "normalized_roi_to_pixels",
    "project_step",
    "rank_view_candidates_detailed",
    "render_projection_png",
    "render_step_view",
    "score_aligned_geometry",
    "score_aligned_geometry_v2",
    "write_comparison_sheet",
]
