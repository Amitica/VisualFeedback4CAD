"""Conservative changes in continuous, direction-compatible target line support.

This compares binary ink in one shared pixel frame. It does not distinguish
physical outlines from dimension lines or text. Short contacts, ambiguous local
directions, and improvements that remain far from the target are left uncolored.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.ndimage import correlate, distance_transform_edt, find_objects, label


def _local_directions(
    ink: np.ndarray, radius: int, minimum_coherence: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate the principal ink direction in a square neighborhood by PCA."""
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    samples = ink.astype(np.float64)
    count = correlate(samples, np.ones(xx.shape), mode="constant", cval=0.0)
    denominator = np.maximum(count, 1.0)

    def moment(kernel: np.ndarray) -> np.ndarray:
        return (
            correlate(samples, kernel.astype(np.float64), mode="constant", cval=0.0)
            / denominator
        )

    mean_x, mean_y = moment(xx), moment(yy)
    cov_xx = np.maximum(moment(xx * xx) - mean_x * mean_x, 0.0)
    cov_yy = np.maximum(moment(yy * yy) - mean_y * mean_y, 0.0)
    cov_xy = moment(xx * yy) - mean_x * mean_y
    trace = cov_xx + cov_yy
    eigenvalue_gap = np.hypot(cov_xx - cov_yy, 2.0 * cov_xy)
    coherence = eigenvalue_gap / np.maximum(trace, np.finfo(np.float64).eps)
    major_variance = (trace + eigenvalue_gap) / 2.0
    angle = 0.5 * np.arctan2(2.0 * cov_xy, cov_xx - cov_yy)
    reliable = (
        ink & (count >= 5) & (major_variance >= 2.0) & (coherence >= minimum_coherence)
    )
    return np.cos(angle), np.sin(angle), reliable


def _shifted_slices(length: int, offset: int) -> tuple[slice, slice]:
    """Pair target and neighbor slices without wrapping image boundaries."""
    if offset >= 0:
        return slice(0, length - offset), slice(offset, length)
    return slice(-offset, length), slice(0, length + offset)


def _compatible_distances(
    target_directions: tuple[np.ndarray, np.ndarray, np.ndarray],
    cad_directions: tuple[np.ndarray, np.ndarray, np.ndarray],
    radius: float,
    angle_tolerance_deg: float,
) -> np.ndarray:
    target_x, target_y, target_valid = target_directions
    cad_x, cad_y, cad_valid = cad_directions
    height, width = target_valid.shape
    distances = np.full((height, width), np.inf, dtype=np.float64)
    cosine_limit = math.cos(math.radians(angle_tolerance_deg))
    integer_radius = math.ceil(radius)
    for dy in range(-integer_radius, integer_radius + 1):
        for dx in range(-integer_radius, integer_radius + 1):
            distance = math.hypot(dx, dy)
            if distance > radius or abs(dy) >= height or abs(dx) >= width:
                continue
            target_rows, cad_rows = _shifted_slices(height, dy)
            target_columns, cad_columns = _shifted_slices(width, dx)
            target_slice = np.s_[target_rows, target_columns]
            cad_slice = np.s_[cad_rows, cad_columns]
            # Principal directions are unoriented: opposite tangents agree.
            direction_agreement = np.abs(
                target_x[target_slice] * cad_x[cad_slice]
                + target_y[target_slice] * cad_y[cad_slice]
            )
            compatible = (
                target_valid[target_slice]
                & cad_valid[cad_slice]
                & (direction_agreement >= cosine_limit)
            )
            existing = distances[target_slice]
            np.minimum(existing, distance, out=existing, where=compatible)
    return distances


def _continuous_segments(mask: np.ndarray, minimum_length: float) -> np.ndarray:
    """Retain connected evidence with sufficient spatial span, not pixel area."""
    components, count = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    keep = np.zeros(count + 1, dtype=bool)
    for component, bounds in enumerate(find_objects(components), start=1):
        if bounds is None:
            continue
        rows, columns = bounds
        span = (
            math.hypot(rows.stop - rows.start - 1, columns.stop - columns.start - 1)
            + 1.0
        )
        keep[component] = span >= minimum_length
    return keep[components]


def _geometric_distances(ink: np.ndarray) -> np.ndarray:
    if not np.any(ink):
        return np.full(ink.shape, np.inf, dtype=np.float64)
    return distance_transform_edt(~ink)


def conservative_target_support(
    reference_ink: np.ndarray,
    current_ink: np.ndarray,
    preview_ink: np.ndarray,
    *,
    support_distance_px: float = 2.0,
    change_tolerance_px: float = 1.0,
    direction_tolerance_deg: float = 20.0,
    minimum_segment_length_px: float = 12.0,
    direction_radius_px: int = 3,
    minimum_direction_coherence: float = 0.6,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return green/red target masks and diagnostics in the full input frame.

    A colored target segment must have nearby, direction-compatible CAD ink on
    its better-supported side, and improve/worsen by more than the raster
    tolerance. Both the support and the changed portion must be continuous.
    An ordinary nearest-ink distance change is also required: losing a reliable
    PCA direction at a crossing cannot masquerade as losing an existing edge.
    Swapping Current and Preview swaps green and red exactly.

    Distances beyond ``support_distance_px + change_tolerance_px`` are unknown,
    not a score: they can only matter when the opposite side is already close.
    Image compositors may crop these masks after this full-frame calculation.
    """
    masks = tuple(
        np.asarray(value) for value in (reference_ink, current_ink, preview_ink)
    )
    if any(mask.ndim != 2 or mask.dtype != np.bool_ for mask in masks):
        raise ValueError("Line support inputs must be two-dimensional boolean ink masks")
    if any(mask.shape != masks[0].shape for mask in masks) or not all(masks[0].shape):
        raise ValueError("Line support inputs must share one non-empty pixel frame")
    positive = (support_distance_px, minimum_segment_length_px)
    if any(not math.isfinite(value) or value <= 0 for value in positive):
        raise ValueError(
            "Support distance and minimum segment length must be positive and finite"
        )
    if not math.isfinite(change_tolerance_px) or change_tolerance_px < 0:
        raise ValueError("Change tolerance must be finite and nonnegative")
    if (
        not math.isfinite(direction_tolerance_deg)
        or not 0 < direction_tolerance_deg < 90
    ):
        raise ValueError("Direction tolerance must lie strictly between 0 and 90 degrees")
    if (
        isinstance(direction_radius_px, bool)
        or not isinstance(direction_radius_px, int)
        or direction_radius_px < 2
    ):
        raise ValueError("Direction radius must be an integer of at least two pixels")
    if (
        not math.isfinite(minimum_direction_coherence)
        or not 0 < minimum_direction_coherence <= 1
    ):
        raise ValueError("Minimum direction coherence must lie in (0, 1]")

    reference, current, preview = masks
    target_directions = _local_directions(
        reference, direction_radius_px, minimum_direction_coherence
    )
    search_radius = support_distance_px + change_tolerance_px
    current_distance, preview_distance = (
        _compatible_distances(
            target_directions,
            _local_directions(cad, direction_radius_px, minimum_direction_coherence),
            search_radius,
            direction_tolerance_deg,
        )
        for cad in (current, preview)
    )
    current_supported = _continuous_segments(
        current_distance <= support_distance_px, minimum_segment_length_px
    )
    preview_supported = _continuous_segments(
        preview_distance <= support_distance_px, minimum_segment_length_px
    )
    with np.errstate(invalid="ignore"):
        change = current_distance - preview_distance
        geometric_change = _geometric_distances(current) - _geometric_distances(preview)
    green = _continuous_segments(
        preview_supported
        & (change > change_tolerance_px)
        & (geometric_change > change_tolerance_px),
        minimum_segment_length_px,
    )
    red = _continuous_segments(
        current_supported
        & (change < -change_tolerance_px)
        & (geometric_change < -change_tolerance_px),
        minimum_segment_length_px,
    )
    diagnostics = {
        "method": "continuous_directional_target_support_v1",
        "distance_domain": "complete_input_frame",
        "support_distance_px": support_distance_px,
        "change_tolerance_px_exclusive": change_tolerance_px,
        "direction_tolerance_deg": direction_tolerance_deg,
        "minimum_segment_length_px": minimum_segment_length_px,
        "direction_radius_px": direction_radius_px,
        "minimum_direction_coherence": minimum_direction_coherence,
        "requires_geometric_distance_change": True,
        "reference_ink_pixels": int(np.count_nonzero(reference)),
        "directional_reference_ink_pixels": int(np.count_nonzero(target_directions[2])),
        "current_supported_target_pixels": int(np.count_nonzero(current_supported)),
        "preview_supported_target_pixels": int(np.count_nonzero(preview_supported)),
        "green_pixels": int(np.count_nonzero(green)),
        "red_pixels": int(np.count_nonzero(red)),
    }
    return green, red, diagnostics
