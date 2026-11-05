"""Partial three-way contour correspondence in one frozen image frame.

Colors describe changes in raster contour agreement, not CAD correctness.
Target ink may include dimensions, centerlines and text; this module does not
claim to identify those semantics. Ambiguous correspondences abstain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

_MIN_LENGTH = 12.0
_CHANGE_TOLERANCE = 1.0
_ASSIGNMENT_MARGIN = 0.12


def _thin(ink: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning of ink strokes, rather than their two boundaries."""
    pixels = np.pad(ink, 1).copy()
    for _ in range(max(ink.shape)):
        changed = False
        for phase in (0, 1):
            p = pixels[1:-1, 1:-1]
            neighbors = (
                pixels[:-2, 1:-1],
                pixels[:-2, 2:],
                pixels[1:-1, 2:],
                pixels[2:, 2:],
                pixels[2:, 1:-1],
                pixels[2:, :-2],
                pixels[1:-1, :-2],
                pixels[:-2, :-2],
            )
            count = np.sum(neighbors, axis=0)
            transitions = sum(
                (~neighbors[index] & neighbors[(index + 1) % 8]).astype(np.uint8)
                for index in range(8)
            )
            north, east, south, west = (
                neighbors[0],
                neighbors[2],
                neighbors[4],
                neighbors[6],
            )
            if phase == 0:
                protected = (north & east & south) | (east & south & west)
            else:
                protected = (north & east & west) | (north & south & west)
            remove = p & (count >= 2) & (count <= 6) & (transitions == 1) & ~protected
            if np.any(remove):
                p[remove] = False
                changed = True
        if not changed:
            break
    return pixels[1:-1, 1:-1]


def _trace_ink(ink: np.ndarray) -> list[np.ndarray]:
    skeleton = _thin(ink)
    height, width = skeleton.shape
    nodes = set(np.flatnonzero(skeleton).tolist())
    adjacency: dict[int, list[int]] = {}
    for node in sorted(nodes):
        y, x = divmod(node, width)
        neighbors = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if not (dx or dy) or not (0 <= y + dy < height and 0 <= x + dx < width):
                    continue
                other = (y + dy) * width + x + dx
                if other not in nodes:
                    continue
                # Avoid triangular graph shortcuts at raster right angles.
                if dx and dy and (node + dx in nodes or node + dy * width in nodes):
                    continue
                neighbors.append(other)
        adjacency[node] = sorted(neighbors)
    visited: set[tuple[int, int]] = set()
    curves = []
    starts = [node for node in adjacency if len(adjacency[node]) != 2]
    starts.extend(node for node in adjacency if len(adjacency[node]) == 2)
    for start in starts:
        for neighbor in adjacency[start]:
            if (min(start, neighbor), max(start, neighbor)) in visited:
                continue
            chain = [start]
            previous, current = start, neighbor
            while True:
                visited.add((min(previous, current), max(previous, current)))
                chain.append(current)
                if current == start or len(adjacency[current]) != 2:
                    break
                following = next(
                    node for node in adjacency[current] if node != previous
                )
                if (min(current, following), max(current, following)) in visited:
                    break
                previous, current = current, following
            curves.append(
                np.asarray(
                    [(node % width, node // width) for node in chain], dtype=float
                )
            )
    return curves


def _length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _resample(
    points: np.ndarray, spacing: float = 1.5, maximum: int = 384
) -> np.ndarray:
    lengths = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    keep = np.r_[True, np.diff(lengths) > 1e-8]
    points, lengths = points[keep], lengths[keep]
    if len(points) < 2:
        return points
    distances = np.linspace(
        0, lengths[-1], min(maximum, max(2, math.ceil(lengths[-1] / spacing) + 1))
    )
    return np.column_stack(
        [np.interp(distances, lengths, points[:, axis]) for axis in (0, 1)]
    )


def _split_corners(points: np.ndarray) -> list[np.ndarray]:
    points = _resample(points)
    closed = len(points) > 3 and np.linalg.norm(points[0] - points[-1]) < 1.6
    if closed:
        points = points[:-1]
    if len(points) < 9:
        return [np.vstack((points, points[0])) if closed else points]
    indices = np.arange(len(points)) if closed else np.arange(3, len(points) - 3)
    before = points[indices] - points[(indices - 3) % len(points)]
    after = points[(indices + 3) % len(points)] - points[indices]
    cosine = np.sum(before * after, axis=1) / np.maximum(
        np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1), 1e-8
    )
    corners = []
    for index in indices[np.argsort(cosine)]:
        location = int(np.flatnonzero(indices == index)[0])
        if cosine[location] > math.cos(math.radians(65)):
            break
        if all(
            min(abs(index - old), len(points) - abs(index - old)) > 4 for old in corners
        ):
            corners.append(int(index))
    if not corners:
        return [np.vstack((points, points[0])) if closed else points]
    corners.sort()
    if closed:
        doubled = np.vstack((points, points))
        stops = [*corners[1:], corners[0] + len(points)]
        return [
            doubled[start : stop + 1]
            for start, stop in zip(corners, stops, strict=True)
        ]
    boundaries = [0, *corners, len(points) - 1]
    return [points[start : stop + 1] for start, stop in pairwise(boundaries)]


def _join_fragments(curves: list[np.ndarray]) -> list[np.ndarray]:
    """Join mutual, unambiguous nearby endpoints with compatible tangents."""
    curves = [curve for curve in curves if len(curve) >= 2 and _length(curve) >= 3]
    for _ in range(len(curves)):
        endpoints = []
        for index, curve in enumerate(curves):
            if np.linalg.norm(curve[0] - curve[-1]) < 1.6:
                continue
            for end in (0, 1):
                oriented = curve if end == 0 else curve[::-1]
                tangent = oriented[min(3, len(oriented) - 1)] - oriented[0]
                norm = np.linalg.norm(tangent)
                if norm > 1e-6:
                    endpoints.append((index, end, oriented[0], tangent / norm))
        if len(endpoints) < 2:
            break
        tree = cKDTree(np.asarray([item[2] for item in endpoints]))
        choices: dict[int, tuple[float, int]] = {}
        for first, item in enumerate(endpoints):
            proposals = []
            for second in tree.query_ball_point(item[2], 3.0):
                other = endpoints[second]
                if first == second or item[0] == other[0]:
                    continue
                agreement = -float(np.dot(item[3], other[3]))
                if agreement < math.cos(math.radians(25)):
                    continue
                cost = float(np.linalg.norm(item[2] - other[2])) + 3 * (1 - agreement)
                proposals.append((cost, second))
            proposals.sort()
            if proposals and (
                len(proposals) == 1 or proposals[1][0] - proposals[0][0] > 0.5
            ):
                choices[first] = proposals[0]
        pair = next(
            (
                (first, second)
                for first, (_, second) in sorted(choices.items())
                if choices.get(second, (0, -1))[1] == first
            ),
            None,
        )
        if pair is None:
            break
        first, second = (endpoints[index] for index in pair)
        left = curves[first[0]] if first[1] else curves[first[0]][::-1]
        right = curves[second[0]] if not second[1] else curves[second[0]][::-1]
        merged = np.vstack((left, right))
        curves = [
            curve
            for index, curve in enumerate(curves)
            if index not in (first[0], second[0])
        ]
        curves.append(merged)
    return curves


@dataclass
class _Curve:
    points: np.ndarray
    length: float
    closed: bool
    tangents: np.ndarray
    tree: cKDTree

    @classmethod
    def create(cls, points: np.ndarray) -> _Curve:
        sampled = _resample(points)
        closed = np.linalg.norm(sampled[0] - sampled[-1]) < 1.6
        if closed:
            sampled = sampled[:-1]
            tangents = np.roll(sampled, -2, axis=0) - np.roll(sampled, 2, axis=0)
        else:
            tangents = np.gradient(sampled, axis=0)
        tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-8)
        return cls(sampled, _length(points), closed, tangents, cKDTree(sampled))


def _point_curve_distances(points: np.ndarray, curve: _Curve) -> np.ndarray:
    """Distance to adjacent polyline segments, insensitive to sampling phase."""
    _, nearest = curve.tree.query(points)
    result = np.full(len(points), np.inf)
    for offset in (-1, 0):
        start = nearest + offset
        if curve.closed:
            start %= len(curve.points)
            end = (start + 1) % len(curve.points)
        else:
            start = np.clip(start, 0, len(curve.points) - 2)
            end = start + 1
        a, b = curve.points[start], curve.points[end]
        vector = b - a
        fraction = np.clip(
            np.sum((points - a) * vector, axis=1)
            / np.maximum(np.sum(vector * vector, axis=1), 1e-10),
            0,
            1,
        )
        result = np.minimum(
            result, np.linalg.norm(points - a - fraction[:, None] * vector, axis=1)
        )
    return result


def _deduplicate(curves: list[_Curve]) -> list[_Curve]:
    """Drop coincident HLR paths or contained fragments, not nearby double lines."""
    retained: list[_Curve] = []
    for curve in sorted(curves, key=lambda item: -item.length):
        duplicate = False
        for other in retained:
            if np.any(
                curve.points.min(axis=0) < other.points.min(axis=0) - 0.35
            ) or np.any(curve.points.max(axis=0) > other.points.max(axis=0) + 0.35):
                continue
            distances = _point_curve_distances(curve.points, other)
            if np.mean(distances <= 0.35) >= 0.98 and float(np.max(distances)) <= 0.7:
                duplicate = True
                break
        if not duplicate:
            retained.append(curve)
    return retained


def _closed_boundaries(curves: list[np.ndarray]) -> list[np.ndarray]:
    """Assemble an unbranched native edge cycle, including sharp corners.

    Only unique, effectively coincident endpoints may turn a sharp corner.
    The looser tangent-based join below remains responsible for raster gaps.
    Two nearby parallel edges must never be collapsed into a thin fake loop.
    """
    ends = [
        (index, end, points[0 if end == 0 else -1])
        for index, points in enumerate(curves)
        if np.linalg.norm(points[0] - points[-1]) >= 1.6
        for end in (0, 1)
    ]
    if len(ends) < 6:
        return curves
    tree = cKDTree(np.asarray([value[2] for value in ends]))
    links = {}
    for index, (curve, end, point) in enumerate(ends):
        options = [
            other
            for other in tree.query_ball_point(point, 0.25)
            if ends[other][0] != curve
        ]
        if len(options) == 1:
            other = ends[options[0]]
            links[curve, end] = (other[0], other[1])
    consumed, loops = set(), []
    for start in range(len(curves)):
        if start in consumed or (start, 0) not in links or (start, 1) not in links:
            continue
        chain, seen, state = [], set(), (start, 0)
        while state[0] not in seen:
            index, enter = state
            seen.add(index)
            points = curves[index] if enter == 0 else curves[index][::-1]
            chain.append(points)
            exit_state = (index, 1 - enter)
            following = links.get(exit_state)
            if following is None or links.get(following) != exit_state:
                break
            state = following
        else:
            if state == (start, 0) and len(seen) >= 3:
                joined = np.vstack(chain)
                loops.append(np.vstack((joined, joined[0])))
                consumed.update(seen)
    return [
        points for index, points in enumerate(curves) if index not in consumed
    ] + loops


def _curves(
    polylines: Any,
    ink: np.ndarray,
    *,
    statistics: dict[str, Any] | None = None,
) -> list[_Curve]:
    raw = _trace_ink(ink) if polylines is None else polylines
    if not isinstance(raw, (tuple, list)):
        raise TypeError(
            "CAD polylines must be a list/tuple of finite Nx2 pixel coordinates or None"
        )
    prepared = []
    for value in raw:
        points = np.asarray(value, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
            raise ValueError("Each polyline must contain finite Nx2 pixel coordinates")
        if len(points) >= 2:
            # Keep an uninterrupted closed boundary as one identity across
            # sharp-corner, chamfer and fillet edits. Splitting only the square
            # version into four sides makes those valid edits incomparable.
            if len(points) > 3 and np.linalg.norm(points[0] - points[-1]) < 1.6:
                prepared.append(points)
            else:
                prepared.extend(_split_corners(points))
    unique = _deduplicate(
        [
            _Curve.create(points)
            for points in prepared
            if len(points) >= 2 and _length(points) >= 3
        ]
    )
    joined = _join_fragments(
        _closed_boundaries(
            [
                np.vstack((curve.points, curve.points[0]))
                if curve.closed
                else curve.points
                for curve in unique
            ]
        )
    )
    if statistics is not None:
        statistics["short_segments_omitted"] = sum(
            _length(points) < _MIN_LENGTH for points in joined
        )
    return _deduplicate(
        [_Curve.create(points) for points in joined if _length(points) >= _MIN_LENGTH]
    )


def _transfer_closed_identity(
    preview: list[_Curve],
    current: list[_Curve],
    matches: dict[int, int],
    ambiguous: set[int],
    records: list[dict[str, Any]],
    current_distance: np.ndarray,
    preview_distance: np.ndarray,
) -> set[int]:
    """Recover unique translated closed curves, without moving the score frame.

    Only disappearing Current curves and novel Preview curves participate.
    Surviving copies cannot be reused as the identity of a newly added hole.
    Repeated equal shapes with competing identities abstain rather than being
    called additions. Shape here is translation invariant, never scale/rotation
    invariant; all Target distances below still use the frozen pixel frame.
    """
    available = [
        c
        for c, curve in enumerate(current)
        if c not in matches.values()
        and curve.closed
        and _novel_fraction(curve, preview_distance) > 0.2
    ]
    options: dict[int, list[tuple[int, dict[str, float]]]] = {}
    for p, curve in enumerate(preview):
        if (
            p in matches
            or not curve.closed
            or _novel_fraction(curve, current_distance) <= 0.2
        ):
            continue
        for c in available:
            metrics = _pair(curve, current[c], 2.5, identity=True)
            if (
                0.98 <= metrics["length_ratio"] <= 1 / 0.98
                and metrics["centered_mean_px"] <= 0.8
                and metrics["centered_direction_agreement"] >= 0.98
            ):
                options.setdefault(p, []).append((c, metrics))
    unresolved = set()
    for p, choices in options.items():
        c, metrics = choices[0]
        unique = (
            len(choices) == 1
            and sum(
                any(other_c == c for other_c, _ in other) for other in options.values()
            )
            == 1
        )
        if unique and p not in ambiguous:
            matches[p] = c
            records.append(
                {
                    "from": p,
                    "to": c,
                    "accepted": True,
                    "reason": "unique_disappearing_closed_shape_translation",
                    **metrics,
                }
            )
        else:
            unresolved.add(p)
            ambiguous.add(p)
            records.append(
                {
                    "from": p,
                    "to": c,
                    "accepted": False,
                    "reason": "competing_disappearing_closed_shapes",
                    **metrics,
                }
            )
    return unresolved


def _transfer_attached_identity(
    preview: list[_Curve],
    current: list[_Curve],
    matches: dict[int, int],
    ambiguous: set[int],
    records: list[dict[str, Any]],
    current_distance: np.ndarray,
    preview_distance: np.ndarray,
) -> None:
    """Carry a hole's attached chord with its unique translated boundary.

    A projected bore may contain a circle and a diameter edge. Transferring only
    the circle would incorrectly label its old, jointly moved diameter as new.
    Both endpoints must touch the matched boundary, and the same translation
    must explain the complete attached curve in both versions.
    """
    available = [
        c
        for c, curve in enumerate(current)
        if c not in matches.values()
        and not curve.closed
        and _novel_fraction(curve, preview_distance) > 0.2
    ]
    options: dict[int, list[tuple[int, int, int, float]]] = {}
    anchors = [
        (p, c) for p, c in matches.items() if preview[p].closed and current[c].closed
    ]
    for ap, ac in anchors:
        anchor = _pair(preview[ap], current[ac], 2.5, identity=True)
        if anchor["centered_mean_px"] > 0.8:
            continue
        shift = current[ac].points.mean(axis=0) - preview[ap].points.mean(axis=0)
        for p, curve in enumerate(preview):
            if p in matches or p in ambiguous or curve.closed:
                continue
            if (
                float(_point_curve_distances(curve.points[[0, -1]], preview[ap]).max())
                > 2.5
            ):
                continue
            for c in available:
                old = current[c]
                if not (0.98 <= curve.length / old.length <= 1 / 0.98):
                    continue
                if (
                    float(
                        _point_curve_distances(old.points[[0, -1]], current[ac]).max()
                    )
                    > 2.5
                ):
                    continue
                error = max(
                    float(_point_curve_distances(curve.points + shift, old).max()),
                    float(_point_curve_distances(old.points - shift, curve).max()),
                )
                if error <= 0.8:
                    options.setdefault(p, []).append((c, ap, ac, error))
    for p, choices in options.items():
        c, ap, ac, error = choices[0]
        if (
            len({item[0] for item in choices}) != 1
            or sum(any(item[0] == c for item in other) for other in options.values())
            != 1
        ):
            ambiguous.add(p)
            continue
        matches[p] = c
        records.append(
            {
                "from": p,
                "to": c,
                "accepted": True,
                "reason": "attached_curve_follows_closed_shape_translation",
                "anchor_from": ap,
                "anchor_to": ac,
                "translated_max_error_px": error,
            }
        )


def _pair(
    a: _Curve, b: _Curve, gate: float, *, identity: bool = False
) -> dict[str, float]:
    _, bi = b.tree.query(a.points)
    _, ai = a.tree.query(b.points)
    ab = _point_curve_distances(a.points, b)
    ba = _point_curve_distances(b.points, a)
    agreement = min(
        float(np.mean(np.abs(np.sum(a.tangents * b.tangents[bi], axis=1)))),
        float(np.mean(np.abs(np.sum(b.tangents * a.tangents[ai], axis=1)))),
    )
    metrics = {
        "forward_mean_px": float(ab.mean()),
        "reverse_mean_px": float(ba.mean()),
        "symmetric_mean_px": float((ab.mean() + ba.mean()) / 2),
        "p90_px": max(float(np.percentile(ab, 90)), float(np.percentile(ba, 90))),
        "forward_coverage": float(np.mean(ab <= gate)),
        "reverse_coverage": float(np.mean(ba <= gate)),
        "direction_agreement": agreement,
        "length_ratio": a.length / max(b.length, 1e-8),
    }
    if identity:
        # Translation changes the nearest tangent on a small circle even when
        # the complete shape is identical. Centered shape evidence may establish
        # identity; all coverage, costs and change scores remain in the original
        # pixel frame. Neither rotation nor scale is fitted here.
        shift = b.points.mean(axis=0) - a.points.mean(axis=0)
        centered_ab, centered_bi = b.tree.query(a.points + shift)
        centered_ba, centered_ai = a.tree.query(b.points - shift)
        centered_agreement = min(
            float(
                np.mean(np.abs(np.sum(a.tangents * b.tangents[centered_bi], axis=1)))
            ),
            float(
                np.mean(np.abs(np.sum(b.tangents * a.tangents[centered_ai], axis=1)))
            ),
        )
        centered_error = float((centered_ab.mean() + centered_ba.mean()) / 2)
        metrics.update(
            spatial_direction_agreement=agreement,
            centered_direction_agreement=centered_agreement,
            centered_mean_px=centered_error,
        )
        if centered_error <= 2.5:
            metrics["direction_agreement"] = max(agreement, centered_agreement)
    return metrics


def _straight_extent_identity(a: _Curve, b: _Curve) -> dict[str, float] | None:
    """Associate collinear open segments whose endpoint extents changed.

    A trim has no novel Preview ink. It must still be compared with the longer
    Current segment. This exception applies only to near-exact straight support
    lines with substantial interval overlap, never to partial circles/arcs.
    Actual before/after errors continue to include unmatched endpoints.
    """
    if a.closed or b.closed:
        return None
    axes = []
    for curve in (a, b):
        vector = curve.points[-1] - curve.points[0]
        norm = float(np.linalg.norm(vector))
        if norm < _MIN_LENGTH:
            return None
        axis = vector / norm
        perpendicular = np.array([-axis[1], axis[0]])
        if (
            float(np.max(np.abs((curve.points - curve.points[0]) @ perpendicular)))
            > 0.8
        ):
            return None
        axes.append(axis)
    if abs(float(np.dot(*axes))) < math.cos(math.radians(3)):
        return None
    axis = axes[0]
    normal = np.array([-axis[1], axis[0]])
    offset = abs(float((a.points.mean(axis=0) - b.points.mean(axis=0)) @ normal))
    if offset > 1.5:
        return None
    first, second = a.points @ axis, b.points @ axis
    overlap = max(
        0.0,
        min(float(first.max()), float(second.max()))
        - max(float(first.min()), float(second.min())),
    )
    shorter, longer = sorted((a.length, b.length))
    if shorter / longer < 0.3 or overlap / shorter < 0.8:
        return None
    return {"support_offset_px": offset, "overlap_fraction": overlap / longer}


def _partial_assignment(
    a: list[_Curve], b: list[_Curve], gate: float
) -> tuple[dict[int, int], set[int], list[dict[str, Any]]]:
    if not a or not b:
        return {}, set(), []
    costs = np.full((len(a), len(b)), 1e6)
    pair_metrics = {}
    rejected: dict[int, dict[str, Any]] = {}
    for i, first in enumerate(a):
        for j, second in enumerate(b):
            # Cheap bounding-box rejection before nearest-curve queries.
            if np.any(
                first.points.min(axis=0) > second.points.max(axis=0) + 2 * gate
            ) or np.any(
                second.points.min(axis=0) > first.points.max(axis=0) + 2 * gate
            ):
                continue
            metrics = _pair(first, second, gate, identity=True)
            extent_identity = _straight_extent_identity(first, second)
            reason = None
            if extent_identity is not None:
                metrics.update(extent_identity=1.0, **extent_identity)
            elif not (0.55 <= metrics["length_ratio"] <= 1 / 0.55):
                reason = "length_ratio"
            elif min(metrics["forward_coverage"], metrics["reverse_coverage"]) < 0.85:
                reason = "bidirectional_coverage"
            elif metrics["direction_agreement"] < 0.72:
                reason = "direction"
            cost = (
                metrics["symmetric_mean_px"] / gate
                + 0.3 * abs(math.log(metrics["length_ratio"]))
                + 0.5 * (1 - metrics["direction_agreement"])
            )
            if extent_identity is not None:
                # Identity uses common support and interval overlap. This cost
                # is never substituted for the full-extent geometric error.
                cost = (
                    0.25 * extent_identity["support_offset_px"] / 1.5
                    + 0.45 * (1 - extent_identity["overlap_fraction"])
                    + 0.2 * abs(math.log(metrics["length_ratio"]))
                )
            if reason is None and cost < 1.15:
                costs[i, j] = cost
                pair_metrics[i, j] = metrics
            elif i not in rejected or cost < rejected[i]["cost"]:
                rejected[i] = {
                    "from": i,
                    "to": j,
                    "accepted": False,
                    "reason": reason or "unmatched_cost",
                    "cost": float(cost),
                    **metrics,
                }
    matrix = np.concatenate((costs, np.full((len(a), len(a)), 1.15)), axis=1)
    rows, columns = linear_sum_assignment(matrix)
    accepted, ambiguous, records = {}, set(), []
    for i, j in zip(rows.tolist(), columns.tolist(), strict=True):
        if j >= len(b):
            if np.any(costs[i] < 1.15):
                ambiguous.add(i)
                candidate = int(np.argmin(costs[i]))
                records.append(
                    {
                        "from": i,
                        "to": candidate,
                        "accepted": False,
                        "reason": "target_capacity",
                        "cost": float(costs[i, candidate]),
                        **pair_metrics[i, candidate],
                    }
                )
            elif i in rejected:
                records.append(rejected[i])
            continue
        row_alternative = float(np.min(np.delete(costs[i], j))) if len(b) > 1 else 1.15
        col_alternative = (
            float(np.min(np.delete(costs[:, j], i))) if len(a) > 1 else 1.15
        )
        margin = min(row_alternative, col_alternative, 1.15) - float(costs[i, j])
        reliable = margin >= _ASSIGNMENT_MARGIN
        records.append(
            {
                "from": i,
                "to": j,
                "accepted": reliable,
                "cost": float(costs[i, j]),
                "margin": float(margin),
                "reason": "accepted" if reliable else "competing_match",
                **pair_metrics[i, j],
            }
        )
        if reliable:
            accepted[i] = j
        else:
            ambiguous.add(i)
    # Nearby, well-aligned matches anchor ordering and contact relationships.
    # This is a consistency veto, not a semantic or graph-isomorphism claim.
    anchors = [
        record
        for record in records
        if record["accepted"]
        and record["symmetric_mean_px"] <= 2.5
        and record["margin"] >= 0.2
    ]
    for record in records:
        if not record["accepted"]:
            continue
        i, j = record["from"], record["to"]
        neighbors = sorted(
            (anchor for anchor in anchors if anchor["from"] != i),
            key=lambda anchor: np.linalg.norm(
                b[j].points.mean(axis=0) - b[anchor["to"]].points.mean(axis=0)
            ),
        )[:3]
        for anchor in neighbors:
            source_delta = a[i].points.mean(axis=0) - a[anchor["from"]].points.mean(
                axis=0
            )
            target_delta = b[j].points.mean(axis=0) - b[anchor["to"]].points.mean(
                axis=0
            )
            inverted = np.any(
                (np.abs(source_delta) > gate)
                & (np.abs(target_delta) > gate)
                & (source_delta * target_delta < 0)
            )
            source_gap = float(
                _point_curve_distances(a[i].points, a[anchor["from"]]).min()
            )
            target_gap = float(
                _point_curve_distances(b[j].points, b[anchor["to"]]).min()
            )
            contact_changed = (target_gap <= 2.5 and source_gap > gate) or (
                source_gap <= 2.5 and target_gap > gate
            )
            if inverted or contact_changed:
                accepted.pop(i, None)
                ambiguous.add(i)
                record.update(
                    accepted=False,
                    structural_conflict="relative_order" if inverted else "contact",
                    anchor_from=anchor["from"],
                    anchor_to=anchor["to"],
                )
                break
    return accepted, ambiguous, records


def _distance_to_ink(ink: np.ndarray) -> np.ndarray:
    if np.any(ink):
        return distance_transform_edt(~ink)
    return np.full(ink.shape, math.hypot(*ink.shape))


def _novel_fraction(curve: _Curve, distances: np.ndarray) -> float:
    xy = np.rint(curve.points).astype(int)
    valid = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] < distances.shape[1])
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < distances.shape[0])
    )
    if not np.any(valid):
        return 0.0
    return float(np.mean(distances[xy[valid, 1], xy[valid, 0]] > 1.5))


def _partial_target_support(curve: _Curve, target: list[_Curve]) -> float:
    """Local support can justify abstention even when whole-curve matching fails."""
    supported = np.zeros(len(curve.points), dtype=bool)
    for other in target:
        distances, indices = other.tree.query(curve.points)
        agreement = np.abs(np.sum(curve.tangents * other.tangents[indices], axis=1))
        supported |= (distances <= 2.5) & (agreement >= math.cos(math.radians(30)))
    return float(np.mean(supported))


def _attached_open_contours(index: int, target: list[_Curve]) -> list[int]:
    """Flag an attached leader/axis ambiguity without claiming line semantics."""
    curve = target[index]
    if not curve.closed:
        return []
    lo, hi = curve.points.min(axis=0) - 2.5, curve.points.max(axis=0) + 2.5
    attached = []
    for j, other in enumerate(target):
        if j == index or other.closed:
            continue
        # An open stroke attached to and extending outside a closed mark may
        # be a callout leader, a centerline, or real geometry. Ink alone cannot
        # establish that adding the mark restores a missing CAD contour.
        if not np.any((other.points < lo) | (other.points > hi)):
            continue
        if float(_point_curve_distances(other.points, curve).min()) <= 2.5:
            attached.append(j)
    return attached


def _competing_closed_subcontours(
    curve: _Curve,
    selected: int,
    target: list[_Curve],
    gate: float,
) -> list[int]:
    """Do not prefer an unrelated straight edge over a plausible loop fragment.

    Whole-chain assignment can exclude the correct closed Target boundary on
    length alone when HLR represents only one rounded side of it. A nearby rib
    then wins a deceptively large margin. Retain the possible partial-loop
    identity as ambiguity; do not silently let it bypass coverage/capacity and
    claim a match. This check does not force Current/Preview curve types equal.
    """
    line = target[selected]
    if curve.closed or line.closed:
        return []
    line_axis = line.points[-1] - line.points[0]
    line_length = float(np.linalg.norm(line_axis))
    curve_axis = curve.points[-1] - curve.points[0]
    curve_length = float(np.linalg.norm(curve_axis))
    if min(line_length, curve_length) < _MIN_LENGTH:
        return []
    line_normal = np.array([-line_axis[1], line_axis[0]]) / line_length
    curve_normal = np.array([-curve_axis[1], curve_axis[0]]) / curve_length
    if float(np.max(np.abs((line.points - line.points[0]) @ line_normal))) > 0.8:
        return []
    if float(np.max(np.abs((curve.points - curve.points[0]) @ curve_normal))) < 2.5:
        return []
    alternatives = []
    for index, other in enumerate(target):
        if index == selected or not other.closed or other.length < 1.4 * curve.length:
            continue
        # Directed support deliberately does not claim full-loop coverage. It
        # only vetoes confidence in a structurally different straight neighbor.
        distances = _point_curve_distances(curve.points, other)
        _, nearest = other.tree.query(curve.points)
        agreement = np.abs(np.sum(curve.tangents * other.tangents[nearest], axis=1))
        # Partial support is evidence of ambiguity, not sufficient evidence for
        # a correspondence. Keep the stricter 85% gate for accepted assignments.
        if float(np.mean((distances <= gate) & (agreement >= 0.8))) >= 0.65:
            alternatives.append(index)
    return alternatives


def _local_change_evidence(
    target: _Curve, current: _Curve, preview: _Curve
) -> dict[str, Any]:
    """Veto whole-curve colors when sustained Target spans move both ways."""
    change = _point_curve_distances(target.points, current) - _point_curve_distances(
        target.points, preview
    )
    improving, worsening = change > _CHANGE_TOLERANCE, change < -_CHANGE_TOLERANCE

    def longest_span(supported: np.ndarray) -> float:
        if target.closed:
            lengths = np.linalg.norm(
                np.roll(target.points, -1, axis=0) - target.points, axis=1
            )
            edges = supported & np.roll(supported, -1)
            lengths, edges = np.tile(lengths, 2), np.tile(edges, 2)
        else:
            lengths = np.linalg.norm(np.diff(target.points, axis=0), axis=1)
            edges = supported[:-1] & supported[1:]
        longest = running = 0.0
        for length, supported_edge in zip(lengths, edges, strict=True):
            running = running + float(length) if supported_edge else 0.0
            longest = max(longest, running)
        return min(longest, target.length)

    improved_span, worsened_span = longest_span(improving), longest_span(worsening)
    # A local fillet must not disappear merely because its parent perimeter is
    # long. Require a contiguous 6–12 px span, not an unbounded global fraction.
    minimum_span = max(6.0, min(12.0, 0.05 * target.length))
    return {
        "improved_fraction": float(np.mean(improving)),
        "worsened_fraction": float(np.mean(worsening)),
        "longest_improved_span_px": improved_span,
        "longest_worsened_span_px": worsened_span,
        "minimum_sustained_span_px": minimum_span,
        "sustained_improvement": improved_span >= minimum_span,
        "sustained_worsening": worsened_span >= minimum_span,
        "mixed": improved_span >= minimum_span and worsened_span >= minimum_span,
    }


def _paint(
    labels: np.ndarray,
    conflicts: np.ndarray,
    ink: np.ndarray,
    curve: _Curve,
    color: int,
) -> None:
    lo = np.maximum(np.floor(curve.points.min(axis=0) - 2).astype(int), 0)
    hi = np.minimum(
        np.ceil(curve.points.max(axis=0) + 3).astype(int), [ink.shape[1], ink.shape[0]]
    )
    if np.any(hi <= lo):
        return
    yy, xx = np.nonzero(ink[lo[1] : hi[1], lo[0] : hi[0]])
    xx, yy = xx + lo[0], yy + lo[1]
    distances, _ = curve.tree.query(np.column_stack((xx, yy)))
    xx, yy = xx[distances <= 1.8], yy[distances <= 1.8]
    disagreement = (labels[yy, xx] != 0) & (labels[yy, xx] != color)
    conflicts[yy[disagreement], xx[disagreement]] = True
    labels[yy, xx] = color
    labels[conflicts] = 0


def analyze_contour_changes(
    reference_ink: np.ndarray,
    current_polylines: list[np.ndarray] | tuple[np.ndarray, ...] | None,
    preview_polylines: list[np.ndarray] | tuple[np.ndarray, ...] | None,
    *,
    current_ink: np.ndarray,
    preview_ink: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return Preview-only labels (0 unchanged/unknown, 1 green, 2 red, 3 yellow).

    One-to-one partial curve assignments use bidirectional coverage, distance,
    direction and length, with competing matches rejected. Target identities
    may be established from either Current or Preview. New target-matched
    curves can be green; newly unsupported curves are yellow, not incorrect.
    Removed curves are reported in diagnostics because Preview has no ink there.
    """
    masks = [np.asarray(value) for value in (reference_ink, current_ink, preview_ink)]
    if (
        any(mask.ndim != 2 or mask.dtype != np.bool_ for mask in masks)
        or any(mask.shape != masks[0].shape for mask in masks)
        or not all(masks[0].shape)
    ):
        raise ValueError(
            "Contour inputs must be same-size nonempty two-dimensional boolean masks"
        )
    reference, current_mask, preview_mask = masks
    extraction_details: dict[str, dict[str, Any]] = {
        key: {} for key in ("target", "current", "preview")
    }
    target = _curves(None, reference, statistics=extraction_details["target"])
    current = _curves(
        current_polylines, current_mask, statistics=extraction_details["current"]
    )
    preview = _curves(
        preview_polylines, preview_mask, statistics=extraction_details["preview"]
    )
    labels = np.zeros(reference.shape, dtype=np.uint8)
    conflicts = np.zeros(reference.shape, dtype=bool)
    counts = dict.fromkeys(
        (
            "improved",
            "worsened",
            "added_matched",
            "added_unmatched",
            "removed",
            "ambiguous",
            "unchanged",
        ),
        0,
    )
    gate = max(6.0, min(18.0, max(reference.shape) * 0.025))
    diagnostics: dict[str, Any] = {
        "status": "ok",
        "method": "partial_three_way_contour_assignment_v1",
        "unit": "contour_segments",
        "counts": counts,
        "extraction": {
            "target": len(target),
            "current": len(current),
            "preview": len(preview),
            "target_method": "thinned_ink_graph",
            "short_segments_omitted": {
                key: value["short_segments_omitted"]
                for key, value in extraction_details.items()
            },
        },
        "parameters": {
            "association_distance_px": gate,
            "change_tolerance_px": _CHANGE_TOLERANCE,
            "minimum_length_px": _MIN_LENGTH,
            "assignment_margin": _ASSIGNMENT_MARGIN,
        },
        "matches": [],
        "removed": [],
        "warnings": [
            "Target ink includes annotations; correspondence is geometric, not semantic.",
            "Uncolored contours include uncertain or unresolved changes; yellow means newly unmatched, not wrong.",
        ],
    }
    if np.array_equal(current_mask, preview_mask):
        counts["unchanged"] = len(preview)
        return labels, diagnostics
    omitted = (
        extraction_details["current"]["short_segments_omitted"]
        + extraction_details["preview"]["short_segments_omitted"]
    )
    if omitted:
        diagnostics["warnings"].append(
            f"{omitted} Current/Preview segment(s) shorter than {_MIN_LENGTH:g} px were omitted; changes confined to them are unresolved."
        )
    ct, _, ct_records = _partial_assignment(current, target, gate)
    pt, pt_ambiguous, pt_records = _partial_assignment(preview, target, gate)
    pc, pc_ambiguous, pc_records = _partial_assignment(preview, current, gate * 1.8)
    current_distance = _distance_to_ink(current_mask)
    preview_distance = _distance_to_ink(preview_mask)
    _transfer_closed_identity(
        preview,
        current,
        pc,
        pc_ambiguous,
        pc_records,
        current_distance,
        preview_distance,
    )
    _transfer_attached_identity(
        preview,
        current,
        pc,
        pc_ambiguous,
        pc_records,
        current_distance,
        preview_distance,
    )
    target_current = {t: c for c, t in ct.items()}
    used_current = set(pc.values())
    diagnostics["assignments"] = {
        "current_target": ct_records,
        "preview_target": pt_records,
        "preview_current": pc_records,
    }
    for p, curve in enumerate(preview):
        c, t = pc.get(p), pt.get(p)
        # Independent reliable target matches can connect curves even when a
        # direct Current/Preview match is too distant or changes its shape.
        if c is None and t in target_current:
            proposed_current = target_current[t]
            if proposed_current not in used_current:
                c = proposed_current
                used_current.add(c)
        old_target = ct.get(c) if c is not None else None
        record: dict[str, Any] = {
            "preview_segment": p,
            "current_segment": c,
            "target_segment": t if t is not None else old_target,
            "preview_bbox_px": [
                *curve.points.min(axis=0).tolist(),
                *curve.points.max(axis=0).tolist(),
            ],
        }
        color, category = 0, "ambiguous"
        novelty = _novel_fraction(curve, current_distance)
        record["novel_fraction"] = novelty
        previous_loss = (
            _novel_fraction(current[c], preview_distance) if c is not None else None
        )
        record["current_loss_fraction"] = previous_loss
        if novelty == 0 and previous_loss == 0:
            category = "unchanged"
        elif c is not None:
            if t is not None and old_target is not None and t != old_target:
                record["reason"] = "Current and Preview disagree on the target identity"
            else:
                anchor = t if t is not None else old_target
                if anchor is not None and p not in pt_ambiguous:
                    before, after = (
                        _pair(current[c], target[anchor], 2.5),
                        _pair(curve, target[anchor], 2.5),
                    )
                    local_change = _local_change_evidence(
                        target[anchor], current[c], curve
                    )
                    competing_subcontours = sorted(
                        set(
                            _competing_closed_subcontours(curve, anchor, target, gate)
                            + _competing_closed_subcontours(
                                current[c], anchor, target, gate
                            )
                        )
                    )
                    record.update(before=before, after=after, local_change=local_change)
                    closure_lost = (
                        target[anchor].closed or current[c].closed
                    ) and not curve.closed
                    complete = curve.closed or curve.length >= 0.85 * min(
                        target[anchor].length, current[c].length
                    )
                    change = before["symmetric_mean_px"] - after["symmetric_mean_px"]
                    forward_change = (
                        before["forward_mean_px"] - after["forward_mean_px"]
                    )
                    reverse_change = (
                        before["reverse_mean_px"] - after["reverse_mean_px"]
                    )
                    if competing_subcontours:
                        record["competing_target_subcontours"] = competing_subcontours
                        record["reason"] = (
                            "A nearby closed Target contour also supports this curved fragment; straight-edge identity is ambiguous"
                        )
                    elif closure_lost:
                        record["reason"] = "Preview loses contour closure"
                    elif local_change["mixed"]:
                        record["reason"] = (
                            "Sustained spans of the same Target contour improve and worsen"
                        )
                    elif (
                        (
                            change > _CHANGE_TOLERANCE
                            or (
                                local_change["sustained_improvement"]
                                and not local_change["sustained_worsening"]
                                and change > 0.1
                            )
                        )
                        and complete
                        and min(forward_change, reverse_change) >= -0.35
                        and after["reverse_coverage"]
                        >= before["reverse_coverage"] - 0.05
                    ):
                        color, category = 1, "improved"
                    elif (
                        (
                            change < -_CHANGE_TOLERANCE
                            or (
                                local_change["sustained_worsening"]
                                and not local_change["sustained_improvement"]
                                and change < -0.1
                            )
                        )
                        and max(forward_change, reverse_change) <= 0.35
                        and before["reverse_coverage"]
                        >= after["reverse_coverage"] - 0.05
                    ):
                        color, category = 2, "worsened"
                    elif abs(change) <= _CHANGE_TOLERANCE:
                        category = "unchanged"
                    elif not complete:
                        record["reason"] = "Preview loses contour length coverage"
                    else:
                        record["reason"] = (
                            "Bidirectional distance and target coverage disagree"
                        )
        elif p not in pc_ambiguous and novelty > 0.6:
            if t is not None and t not in target_current:
                support = _pair(curve, target[t], 2.5)
                record["after"] = support
                attachments = _attached_open_contours(t, target)
                if attachments:
                    record["attached_target_segments"] = attachments
                    record["reason"] = (
                        "An attached open Target stroke makes this new closed correspondence ambiguous (leader, centerline or geometry)"
                    )
                elif target[t].closed and not curve.closed:
                    record["reason"] = (
                        "An open fragment does not establish the missing closed target contour"
                    )
                elif (
                    min(support["forward_coverage"], support["reverse_coverage"])
                    >= 0.85
                ):
                    color, category = 1, "added_matched"
                else:
                    record["reason"] = (
                        "Target identity is plausible, but bidirectional coverage within 2.5 px is insufficient"
                    )
            elif p not in pt_ambiguous:
                partial_support = _partial_target_support(curve, target)
                record["partial_target_support"] = partial_support
                if partial_support >= 0.65:
                    record["reason"] = (
                        "Local Target support exists, but whole-contour correspondence is incomplete"
                    )
                else:
                    color, category = 3, "added_unmatched"
        if category == "ambiguous" and "reason" not in record:
            if p in pc_ambiguous:
                record["reason"] = (
                    "Current counterpart has competing or inconsistent matches"
                )
            elif p in pt_ambiguous:
                record["reason"] = (
                    "Target counterpart has competing or inconsistent matches"
                )
            elif c is not None:
                record["reason"] = (
                    "Current and Preview have no reliable shared Target correspondence"
                )
            else:
                record["reason"] = (
                    "Partial raster change does not establish a reliable new contour"
                )
        record["category"] = category
        diagnostics["matches"].append(record)
        counts[category] += 1
        if color:
            paint_ink = (
                preview_mask & (current_distance > 1.5) if color == 3 else preview_mask
            )
            _paint(labels, conflicts, paint_ink, curve, color)
    for c, curve in enumerate(current):
        if c not in used_current and _novel_fraction(curve, preview_distance) > 0.6:
            counts["removed"] += 1
            diagnostics["removed"].append(
                {
                    "current_segment": c,
                    "target_segment": ct.get(c),
                    "reason": "Current contour has no reliable Preview counterpart and is absent from Preview ink",
                }
            )
    diagnostics["pixel_counts"] = {
        "green": int(np.count_nonzero(labels == 1)),
        "red": int(np.count_nonzero(labels == 2)),
        "yellow": int(np.count_nonzero(labels == 3)),
    }
    if counts["removed"]:
        diagnostics["warnings"].append(
            f"{counts['removed']} Current contour segment(s) were removed; these cannot be colored on Preview."
        )
    return labels, diagnostics
