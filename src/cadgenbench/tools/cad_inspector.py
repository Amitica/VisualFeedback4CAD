"""Select, highlight and measure STEP entities in an existing aligned view.

Run ``python -m cadgenbench.tools.cad_inspector --help``. No model is used.
Geometry is measured in the imported STEP's millimetre coordinate system;
pixel coordinates are used only for selection and annotation placement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import minimum_filter

from .cad_inspector_geometry import GeometryCatalog
from .drawing_view import (
    VIEW_PRESETS,
    SimilarityTransform,
    _fit_projection_transform,
    _render_cad_view,
    project_step,
)

BLUE = (32, 102, 162)
COLORS = [BLUE, (204, 102, 36), (24, 137, 117), (133, 80, 171)]


def _font(size=18, bold=False):
    try:
        return ImageFont.truetype(
            "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size
        )
    except OSError:
        return ImageFont.load_default()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _unit(vector):
    result = np.asarray(vector, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("Camera/direction must contain three finite numbers")
    length = np.linalg.norm(result)
    if length < 1e-12:
        raise ValueError("Camera/direction cannot be zero")
    return result / length


class CADInspector:
    """A version-bound selection and measurement session for one view.

    Public workflow: ``prepare`` -> inspect numbered image -> ``pick`` or ID
    selection -> ``measure_and_render``. IDs refer to this STEP version only.
    Sections are explicitly rejected until cut-generated topology is mapped.
    """

    def __init__(
        self,
        step_path,
        *,
        view="front",
        direction=None,
        up=None,
        origin=None,
        transform=None,
        image_size=(1000, 760),
        reference_path=None,
        expected_sha256=None,
    ):
        self.step_path = Path(step_path).resolve()
        if expected_sha256 is not None:
            digest = hashlib.sha256(self.step_path.read_bytes()).hexdigest()
            if digest != expected_sha256:
                raise ValueError(
                    "STEP hash changed; prepare a new session and reselect IDs"
                )
        if (direction is None) != (up is None):
            raise ValueError("Provide both direction and up")
        if direction is None:
            if view not in VIEW_PRESETS:
                raise ValueError(f"Unknown view: {view}")
            direction, up = VIEW_PRESETS[view].direction, VIEW_PRESETS[view].up
        self.direction, self.up = _unit(direction), _unit(up)
        self.x_axis = _unit(np.cross(self.up, self.direction))
        self.y_axis = _unit(np.cross(self.direction, self.x_axis))
        self.image_size = tuple(image_size)
        # Aligned drawing crops can be thin (e.g. a side view 84 pixels wide).
        # Keep their exact canvas and transform; 32 pixels still leaves a
        # positive drawing area for the default fit margin below.
        if len(self.image_size) != 2 or any(
            not isinstance(v, int) or isinstance(v, bool) or not 32 <= v <= 6000
            for v in self.image_size
        ):
            raise ValueError("Image dimensions must be integers in [32, 6000]")
        self.catalog = GeometryCatalog(self.step_path)
        self.origin = np.asarray(
            origin if origin is not None else tuple(self.catalog.part.center()),
            dtype=float,
        )
        if self.origin.shape != (3,) or not np.isfinite(self.origin).all():
            raise ValueError("Projection origin must contain three finite numbers")
        self.projection = project_step(
            self.step_path,
            view=view,
            direction=self.direction,
            up=self.up,
            look_at=self.origin,
        )
        self.transform = (
            SimilarityTransform(**transform)
            if isinstance(transform, dict)
            else transform
        ) or _fit_projection_transform(
            self.projection,
            image_size=self.image_size,
            margin=min(90, min(self.image_size) // 2 - 10),
        )
        transform_values = asdict(self.transform)
        if (
            not all(math.isfinite(float(v)) for v in transform_values.values())
            or self.transform.scale_px_per_unit <= 0
        ):
            raise ValueError("Invalid similarity transform")
        self.reference_path = Path(reference_path).resolve() if reference_path else None
        if self.reference_path:
            with Image.open(self.reference_path) as reference:
                if reference.size != self.image_size:
                    raise ValueError("Reference image size differs from aligned view")
        self.base_image = _render_cad_view(
            self.image_size, self.projection, self.transform
        )
        self.face_ids = [
            k for k, e in self.catalog.entities.items() if e["kind"] == "face"
        ]
        self.depth, self.face_map = self._rasterize_faces()
        # Conservative neighbourhood visibility test for sampled boundary edges.
        finite_depth = np.where(np.isfinite(self.depth), self.depth, np.inf)
        self.edge_depth = minimum_filter(
            finite_depth, size=3, mode="constant", cval=np.inf
        )
        self.edge_pixels = {}
        self.objects = self._view_objects()

    @classmethod
    def from_manifest(cls, step_path, manifest_path, region_id, *, reference_path=None):
        path = Path(manifest_path).resolve()
        manifest = json.loads(path.read_text())
        regions = [r for r in manifest["regions"] if r["region_id"] == region_id]
        if len(regions) != 1:
            raise ValueError(f"Expected one region {region_id!r}; found {len(regions)}")
        region = regions[0]
        selected = region["selected"]
        spec = selected["view_spec"]
        if region.get("view_type") == "section" or any(
            spec.get(k) is not None
            for k in ("section_origin", "section_offset_fraction", "section_keep")
        ):
            raise ValueError(
                "Section views require cut-topology mapping and are not supported yet"
            )
        if reference_path is None and region.get("crop_path"):
            # Manifests use run-root-relative crop paths; locate without assumptions
            # about the nesting depth of drawing_feedback/<calibration>.
            for parent in path.parents:
                candidate = parent / region["crop_path"]
                if candidate.is_file():
                    reference_path = candidate
                    break
        return cls(
            step_path,
            view=spec["name"],
            direction=spec["direction"],
            up=spec["up"],
            origin=selected.get("projection_origin"),
            transform=selected["transform"],
            image_size=(region["width"], region["height"]),
            reference_path=reference_path,
            expected_sha256=manifest.get("step_sha256"),
        )

    @classmethod
    def load(cls, session_path):
        saved = json.loads(Path(session_path).read_text())
        if saved.get("version") != 1:
            raise ValueError("Unsupported inspector session version")
        return cls(**saved["configuration"], expected_sha256=saved["model_sha256"])

    def project_points(self, points):
        """World points -> image pixels, using the exact build123d HLR basis."""
        centred = np.asarray(points, dtype=float).reshape(-1, 3) - self.origin
        projected = np.column_stack((centred @ self.x_axis, centred @ self.y_axis))
        return self.transform.apply(projected)

    def _rasterize_faces(self):
        width, height = self.image_size
        depth = np.full((height, width), -np.inf)
        face_map = np.full((height, width), -1, dtype=np.int32)
        tolerance = min(0.08, 0.25 / self.transform.scale_px_per_unit)
        for index, entity_id in enumerate(self.face_ids):
            vertices, triangles = self.catalog.shapes[entity_id].tessellate(
                tolerance, 0.12
            )
            if not triangles:
                continue
            world = np.array([tuple(v) for v in vertices])
            pixels = self.project_points(world)
            depths = (world - self.origin) @ self.direction
            for triangle in triangles:
                pts, z = pixels[list(triangle)], depths[list(triangle)]
                lower = np.maximum(np.floor(pts.min(axis=0)).astype(int), (0, 0))
                upper = np.minimum(
                    np.ceil(pts.max(axis=0)).astype(int), (width - 1, height - 1)
                )
                if np.any(upper < lower):
                    continue
                a, b, c = pts
                denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (
                    a[1] - c[1]
                )
                if abs(denominator) < 1e-9:
                    continue
                xx, yy = np.meshgrid(
                    np.arange(lower[0], upper[0] + 1) + 0.5,
                    np.arange(lower[1], upper[1] + 1) + 0.5,
                )
                wa = (
                    (b[1] - c[1]) * (xx - c[0]) + (c[0] - b[0]) * (yy - c[1])
                ) / denominator
                wb = (
                    (c[1] - a[1]) * (xx - c[0]) + (a[0] - c[0]) * (yy - c[1])
                ) / denominator
                wc = 1 - wa - wb
                zz = wa * z[0] + wb * z[1] + wc * z[2]
                region = np.s_[lower[1] : upper[1] + 1, lower[0] : upper[0] + 1]
                mask = (
                    (wa >= -1e-8) & (wb >= -1e-8) & (wc >= -1e-8) & (zz > depth[region])
                )
                depth[region][mask] = zz[mask]
                face_map[region][mask] = index
        return depth, face_map

    def _view_objects(self):
        objects = {}
        width, height = self.image_size
        depth_tolerance = max(0.12, 1.5 / self.transform.scale_px_per_unit)
        for entity_id, entity in self.catalog.entities.items():
            item = dict(entity)
            if entity["kind"] == "face":
                index = self.face_ids.index(entity_id)
                yy, xx = np.nonzero(self.face_map == index)
                item["visible_pixels"] = len(xx)
                visible = len(xx) >= 6
                if visible:
                    points = np.column_stack((xx, yy))
                    centre = points.mean(axis=0)
                    # Anchor on the actual visible face, never in a central hole.
                    anchor = points[np.argmin(np.sum((points - centre) ** 2, axis=1))]
            else:
                length = float(
                    entity.get("length", self.catalog.shapes[entity_id].length)
                )
                count = int(
                    np.clip(length * self.transform.scale_px_per_unit / 2, 48, 1200)
                )
                world = self.catalog.sample_edge(entity_id, count=count)
                pixels = self.project_points(world)
                z = (world - self.origin) @ self.direction
                indices = np.floor(pixels).astype(int)
                inside = (
                    (indices[:, 0] >= 0)
                    & (indices[:, 0] < width)
                    & (indices[:, 1] >= 0)
                    & (indices[:, 1] < height)
                )
                shown = np.zeros(len(world), dtype=bool)
                i = indices[inside]
                shown[inside] = (
                    z[inside] >= self.edge_depth[i[:, 1], i[:, 0]] - depth_tolerance
                )
                self.edge_pixels[entity_id] = (pixels, shown)
                item["visible_fraction"] = float(shown.mean())
                # Very short / degenerate silhouette edges are not useful labels.
                visible = (
                    shown.sum() >= 3
                    and np.linalg.norm(np.ptp(pixels[shown], axis=0)) > 3
                )
                if visible:
                    positions = pixels[shown]
                    anchor = positions[len(positions) // 2]
            item["visible"] = bool(visible)
            item["anchor_px"] = [float(v) for v in anchor] if visible else None
            objects[entity_id] = item
        return objects

    def pick(self, x, y, *, kind="edge", radius_px=12):
        """Return nearby visible IDs; coincident geometry remains explicit."""
        if kind not in ("edge", "face"):
            raise ValueError("Pick kind must be edge or face")
        if not all(math.isfinite(v) for v in (x, y, radius_px)) or radius_px <= 0:
            raise ValueError("Pick position must be finite and radius positive")
        if not (0 <= x < self.image_size[0] and 0 <= y < self.image_size[1]):
            raise ValueError("Pick position is outside the image")
        candidates = []
        if kind == "face":
            index = self.face_map[int(y), int(x)]
            if index >= 0:
                candidates.append(
                    {"entity_id": self.face_ids[index], "distance_px": 0.0}
                )
        else:
            for entity_id, (pixels, shown) in self.edge_pixels.items():
                if not self.objects[entity_id]["visible"]:
                    continue
                distance = float(np.linalg.norm(pixels[shown] - [x, y], axis=1).min())
                if distance <= radius_px:
                    candidates.append({"entity_id": entity_id, "distance_px": distance})
        candidates.sort(key=lambda v: v["distance_px"])
        return {
            "model_sha256": self.catalog.model_sha256,
            "position_px": [x, y],
            "candidates": candidates,
        }

    def highlight(self, entity_ids, *, colors=None):
        image = self.base_image.copy()
        for index, entity_id in enumerate(entity_ids):
            if entity_id not in self.objects:
                raise ValueError(f"Unknown entity ID {entity_id}")
            color = (
                colors.get(entity_id, COLORS[index % len(COLORS)])
                if colors
                else COLORS[index % len(COLORS)]
            )
            if self.objects[entity_id]["kind"] == "face":
                mask = Image.fromarray(
                    (self.face_map == self.face_ids.index(entity_id)).astype(np.uint8)
                    * 80
                )
                image = Image.composite(
                    Image.new("RGB", image.size, color), image, mask
                )
            else:
                draw = ImageDraw.Draw(image)
                pixels, shown = self.edge_pixels[entity_id]
                for j in range(len(pixels) - 1):
                    if shown[j] and shown[j + 1]:
                        draw.line(
                            [tuple(pixels[j]), tuple(pixels[j + 1])],
                            fill=color,
                            width=4,
                        )
        return image

    def _label(self, image, entity_ids):
        draw = ImageDraw.Draw(image)
        used = []
        width, height = image.size
        foreground = np.asarray(self.base_image).min(axis=2) < 180
        for entity_id in entity_ids:
            item = self.objects[entity_id]
            if not item["visible"]:
                continue
            anchor = np.array(item["anchor_px"])
            text_box = draw.textbbox((0, 0), entity_id, font=_font(17, True))
            bw, bh = text_box[2] - text_box[0] + 16, 29
            options = []
            for radius in (35, 58, 85, 120):
                for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                    centre = anchor + radius * np.array([np.cos(angle), np.sin(angle)])
                    x = int(np.clip(centre[0] - bw / 2, 5, width - bw - 5))
                    y = int(np.clip(centre[1] - bh / 2, 5, height - bh - 5))
                    box = (x, y, x + bw, y + bh)
                    overlap = sum(
                        max(0, min(box[2], b[2]) - max(box[0], b[0]))
                        * max(0, min(box[3], b[3]) - max(box[1], b[1]))
                        for b in used
                    )
                    cost = (
                        overlap * 100
                        + foreground[y : y + bh, x : x + bw].sum() * 3
                        + radius
                    )
                    options.append((cost, box))
            _, box = min(options, key=lambda v: v[0])
            used.append(box)
            centre = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            draw.line([tuple(anchor), centre], fill=BLUE, width=1)
            draw.ellipse(
                (anchor[0] - 3, anchor[1] - 3, anchor[0] + 3, anchor[1] + 3), fill=BLUE
            )
            draw.rounded_rectangle(box, radius=5, fill="white", outline=BLUE, width=2)
            draw.text(
                (box[0] + 8, box[1] + 4), entity_id, fill=BLUE, font=_font(17, True)
            )
        return image

    def prepare(
        self,
        output_dir,
        *,
        kind="edge",
        geometry_type=None,
        max_labels=16,
        facing_only=False,
    ):
        if kind not in ("edge", "face", "all"):
            raise ValueError("kind must be edge, face or all")
        if not isinstance(max_labels, int) or max_labels < 1:
            raise ValueError("max_labels must be a positive integer")
        out = Path(output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        chosen = [
            k
            for k, v in self.objects.items()
            if v["visible"]
            and (kind == "all" or v["kind"] == kind)
            and (not geometry_type or v["geometry_type"] == geometry_type.upper())
        ]
        if facing_only:
            chosen = [
                k
                for k in chosen
                if abs(
                    np.dot(
                        self.objects[k].get(
                            "axis3d", self.objects[k].get("normal3d", [0, 0, 0])
                        ),
                        self.direction,
                    )
                )
                > 0.999
            ]
        # Label the nearest coincident circular rim; retain all topology IDs.
        groups = {}
        for key in chosen:
            entity = self.objects[key]
            if (
                entity["geometry_type"] != "CIRCLE"
                or abs(np.dot(entity["axis3d"], self.direction)) < 0.999
            ):
                continue
            centre = self.project_points([entity["center3d"]])[0]
            signature = tuple(
                np.round(
                    np.r_[centre, entity["radius"] * self.transform.scale_px_per_unit],
                    2,
                )
            )
            groups.setdefault(signature, []).append(key)
        suppressed = set()
        for keys in groups.values():
            if len(keys) < 2:
                continue
            primary = max(
                keys,
                key=lambda k: (
                    np.dot(self.objects[k]["center3d"], self.direction),
                    self.objects[k]["length"],
                ),
            )
            self.objects[primary]["same_projected_circle_ids"] = [
                k for k in keys if k != primary
            ]
            suppressed.update(k for k in keys if k != primary)
        chosen = [k for k in chosen if k not in suppressed]
        chosen.sort(
            key=lambda k: (
                self.objects[k]["anchor_px"][1],
                self.objects[k]["anchor_px"][0],
            )
        )
        self.base_image.save(out / "cad.png")
        pages = []
        for start in range(0, max(1, len(chosen)), max_labels):
            ids = chosen[start : start + max_labels]
            image = self._label(self.base_image.copy(), ids)
            name = f"numbered_{len(pages) + 1:02}.png"
            image.save(out / name)
            if not pages:
                image.save(out / "numbered.png")
            pages.append({"image": name, "entity_ids": ids})
        for old_page in out.glob("numbered_[0-9][0-9].png"):
            if int(old_page.stem.split("_")[-1]) > len(pages):
                old_page.unlink()
        reference = None
        if self.reference_path:
            with Image.open(self.reference_path) as image:
                image.convert("RGB").save(out / "reference.png")
            reference = str(out / "reference.png")
        _write_json(
            out / "entities.json",
            {
                "model_sha256": self.catalog.model_sha256,
                "unit": "mm",
                "objects": self.objects,
                "pages": pages,
                "visibility": "tessellated depth buffer; exact measurements use B-rep",
            },
        )
        config = {
            "step_path": str(self.step_path),
            "view": self.projection.view_name,
            "direction": self.direction.tolist(),
            "up": self.up.tolist(),
            "origin": self.origin.tolist(),
            "transform": asdict(self.transform),
            "image_size": list(self.image_size),
            "reference_path": reference,
        }
        _write_json(
            out / "session.json",
            {
                "version": 1,
                "model_sha256": self.catalog.model_sha256,
                "configuration": config,
            },
        )
        return {
            "session": str(out / "session.json"),
            "entities": str(out / "entities.json"),
            "visible_selected_count": len(chosen),
            "pages": pages,
        }

    def _dimension_anchors(self, result):
        if result["quantity"] in ("diameter", "radius"):
            entity = self.catalog.entities[result["entity_ids"][0]]
            centre = np.asarray(entity["center3d"])
            if result["quantity"] == "radius" and entity["kind"] == "edge":
                return np.array(
                    [centre, self.catalog.sample_edge(entity["id"], count=3)[1]]
                )
            axis = _unit(entity["axis3d"])
            radial = self.x_axis - np.dot(self.x_axis, axis) * axis
            if np.linalg.norm(radial) < 1e-8:
                radial = self.y_axis - np.dot(self.y_axis, axis) * axis
            radial = _unit(radial) * entity["radius"]
            return np.array(
                [
                    centre if result["quantity"] == "radius" else centre - radial,
                    centre + radial,
                ]
            )
        return np.asarray(result.get("anchors3d", []), dtype=float)

    @staticmethod
    def _arrow(draw, tip, other, color):
        direction = other - tip
        length = np.linalg.norm(direction)
        if length < 1:
            return
        direction /= length
        perp = np.array([-direction[1], direction[0]])
        size = min(9, length / 3)
        draw.polygon(
            [
                tuple(tip),
                tuple(tip + size * direction + 3 * perp),
                tuple(tip + size * direction - 3 * perp),
            ],
            fill=color,
        )

    def measure_and_render(self, requests, output_dir):
        if not isinstance(requests, list) or not requests:
            raise ValueError("Provide a nonempty list of measurement requests")
        results = []
        for request in requests:
            result = self.catalog.measure(
                request["quantity"],
                request["entity_ids"],
                direction=request.get("direction"),
                expected_sha256=self.catalog.model_sha256,
            )
            results.append(result)
        ids = list(dict.fromkeys(k for r in results for k in r["entity_ids"]))
        entity_colors = {
            key: COLORS[i % len(COLORS)]
            for i, result in enumerate(results)
            for key in result["entity_ids"]
        }
        image = self.highlight(ids, colors=entity_colors)
        draw = ImageDraw.Draw(image)
        for index, (request, result) in enumerate(zip(requests, results)):
            color = COLORS[index % len(COLORS)]
            anchors = (
                np.asarray(result["anchors3d"], dtype=float)
                if request.get("direction") is not None
                and result["quantity"] in ("diameter", "radius")
                else self._dimension_anchors(result)
            )
            result["annotation_anchors3d"] = anchors.tolist()
            visible = all(self.objects[k]["visible"] for k in result["entity_ids"])
            result["entities_visible_in_view"] = visible
            quantity = result["quantity"]
            value = f"{result['value']:.3f}".rstrip("0").rstrip(".")
            prefix = {
                "diameter": "Ø",
                "radius": "R",
                "center_distance": "c–c ",
                "edge_length": "L ",
                "face_distance": "plane gap ",
                "directional_distance": "dir. ",
            }.get(quantity, "")
            result["annotation_status"] = "numeric_report_only"
            if (
                anchors.shape != (2, 3)
                or quantity in ("angle", "face_distance")
                or not visible
            ):
                continue
            entity = self.objects[result["entity_ids"][0]]
            if quantity in ("diameter", "radius") and (
                entity["kind"] == "face"
                or not entity.get("is_full_circle", False)
                or entity.get("radius", 0) * 2 * self.transform.scale_px_per_unit < 75
                or request.get("annotation_style") == "leader"
            ):
                if entity["kind"] == "edge":
                    pixels, shown = self.edge_pixels[entity["id"]]
                    indices = np.flatnonzero(shown)
                    point = pixels[indices[len(indices) // 2]]
                    world = self.catalog.sample_edge(entity["id"], count=len(pixels))
                    result["leader_witness3d"] = world[
                        indices[len(indices) // 2]
                    ].tolist()
                else:
                    point = np.asarray(entity["anchor_px"])
                label = f"{prefix}{value} {result['unit']}"
                default = point + np.array([55, -55])
                position = np.asarray(
                    request.get("label_position_px", default), dtype=float
                )
                if position.shape != (2,) or not np.isfinite(position).all():
                    raise ValueError(
                        "label_position_px must contain two finite numbers"
                    )
                tw = draw.textlength(label, font=_font(18, True))
                position = np.clip(
                    position, [5, 5], np.array(self.image_size) - [tw + 5, 30]
                )
                end = position + np.array([tw / 2, 25])
                draw.line([tuple(point), tuple(end)], fill=color, width=2)
                self._arrow(draw, point, end, color)
                draw.rounded_rectangle(
                    (
                        position[0] - 4,
                        position[1] - 3,
                        position[0] + tw + 4,
                        position[1] + 24,
                    ),
                    radius=3,
                    fill="white",
                )
                draw.text(tuple(position), label, fill=color, font=_font(18, True))
                result["annotation_status"] = "leader_drawn"
                result["annotation_anchor_pixels"] = [point.tolist()]
                continue
            points = self.project_points(anchors)
            vector = points[1] - points[0]
            span = np.linalg.norm(vector)
            if span < 4:
                continue
            projected = np.linalg.norm(
                (anchors[1] - anchors[0]) @ np.column_stack([self.x_axis, self.y_axis])
            )
            actual = np.linalg.norm(anchors[1] - anchors[0])
            foreshortened = actual > 1e-8 and projected / actual < 0.995
            if (
                quantity == "edge_length"
                and self.objects[result["entity_ids"][0]]["geometry_type"] != "LINE"
            ):
                result["annotation_status"] = "curve_length_in_report"
                continue
            label = f"{prefix}{value} {result['unit']}" + (
                " (3D)" if foreshortened else ""
            )
            perp = np.array([-vector[1], vector[0]]) / span
            offset = float(request.get("offset_px", 40 + index * 22))
            if not math.isfinite(offset):
                raise ValueError("offset_px must be finite")
            dimension = points + perp * offset
            # Keep dimension line in the original frame; shorten offset if needed.
            for _ in range(12):
                if np.all(dimension >= [12, 18]) and np.all(
                    dimension <= np.array(self.image_size) - [12, 18]
                ):
                    break
                offset *= 0.7
                dimension = points + perp * offset
            witnesses = self.project_points(result.get("witness_anchors3d", anchors))
            result["witness_anchor_pixels"] = witnesses.tolist()
            for point, end in zip(witnesses, dimension):
                draw.line(
                    [tuple(point), tuple(end + perp * np.sign(offset) * 5)],
                    fill=color,
                    width=1,
                )
                draw.line(
                    [(point[0] - 4, point[1]), (point[0] + 4, point[1])],
                    fill=color,
                    width=1,
                )
                draw.line(
                    [(point[0], point[1] - 4), (point[0], point[1] + 4)],
                    fill=color,
                    width=1,
                )
            draw.line([tuple(dimension[0]), tuple(dimension[1])], fill=color, width=2)
            self._arrow(draw, dimension[0], dimension[1], color)
            self._arrow(draw, dimension[1], dimension[0], color)
            middle = dimension.mean(axis=0)
            tw = draw.textlength(label, font=_font(18, True))
            tx = float(np.clip(middle[0] - tw / 2, 5, self.image_size[0] - tw - 5))
            ty = float(np.clip(middle[1] - 28, 3, self.image_size[1] - 27))
            draw.rounded_rectangle(
                (tx - 4, ty - 3, tx + tw + 4, ty + 25), radius=3, fill="white"
            )
            draw.text((tx, ty), label, fill=color, font=_font(18, True))
            result["annotation_status"] = "dimension_drawn"
            result["annotation_anchor_pixels"] = points.tolist()
            result["foreshortened"] = bool(foreshortened)
        out = Path(output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        image.save(out / "annotated.png")
        self._label(self.highlight(ids, colors=entity_colors), ids).save(
            out / "selection.png"
        )
        _write_json(out / "requests.json", requests)
        report = {
            "model_sha256": self.catalog.model_sha256,
            "source_step": str(self.step_path),
            "unit": "mm",
            "results": results,
        }
        _write_json(out / "measurements.json", report)
        self._comparison(image, results).save(out / "comparison.png")
        return report

    def _comparison(self, annotated, results):
        width, height = self.image_size
        panels = 2 if self.reference_path else 1
        header = 55
        footer = 35 + 31 * len(results)
        sheet = Image.new(
            "RGB", (width * panels, height + header + footer), (247, 249, 252)
        )
        draw = ImageDraw.Draw(sheet)
        if self.reference_path:
            with Image.open(self.reference_path) as reference:
                sheet.paste(reference.convert("RGB"), (0, header))
            draw.text(
                (20, 15), "Reference drawing", fill=(30, 40, 55), font=_font(23, True)
            )
        sheet.paste(annotated, ((panels - 1) * width, header))
        draw.text(
            ((panels - 1) * width + 20, 15),
            "Current CAD · measured geometry",
            fill=(30, 40, 55),
            font=_font(23, True),
        )
        for i, result in enumerate(results):
            text = f"{', '.join(result['entity_ids'])}   {result['quantity']}: {result['value']:.4g} {result['unit']}"
            if result["annotation_status"] not in ("dimension_drawn", "leader_drawn"):
                text += f"   [{result['annotation_status']}]"
            draw.text(
                (20, height + header + 15 + 31 * i),
                text,
                fill=COLORS[i % len(COLORS)],
                font=_font(18),
            )
        return sheet


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser(
        "prepare", help="Create numbered view, geometry catalog and session"
    )
    prepare.add_argument("--step", required=True)
    prepare.add_argument("--manifest")
    prepare.add_argument("--region")
    prepare.add_argument("--reference")
    prepare.add_argument("--view", default="front", choices=list(VIEW_PRESETS))
    prepare.add_argument("--kind", default="edge", choices=["edge", "face", "all"])
    prepare.add_argument(
        "--geometry", help="Optional analytic type: CIRCLE, LINE, CYLINDER, PLANE, ..."
    )
    prepare.add_argument("--max-labels", default=16, type=int)
    prepare.add_argument(
        "--facing-only",
        action="store_true",
        help="Only axes/normals parallel to the viewing direction",
    )
    prepare.add_argument("--output", required=True)
    measure = sub.add_parser(
        "measure", help="Select IDs, calculate values and draw dimensions"
    )
    measure.add_argument("--session", required=True)
    measure.add_argument("--quantity")
    measure.add_argument("--ids", nargs="+")
    measure.add_argument("--direction", nargs=3, type=float)
    measure.add_argument("--offset-px", type=float, default=40)
    measure.add_argument(
        "--requests", help="JSON list for multiple measurements on one view"
    )
    measure.add_argument("--output", required=True)
    pick = sub.add_parser("pick", help="Resolve an image point to visible geometry IDs")
    pick.add_argument("--session", required=True)
    pick.add_argument("--x", type=float, required=True)
    pick.add_argument("--y", type=float, required=True)
    pick.add_argument("--kind", choices=["edge", "face"], default="edge")
    pick.add_argument("--radius-px", type=float, default=12)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            if bool(args.manifest) != bool(args.region):
                raise ValueError("--manifest and --region must be supplied together")
            inspector = (
                CADInspector.from_manifest(
                    args.step, args.manifest, args.region, reference_path=args.reference
                )
                if args.manifest
                else CADInspector(
                    args.step, view=args.view, reference_path=args.reference
                )
            )
            result = inspector.prepare(
                args.output,
                kind=args.kind,
                geometry_type=args.geometry,
                max_labels=args.max_labels,
                facing_only=args.facing_only,
            )
        else:
            inspector = CADInspector.load(args.session)
            if args.command == "pick":
                result = inspector.pick(
                    args.x, args.y, kind=args.kind, radius_px=args.radius_px
                )
            else:
                if args.requests and (args.quantity or args.ids or args.direction):
                    raise ValueError(
                        "Use --requests or a single --quantity/--ids measurement"
                    )
                if args.requests:
                    requests = json.loads(Path(args.requests).read_text())
                else:
                    if not args.quantity or not args.ids:
                        raise ValueError("Provide --quantity and --ids, or --requests")
                    requests = [
                        {
                            "quantity": args.quantity,
                            "entity_ids": args.ids,
                            "direction": args.direction,
                            "offset_px": args.offset_px,
                        }
                    ]
                result = inspector.measure_and_render(requests, args.output)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"cad-inspector: {exc}\n")


if __name__ == "__main__":
    main()
