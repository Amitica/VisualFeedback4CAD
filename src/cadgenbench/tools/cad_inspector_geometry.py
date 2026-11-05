"""Address and measure analytic geometry in one immutable STEP snapshot.

Entity IDs address imported BRep topology, not semantic features: a cylindrical
face can belong to either a hole or a boss. IDs are only valid for the file hash
in the catalog. All coordinates and linear measurements use the millimetre
working units of the build123d/OpenCascade STEP importer.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np


def _xyz(point) -> list[float]:
    return [float(v) for v in point.Coord()]


def _unit(direction: Sequence[float]) -> np.ndarray:
    array = np.asarray(direction, dtype=float)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError("direction must contain three finite numbers")
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("direction must be nonzero")
    return array / norm


def _center(shape, kind: str) -> list[float]:
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    props = GProp_GProps()
    if kind == "face":
        BRepGProp.SurfaceProperties_s(shape.wrapped, props)
    else:
        BRepGProp.LinearProperties_s(shape.wrapped, props)
    return _xyz(props.CentreOfMass())


class GeometryCatalog:
    """Imported shapes, JSON descriptions, and exact analytic measurements."""

    def __init__(self, step_path: str | Path):
        from build123d import import_step

        self.step_path = Path(step_path).resolve()
        self.model_sha256 = hashlib.sha256(self.step_path.read_bytes()).hexdigest()
        self.part = import_step(str(self.step_path))
        if self.part is None or self.part.wrapped is None:
            raise ValueError(f"No geometry in STEP file: {self.step_path}")
        self.shapes: dict[str, object] = {}
        self.entities: dict[str, dict] = {}
        for kind, prefix, shapes in (
            ("face", "F", self.part.faces()),
            ("edge", "E", self.part.edges()),
        ):
            for number, shape in enumerate(shapes, 1):
                entity_id = f"{prefix}{number:03d}"
                self.shapes[entity_id] = shape
                self.entities[entity_id] = self._describe(entity_id, kind, shape)
        # Face-to-edge incidence lets renderers show boundaries of a selected face.
        edges = [(key, shape) for key, shape in self.shapes.items() if key.startswith("E")]
        for key, entity in self.entities.items():
            if entity["kind"] == "face":
                entity["edge_ids"] = [
                    edge_id
                    for edge_id, edge in edges
                    if any(edge.wrapped.IsSame(item.wrapped) for item in self.shapes[key].edges())
                ]

    def _describe(self, entity_id: str, kind: str, shape) -> dict:
        from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface

        result = {
            "id": entity_id,
            "kind": kind,
            "geometry_type": shape.geom_type.name,
            "center3d": _center(shape, kind),
        }
        if kind == "face":
            result["area"] = float(shape.area)
            adaptor = BRepAdaptor_Surface(shape.wrapped)
            if result["geometry_type"] == "PLANE":
                plane = adaptor.Plane()
                result["normal3d"] = _xyz(plane.Axis().Direction())
                result["plane_origin3d"] = _xyz(plane.Location())
            elif result["geometry_type"] == "CYLINDER":
                cylinder = adaptor.Cylinder()
                origin = np.asarray(_xyz(cylinder.Location()))
                axis = np.asarray(_xyz(cylinder.Axis().Direction()))
                center = np.asarray(result["center3d"])
                result.update(
                    radius=float(cylinder.Radius()),
                    axis3d=axis.tolist(),
                    radial_direction3d=_xyz(cylinder.XAxis().Direction()),
                    cylinder_origin3d=origin.tolist(),
                    surface_centroid3d=center.tolist(),
                    center3d=(origin + np.dot(center - origin, axis) * axis).tolist(),
                )
        else:
            result["length"] = float(shape.length)
            adaptor = BRepAdaptor_Curve(shape.wrapped)
            first, last = adaptor.FirstParameter(), adaptor.LastParameter()
            result["endpoints3d"] = [_xyz(adaptor.Value(first)), _xyz(adaptor.Value(last))]
            if result["geometry_type"] == "CIRCLE":
                circle = adaptor.Circle()
                result.update(
                    radius=float(circle.Radius()),
                    axis3d=_xyz(circle.Axis().Direction()),
                    radial_direction3d=_xyz(circle.XAxis().Direction()),
                    circle_center3d=_xyz(circle.Location()),
                    center3d=_xyz(circle.Location()),
                    parameter_range=[float(first), float(last)],
                    is_full_circle=math.isclose(last - first, 2 * math.pi, abs_tol=1e-7),
                )
            elif result["geometry_type"] == "LINE":
                result["axis3d"] = _xyz(adaptor.Line().Direction())
        return result

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "step_path": str(self.step_path),
            "model_sha256": self.model_sha256,
            "length_unit": "mm",
            "unit_note": "OpenCascade STEP import working units; dimensions use imported geometry.",
            "id_scope": "Entity IDs are valid only for this model_sha256.",
            "entities": self.entities,
        }

    def sample_edge(self, entity_id: str, count: int = 64) -> np.ndarray:
        """Return world-coordinate samples; a full circle includes its closure."""
        from OCP.BRepAdaptor import BRepAdaptor_Curve

        entity = self._get(entity_id)
        if entity["kind"] != "edge":
            raise ValueError(f"{entity_id} is not an edge")
        if count < 2:
            raise ValueError("count must be at least 2")
        adaptor = BRepAdaptor_Curve(self.shapes[entity_id].wrapped)
        parameters = np.linspace(adaptor.FirstParameter(), adaptor.LastParameter(), count)
        return np.asarray([_xyz(adaptor.Value(float(t))) for t in parameters])

    def _get(self, entity_id: str) -> dict:
        try:
            return self.entities[entity_id]
        except KeyError:
            raise ValueError(f"Unknown entity ID: {entity_id}") from None

    def measure(
        self,
        quantity: str,
        entity_ids: Sequence[str],
        *,
        direction: Sequence[float] | None = None,
        expected_sha256: str | None = None,
    ) -> dict:
        """Measure selected BRep entities, returning value and world anchors.

        ``face_distance`` is the separation of supporting parallel planes,
        not a minimum distance between trimmed face regions. ``angle`` is the
        acute unoriented angle between two line directions or two plane normals.
        ``directional_distance`` is absolute projected circle-centre separation.
        """
        if expected_sha256 is not None and expected_sha256 != self.model_sha256:
            raise ValueError("Stale entity selection: model_sha256 does not match this STEP")
        counts = {
            "diameter": 1, "radius": 1, "edge_length": 1,
            "center_distance": 2, "directional_distance": 2,
            "face_distance": 2, "angle": 2,
        }
        if quantity not in counts:
            raise ValueError(f"Unsupported quantity: {quantity}; choose one of {', '.join(counts)}")
        if isinstance(entity_ids, str):
            entity_ids = [entity_ids]
        if len(entity_ids) != counts[quantity]:
            raise ValueError(f"{quantity} requires {counts[quantity]} entity IDs")
        if len(set(entity_ids)) != len(entity_ids):
            raise ValueError("Select distinct entities")
        if direction is not None and quantity not in {"directional_distance", "diameter", "radius"}:
            raise ValueError(f"direction is not used by {quantity}")
        entities = [self._get(key) for key in entity_ids]
        first = entities[0]
        result = {
            "quantity": quantity,
            "entity_ids": list(entity_ids),
            "model_sha256": self.model_sha256,
            "unit": "mm",
        }

        if quantity in {"diameter", "radius"}:
            if first["geometry_type"] not in {"CIRCLE", "CYLINDER"}:
                raise ValueError(f"{quantity} requires a circular edge or cylindrical face")
            axis = np.asarray(first["axis3d"])
            radial = np.asarray(first["radial_direction3d"])
            if direction is not None:
                requested = _unit(direction)
                radial = requested - np.dot(requested, axis) * axis
                if np.linalg.norm(radial) < 1e-8:
                    raise ValueError("Radial direction must not be parallel to the circle/cylinder axis")
                radial = _unit(radial)
            center = np.asarray(first["center3d"])
            radius = first["radius"]
            result.update(
                value=radius * (2 if quantity == "diameter" else 1),
                anchors3d=[
                    (center - radius * radial if quantity == "diameter" else center).tolist(),
                    (center + radius * radial).tolist(),
                ],
                center3d=center.tolist(), axis3d=axis.tolist(),
                geometry_scope="supporting circle or cylinder",
            )
        elif quantity == "edge_length":
            if first["kind"] != "edge":
                raise ValueError("edge_length requires an edge")
            result.update(value=first["length"], anchors3d=first["endpoints3d"],
                          geometry_scope="length along the trimmed edge")
        elif quantity in {"center_distance", "directional_distance"}:
            if any(e["kind"] != "edge" or e["geometry_type"] != "CIRCLE" for e in entities):
                raise ValueError(f"{quantity} requires two circular edges")
            a, b = [np.asarray(e["circle_center3d"]) for e in entities]
            delta = b - a
            result.update(anchors3d=[a.tolist(), b.tolist()], geometry_scope="circle centres")
            if quantity == "center_distance":
                result["value"] = float(np.linalg.norm(delta))
            else:
                if direction is None:
                    raise ValueError("directional_distance requires an explicit world-coordinate direction")
                axis = _unit(direction)
                signed_distance = float(np.dot(delta, axis))
                projected_b = a + signed_distance * axis
                result.update(value=abs(signed_distance), signed_value=signed_distance,
                              direction3d=axis.tolist(),
                              witness_anchors3d=[a.tolist(), b.tolist()],
                              anchors3d=[a.tolist(), projected_b.tolist()])
        elif quantity == "face_distance":
            if any(e["kind"] != "face" or e["geometry_type"] != "PLANE" for e in entities):
                raise ValueError("face_distance requires two planar faces")
            normal = np.asarray(first["normal3d"])
            other_normal = np.asarray(entities[1]["normal3d"])
            if np.linalg.norm(np.cross(normal, other_normal)) > 1e-7:
                raise ValueError("face_distance requires parallel planar faces")
            a, b = [np.asarray(e["center3d"]) for e in entities]
            signed_distance = float(np.dot(b - a, normal))
            result.update(value=abs(signed_distance),
                          anchors3d=[a.tolist(), (a + signed_distance * normal).tolist()],
                          direction3d=normal.tolist(),
                          geometry_scope="separation of supporting parallel planes")
        elif quantity == "angle":
            types = {(e["kind"], e["geometry_type"]) for e in entities}
            if types == {("face", "PLANE")}:
                vectors = [e["normal3d"] for e in entities]
                scope = "acute angle between plane normals"
            elif types == {("edge", "LINE")}:
                vectors = [e["axis3d"] for e in entities]
                scope = "acute angle between line directions"
            else:
                raise ValueError("angle requires two planar faces or two straight edges")
            cosine = float(np.clip(abs(np.dot(_unit(vectors[0]), _unit(vectors[1]))), 0, 1))
            result.update(value=math.degrees(math.acos(cosine)), unit="deg",
                          anchors3d=[e["center3d"] for e in entities],
                          directions3d=vectors, geometry_scope=scope)
        return result
