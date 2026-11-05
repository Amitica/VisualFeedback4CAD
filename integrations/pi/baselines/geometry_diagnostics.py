"""Read-only execution diagnostics for compiler repair; never heals a STEP."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def boundary_edge_locations(vertices, triangles, limit=8):
    edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    bad = np.flatnonzero(counts != 2)
    return [{"vertex_ids": unique[i].tolist(), "endpoints_mm": np.round(vertices[unique[i]], 6).tolist(), "incident_triangles": int(counts[i])} for i in bad[:limit]]


def diagnose(path):
    from build123d import import_step
    from cadgenbench.common.mesh import tessellate_shape
    from cadgenbench.common.validity import _tessellation_deflection
    shape = import_step(str(path))
    deflection = _tessellation_deflection(shape.wrapped, None)
    # Inspect the same topology-based tessellator without accepting its mesh.
    # The actual worker validation result remains authoritative and unchanged.
    mesh = tessellate_shape(shape.wrapped, deflection)
    return {"diagnostics_only": True, "geometry_modified": False, "step_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "coordinates": "CAD world coordinates in millimetres", "deflection_mm": deflection, "boundary_edge_locations": boundary_edge_locations(mesh.vertices, mesh.triangles), "hint": "An edge with more than two incident triangles can indicate zero-thickness or tangent contacts. Inspect intersecting bores, point/edge-only joins, and coincident boundaries near these coordinates; preserve drawing geometry and do not disable validity checks."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", type=Path)
    args = parser.parse_args()
    try:
        data = diagnose(args.step)
    except Exception as error:
        data = {"diagnostics_only": True, "geometry_modified": False, "diagnostic_error": f"{type(error).__name__}: {error}"}
    print(json.dumps(data))


if __name__ == "__main__":
    main()
