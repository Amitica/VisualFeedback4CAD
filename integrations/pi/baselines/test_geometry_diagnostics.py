"""Synthetic topology checks for the read-only compiler diagnostic helper."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np


spec = importlib.util.spec_from_file_location("geometry_diagnostics", Path(__file__).with_name("geometry_diagnostics.py"))
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


class BoundaryEdgeLocationsTest(unittest.TestCase):
    def test_closed_tetrahedron_has_no_flagged_edge(self):
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        triangles = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
        self.assertEqual(diagnostics.boundary_edge_locations(vertices, triangles), [])

    def test_two_solids_touching_along_an_edge_report_world_coordinates_without_mutation(self):
        vertices = np.array([[10., 20., 30.], [11., 20., 30.], [10., 21., 30.],
                             [10., 20., 31.], [10., 19., 30.], [10., 20., 29.]])
        triangles = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3],
                              [0, 4, 1], [0, 1, 5], [1, 4, 5], [4, 0, 5]])
        before_vertices, before_triangles = vertices.copy(), triangles.copy()
        rows = diagnostics.boundary_edge_locations(vertices, triangles)
        self.assertEqual(rows, [{"vertex_ids": [0, 1], "endpoints_mm": [[10., 20., 30.], [11., 20., 30.]], "incident_triangles": 4}])
        np.testing.assert_array_equal(vertices, before_vertices)
        np.testing.assert_array_equal(triangles, before_triangles)

    def test_open_quad_reports_only_open_boundary_and_limits_diagnostic_size(self):
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
        triangles = np.array([[0, 1, 2], [0, 2, 3]])
        rows = diagnostics.boundary_edge_locations(vertices, triangles)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["incident_triangles"] == 1 for row in rows))
        self.assertNotIn([0, 2], [row["vertex_ids"] for row in rows])
        self.assertEqual(len(diagnostics.boundary_edge_locations(vertices, triangles, limit=1)), 1)


if __name__ == "__main__":
    unittest.main()
