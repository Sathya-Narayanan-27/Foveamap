"""Unit tests for Multi-Level Surface and Persistent Bayesian mapping."""
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from src import config as C
from src.mapping import SurfaceInterval, MLSCell, MultiLevelSurfaceMap, PersistentBayesianMap


class TestMapping(unittest.TestCase):
    def test_mls_interval_and_cell(self):
        """Verify surface interval merging and multi-layer creation."""
        cell = MLSCell()
        # Add road surface at z = 0.05
        cell.insert(0.05, label=0, gap_thresh=0.40)
        # Add slightly uneven ground at z = 0.12 (should merge)
        cell.insert(0.12, label=0, gap_thresh=0.40)
        self.assertEqual(len(cell.intervals), 1)
        self.assertAlmostEqual(cell.intervals[0].z_min, 0.05, places=3)
        self.assertAlmostEqual(cell.intervals[0].z_max, 0.12, places=3)

        # Add an overpass deck at z = 4.50 (should create new layer)
        cell.insert(4.50, label=5, gap_thresh=0.40)
        self.assertEqual(len(cell.intervals), 2)
        self.assertAlmostEqual(cell.intervals[1].z_min, 4.50, places=3)

    def test_mls_map_queries(self):
        """Verify MultiLevelSurfaceMap coordinate ingestion and query."""
        mls = MultiLevelSurfaceMap(extent=20.0, resolution=0.25, gap_thresh=0.50)
        xy = np.array([[2.0, 3.0], [2.0, 3.0]], dtype=np.float32)
        z = np.array([0.0, 5.0], dtype=np.float32)
        labels = np.array([0, 5], dtype=np.int64)

        mls.insert_points(xy, z, labels)
        gnd, ceil = mls.get_ground_and_ceiling(2.0, 3.0)
        self.assertIsNotNone(gnd)
        self.assertIsNotNone(ceil)
        self.assertAlmostEqual(gnd, 0.0, places=2)
        self.assertAlmostEqual(ceil, 5.0, places=2)

    def test_persistent_bayesian_map(self):
        """Verify Bayesian log-odds occupancy and categorical Dirichlet fusion."""
        bmap = PersistentBayesianMap(world_extent=40.0, resolution=0.50)
        xy = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        z = np.array([0.1, 0.2], dtype=np.float32)
        labels = np.array([0, 0], dtype=np.int64)  # road
        pose = (0.0, 0.0, 0.0, 0.0)

        bmap.integrate_sweep(xy, z, labels, pose)
        occ = bmap.get_occupancy_probability()
        sem = bmap.get_semantic_map()

        self.assertGreater(occ.max(), 0.50)
        # Check majority class at observed location
        ix, iy, _ = bmap._world_to_grid(np.array([1.0]), np.array([1.0]))
        self.assertEqual(sem[iy[0], ix[0]], 0)


if __name__ == "__main__":
    unittest.main()
