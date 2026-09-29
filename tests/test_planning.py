"""Unit tests for costmap generation and A* trajectory planning."""
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from src import config as C
from src.planning import CostmapConfig, TraversabilityCostmap, AStarPlanner, TrajectoryGenerator


class TestPlanning(unittest.TestCase):
    def setUp(self):
        self.cfg = CostmapConfig(extent=30.0, resolution=0.20, inflation_radius=0.60)
        self.cm = TraversabilityCostmap(self.cfg)

    def test_costmap_generation(self):
        """Verify costmap creation from rings."""
        mock_rings = []
        for r_in, r_out, cell in C.GRID_RINGS:
            tile = C.RING_TILE
            cls_map = np.zeros((tile, tile), dtype=np.uint8)  # road (0)
            # Create a blocked zone
            cls_map[180:220, 210:230] = 5  # structure
            val_map = np.ones((tile, tile), dtype=bool)
            mock_rings.append({
                "cls": cls_map,
                "valid": val_map,
                "zmean": np.zeros((tile, tile), dtype=np.float32),
                "zmax": np.zeros((tile, tile), dtype=np.float32),
                "dyn": np.zeros((tile, tile), dtype=np.uint8),
            })

        cost = self.cm.build(mock_rings)
        self.assertEqual(cost.shape, (self.cfg.size, self.cfg.size))
        self.assertEqual(cost.dtype, np.uint8)
        self.assertTrue((cost >= 0).all() and (cost <= 255).all())

    def test_astar_planner_and_smoothing(self):
        """Verify A* finds a collision-free path and smooths trajectory."""
        n = self.cfg.size
        costmap = np.zeros((n, n), dtype=np.uint8)
        # Place an obstacle blocking the direct line
        costmap[n//2 - 5 : n//2 + 5, n//2 + 15 : n//2 + 25] = 254

        planner = AStarPlanner(self.cm)
        path = planner.plan(start_ego=(0.0, 0.0), goal_ego=(8.0, 0.0), costmap=costmap)
        self.assertIsNotNone(path)
        self.assertGreater(len(path), 2)

        traj = TrajectoryGenerator.smooth(path)
        self.assertEqual(len(traj), len(path))
        # Verify first and last waypoints match
        self.assertAlmostEqual(traj[0][0], path[0][0], places=1)
        self.assertAlmostEqual(traj[-1][0], path[-1][0], places=1)


if __name__ == "__main__":
    unittest.main()
