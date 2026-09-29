"""Integration and unit tests for FoveaMap inference pipeline."""
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from src import config as C
from src.grids import FoveatedGrid
from src.model import FoveaNet
from src.pipeline import infer_points, motion_score, project_uniform


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.model = FoveaNet(ch=(16, 32, 64, 96))
        self.model.eval()

        # Mock frame
        N = 5000
        self.xy = (self.rng.random((N, 2)) * 60 - 30).astype(np.float32)
        self.z = (self.rng.random(N) * 2 - 1).astype(np.float32)
        self.inten = self.rng.integers(0, 255, N).astype(np.uint8)
        self.label = self.rng.integers(0, C.NUM_CLASSES, N).astype(np.uint8)
        self.mock_frame = dict(
            xy=self.xy,
            z=self.z,
            inten=self.inten,
            label=self.label,
        )

    def test_infer_points(self):
        """Verify complete infer_points pipeline returns valid predictions and rings."""
        res = infer_points(self.model, self.mock_frame, timing=True)
        self.assertIn("pred", res)
        self.assertIn("prob", res)
        self.assertIn("rings", res)
        self.assertIn("times", res)

        self.assertEqual(len(res["pred"]), len(self.z))
        self.assertEqual(len(res["prob"]), len(self.z))
        self.assertEqual(len(res["rings"]), len(C.GRID_RINGS))

        # Check probability bounds
        self.assertTrue((res["prob"] >= 0.0).all() and (res["prob"] <= 1.0).all())

    def test_motion_score(self):
        """Verify ego-motion compensated temporal differencing."""
        f1 = self.mock_frame
        # Frame 2 with small vehicle movement
        f2 = dict(
            xy=self.xy + 0.1,
            z=self.z,
            inten=self.inten,
            label=self.label,
        )
        pose1 = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        pose2 = np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float64)

        mot = motion_score(f1, f2, prev_pose=pose1, cur_pose=pose2)
        self.assertEqual(len(mot), len(self.z))
        self.assertTrue((mot >= 0.0).all() and (mot <= 1.0).all())

    def test_project_uniform(self):
        """Verify baseline uniform grid projection."""
        uni = project_uniform(self.xy, self.z, self.label, np.ones(len(self.z), dtype=np.float32),
                              np.zeros(len(self.z), dtype=np.float32), cell=0.50)
        self.assertIn("tile", uni)
        self.assertIn("valid", uni["tile"])
        self.assertTrue(uni["tile"]["valid"].sum() > 0)


if __name__ == "__main__":
    unittest.main()
