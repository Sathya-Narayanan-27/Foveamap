"""Unit tests for dataset adapters and taxonomy remapping."""
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from src import config as C
from src.adapters import remap_kitti_labels, load_point_cloud


class TestAdapters(unittest.TestCase):
    def test_kitti_label_remapping(self):
        """Verify SemanticKITTI 19/28-class to FoveaMap 11-class mapping."""
        kitti_in = np.array([10, 40, 48, 50, 70, 80, 30, 11, 0, 99], dtype=np.uint32)
        expected = np.array([8, 0, 2, 5, 7, 6, 9, 10, 255, 255], dtype=np.uint8)

        out = remap_kitti_labels(kitti_in)
        np.testing.assert_array_equal(out, expected)

    def test_generic_point_cloud_loader(self):
        """Verify generic loader with synthetic .npy or .npz arrays."""
        tmp_path = os.path.join(C.ROOT, "outputs", "test_mock_cloud.npy")
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        try:
            arr = np.array([
                [1.0, 2.0, 0.5, 120.0, 0.0],
                [3.0, 4.0, 1.2, 200.0, 8.0],
            ], dtype=np.float32)
            np.save(tmp_path, arr)

            d = load_point_cloud(tmp_path)
            self.assertEqual(len(d["xy"]), 2)
            self.assertEqual(len(d["z"]), 2)
            self.assertEqual(len(d["inten"]), 2)
            self.assertEqual(len(d["label"]), 2)
            self.assertEqual(d["label"][1], 8)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
