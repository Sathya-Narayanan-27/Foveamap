"""Sanity tests for the foveated 2.5D grid engine and the BEV rasteriser."""
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src import config as C
from src.grids import FoveatedGrid, rasterize_bev, uniform_occupancy, ring_stats


class TestFoveatedGrid(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_ring_topology(self):
        """Verify ring topology properties and theoretical cell ratios."""
        a, u = C.theoretical_cell_counts()
        self.assertEqual(C.GRID_RINGS[0][0], 0.0)
        self.assertEqual(C.GRID_RINGS[-1][1], 100.0)
        self.assertTrue(all(C.GRID_RINGS[i][2] <= C.GRID_RINGS[i + 1][2] for i in range(4)))
        self.assertTrue(all(abs(2 * r[1] / r[2] - C.RING_TILE) < 1e-9 for r in C.GRID_RINGS))
        self.assertGreater(u["uniform_5cm"] / a, 20.0)

    def test_projection_and_alignment(self):
        """Verify no lost points, bounds check, and cell-centre alignment precision."""
        N = 100000
        xy = (self.rng.random((N, 2)) * 200 - 100).astype(np.float32)
        r = np.hypot(xy[:, 0], xy[:, 1])
        keep = r < 99.9
        xy = xy[keep]
        r = r[keep]
        z = (self.rng.random(len(xy)) * 2 - 1).astype(np.float32)
        cls = self.rng.integers(0, C.NUM_CLASSES, len(xy)).astype(np.int64)

        g = FoveatedGrid()
        rings = g.project(xy, z, cls)
        tot = sum(int(rg["valid"].sum()) for rg in rings)

        self.assertTrue(all(rg["valid"].sum() > 0 for rg in rings))
        self.assertLessEqual(tot, len(xy))

        # Precision test: maximum point-to-cell-centre error
        for k, rg in enumerate(rings):
            cell, rout = C.GRID_RINGS[k][2], C.GRID_RINGS[k][1]
            m = np.flatnonzero(C.ring_index_of(r) == k)
            ix = np.clip(((xy[m, 0] + rout) / cell).astype(np.int64), 0, C.RING_TILE - 1)
            iy = np.clip(((xy[m, 1] + rout) / cell).astype(np.int64), 0, C.RING_TILE - 1)
            cx = (ix + 0.5) * cell - rout
            cy = (iy + 0.5) * cell - rout
            err = np.hypot(xy[m, 0] - cx, xy[m, 1] - cy).max()
            lim = cell * 0.7072
            self.assertLessEqual(err, lim)
            self.assertTrue(bool(rg["valid"][iy, ix].all()))

    def test_ring_membership(self):
        """Verify ring membership and radius routing."""
        N = 50000
        xy = (self.rng.random((N, 2)) * 190 - 95).astype(np.float32)
        r = np.hypot(xy[:, 0], xy[:, 1])
        keep = r < 99.9
        xy = xy[keep]
        r = r[keep]
        z = np.zeros(len(xy), np.float32)
        cls = np.zeros(len(xy), np.int64)

        g = FoveatedGrid()
        rings = g.project(xy, z, cls)
        cnt = sum(len(rg["pt_idx"]) for rg in rings)
        self.assertEqual(cnt, len(xy))

        lvl = C.ring_index_of(r)
        self.assertTrue(all((lvl[rg["pt_idx"]] == k).all() for k, rg in enumerate(rings)))

    def test_bev_rasterizer(self):
        """Verify 8-channel BEV rasterization channels and validity masks."""
        extent, px, res = C.PYRAMID[0]
        x = np.array([0.0, 0.0, 1.0], np.float32)
        y = np.array([0.0, 0.0, 0.0], np.float32)
        zz = np.array([-0.5, 1.5, 0.25], np.float32)
        f, lab, val = rasterize_bev(
            np.stack([x, y], 1), zz, np.zeros(3, np.float32),
            np.array([0, 0, 3], np.int64), extent, px
        )
        c = int((0.0 + extent / 2) / res)
        lo, hi = C.Z_RANGE
        self.assertAlmostEqual(f[0, c, c], (1.5 - lo) / (hi - lo), places=5)
        self.assertAlmostEqual(f[2, c, c], (-0.5 - lo) / (hi - lo), places=5)
        self.assertAlmostEqual(f[1, c, c], ((-0.5 + 1.5) / 2 - lo) / (hi - lo), places=5)

        c2x = int((1.0 + extent / 2) / res)
        self.assertEqual(int(val.sum()), 2)
        self.assertTrue(bool(val[c, c2x]))
        self.assertFalse(bool(val[c, c + 40]))
        self.assertEqual(lab[c, c], 0)


if __name__ == "__main__":
    unittest.main()

