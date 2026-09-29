"""
FoveaMap - Multi-Level Surface (MLS) & Persistent Bayesian Mapping.

Provides:
1. MultiLevelSurfaceMap: 2.5D elevation map supporting multiple vertical layers
   per cell (e.g. ground under bridges, tree canopies, multi-story overhangs).
2. PersistentBayesianMap: Multi-sweep rolling world-frame accumulator with
   recursive Bayesian log-odds occupancy and Dirichlet categorical semantics.
"""

from __future__ import annotations
import math
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

from . import config as C


# ==========================================================================
# 1. Multi-Level Surface (MLS) Mapping
# ==========================================================================
class SurfaceInterval:
    """A vertical interval [z_min, z_max] in an MLS cell."""
    __slots__ = ("z_min", "z_max", "n_pts", "cls_counts")

    def __init__(self, z: float, label: int = 0, num_classes: int = C.NUM_CLASSES):
        self.z_min = float(z)
        self.z_max = float(z)
        self.n_pts = 1
        self.cls_counts = np.zeros(num_classes, dtype=np.int32)
        if 0 <= label < num_classes:
            self.cls_counts[label] = 1

    def update(self, z: float, label: int = 0, num_classes: int = C.NUM_CLASSES):
        self.z_min = min(self.z_min, float(z))
        self.z_max = max(self.z_max, float(z))
        self.n_pts += 1
        if 0 <= label < num_classes:
            self.cls_counts[label] += 1

    @property
    def majority_class(self) -> int:
        return int(np.argmax(self.cls_counts))


class MLSCell:
    """Cell containing one or more distinct vertical surface intervals."""
    __slots__ = ("intervals",)

    def __init__(self):
        self.intervals: List[SurfaceInterval] = []

    def insert(self, z: float, label: int = 0, gap_thresh: float = 0.50):
        """Insert a point z, merging with nearby intervals or creating a new layer."""
        merged = False
        for iv in self.intervals:
            # If point sits within or close to an existing interval
            if (iv.z_min - gap_thresh) <= z <= (iv.z_max + gap_thresh):
                iv.update(z, label)
                merged = True
                break
        if not merged:
            self.intervals.append(SurfaceInterval(z, label))
            # Keep intervals sorted by altitude
            self.intervals.sort(key=lambda x: x.z_min)


class MultiLevelSurfaceMap:
    """2.5D multi-level surface grid covering a bounding area."""

    def __init__(self, extent: float = 100.0, resolution: float = 0.20, gap_thresh: float = 0.50):
        self.extent = extent
        self.res = resolution
        self.gap_thresh = gap_thresh
        self.size = int(round(extent / resolution))
        self.grid: List[List[MLSCell]] = [[MLSCell() for _ in range(self.size)] for _ in range(self.size)]

    def insert_points(self, xy: np.ndarray, z: np.ndarray, labels: Optional[np.ndarray] = None):
        """Insert Nx2 xy and Nx1 z points into the MLS grid."""
        n = self.size
        h_ext = self.extent * 0.5
        ix = ((xy[:, 0] + h_ext) / self.res).astype(np.int64)
        iy = ((xy[:, 1] + h_ext) / self.res).astype(np.int64)
        valid = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)

        labs = labels if labels is not None else np.zeros(len(z), dtype=np.int64)
        for i in np.flatnonzero(valid):
            cx, cy = int(ix[i]), int(iy[i])
            self.grid[cy][cx].insert(float(z[i]), int(labs[i]), self.gap_thresh)

    def get_ground_and_ceiling(self, x: float, y: float) -> Tuple[Optional[float], Optional[float]]:
        """Get (ground_z, lowest_overhang_z) for a query coordinate."""
        h_ext = self.extent * 0.5
        cx = int(round((x + h_ext) / self.res))
        cy = int(round((y + h_ext) / self.res))
        if not (0 <= cx < self.size and 0 <= cy < self.size):
            return None, None
        cell = self.grid[cy][cx]
        if not cell.intervals:
            return None, None
        ground = cell.intervals[0].z_max
        ceiling = cell.intervals[1].z_min if len(cell.intervals) > 1 else None
        return ground, ceiling


# ==========================================================================
# 2. Persistent Bayesian Multi-Sweep Map
# ==========================================================================
class PersistentBayesianMap:
    """World-frame persistent map with recursive Bayesian evidence fusion."""

    def __init__(
        self,
        world_extent: float = 300.0,
        resolution: float = 0.25,
        p_hit: float = 0.70,
        p_miss: float = 0.40,
        num_classes: int = C.NUM_CLASSES,
    ):
        self.extent = world_extent
        self.res = resolution
        self.size = int(round(world_extent / resolution))
        self.num_classes = num_classes

        # Log-odds update values
        self.l_hit = math.log(p_hit / (1.0 - p_hit))
        self.l_miss = math.log(p_miss / (1.0 - p_miss))
        self.l_max = 5.0
        self.l_min = -5.0

        n = self.size
        self.log_odds = np.zeros((n, n), dtype=np.float32)
        self.z_mean = np.zeros((n, n), dtype=np.float32)
        self.z_m2 = np.zeros((n, n), dtype=np.float32)
        self.count = np.zeros((n, n), dtype=np.int32)
        self.class_counts = np.zeros((n, n, num_classes), dtype=np.int32)

    def _world_to_grid(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h_ext = self.extent * 0.5
        ix = np.clip(((x + h_ext) / self.res).astype(np.int64), 0, self.size - 1)
        iy = np.clip(((y + h_ext) / self.res).astype(np.int64), 0, self.size - 1)
        valid = (x >= -h_ext) & (x < h_ext) & (y >= -h_ext) & (y < h_ext)
        return ix, iy, valid

    def integrate_sweep(
        self,
        ego_xy: np.ndarray,
        ego_z: np.ndarray,
        labels: np.ndarray,
        pose: Tuple[float, float, float, float],
    ):
        """Integrate an ego sweep (x, y, z, labels) given pose=(x, y, z, yaw)."""
        px, py, pz, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)

        # Transform to world coordinates
        wx = px + ego_xy[:, 0] * c - ego_xy[:, 1] * s
        wy = py + ego_xy[:, 0] * s + ego_xy[:, 1] * c
        wz = pz + ego_z

        ix, iy, valid = self._world_to_grid(wx, wy)
        if not np.any(valid):
            return

        ix = ix[valid]
        iy = iy[valid]
        wz = wz[valid]
        labs = labels[valid].astype(np.int64)

        # Update log-odds occupancy and online elevation (Welford algorithm)
        for i in range(len(ix)):
            gx, gy = ix[i], iy[i]
            z_val = float(wz[i])
            c_val = int(labs[i])

            self.log_odds[gy, gx] = min(self.l_max, self.log_odds[gy, gx] + self.l_hit)
            self.count[gy, gx] += 1
            n = self.count[gy, gx]

            # Online mean & variance update
            delta = z_val - self.z_mean[gy, gx]
            self.z_mean[gy, gx] += delta / n
            delta2 = z_val - self.z_mean[gy, gx]
            self.z_m2[gy, gx] += delta * delta2

            # Dirichlet semantic class count accumulation
            if 0 <= c_val < self.num_classes:
                self.class_counts[gy, gx, c_val] += 1

    def get_occupancy_probability(self) -> np.ndarray:
        """Returns P(occupied) in [0, 1] per cell."""
        return 1.0 / (1.0 + np.exp(-self.log_odds))

    def get_semantic_map(self) -> np.ndarray:
        """Returns majority semantic class per cell (255 if unobserved)."""
        tot = self.class_counts.sum(axis=-1)
        majority = np.argmax(self.class_counts, axis=-1).astype(np.uint8)
        return np.where(tot > 0, majority, 255)


# ==========================================================================
# Quick Self-Test
# ==========================================================================
if __name__ == "__main__":
    print("[mapping.py] Testing MLS and Persistent Bayesian mapping...")
    mls = MultiLevelSurfaceMap(extent=20.0, resolution=0.20, gap_thresh=0.50)

    # Insert a ground point and an overhanging bridge point at the same (x, y)
    xy_mock = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    z_mock = np.array([0.1, 4.5], dtype=np.float32)  # 10 cm ground, 4.5 m bridge deck
    labels_mock = np.array([0, 5], dtype=np.int64)    # road, structure

    mls.insert_points(xy_mock, z_mock, labels_mock)
    gnd, ceil = mls.get_ground_and_ceiling(0.0, 0.0)
    print(f"  MLS cell (0,0): Ground elevation = {gnd:.2f} m, Ceiling = {ceil:.2f} m")
    assert gnd is not None and ceil is not None and abs(ceil - 4.5) < 0.1

    # Test Bayesian persistent mapping
    bmap = PersistentBayesianMap(world_extent=50.0, resolution=0.50)
    pose1 = (0.0, 0.0, 0.0, 0.0)
    bmap.integrate_sweep(xy_mock, z_mock, labels_mock, pose1)
    occ = bmap.get_occupancy_probability()
    sem = bmap.get_semantic_map()
    print(f"  Bayesian map updated. Max occupancy probability = {occ.max():.2f}")
    print("[mapping.py] All checks passed.")
