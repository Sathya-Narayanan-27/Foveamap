"""
FoveaMap - Autonomous Costmap Generation & Trajectory Planning.

Converts the variable-resolution 2.5D elevation + semantic map into an
ego-centric traversability costmap and plans collision-free trajectories
using an 8-connected A* search with kinodynamic smoothing.
"""

from __future__ import annotations
import heapq
import math
from typing import List, Tuple, Optional, Dict

import numpy as np
from scipy.ndimage import distance_transform_edt

from . import config as C


# ==========================================================================
# 1. Costmap Configuration & Builder
# ==========================================================================
# Base costs for semantic classes (0: free / drivable, 255: lethal obstacle)
DEFAULT_CLASS_COSTS = {
    0: 0,       # road (free)
    1: 40,      # pothole (drivable with penalty)
    2: 180,     # sidewalk (traversable only in emergency)
    3: 240,     # curb (severe mechanical hazard)
    4: 100,     # grass (low traction)
    5: 255,     # structure (lethal)
    6: 255,     # pole (lethal)
    7: 220,     # vegetation (dense obstacle)
    8: 255,     # vehicle (lethal obstacle)
    9: 255,     # pedestrian (lethal obstacle)
    10: 255,    # cyclist (lethal obstacle)
    255: 50,    # unobserved (moderate exploration cost)
}


class CostmapConfig:
    def __init__(
        self,
        extent: float = 60.0,          # metres in x and y (square window [-30, +30] or [0, 60])
        resolution: float = 0.15,      # metres per cell (e.g. 15 cm planning grid)
        inflation_radius: float = 1.0, # metres to inflate lethal obstacles
        step_threshold: float = 0.10,  # metres elevation step that causes penalty
        pothole_depth_penalty: float = 400.0, # cost per metre of depression
        lethal_cost: int = 254,
    ):
        self.extent = extent
        self.resolution = resolution
        self.inflation_radius = inflation_radius
        self.step_threshold = step_threshold
        self.pothole_depth_penalty = pothole_depth_penalty
        self.lethal_cost = lethal_cost
        self.size = int(round(extent / resolution))


class TraversabilityCostmap:
    """Generates an ego-centric 2D costmap from FoveaMap's multi-ring 2.5D output."""

    def __init__(self, cfg: Optional[CostmapConfig] = None):
        self.cfg = cfg or CostmapConfig()
        self.size = self.cfg.size
        self.res = self.cfg.resolution
        self.extent = self.cfg.extent

    def build(self, rings: List[Dict], point_cloud: Optional[Dict] = None) -> np.ndarray:
        """Build costmap (0..255) from projected rings.
        
        Grid frame: cell (ix, iy) maps to ego (x, y) where:
            x = (iy - size/2) * res  (forward/backward)
            y = -(ix - size/2) * res (left/right)
        """
        n = self.size
        costmap = np.full((n, n), DEFAULT_CLASS_COSTS[255], dtype=np.float32)
        valid_mask = np.zeros((n, n), dtype=bool)
        elev_map = np.full((n, n), np.nan, dtype=np.float32)

        # Coordinate grid for the costmap in ego frame (x: forward, y: left)
        coords_1d = (np.arange(n) - n // 2) * self.res
        gx, gy = np.meshgrid(coords_1d, coords_1d)
        gr = np.hypot(gx, gy)

        # Iterate rings from outer (coarse) to inner (fine) so fine details overwrite coarse
        for k in range(len(C.GRID_RINGS) - 1, -1, -1):
            r_in, r_out, cell_size = C.GRID_RINGS[k]
            rg = rings[k]
            if not rg.get("valid", np.zeros(1, bool)).any():
                continue

            ring_mask = (gr >= r_in) & (gr < r_out)
            if not ring_mask.any():
                continue

            # Project ring cell values to costmap coords
            r_tile = C.RING_TILE
            rx = gx[ring_mask]
            ry = gy[ring_mask]
            rix = np.clip(((rx + r_out) / cell_size).astype(np.int64), 0, r_tile - 1)
            riy = np.clip(((ry + r_out) / cell_size).astype(np.int64), 0, r_tile - 1)

            cls_arr = rg.get("cls_ref", rg.get("cls"))
            val_arr = rg.get("valid")
            z_arr = rg.get("zmean", rg.get("zmax"))
            dyn_arr = rg.get("dyn", np.zeros_like(cls_arr))

            pix_val = val_arr[riy, rix]
            pix_cls = cls_arr[riy, rix]
            pix_dyn = dyn_arr[riy, rix] > 64

            # Map semantic classes to base costs
            c_base = np.array([DEFAULT_CLASS_COSTS.get(int(c), 150) for c in pix_cls], dtype=np.float32)

            # Dynamic obstacle extra penalty
            c_base[pix_dyn] = self.cfg.lethal_cost

            costmap[ring_mask] = np.where(pix_val, c_base, costmap[ring_mask])
            valid_mask[ring_mask] = pix_val
            elev_map[ring_mask] = np.where(pix_val, z_arr[riy, rix], np.nan)

        # Geometric roughness and step penalty on valid terrain
        if np.any(valid_mask):
            elev_clean = np.nan_to_num(elev_map, nan=0.0)
            grad_y, grad_x = np.gradient(elev_clean, self.res)
            slope = np.hypot(grad_x, grad_y)
            step_hazard = np.maximum(0.0, slope - self.cfg.step_threshold / self.res) * 50.0
            costmap += np.where(valid_mask, np.clip(step_hazard, 0, 100), 0)

        # Obstacle inflation layer
        lethal_mask = costmap >= 220
        if lethal_mask.any():
            dist_cells = distance_transform_edt(~lethal_mask)
            infl_cells = self.cfg.inflation_radius / self.res
            infl_cost = np.clip(1.0 - (dist_cells / infl_cells), 0.0, 1.0) * 180.0
            costmap = np.maximum(costmap, infl_cost)

        return np.clip(costmap, 0, 255).astype(np.uint8)

    def ego_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert ego coordinates (x forward, y left) to costmap grid (gx, gy)."""
        gx = int(round(x / self.res + self.size // 2))
        gy = int(round(y / self.res + self.size // 2))
        return gx, gy

    def grid_to_ego(self, gx: int, gy: int) -> Tuple[float, float]:
        """Convert costmap grid index (gx, gy) to ego coordinates."""
        x = (gx - self.size // 2) * self.res
        y = (gy - self.size // 2) * self.res
        return x, y


# ==========================================================================
# 2. A* Path Planner
# ==========================================================================
class AStarPlanner:
    """8-connected grid A* path search with traversability cost penalty."""

    def __init__(self, costmap_obj: TraversabilityCostmap):
        self.cm = costmap_obj

    def plan(
        self,
        start_ego: Tuple[float, float],
        goal_ego: Tuple[float, float],
        costmap: np.ndarray,
        lethal_cost_thresh: int = 220,
    ) -> Optional[List[Tuple[float, float]]]:
        """Plan a path from start_ego to goal_ego in ego coordinates."""
        sx, sy = self.cm.ego_to_grid(*start_ego)
        gx, gy = self.cm.ego_to_grid(*goal_ego)
        n = self.cm.size

        if not (0 <= sx < n and 0 <= sy < n and 0 <= gx < n and 0 <= gy < n):
            return None

        # 8-connected motion primitives (dx, dy, step_cost)
        motions = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414),
        ]

        open_set = []
        heapq.heappush(open_set, (0.0, 0.0, (sx, sy)))
        came_from = {}
        g_score = { (sx, sy): 0.0 }

        while open_set:
            _, current_g, current = heapq.heappop(open_set)

            if current == (gx, gy):
                # Reconstruct path
                path_grid = [current]
                while current in came_from:
                    current = came_from[current]
                    path_grid.append(current)
                path_grid.reverse()
                return [self.cm.grid_to_ego(px, py) for px, py in path_grid]

            cx, cy = current
            for dx, dy, step_dist in motions:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < n and 0 <= ny < n):
                    continue

                cell_cost = int(costmap[ny, nx])
                if cell_cost >= lethal_cost_thresh:
                    continue

                # Traversal cost: distance + normalized costmap cell penalty
                cost_penalty = (cell_cost / 25.0) * step_dist
                tentative_g = current_g + step_dist + cost_penalty

                if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                    came_from[(nx, ny)] = current
                    g_score[(nx, ny)] = tentative_g
                    h = math.hypot(nx - gx, ny - gy)
                    f = tentative_g + h
                    heapq.heappush(open_set, (f, tentative_g, (nx, ny)))

        return None


# ==========================================================================
# 3. Trajectory Generator & Smoother
# ==========================================================================
class TrajectoryGenerator:
    """Smooths discrete A* waypoints and computes vehicle control targets."""

    @staticmethod
    def smooth(waypoints: List[Tuple[float, float]], window: int = 5) -> List[Tuple[float, float, float, float]]:
        """Applies moving average smoothing and computes heading (yaw) and curvature."""
        if not waypoints or len(waypoints) < 2:
            return []

        pts = np.array(waypoints, dtype=np.float32)
        n = len(pts)
        if n < window:
            smooth_pts = pts
        else:
            w = min(window, n)
            kernel = np.ones(w) / w
            smooth_x = np.convolve(pts[:, 0], kernel, mode="same")
            smooth_y = np.convolve(pts[:, 1], kernel, mode="same")
            smooth_pts = np.column_stack([smooth_x, smooth_y])
            smooth_pts[0] = pts[0]
            smooth_pts[-1] = pts[-1]

        # Compute heading angles and curvature
        dx = np.gradient(smooth_pts[:, 0])
        dy = np.gradient(smooth_pts[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        yaws = np.arctan2(dy, dx)
        curvatures = np.abs(dx * ddy - dy * ddx) / np.maximum((dx**2 + dy**2)**1.5, 1e-4)

        trajectory = []
        for i in range(len(smooth_pts)):
            trajectory.append((
                float(smooth_pts[i, 0]),
                float(smooth_pts[i, 1]),
                float(yaws[i]),
                float(curvatures[i]),
            ))
        return trajectory


# ==========================================================================
# Quick Self-Test
# ==========================================================================
if __name__ == "__main__":
    print("[planning.py] Testing costmap generation and A* trajectory planning...")
    cm_cfg = CostmapConfig(extent=40.0, resolution=0.20)
    cm = TraversabilityCostmap(cm_cfg)

    # Synthetic mock ring
    mock_rings = []
    for r_in, r_out, cell in C.GRID_RINGS:
        tile = C.RING_TILE
        cls_map = np.zeros((tile, tile), dtype=np.uint8)
        # Add a block of static obstacle
        cls_map[180:220, 210:230] = 5  # structure
        val_map = np.ones((tile, tile), dtype=bool)
        mock_rings.append({
            "cls": cls_map,
            "valid": val_map,
            "zmean": np.zeros((tile, tile), dtype=np.float32),
            "zmax": np.zeros((tile, tile), dtype=np.float32),
            "dyn": np.zeros((tile, tile), dtype=np.uint8),
        })

    cost = cm.build(mock_rings)
    planner = AStarPlanner(cm)
    path = planner.plan(start_ego=(0.0, 0.0), goal_ego=(15.0, 0.0), costmap=cost)

    if path is not None:
        traj = TrajectoryGenerator.smooth(path)
        print(f"  A* planned path with {len(path)} waypoints, smoothed to {len(traj)} trajectory states.")
        print(f"  Start: {traj[0][:2]}, Goal: {traj[-1][:2]}")
        print("[planning.py] All checks passed.")
    else:
        print("[planning.py] Path planning failed (no path found).")
