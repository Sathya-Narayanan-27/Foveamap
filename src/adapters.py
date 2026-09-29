"""
FoveaMap - Real-World Dataset Loaders & Adapters.

Provides ingestion utilities and taxonomy remapping for real-world LiDAR
datasets (SemanticKITTI, nuScenes, Waymo, raw XYZ/XYZI binary/text formats)
to run seamlessly through the FoveaMap pipeline.
"""

from __future__ import annotations
import os
from typing import Dict, Tuple, Optional, Union

import numpy as np

from . import config as C
from .grids import FoveatedGrid
from .pipeline import infer_points


# ==========================================================================
# SemanticKITTI (19/28-class) -> FoveaMap (11-class) Taxonomy Remapping
# ==========================================================================
# FoveaMap class IDs:
#   0: road, 1: pothole, 2: sidewalk, 3: curb, 4: grass, 5: structure,
#   6: pole, 7: vegetation, 8: vehicle, 9: pedestrian, 10: cyclist, 255: ignore
SEMANTICKITTI_TO_FOVEAMAP = {
    0: 255,     # "unlabeled"
    1: 255,     # "outlier"
    10: 8,      # "car" -> vehicle
    11: 10,     # "bicycle" -> cyclist
    13: 8,      # "bus" -> vehicle
    15: 8,      # "motorcycle" -> vehicle
    16: 8,      # "on-rails" -> vehicle
    18: 8,      # "truck" -> vehicle
    20: 8,      # "other-vehicle" -> vehicle
    30: 9,      # "person" -> pedestrian
    31: 10,     # "bicyclist" -> cyclist
    32: 10,     # "motorcyclist" -> cyclist
    40: 0,      # "road" -> road
    44: 0,      # "parking" -> road
    48: 2,      # "sidewalk" -> sidewalk
    49: 4,      # "other-ground" -> grass/terrain
    50: 5,      # "building" -> structure
    51: 5,      # "fence" -> structure
    52: 5,      # "other-structure" -> structure
    60: 4,      # "lane-marking" -> road
    70: 7,      # "vegetation" -> vegetation
    71: 7,      # "trunk" -> vegetation
    72: 4,      # "terrain" -> grass
    80: 6,      # "pole" -> pole
    81: 6,      # "traffic-sign" -> pole
    99: 255,    # "other-object"
    255: 255,   # ignore
}


def remap_kitti_labels(raw_labels: np.ndarray) -> np.ndarray:
    """Remap SemanticKITTI uint32 raw labels into FoveaMap 11-class IDs."""
    sem_ids = (raw_labels.astype(np.uint32) & 0xFFFF).astype(np.int64)
    # Fast vectorized remap via array lookup or fallback
    max_id = max(SEMANTICKITTI_TO_FOVEAMAP.keys())
    lut = np.full(max_id + 1, 255, dtype=np.uint8)
    for k, v in SEMANTICKITTI_TO_FOVEAMAP.items():
        if k <= max_id:
            lut[k] = v

    in_range = (sem_ids >= 0) & (sem_ids <= max_id)
    out = np.full_like(sem_ids, 255, dtype=np.uint8)
    out[in_range] = lut[sem_ids[in_range]]
    return out


def load_kitti_bin(bin_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load SemanticKITTI .bin LiDAR file -> (xy, z, intensity)."""
    scan = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    xyz = scan[:, :3]
    xy = xyz[:, :2].astype(np.float32)
    z = xyz[:, 2].astype(np.float32)
    # KITTI remission is in [0, 1] -> scale to [0, 255]
    intensity = np.clip(scan[:, 3] * 255.0, 0, 255).astype(np.uint8)
    return xy, z, intensity


def load_kitti_label(label_path: str) -> np.ndarray:
    """Load SemanticKITTI .label file -> FoveaMap uint8 labels."""
    raw = np.fromfile(label_path, dtype=np.uint32)
    return remap_kitti_labels(raw)


def load_point_cloud(path: str) -> Dict[str, np.ndarray]:
    """Generic loader supporting .bin, .npy, .npz, .txt, .csv formats."""
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".bin":
        xy, z, inten = load_kitti_bin(path)
        return dict(xy=xy, z=z, inten=inten, label=np.zeros(len(z), dtype=np.uint8))
    elif ext == ".npy":
        arr = np.load(path)
        return dict(
            xy=arr[:, :2].astype(np.float32),
            z=arr[:, 2].astype(np.float32),
            inten=(arr[:, 3] if arr.shape[1] > 3 else np.full(len(arr), 128, dtype=np.uint8)),
            label=(arr[:, 4].astype(np.uint8) if arr.shape[1] > 4 else np.zeros(len(arr), dtype=np.uint8)),
        )
    elif ext == ".npz":
        d = np.load(path)
        if "xyz" in d:
            xyz = d["xyz"].astype(np.float32)
            scale = 0.01 if d["xyz"].dtype == np.int16 else 1.0
            return dict(
                xy=xyz[:, :2] * scale,
                z=xyz[:, 2] * scale,
                inten=d.get("inten", np.full(len(xyz), 128, dtype=np.uint8)),
                label=d.get("label", np.zeros(len(xyz), dtype=np.uint8)),
            )
        return dict(
            xy=d["xy"].astype(np.float32),
            z=d["z"].astype(np.float32),
            inten=d.get("inten", np.full(len(d["z"]), 128, dtype=np.uint8)),
            label=d.get("label", np.zeros(len(d["z"]), dtype=np.uint8)),
        )
    elif ext in (".txt", ".csv", ".xyz"):
        arr = np.loadtxt(path, delimiter="," if ext == ".csv" else None, dtype=np.float32)
        return dict(
            xy=arr[:, :2],
            z=arr[:, 2],
            inten=(arr[:, 3].astype(np.uint8) if arr.shape[1] > 3 else np.full(len(arr), 128, dtype=np.uint8)),
            label=np.zeros(len(arr), dtype=np.uint8),
        )
    else:
        raise ValueError(f"Unsupported point cloud file format: {ext}")


def run_foveamap_on_cloud(model, frame_data: Dict[str, np.ndarray], grid: Optional[FoveatedGrid] = None) -> Dict:
    """Run full FoveaMap pipeline on a generic point cloud dictionary."""
    return infer_points(model, frame_data, grid=grid or FoveatedGrid(), timing=True)


# ==========================================================================
# Quick Self-Test
# ==========================================================================
if __name__ == "__main__":
    print("[adapters.py] Testing SemanticKITTI taxonomy remapping and loader...")
    raw_mock = np.array([10, 40, 48, 50, 70, 80, 30, 0, 99], dtype=np.uint32)
    remapped = remap_kitti_labels(raw_mock)
    print(f"  Raw KITTI IDs:      {raw_mock.tolist()}")
    print(f"  Remapped Fovea IDs: {remapped.tolist()}")
    assert remapped[0] == 8   # car -> vehicle
    assert remapped[1] == 0   # road -> road
    assert remapped[2] == 2   # sidewalk -> sidewalk
    assert remapped[3] == 5   # building -> structure
    assert remapped[4] == 7   # vegetation -> vegetation
    assert remapped[5] == 6   # pole -> pole
    assert remapped[6] == 9   # person -> pedestrian
    print("[adapters.py] All checks passed.")
