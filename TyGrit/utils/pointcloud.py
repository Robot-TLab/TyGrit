"""Point-cloud utilities: downsample, merge, crop, filter.

Ported from ``grasp_anywhere.observation.scene.Scene`` static methods.
All functions are **pure** -- they take arrays in and return arrays out.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def voxel_downsample(
    points: npt.NDArray[np.float32],
    voxel_size: float,
) -> npt.NDArray[np.float32]:
    """Down-sample *points* (N, 3) with a voxel grid of size *voxel_size*.

    Keeps one representative point per voxel (the first encountered).
    Pure-numpy implementation -- no Open3D dependency.
    """
    if points.shape[0] == 0 or voxel_size <= 0:
        return points

    q = max(voxel_size, 1e-9)
    keys = np.floor(points / q).astype(np.int64)

    # np.unique on rows -- returns the *first* occurrence index
    _, idx = np.unique(keys, axis=0, return_index=True)
    idx.sort()
    return points[idx]


def merge_dedup(
    base: npt.NDArray[np.float32],
    add: npt.NDArray[np.float32],
    radius: float,
) -> npt.NDArray[np.float32]:
    """Merge *add* into *base*, de-duplicating via a voxel grid of size *radius*.

    Points in *base* always take priority (are kept).
    Ported from ``Scene._merge_dedup``.
    """
    if add.shape[0] == 0:
        return base
    if base.shape[0] == 0:
        return add

    q = max(radius, 1e-6)

    base_keys = np.floor(base / q).astype(np.int64)
    add_keys = np.floor(add / q).astype(np.int64)

    all_keys = np.vstack((base_keys, add_keys))
    _, unique_indices = np.unique(all_keys, axis=0, return_index=True)

    n_base = len(base)
    add_indices = unique_indices[unique_indices >= n_base] - n_base

    if len(add_indices) == 0:
        return base

    add_indices.sort()
    return np.vstack((base, add[add_indices]))


def crop_sphere(
    points: npt.NDArray[np.float32],
    center: npt.NDArray[np.float32],
    radius: float,
) -> npt.NDArray[np.float32]:
    """Keep only points within a 2-D cylinder (XY distance ≤ *radius*) from *center*.

    Ported from ``Scene._crop_sphere``.
    """
    if points.shape[0] == 0:
        return points
    diffs = points[:, :2] - center[:2]
    d2 = np.sum(diffs * diffs, axis=1)
    return points[d2 <= radius * radius]


def filter_ground(
    points: npt.NDArray[np.float32],
    z_threshold: float,
) -> npt.NDArray[np.float32]:
    """Remove points at or below *z_threshold* (the ground plane)."""
    if points.shape[0] == 0:
        return points
    return points[points[:, 2] > z_threshold]


def points_in_frustum_mask(
    points: npt.NDArray[np.float32],
    intrinsics: npt.NDArray[np.float64],
    extrinsics: npt.NDArray[np.float64],
    z_range: tuple[float, float],
    image_width: int = 640,
    image_height: int = 480,
) -> npt.NDArray[np.bool_]:
    """Return a boolean mask for *points* that fall inside the camera frustum.

    Args:
        points: (N, 3) world-frame points.
        intrinsics: (3, 3) camera intrinsic matrix K.
        extrinsics: (4, 4) world-to-camera T_wc.
        z_range: (z_min, z_max) depth bounds in camera frame.
        image_width: Sensor width in pixels.
        image_height: Sensor height in pixels.

    Ported from ``Scene._points_in_frustum_mask``.
    """
    rot = extrinsics[:3, :3].astype(np.float32)
    t = extrinsics[:3, 3].astype(np.float32)
    pc = (points - t) @ rot  # world → camera

    z = pc[:, 2]
    zmin, zmax = z_range
    mask_z = (z >= zmin) & (z <= zmax)
    if not np.any(mask_z):
        return np.zeros(len(points), dtype=bool)

    pc_ok = pc[mask_z]
    z_ok = pc_ok[:, 2]

    u = (pc_ok[:, 0] * intrinsics[0, 0] / z_ok) + intrinsics[0, 2]
    v = (pc_ok[:, 1] * intrinsics[1, 1] / z_ok) + intrinsics[1, 2]

    mask_uv = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)

    final = np.zeros(len(points), dtype=bool)
    final[np.where(mask_z)[0]] = mask_uv
    return final
