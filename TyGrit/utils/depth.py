"""Depth-image utilities: back-projection to point clouds, image projection.

Ported from ``grasp_anywhere.utils.perception_utils``.
All functions are **pure** -- no robot or ROS references.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def depth_to_pointcloud(
    depth: npt.NDArray[np.float32],
    intrinsics: npt.NDArray[np.float64],
    stride: int = 1,
) -> npt.NDArray[np.float32]:
    """Back-project a depth image to a camera-frame point cloud.

    Args:
        depth: (H, W) depth map in metres.
        intrinsics: (3, 3) camera intrinsic matrix K.
        stride: Decimation factor (1 = no decimation).

    Returns:
        (N, 3) point cloud in the camera frame.  Points with depth ≤ 0 are
        excluded.

    Ported from ``grasp_anywhere.utils.perception_utils.depth2pc``.
    """
    K = intrinsics
    if stride > 1:
        depth = depth[::stride, ::stride]
        K = K.copy()
        K[0, :] /= stride
        K[1, :] /= stride

    mask = np.where(depth > 0)
    y, x = mask[0], mask[1]

    norm_x = x.astype(np.float32) - K[0, 2]
    norm_y = y.astype(np.float32) - K[1, 2]
    d = depth[y, x]

    world_x = norm_x * d / K[0, 0]
    world_y = norm_y * d / K[1, 1]
    world_z = d

    return np.vstack((world_x, world_y, world_z)).T.astype(np.float32)


def depth_to_world_pointcloud(
    depth: npt.NDArray[np.float32],
    intrinsics: npt.NDArray[np.float64],
    extrinsics: npt.NDArray[np.float64],
    z_range: tuple[float, float] = (0.2, 3.0),
    stride: int = 4,
) -> npt.NDArray[np.float32]:
    """Back-project depth and transform to world frame in one call.

    Args:
        depth: (H, W) depth map in metres (NaN-safe).
        intrinsics: (3, 3) camera intrinsic matrix K.
        extrinsics: (4, 4) camera-to-world matrix T_wc.
        z_range: (z_min, z_max) depth bounds in camera frame.
        stride: Decimation factor.

    Returns:
        (N, 3) world-frame point cloud (float32).
    """
    depth = np.nan_to_num(depth)
    pc_cam = depth_to_pointcloud(depth, intrinsics, stride=stride)
    if pc_cam.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32)

    zmin, zmax = z_range
    pc_cam = pc_cam[(pc_cam[:, 2] >= zmin) & (pc_cam[:, 2] <= zmax)]
    if pc_cam.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32)

    rot = extrinsics[:3, :3].astype(np.float32)
    t = extrinsics[:3, 3].astype(np.float32)
    pc_world = (pc_cam @ rot.T) + t
    return pc_world.astype(np.float32)


def pointcloud_from_mask(
    depth: npt.NDArray[np.float32],
    mask: npt.NDArray[np.bool_],
    intrinsics: npt.NDArray[np.float64],
) -> npt.NDArray[np.float32]:
    """Extract a camera-frame point cloud for pixels where *mask* is ``True``.

    Ported from ``grasp_anywhere.utils.perception_utils.get_pcd_from_mask``.

    Returns:
        (N, 3) point cloud, or an empty (0, 3) array when no valid points exist.
    """
    y_coords, x_coords = np.where(mask)
    d = depth[y_coords, x_coords]

    valid = d > 0
    x_coords = x_coords[valid]
    y_coords = y_coords[valid]
    d = d[valid]

    if len(d) == 0:
        return np.empty((0, 3), dtype=np.float32)

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    px = (x_coords - cx) * d / fx
    py = (y_coords - cy) * d / fy
    return np.vstack((px, py, d)).T.astype(np.float32)


def project_points_to_image(
    points: npt.NDArray[np.float32],
    intrinsics: npt.NDArray[np.float64],
    extrinsics: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float32]]:
    """Project world-frame 3-D points onto the image plane.

    Args:
        points: (N, 3) in world frame.
        intrinsics: (3, 3) camera K.
        extrinsics: (4, 4) world-to-camera T_wc.

    Returns:
        (u, v, depth) -- pixel coordinates and depth in camera frame.
    """
    rot = extrinsics[:3, :3].astype(np.float32)
    t = extrinsics[:3, 3].astype(np.float32)
    pc = (points - t) @ rot

    z = pc[:, 2]
    u = (pc[:, 0] * intrinsics[0, 0] / z) + intrinsics[0, 2]
    v = (pc[:, 1] * intrinsics[1, 1] / z) + intrinsics[1, 2]

    return u, v, z
