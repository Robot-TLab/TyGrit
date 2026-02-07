"""Coordinate-frame transform utilities.

Ported from ``grasp_anywhere.robot.utils.transform_utils``.
All quaternions use **[x, y, z, w]** convention.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R

from TyGrit.utils.math import quaternion_to_matrix

# ── helpers ──────────────────────────────────────────────────────────────────


def create_pose_matrix(
    position: npt.NDArray[np.float64],
    quaternion: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Build a 4x4 homogeneous matrix from *position* and *quaternion* [x,y,z,w]."""
    T = quaternion_to_matrix(quaternion)
    T[:3, 3] = position
    return T


def create_transform_matrix(
    translation: npt.NDArray[np.float64],
    rotation_matrix: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Build a 4x4 homogeneous matrix from a 3-vector translation and a 3x3 rotation."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = translation
    return T


# ── frame conversions ────────────────────────────────────────────────────────


def pose_to_world(
    base_pos: npt.NDArray[np.float64],
    base_yaw: float,
    ee_pos: npt.NDArray[np.float64],
    ee_quat: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Transform a pose from the robot base frame to the world frame.

    Args:
        base_pos: [x, y, z] of the base in world frame.
        base_yaw: Heading of the base (radians).
        ee_pos: [x, y, z] in base frame.
        ee_quat: [x, y, z, w] in base frame.

    Returns:
        (world_pos, world_quat) -- both as numpy arrays.
    """
    base_rot = R.from_euler("z", base_yaw)
    base_rot_matrix = base_rot.as_matrix()

    ee_pos_rotated = base_rot_matrix @ np.asarray(ee_pos, dtype=np.float64)
    world_pos = np.array(
        [
            base_pos[0] + ee_pos_rotated[0],
            base_pos[1] + ee_pos_rotated[1],
            ee_pos_rotated[2],
        ],
        dtype=np.float64,
    )

    ee_rot = R.from_quat(ee_quat)
    world_rot = base_rot * ee_rot
    world_quat = world_rot.as_quat()  # xyzw

    return world_pos, world_quat


def pose_to_base(
    world_pos: npt.NDArray[np.float64],
    world_quat: npt.NDArray[np.float64],
    base_pos: npt.NDArray[np.float64],
    base_yaw: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Transform a pose from the world frame to the robot base frame.

    Args:
        world_pos: [x, y, z] in world frame.
        world_quat: [x, y, z, w] in world frame.
        base_pos: [x, y, z] of the base in world frame.
        base_yaw: Heading of the base (radians).

    Returns:
        (ee_pos, ee_quat) -- in the base frame.
    """
    base_rot = R.from_euler("z", base_yaw)
    base_rot_inv = base_rot.inv()

    pos_rel = np.array(
        [
            world_pos[0] - base_pos[0],
            world_pos[1] - base_pos[1],
            world_pos[2] - base_pos[2],
        ],
        dtype=np.float64,
    )
    ee_pos = base_rot_inv.apply(pos_rel)

    world_rot = R.from_quat(world_quat)
    ee_rot = base_rot_inv * world_rot
    ee_quat = ee_rot.as_quat()

    return ee_pos, ee_quat


def se2_to_matrix(x: float, y: float, theta: float) -> npt.NDArray[np.float64]:
    """Convert an SE(2) pose to a 4x4 homogeneous matrix (rotation about Z)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [c, -s, 0.0, x],
            [s, c, 0.0, y],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
