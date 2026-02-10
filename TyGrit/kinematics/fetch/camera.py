"""Camera extrinsic computation from robot state.

Computes the camera-to-world transform for the Fetch head camera using our
own FK chain, keeping the entire FK→camera→grasp→IK pipeline internally
consistent (no reliance on ManiSkill kinematic queries at inference time).

Camera frame uses **OpenCV convention**: +X right, +Y down, +Z forward.
This matches the frame produced by ``depth_to_pointcloud`` and consumed by
GraspGen.

Verified against ManiSkill3/SAPIEN link poses — see ``scripts/verify_fk_camera.py``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from TyGrit.kinematics.fetch.constants import HEAD_CAMERA_OFFSET, R_CV_TO_CAMERA_LINK
from TyGrit.kinematics.fetch.fk_numpy import forward_kinematics
from TyGrit.types.robot import RobotState
from TyGrit.utils.transforms import se2_to_matrix


def compute_camera_pose(state: RobotState) -> npt.NDArray[np.float64]:
    """Compute the 4x4 camera-to-world transform (OpenCV convention).

    The returned matrix maps points from the camera's OpenCV frame
    (X right, Y down, Z forward) into the world frame::

        p_world = cam2world @ [x_cv, y_cv, z_cv, 1]

    Args:
        state: Current robot state (base pose + joint angles).

    Returns:
        (4, 4) cam2world matrix (OpenCV convention).
    """
    bp = state.base_pose
    fk_joints = np.array(
        [*state.planning_joints, *state.head_joints],
        dtype=np.float64,
    )

    # FK in base_link frame
    link_poses = forward_kinematics(fk_joints)
    T_base_tilt = link_poses["head_tilt_link"]

    # head_tilt_link → head_camera_link (pure translation, same orientation)
    T_tilt_to_camera = np.eye(4, dtype=np.float64)
    T_tilt_to_camera[:3, 3] = HEAD_CAMERA_OFFSET
    T_base_camera = T_base_tilt @ T_tilt_to_camera

    # base_link → world
    T_world_base = se2_to_matrix(bp.x, bp.y, bp.theta)

    # cam2world (OpenCV): world ← base ← camera_link ← cv_frame
    return T_world_base @ T_base_camera @ R_CV_TO_CAMERA_LINK
