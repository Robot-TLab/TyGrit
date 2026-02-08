"""Fetch skeleton FK — pure NumPy, all 15 links.

Ported from ``grasp_anywhere/robot/kinematics.py``.
Uses ``create_transform_matrix`` from ``TyGrit.utils.transforms``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R

from TyGrit.kinematics.fetch.constants import (
    ELBOW_FLEX_OFFSET,
    FOREARM_ROLL_OFFSET,
    GRIPPER_OFFSET,
    HEAD_PAN_OFFSET,
    HEAD_TILT_OFFSET,
    L_GRIPPER_FINGER_OFFSET,
    R_GRIPPER_FINGER_OFFSET,
    SHOULDER_LIFT_OFFSET,
    SHOULDER_PAN_OFFSET,
    TORSO_BASE_OFFSET,
    TORSO_FIXED_OFFSET,
    UPPERARM_ROLL_OFFSET,
    WRIST_FLEX_OFFSET,
    WRIST_ROLL_OFFSET,
)
from TyGrit.kinematics.fk import SkeletonFKSolver
from TyGrit.utils.transforms import create_transform_matrix


def forward_kinematics(
    joint_angles: npt.NDArray[np.float64],
) -> dict[str, npt.NDArray[np.float64]]:
    """Compute forward kinematics for all relevant Fetch links in base_link frame.

    Args:
        joint_angles: (10,) array of joint values in radians:
            [torso_lift, shoulder_pan, shoulder_lift, upperarm_roll,
             elbow_flex, forearm_roll, wrist_flex, wrist_roll,
             head_pan, head_tilt].

    Returns:
        Dict mapping link names to 4x4 pose matrices in the base_link frame.
    """
    (
        torso_lift,
        shoulder_pan,
        shoulder_lift,
        upperarm_roll,
        elbow_flex,
        forearm_roll,
        wrist_flex,
        wrist_roll,
        head_pan,
        head_tilt,
    ) = joint_angles

    _tf = create_transform_matrix
    _eye3 = np.eye(3)
    _zero3 = np.zeros(3)

    link_poses: dict[str, npt.NDArray[np.float64]] = {}

    # Base link — identity
    T_base = np.eye(4)
    link_poses["base_link"] = T_base

    # Torso fixed link (rigid child of base)
    link_poses["torso_fixed_link"] = T_base @ _tf(TORSO_FIXED_OFFSET, _eye3)

    # 1. Torso lift (prismatic along Z)
    torso_translation = TORSO_BASE_OFFSET.copy()
    torso_translation[2] += torso_lift
    T = T_base @ _tf(torso_translation, _eye3)
    link_poses["torso_lift_link"] = T

    # Head chain (child of torso)
    T_head_pan = (
        T
        @ _tf(HEAD_PAN_OFFSET, _eye3)
        @ _tf(_zero3, R.from_euler("z", head_pan).as_matrix())
    )
    link_poses["head_pan_link"] = T_head_pan

    T_head_tilt = (
        T_head_pan
        @ _tf(HEAD_TILT_OFFSET, _eye3)
        @ _tf(_zero3, R.from_euler("y", head_tilt).as_matrix())
    )
    link_poses["head_tilt_link"] = T_head_tilt

    # 2. Shoulder pan
    T = (
        T
        @ _tf(SHOULDER_PAN_OFFSET, _eye3)
        @ _tf(_zero3, R.from_euler("z", shoulder_pan).as_matrix())
    )
    link_poses["shoulder_pan_link"] = T

    # 3. Shoulder lift
    T = (
        T
        @ _tf(SHOULDER_LIFT_OFFSET, _eye3)
        @ _tf(_zero3, R.from_euler("y", shoulder_lift).as_matrix())
    )
    link_poses["shoulder_lift_link"] = T

    # 4. Upperarm roll
    T = (
        T
        @ _tf(UPPERARM_ROLL_OFFSET, _eye3)
        @ _tf(_zero3, R.from_euler("x", upperarm_roll).as_matrix())
    )
    link_poses["upperarm_roll_link"] = T

    # 5. Elbow flex
    T = (
        T
        @ _tf(ELBOW_FLEX_OFFSET, _eye3)
        @ _tf(_zero3, R.from_euler("y", elbow_flex).as_matrix())
    )
    link_poses["elbow_flex_link"] = T

    # 6. Forearm roll
    T = (
        T
        @ _tf(FOREARM_ROLL_OFFSET, _eye3)
        @ _tf(_zero3, R.from_euler("x", forearm_roll).as_matrix())
    )
    link_poses["forearm_roll_link"] = T

    # 7. Wrist flex
    T = (
        T
        @ _tf(WRIST_FLEX_OFFSET, _eye3)
        @ _tf(_zero3, R.from_euler("y", wrist_flex).as_matrix())
    )
    link_poses["wrist_flex_link"] = T

    # 8. Wrist roll
    T = (
        T
        @ _tf(WRIST_ROLL_OFFSET, _eye3)
        @ _tf(_zero3, R.from_euler("x", wrist_roll).as_matrix())
    )
    link_poses["wrist_roll_link"] = T

    # 9. Gripper (fixed offset from wrist roll)
    T = T @ _tf(GRIPPER_OFFSET, _eye3)
    link_poses["gripper_link"] = T

    # Finger links (fixed children of gripper)
    link_poses["r_gripper_finger_link"] = T @ _tf(R_GRIPPER_FINGER_OFFSET, _eye3)
    link_poses["l_gripper_finger_link"] = T @ _tf(L_GRIPPER_FINGER_OFFSET, _eye3)

    return link_poses


class FetchSkeletonFK(SkeletonFKSolver):
    """All link poses via pure-NumPy FK.

    Input: 10 joints (torso + 7 arm + 2 head).
    Output: all link poses in **base_link** frame.
    """

    @property
    def base_frame(self) -> str:
        return "base_link"

    def solve(
        self, joint_angles: npt.NDArray[np.float64]
    ) -> dict[str, npt.NDArray[np.float64]]:
        return forward_kinematics(joint_angles)
