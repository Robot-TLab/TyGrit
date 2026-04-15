"""Fetch end-effector FK via IKFast C extension."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def ee_forward_kinematics(
    joint_angles: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute end-effector pose using IKFast analytical FK (C extension).

    Args:
        joint_angles: (8,) array — ``[torso_lift, shoulder_pan, shoulder_lift,
            upperarm_roll, elbow_flex, forearm_roll, wrist_flex, wrist_roll]``.

    Returns:
        4×4 homogeneous transform of the gripper link in base_link frame.
    """
    import ikfast_fetch

    position, rotation = ikfast_fetch.get_fk(joint_angles.tolist())
    T = np.eye(4)
    T[:3, :3] = np.array(rotation)
    T[:3, 3] = np.array(position)
    return T


class FetchIKFastFK:
    """Gripper-only pose via IKFast C FK.

    Input: 8 joints (torso + 7 arm).
    Output: 4×4 gripper pose in **base_link** frame.
    """

    @property
    def base_frame(self) -> str:
        return "base_link"

    def solve(self, joint_angles: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return ee_forward_kinematics(joint_angles)
