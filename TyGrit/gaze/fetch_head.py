"""Fetch head-camera gaze control.

Aims a Fetch robot's head pan/tilt at a 3-D world-frame target via
forward kinematics — the inverse-kinematics math lives here (not in
``TyGrit/envs/fetch/core.py``) because gaze control is strictly above
the sensor/actuation adapter layer.

The public :func:`look_at` function takes a :class:`FetchRobotCore`
(or anything with the same ``get_robot_state`` + ``set_head_target``
surface) and the desired world target. It sets the robot's internal
``_head_target`` via :meth:`FetchRobotCore.set_head_target`; the PD
controller baked into :meth:`FetchRobotCore._compute_head_pd` then
drives the head on subsequent :meth:`step` calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from TyGrit.robots.fetch.kinematics.fk_numpy import forward_kinematics

if TYPE_CHECKING:
    from TyGrit.envs.fetch.core import FetchRobotCore


def look_at(
    robot: "FetchRobotCore",
    target: npt.NDArray[np.float64],
    camera_id: str = "head",
) -> None:
    """Aim ``robot``'s head camera at a 3-D world-frame ``target``.

    Uses Fetch-specific forward kinematics to compute the pan/tilt
    joint angles that place ``target`` in the centre of the head
    camera's field of view, then pushes those to the robot via
    :meth:`FetchRobotCore.set_head_target`. The PD controller takes
    over from there.

    Parameters
    ----------
    robot
        A :class:`FetchRobotCore` (or duck-typed equivalent with
        ``get_robot_state`` + ``set_head_target``).
    target
        World-frame ``[x, y, z]`` target.
    camera_id
        Must be ``"head"`` — Fetch's only steerable camera. Raises
        :class:`NotImplementedError` otherwise.
    """
    if camera_id != "head":
        raise NotImplementedError(
            f"fetch_head.look_at: cannot steer camera {camera_id!r}; "
            "only 'head' is steerable on Fetch"
        )

    state = robot.get_robot_state()

    # Build FK input: [torso, 7 arm, pan, tilt]
    fk_joints = np.array(
        [*state.planning_joints, *state.head_joints],
        dtype=np.float64,
    )
    link_poses = forward_kinematics(fk_joints)

    # Transform world target to base frame.
    bp = state.base_pose
    cos_th, sin_th = np.cos(bp.theta), np.sin(bp.theta)
    R_wb = np.array([[cos_th, -sin_th], [sin_th, cos_th]])
    t_wb = np.array([bp.x, bp.y])
    target_base_xy = R_wb.T @ (target[:2] - t_wb)
    target_base = np.array([target_base_xy[0], target_base_xy[1], target[2]])

    # Compute relative pan in head_pan_link frame.
    T_head_pan = link_poses["head_pan_link"]
    T_head_pan_inv = np.linalg.inv(T_head_pan)
    target_head_pan = (T_head_pan_inv @ np.append(target_base, 1.0))[:3]

    x, y, z = target_head_pan
    current_pan = float(state.head_joints[0])
    pan_rel = float(np.arctan2(y, x))

    # Tilt: vector from tilt joint origin to target in pan-aligned frame.
    T_head_tilt = link_poses["head_tilt_link"]
    T_pan_tilt = T_head_pan_inv @ T_head_tilt
    tilt_origin_pan = T_pan_tilt[:3, 3]
    dist_xy = np.sqrt(x**2 + y**2)
    v_tilt_target = np.array([dist_xy, 0.0, z]) - tilt_origin_pan
    tilt_abs = float(np.arctan2(-v_tilt_target[2], v_tilt_target[0]))

    pan = current_pan + pan_rel
    robot.set_head_target(pan, tilt_abs)


__all__ = ["look_at"]
