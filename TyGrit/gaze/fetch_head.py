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
    import torch

    from TyGrit.envs.fetch.core import FetchRobotCore
    from TyGrit.envs.fetch.core_vec import FetchRobotCoreVec


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


def look_at_batched(
    robot: "FetchRobotCoreVec",
    targets: "torch.Tensor",
) -> None:
    """Aim each parallel env's head camera at a 3-D world-frame
    ``targets[i]`` via batched forward kinematics.

    Vec counterpart of :func:`look_at`. The whole per-env head-
    kinematics solve is one batched FK + transform; no Python per-env
    loop. Writes into ``robot._head_target`` via
    :meth:`FetchRobotCoreVec.set_head_target` so the PD controller in
    :meth:`FetchRobotCoreVec._compute_head_pd` consumes it on the
    next step.

    Parameters
    ----------
    robot
        Vec Fetch core on a :class:`SimHandlerVec`.
    targets
        ``(num_envs, 3)`` world-frame targets. Device must match
        ``robot.device`` — the callers in :mod:`TyGrit.rl` allocate
        on the RL loop's GPU.
    """
    import torch as _torch

    from TyGrit.robots.fetch import FETCH_CFG
    from TyGrit.robots.fetch.kinematics.fk_torch import batch_forward_kinematics

    if targets.shape != (robot.num_envs, 3):
        raise ValueError(
            f"fetch_head.look_at_batched: targets shape {tuple(targets.shape)} "
            f"!= ({robot.num_envs}, 3)"
        )

    state = robot.get_robot_state()
    planning = state.planning_joints.to(_torch.float32)  # (N, 8)
    head = state.head_joints.to(_torch.float32)  # (N, 2)
    base = state.base_xy_theta.to(_torch.float32)  # (N, 3)

    _ = FETCH_CFG  # documented point of dispatch; link names are Fetch-standard
    fk_input = _torch.cat([planning, head], dim=1)  # (N, 10)
    link_poses = batch_forward_kinematics(fk_input)

    T_head_pan = link_poses["head_pan_link"].to(_torch.float32)

    targets = targets.to(robot.device).float()
    cos_th = _torch.cos(base[:, 2])
    sin_th = _torch.sin(base[:, 2])
    dx = targets[:, 0] - base[:, 0]
    dy = targets[:, 1] - base[:, 1]
    target_base_x = cos_th * dx + sin_th * dy
    target_base_y = -sin_th * dx + cos_th * dy
    target_base_z = targets[:, 2]
    target_base = _torch.stack([target_base_x, target_base_y, target_base_z], dim=1)

    T_inv = _torch.linalg.inv(T_head_pan)
    target_base_h = _torch.cat(
        [target_base, _torch.ones(target_base.shape[0], 1, device=target_base.device)],
        dim=1,
    )
    target_head_pan = _torch.bmm(T_inv, target_base_h.unsqueeze(2)).squeeze(2)[:, :3]

    x, y, z = target_head_pan[:, 0], target_head_pan[:, 1], target_head_pan[:, 2]
    current_pan = head[:, 0]
    pan_rel = _torch.atan2(y, x)

    T_head_tilt = link_poses["head_tilt_link"].to(_torch.float32)
    T_pan_tilt = _torch.bmm(T_inv, T_head_tilt)
    tilt_origin_pan = T_pan_tilt[:, :3, 3]
    dist_xy = _torch.sqrt(x * x + y * y)
    v_target = _torch.stack([dist_xy, _torch.zeros_like(dist_xy), z], dim=1)
    v_tilt = v_target - tilt_origin_pan
    tilt_abs = _torch.atan2(-v_tilt[:, 2], v_tilt[:, 0])

    pan = current_pan + pan_rel
    robot.set_head_target(pan, tilt_abs)


__all__ = ["look_at", "look_at_batched"]
