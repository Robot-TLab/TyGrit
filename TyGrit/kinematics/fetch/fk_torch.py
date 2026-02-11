"""Batch forward kinematics for the Fetch robot using PyTorch.

Pure-function counterpart of :func:`kinematics.forward_kinematics` that
processes ``(B, 10)`` joint configurations in a single batched call, suitable
for GPU-accelerated gaze optimization and trajectory processing.

Ported from ``grasp_anywhere/observation/gaze_optimizer.py`` with the
following improvements:

* **Pure function** — no class, no ``self.device``.
* **All 15 links** including the head chain (head_pan, head_tilt).
* Device inferred from input tensor (override via *device* kwarg).
* Kinematic offsets cached per-device to avoid repeated allocation.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor

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

# ── Module-level offset cache keyed by device ────────────────────────────────

_offset_cache: dict[torch.device, dict[str, Tensor]] = {}


def _get_offsets(device: torch.device) -> dict[str, Tensor]:
    """Return kinematic offset tensors on *device*, creating them once."""
    if device not in _offset_cache:

        def _t(arr):  # noqa: E301
            return torch.tensor(arr, dtype=torch.float32, device=device)

        _offset_cache[device] = {
            "torso_base": _t(TORSO_BASE_OFFSET),
            "torso_fixed": _t(TORSO_FIXED_OFFSET),
            "shoulder_pan": _t(SHOULDER_PAN_OFFSET),
            "shoulder_lift": _t(SHOULDER_LIFT_OFFSET),
            "upperarm_roll": _t(UPPERARM_ROLL_OFFSET),
            "elbow_flex": _t(ELBOW_FLEX_OFFSET),
            "forearm_roll": _t(FOREARM_ROLL_OFFSET),
            "wrist_flex": _t(WRIST_FLEX_OFFSET),
            "wrist_roll": _t(WRIST_ROLL_OFFSET),
            "gripper": _t(GRIPPER_OFFSET),
            "head_pan": _t(HEAD_PAN_OFFSET),
            "head_tilt": _t(HEAD_TILT_OFFSET),
            "r_finger": _t(R_GRIPPER_FINGER_OFFSET),
            "l_finger": _t(L_GRIPPER_FINGER_OFFSET),
        }
    return _offset_cache[device]


# ── Internal helpers ─────────────────────────────────────────────────────────


def _batch_transform(translation: Tensor, rotation: Tensor) -> Tensor:
    """Build ``(B, 4, 4)`` homogeneous transforms from components.

    Args:
        translation: ``(B, 3)`` translation vectors.
        rotation: ``(B, 3, 3)`` rotation matrices.

    Returns:
        ``(B, 4, 4)`` homogeneous transformation matrices.
    """
    B = translation.shape[0]
    T = torch.zeros(B, 4, 4, dtype=translation.dtype, device=translation.device)
    T[:, :3, :3] = rotation
    T[:, :3, 3] = translation
    T[:, 3, 3] = 1.0
    return T


def _euler_to_rotation(angles: Tensor, axis: str) -> Tensor:
    """Batch single-axis Euler angles to ``(B, 3, 3)`` rotation matrices.

    Args:
        angles: ``(B,)`` angles in radians.
        axis: One of ``'x'``, ``'y'``, ``'z'``.
    """
    c = torch.cos(angles)
    s = torch.sin(angles)
    o = torch.zeros_like(angles)
    ones = torch.ones_like(angles)

    if axis == "x":
        mat = torch.stack([ones, o, o, o, c, -s, o, s, c], dim=1)
    elif axis == "y":
        mat = torch.stack([c, o, s, o, ones, o, -s, o, c], dim=1)
    elif axis == "z":
        mat = torch.stack([c, -s, o, s, c, o, o, o, ones], dim=1)
    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")

    return mat.reshape(-1, 3, 3)


# ── Public API ───────────────────────────────────────────────────────────────


def batch_forward_kinematics(
    joint_angles: Tensor,
    device: torch.device | None = None,
) -> Dict[str, Tensor]:
    """Compute forward kinematics for a batch of Fetch joint configurations.

    Args:
        joint_angles: ``(B, 10)`` tensor — columns are
            ``[torso_lift, shoulder_pan, shoulder_lift, upperarm_roll,
              elbow_flex, forearm_roll, wrist_flex, wrist_roll,
              head_pan, head_tilt]``.
        device: Target device.  ``None`` (default) infers from *joint_angles*.

    Returns:
        Dict mapping 15 link names to ``(B, 4, 4)`` pose matrices in
        the base_link frame.
    """
    if device is not None:
        joint_angles = joint_angles.to(device)
    dev = joint_angles.device
    B = joint_angles.shape[0]
    off = _get_offsets(dev)

    eye3 = torch.eye(3, dtype=torch.float32, device=dev).unsqueeze(0).expand(B, -1, -1)
    zero3 = torch.zeros(B, 3, dtype=torch.float32, device=dev)

    # Unpack joints
    torso_lift = joint_angles[:, 0]
    shoulder_pan = joint_angles[:, 1]
    shoulder_lift = joint_angles[:, 2]
    upperarm_roll = joint_angles[:, 3]
    elbow_flex = joint_angles[:, 4]
    forearm_roll = joint_angles[:, 5]
    wrist_flex = joint_angles[:, 6]
    wrist_roll = joint_angles[:, 7]
    head_pan = joint_angles[:, 8]
    head_tilt = joint_angles[:, 9]

    link_poses: Dict[str, Tensor] = {}

    # Base link — identity
    T_base = (
        torch.eye(4, dtype=torch.float32, device=dev).unsqueeze(0).expand(B, -1, -1)
    )
    link_poses["base_link"] = T_base

    # Torso fixed link (rigid child of base)
    link_poses["torso_fixed_link"] = torch.bmm(
        T_base,
        _batch_transform(off["torso_fixed"].unsqueeze(0).expand(B, -1), eye3),
    )

    # 1. Torso lift (prismatic along Z)
    torso_trans = off["torso_base"].unsqueeze(0).expand(B, -1).clone()
    torso_trans[:, 2] = torso_trans[:, 2] + torso_lift
    T = torch.bmm(T_base, _batch_transform(torso_trans, eye3))
    link_poses["torso_lift_link"] = T

    # Head chain (child of torso)
    T_head_pan = torch.bmm(
        T,
        torch.bmm(
            _batch_transform(off["head_pan"].unsqueeze(0).expand(B, -1), eye3),
            _batch_transform(zero3, _euler_to_rotation(head_pan, "z")),
        ),
    )
    link_poses["head_pan_link"] = T_head_pan

    T_head_tilt = torch.bmm(
        T_head_pan,
        torch.bmm(
            _batch_transform(off["head_tilt"].unsqueeze(0).expand(B, -1), eye3),
            _batch_transform(zero3, _euler_to_rotation(head_tilt, "y")),
        ),
    )
    link_poses["head_tilt_link"] = T_head_tilt

    # 2. Shoulder pan
    T = torch.bmm(
        T,
        torch.bmm(
            _batch_transform(off["shoulder_pan"].unsqueeze(0).expand(B, -1), eye3),
            _batch_transform(zero3, _euler_to_rotation(shoulder_pan, "z")),
        ),
    )
    link_poses["shoulder_pan_link"] = T

    # 3. Shoulder lift
    T = torch.bmm(
        T,
        torch.bmm(
            _batch_transform(off["shoulder_lift"].unsqueeze(0).expand(B, -1), eye3),
            _batch_transform(zero3, _euler_to_rotation(shoulder_lift, "y")),
        ),
    )
    link_poses["shoulder_lift_link"] = T

    # 4. Upperarm roll
    T = torch.bmm(
        T,
        torch.bmm(
            _batch_transform(off["upperarm_roll"].unsqueeze(0).expand(B, -1), eye3),
            _batch_transform(zero3, _euler_to_rotation(upperarm_roll, "x")),
        ),
    )
    link_poses["upperarm_roll_link"] = T

    # 5. Elbow flex
    T = torch.bmm(
        T,
        torch.bmm(
            _batch_transform(off["elbow_flex"].unsqueeze(0).expand(B, -1), eye3),
            _batch_transform(zero3, _euler_to_rotation(elbow_flex, "y")),
        ),
    )
    link_poses["elbow_flex_link"] = T

    # 6. Forearm roll
    T = torch.bmm(
        T,
        torch.bmm(
            _batch_transform(off["forearm_roll"].unsqueeze(0).expand(B, -1), eye3),
            _batch_transform(zero3, _euler_to_rotation(forearm_roll, "x")),
        ),
    )
    link_poses["forearm_roll_link"] = T

    # 7. Wrist flex
    T = torch.bmm(
        T,
        torch.bmm(
            _batch_transform(off["wrist_flex"].unsqueeze(0).expand(B, -1), eye3),
            _batch_transform(zero3, _euler_to_rotation(wrist_flex, "y")),
        ),
    )
    link_poses["wrist_flex_link"] = T

    # 8. Wrist roll
    T = torch.bmm(
        T,
        torch.bmm(
            _batch_transform(off["wrist_roll"].unsqueeze(0).expand(B, -1), eye3),
            _batch_transform(zero3, _euler_to_rotation(wrist_roll, "x")),
        ),
    )
    link_poses["wrist_roll_link"] = T

    # 9. Gripper (fixed offset from wrist roll)
    T = torch.bmm(
        T,
        _batch_transform(off["gripper"].unsqueeze(0).expand(B, -1), eye3),
    )
    link_poses["gripper_link"] = T

    # Finger links (fixed children of gripper)
    link_poses["r_gripper_finger_link"] = torch.bmm(
        T,
        _batch_transform(off["r_finger"].unsqueeze(0).expand(B, -1), eye3),
    )
    link_poses["l_gripper_finger_link"] = torch.bmm(
        T,
        _batch_transform(off["l_finger"].unsqueeze(0).expand(B, -1), eye3),
    )

    return link_poses


# ── Solver wrapper ───────────────────────────────────────────────────────────


class FetchBatchFK:
    """Batched all-link FK via PyTorch.

    Input: ``(B, 10)`` tensor of joints (torso + 7 arm + 2 head).
    Output: dict of ``(B, 4, 4)`` pose tensors in **base_link** frame.
    """

    @property
    def base_frame(self) -> str:
        return "base_link"

    def solve(self, joint_angles: Tensor) -> Dict[str, Tensor]:
        return batch_forward_kinematics(joint_angles)
