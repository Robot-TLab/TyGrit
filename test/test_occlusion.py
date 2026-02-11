"""Tests for TyGrit.checker.occlusion — self-occlusion check."""

import numpy as np

from TyGrit.checker.occlusion import check_self_occlusion
from TyGrit.kinematics.fetch.fk_numpy import forward_kinematics

# Typical Fetch head-camera intrinsics (SAPIEN default).
CAM_K = np.array(
    [[154.10223, 0.0, 320.0], [0.0, 154.10223, 240.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
IMG_SHAPE = (480, 640)


def _fk(
    torso: float = 0.0,
    arm: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    head: tuple[float, float] = (0.0, 0.0),
) -> dict[str, np.ndarray]:
    joints = np.array([torso, *arm, *head], dtype=np.float64)
    return forward_kinematics(joints)


class TestCheckSelfOcclusion:
    """Core behaviour of the self-occlusion checker."""

    def test_no_occlusion_target_to_the_side(self):
        """Arm tucked, target far to the left — no overlap expected."""
        link_poses = _fk()
        # Target far to the left (+Y), well above the base
        result = check_self_occlusion(
            link_poses,
            target_center=np.array([0.5, 0.8, 1.2]),
            target_radius=0.05,
            intrinsics=CAM_K,
            image_shape=IMG_SHAPE,
        )
        assert result is False

    def test_occlusion_arm_blocks_target(self):
        """Arm extended forward, target directly behind the arm — overlap expected."""
        # Extend arm roughly forward: shoulder_pan≈0, shoulder_lift≈0
        # This puts the arm along +X at torso height.
        link_poses = _fk(torso=0.2, arm=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

        # Place target right behind the gripper along the camera's line of sight.
        gripper_pos = link_poses["gripper_link"][:3, 3]
        # Slightly further along the camera forward direction (+X in base frame)
        target = gripper_pos.copy()
        target[0] += 0.05  # just past the gripper

        result = check_self_occlusion(
            link_poses,
            target_center=target,
            target_radius=0.03,
            intrinsics=CAM_K,
            image_shape=IMG_SHAPE,
        )
        assert result is True

    def test_target_behind_camera_returns_false(self):
        """Target behind the camera plane should never register as occluded."""
        link_poses = _fk()
        # Camera looks along +X (base frame). Place target far behind (-X).
        result = check_self_occlusion(
            link_poses,
            target_center=np.array([-5.0, 0.0, 1.0]),
            target_radius=0.1,
            intrinsics=CAM_K,
            image_shape=IMG_SHAPE,
        )
        assert result is False

    def test_empty_link_poses_returns_false(self):
        """No matching links → no spheres → no occlusion."""
        result = check_self_occlusion(
            link_poses={},
            target_center=np.array([1.0, 0.0, 0.8]),
            target_radius=0.1,
            intrinsics=CAM_K,
            image_shape=IMG_SHAPE,
        )
        assert result is False

    def test_missing_head_tilt_returns_false(self):
        """If head_tilt_link is absent, camera pose can't be computed."""
        link_poses = _fk()
        del link_poses["head_tilt_link"]
        result = check_self_occlusion(
            link_poses,
            target_center=np.array([1.0, 0.0, 0.8]),
            target_radius=0.1,
            intrinsics=CAM_K,
            image_shape=IMG_SHAPE,
        )
        assert result is False
