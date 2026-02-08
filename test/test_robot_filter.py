"""Tests for TyGrit.kinematics.robot_filter — robot self-filter."""

import numpy as np
import pytest

from TyGrit.kinematics.fetch.constants import FETCH_SPHERES
from TyGrit.kinematics.fetch.fk import forward_kinematics
from TyGrit.kinematics.robot_filter import filter_robot_points
from TyGrit.utils.transforms import se2_to_matrix


def _world_link_poses(
    joint_angles: np.ndarray,
    base_pose: tuple[float, float, float],
) -> dict[str, np.ndarray]:
    """Run FK and transform all link poses into the world frame."""
    link_poses = forward_kinematics(np.asarray(joint_angles, dtype=np.float64))
    x, y, theta = base_pose
    T_world_base = se2_to_matrix(x, y, theta)
    return {name: T_world_base @ T for name, T in link_poses.items()}


class TestFilterRobotPoints:
    """Test that points on/near the robot body are removed."""

    @pytest.fixture
    def zero_config(self):
        """Zero joint config + base at origin → world-frame link poses."""
        joints = np.zeros(10)
        base = (0.0, 0.0, 0.0)
        return _world_link_poses(joints, base)

    def test_empty_cloud_returns_empty(self, zero_config):
        pts = np.empty((0, 3), dtype=np.float32)
        result = filter_robot_points(pts, zero_config, FETCH_SPHERES)
        assert result.shape == (0, 3)

    def test_far_points_kept(self, zero_config):
        """Points far from the robot should be unchanged."""
        pts = np.array(
            [[5.0, 5.0, 1.0], [10.0, 0.0, 0.5], [-3.0, -3.0, 2.0]],
            dtype=np.float32,
        )
        result = filter_robot_points(pts, zero_config, FETCH_SPHERES)
        assert result.shape[0] == 3

    def test_point_on_base_removed(self, zero_config):
        """A point right at the base link sphere centre should be removed."""
        robot_point = np.array([[0.0, 0.0, 0.2]], dtype=np.float32)
        far_point = np.array([[5.0, 5.0, 1.0]], dtype=np.float32)
        pts = np.vstack([robot_point, far_point])

        result = filter_robot_points(
            pts, zero_config, FETCH_SPHERES, sphere_radius=0.15
        )
        assert result.shape[0] == 1
        np.testing.assert_allclose(result[0], far_point[0], atol=1e-6)

    def test_points_on_arm_removed(self, zero_config):
        """Points placed at FK link origins should be removed."""
        gripper_pos = zero_config["gripper_link"][:3, 3].astype(np.float32)
        elbow_pos = zero_config["elbow_flex_link"][:3, 3].astype(np.float32)
        far_point = np.array([5.0, 5.0, 1.0], dtype=np.float32)

        pts = np.vstack(
            [gripper_pos[np.newaxis], elbow_pos[np.newaxis], far_point[np.newaxis]]
        )
        result = filter_robot_points(
            pts, zero_config, FETCH_SPHERES, sphere_radius=0.12
        )

        assert result.shape[0] == 1
        np.testing.assert_allclose(result[0], far_point, atol=1e-6)

    def test_base_offset_moves_filter(self):
        """Moving the base should shift where points are filtered."""
        joints = np.zeros(10)
        robot_pt = np.array([[0.0, 0.0, 0.2]], dtype=np.float32)

        # With base at origin, this point is on the robot
        poses_origin = _world_link_poses(joints, (0.0, 0.0, 0.0))
        result_at_origin = filter_robot_points(
            robot_pt, poses_origin, FETCH_SPHERES, sphere_radius=0.15
        )
        assert result_at_origin.shape[0] == 0  # Removed

        # With base far away, the point should survive
        poses_far = _world_link_poses(joints, (10.0, 10.0, 0.0))
        result_far = filter_robot_points(
            robot_pt, poses_far, FETCH_SPHERES, sphere_radius=0.15
        )
        assert result_far.shape[0] == 1  # Kept

    def test_rotated_base(self):
        """With base rotated 90 deg, the arm extends in a different direction."""
        joints = np.zeros(10)
        base_rotated = (0.0, 0.0, np.pi / 2)
        poses = _world_link_poses(joints, base_rotated)

        gripper_world = poses["gripper_link"][:3, 3].astype(np.float32)

        pts = np.vstack(
            [gripper_world[np.newaxis], np.array([[5.0, 5.0, 1.0]], dtype=np.float32)]
        )
        result = filter_robot_points(pts, poses, FETCH_SPHERES, sphere_radius=0.12)

        assert result.shape[0] == 1

    def test_preserves_dtype(self, zero_config):
        """Output should be float32."""
        pts = np.array([[5.0, 5.0, 1.0]], dtype=np.float32)
        result = filter_robot_points(pts, zero_config, FETCH_SPHERES)
        assert result.dtype == np.float32

    def test_large_cloud_performance(self, zero_config):
        """Smoke test: filter a 10k point cloud without error."""
        rng = np.random.default_rng(42)
        pts = rng.uniform(-5, 5, (10_000, 3)).astype(np.float32)
        result = filter_robot_points(pts, zero_config, FETCH_SPHERES)
        assert result.shape[0] < pts.shape[0]
        assert result.shape[0] > 0
