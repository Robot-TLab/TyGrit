"""Unit tests for TyGrit.worlds.reachability.

Pure-Python tests for the coordinate-transform helpers and
base-pose sampling. The IK-dependent ``check_reachability`` test
is skipped when the ``ikfast_fetch`` C extension is not available
(CI / default env without the extension compiled).

Run: ``pixi run test test/test_worlds_reachability.py -v``
"""

from __future__ import annotations

import math

import numpy as np
import pytest


class TestSE2To4x4:
    """Tests for the internal _se2_to_4x4 helper."""

    def test_identity_at_origin(self) -> None:
        from TyGrit.worlds.reachability import _se2_to_4x4

        T = _se2_to_4x4(0.0, 0.0, 0.0)
        np.testing.assert_allclose(T, np.eye(4), atol=1e-12)

    def test_translation_only(self) -> None:
        from TyGrit.worlds.reachability import _se2_to_4x4

        T = _se2_to_4x4(1.5, -2.3, 0.0)
        assert T[0, 3] == pytest.approx(1.5)
        assert T[1, 3] == pytest.approx(-2.3)
        assert T[2, 3] == pytest.approx(0.0)
        # Rotation should be identity.
        np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-12)

    def test_90_degree_rotation(self) -> None:
        from TyGrit.worlds.reachability import _se2_to_4x4

        T = _se2_to_4x4(0.0, 0.0, math.pi / 2)
        expected_rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        np.testing.assert_allclose(T[:3, :3], expected_rot, atol=1e-12)


class TestWorldToBaseLink:
    """Tests for world_to_base_link."""

    def test_identity_base_is_noop(self) -> None:
        from TyGrit.worlds.reachability import world_to_base_link

        target = np.eye(4, dtype=np.float64)
        target[0, 3] = 2.0
        result = world_to_base_link(target, 0.0, 0.0, 0.0)
        np.testing.assert_allclose(result, target, atol=1e-12)

    def test_base_translation_subtracts(self) -> None:
        from TyGrit.worlds.reachability import world_to_base_link

        target = np.eye(4, dtype=np.float64)
        target[0, 3] = 3.0
        target[1, 3] = 1.0
        result = world_to_base_link(target, 1.0, 0.5, 0.0)
        assert result[0, 3] == pytest.approx(2.0)
        assert result[1, 3] == pytest.approx(0.5)

    def test_roundtrip_is_identity(self) -> None:
        from TyGrit.worlds.reachability import _se2_to_4x4, world_to_base_link

        target_world = np.eye(4, dtype=np.float64)
        target_world[0, 3] = 5.0
        target_world[1, 3] = 3.0
        bx, by, bt = 2.0, 1.0, 0.7
        T_base = _se2_to_4x4(bx, by, bt)
        target_base = world_to_base_link(target_world, bx, by, bt)
        # Reconstructing world from base should recover original.
        recovered = T_base @ target_base
        np.testing.assert_allclose(recovered, target_world, atol=1e-12)


class TestSampleBasePosesAroundObject:
    """Tests for sample_base_poses_around_object."""

    def test_returns_correct_count(self) -> None:
        from TyGrit.worlds.reachability import sample_base_poses_around_object

        poses = sample_base_poses_around_object(
            (1.0, 2.0, 0.8),
            num_distances=3,
            num_angles=8,
        )
        assert len(poses) == 3 * 8

    def test_poses_within_distance_bounds(self) -> None:
        from TyGrit.worlds.reachability import sample_base_poses_around_object

        ox, oy = 1.0, 2.0
        poses = sample_base_poses_around_object(
            (ox, oy, 0.8),
            min_distance=0.5,
            max_distance=1.5,
            num_distances=4,
            num_angles=12,
        )
        for bx, by, _ in poses:
            d = math.hypot(bx - ox, by - oy)
            assert d >= 0.5 - 1e-6
            assert d <= 1.5 + 1e-6

    def test_face_object_orientation(self) -> None:
        from TyGrit.worlds.reachability import sample_base_poses_around_object

        ox, oy = 3.0, 4.0
        poses = sample_base_poses_around_object(
            (ox, oy, 0.8),
            num_distances=1,
            num_angles=4,
            min_distance=1.0,
            max_distance=1.0,
            face_object=True,
        )
        for bx, by, theta in poses:
            # Theta should point from (bx, by) toward (ox, oy).
            expected_theta = math.atan2(oy - by, ox - bx)
            # Normalize angles for comparison.
            diff = (theta - expected_theta + math.pi) % (2 * math.pi) - math.pi
            assert abs(diff) < 1e-10, f"theta={theta}, expected={expected_theta}"

    def test_no_face_object(self) -> None:
        from TyGrit.worlds.reachability import sample_base_poses_around_object

        poses = sample_base_poses_around_object(
            (0.0, 0.0, 0.8),
            num_distances=2,
            num_angles=4,
            face_object=False,
        )
        for _, _, theta in poses:
            assert theta == 0.0


class TestCheckReachability:
    """Integration test for check_reachability.

    Requires the ikfast_fetch C extension. Skipped if not compiled.
    """

    @pytest.fixture(autouse=True)
    def _skip_without_ikfast(self) -> None:
        try:
            import ikfast_fetch  # noqa: F401
        except ImportError:
            pytest.skip("ikfast_fetch C extension not compiled")

    def test_reachable_pose_in_front(self) -> None:
        """An object 0.7m in front of the robot at torso height should be reachable."""
        from TyGrit.worlds.reachability import check_reachability

        # Object at (0.7, 0, 0.8) in world frame; robot at origin facing +X.
        obj_pose = np.eye(4, dtype=np.float64)
        obj_pose[0, 3] = 0.7
        obj_pose[2, 3] = 0.8

        reachable, sol = check_reachability(obj_pose, 0.0, 0.0, 0.0)
        assert reachable
        assert sol is not None
        assert sol.shape == (8,)

    def test_unreachable_pose_far_away(self) -> None:
        """An object 10m away should be unreachable."""
        from TyGrit.worlds.reachability import check_reachability

        obj_pose = np.eye(4, dtype=np.float64)
        obj_pose[0, 3] = 10.0  # 10m in front
        obj_pose[2, 3] = 0.8

        reachable, sol = check_reachability(obj_pose, 0.0, 0.0, 0.0)
        assert not reachable
        assert sol is None

    def test_unreachable_behind_robot(self) -> None:
        """An object 0.7m directly behind the robot should be unreachable."""
        from TyGrit.worlds.reachability import check_reachability

        obj_pose = np.eye(4, dtype=np.float64)
        obj_pose[0, 3] = -0.7  # behind the robot
        obj_pose[2, 3] = 0.8

        reachable, sol = check_reachability(obj_pose, 0.0, 0.0, 0.0)
        assert not reachable
        assert sol is None
