"""Unit tests for TyGrit.worlds.generators.mobile_grasp.

Tests the pure-logic helpers (pose conversion, object placement sampling)
without simulator imports. The end-to-end generator test requires
ManiSkill and is in a separate test file.

Run: ``pixi run test test/test_worlds_mobile_grasp_generator.py -v``
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from TyGrit.worlds.generators.mobile_grasp import (
    _object_pose_4x4,
    _sample_object_placement,
    _yaw_to_quat_xyzw,
)


class TestYawToQuatXyzw:
    """Tests for _yaw_to_quat_xyzw."""

    def test_zero_yaw_is_identity(self) -> None:
        q = _yaw_to_quat_xyzw(0.0)
        assert q == pytest.approx((0.0, 0.0, 0.0, 1.0))

    def test_90_degrees(self) -> None:
        q = _yaw_to_quat_xyzw(math.pi / 2)
        half = math.pi / 4
        expected = (0.0, 0.0, math.sin(half), math.cos(half))
        assert q == pytest.approx(expected, abs=1e-12)

    def test_180_degrees(self) -> None:
        q = _yaw_to_quat_xyzw(math.pi)
        # sin(pi/2) = 1, cos(pi/2) = 0
        assert q[0] == pytest.approx(0.0)
        assert q[1] == pytest.approx(0.0)
        assert abs(q[2]) == pytest.approx(1.0)
        assert q[3] == pytest.approx(0.0, abs=1e-12)

    def test_unit_quaternion(self) -> None:
        for yaw in [0.0, 0.3, 1.5, math.pi, -0.7]:
            q = _yaw_to_quat_xyzw(yaw)
            norm = math.sqrt(sum(c**2 for c in q))
            assert norm == pytest.approx(1.0, abs=1e-12)


class TestObjectPose4x4:
    """Tests for _object_pose_4x4."""

    def test_identity_at_origin(self) -> None:
        T = _object_pose_4x4(0.0, 0.0, 0.0, 0.0)
        np.testing.assert_allclose(T, np.eye(4), atol=1e-12)

    def test_translation(self) -> None:
        T = _object_pose_4x4(1.5, -2.0, 0.8, 0.0)
        assert T[0, 3] == pytest.approx(1.5)
        assert T[1, 3] == pytest.approx(-2.0)
        assert T[2, 3] == pytest.approx(0.8)

    def test_rotation_z(self) -> None:
        yaw = math.pi / 3
        T = _object_pose_4x4(0.0, 0.0, 0.0, yaw)
        c, s = math.cos(yaw), math.sin(yaw)
        assert T[0, 0] == pytest.approx(c)
        assert T[0, 1] == pytest.approx(-s)
        assert T[1, 0] == pytest.approx(s)
        assert T[1, 1] == pytest.approx(c)

    def test_is_valid_se3(self) -> None:
        T = _object_pose_4x4(1.0, 2.0, 3.0, 1.2)
        # Bottom row.
        np.testing.assert_allclose(T[3, :], [0, 0, 0, 1], atol=1e-12)
        # Rotation is orthonormal.
        R = T[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)


class TestSampleObjectPlacement:
    """Tests for _sample_object_placement."""

    def test_returns_point_on_surface(self) -> None:
        surface = np.array(
            [[1.0, 2.0, 0.8], [1.1, 2.1, 0.81], [0.9, 1.9, 0.79]],
            dtype=np.float32,
        )
        rng = np.random.default_rng(42)
        x, y, z, yaw = _sample_object_placement(surface, rng)
        # The sampled point should be one of the surface points.
        dists = np.linalg.norm(surface - [x, y, z], axis=1)
        assert min(dists) < 1e-6

    def test_yaw_in_range(self) -> None:
        surface = np.random.default_rng(0).uniform(size=(100, 3)).astype(np.float32)
        rng = np.random.default_rng(42)
        for _ in range(50):
            _, _, _, yaw = _sample_object_placement(surface, rng)
            assert 0.0 <= yaw < 2 * math.pi

    def test_deterministic_with_same_rng(self) -> None:
        surface = np.random.default_rng(0).uniform(size=(20, 3)).astype(np.float32)
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        result1 = _sample_object_placement(surface, rng1)
        result2 = _sample_object_placement(surface, rng2)
        assert result1 == pytest.approx(result2)


class TestGenerateDatapointsForScene:
    """Integration test for generate_datapoints_for_scene with synthetic data.

    Uses an IKFast solver if available; skips the test otherwise.
    """

    @pytest.fixture(autouse=True)
    def _skip_without_ikfast(self) -> None:
        try:
            import ikfast_fetch  # noqa: F401
        except ImportError:
            pytest.skip("ikfast_fetch C extension not compiled")

    def test_generates_reachable_datapoints(self) -> None:
        from TyGrit.types.worlds import ObjectSpec, SceneSpec
        from TyGrit.worlds.generators.mobile_grasp import generate_datapoints_for_scene

        scene = SceneSpec(
            scene_id="test/synthetic",
            source="test",
            background_builtin_id="test:synthetic",
        )
        obj = ObjectSpec(
            name="test_cube",
            builtin_id="test:cube",
        )

        # Synthetic placement surface: a table at z=0.8, centered at
        # (0, 0), with enough area to sample from.
        surface = np.zeros((500, 3), dtype=np.float32)
        surface[:, 0] = (
            np.random.default_rng(0).uniform(-0.3, 0.3, 500).astype(np.float32)
        )
        surface[:, 1] = (
            np.random.default_rng(1).uniform(-0.3, 0.3, 500).astype(np.float32)
        )
        surface[:, 2] = 0.8

        rng = np.random.default_rng(42)
        datapoints = generate_datapoints_for_scene(
            scene,
            (obj,),
            [surface],
            num_objects=5,
            rng=rng,
        )

        # We should get at least some reachable datapoints from a
        # table-height surface with the robot searching at 0.6-1.2m.
        assert len(datapoints) > 0
        for dp in datapoints:
            assert dp.scene.scene_id == "test/synthetic"
            assert dp.object.name == "test_cube"
            assert len(dp.base_pose) == 3
            assert dp.init_qpos  # should have planning joint positions
            assert dp.grasp_hint is not None
            assert len(dp.grasp_hint) == 7

    def test_empty_surfaces_returns_empty(self) -> None:
        from TyGrit.types.worlds import ObjectSpec, SceneSpec
        from TyGrit.worlds.generators.mobile_grasp import generate_datapoints_for_scene

        scene = SceneSpec(
            scene_id="test/empty",
            source="test",
            background_builtin_id="test:empty",
        )
        obj = ObjectSpec(name="cube", builtin_id="test:cube")
        rng = np.random.default_rng(0)
        result = generate_datapoints_for_scene(
            scene, (obj,), [], num_objects=5, rng=rng
        )
        assert result == []

    def test_empty_objects_returns_empty(self) -> None:
        from TyGrit.types.worlds import SceneSpec
        from TyGrit.worlds.generators.mobile_grasp import generate_datapoints_for_scene

        scene = SceneSpec(
            scene_id="test/noobj",
            source="test",
            background_builtin_id="test:noobj",
        )
        surface = np.random.default_rng(0).uniform(size=(100, 3)).astype(np.float32)
        rng = np.random.default_rng(0)
        result = generate_datapoints_for_scene(
            scene, (), [surface], num_objects=5, rng=rng
        )
        assert result == []
