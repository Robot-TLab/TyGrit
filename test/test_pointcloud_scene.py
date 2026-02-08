"""Tests for TyGrit.scene.pointcloud_scene — PointCloudScene."""

import numpy as np
import pytest

from TyGrit.scene.config import PointCloudSceneConfig
from TyGrit.scene.pointcloud_scene import PointCloudScene
from TyGrit.types.geometry import SE2Pose
from TyGrit.types.robot import RobotState
from TyGrit.types.sensor import SensorSnapshot

# ── helpers ──────────────────────────────────────────────────────────────────


def _make_snapshot(
    depth: np.ndarray | None = None,
    intrinsics: np.ndarray | None = None,
) -> SensorSnapshot:
    """Create a minimal SensorSnapshot for testing."""
    if depth is None:
        # 10x10 depth image at 1.0m
        depth = np.ones((10, 10), dtype=np.float32)
    if intrinsics is None:
        intrinsics = np.array(
            [[100.0, 0, 5.0], [0, 100.0, 5.0], [0, 0, 1]], dtype=np.float64
        )
    rgb = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
    state = RobotState(
        base_pose=SE2Pose(0.0, 0.0, 0.0),
        planning_joints=tuple([0.0] * 8),
        head_joints=(0.0, 0.0),
    )
    return SensorSnapshot(
        rgb=rgb, depth=depth, intrinsics=intrinsics, robot_state=state
    )


def _identity_cam_pose() -> np.ndarray:
    """Camera at origin, looking along +Z."""
    return np.eye(4, dtype=np.float64)


# ── Tests ────────────────────────────────────────────────────────────────────


class TestPointCloudSceneInit:
    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be one of"):
            PointCloudScene(PointCloudSceneConfig(mode="invalid"))

    def test_valid_modes(self):
        for mode in ("static", "latest", "accumulated", "combine", "ray_casting"):
            scene = PointCloudScene(PointCloudSceneConfig(mode=mode))
            assert scene.mode == mode

    def test_empty_static_map(self):
        scene = PointCloudScene()
        pcd = scene.get_pointcloud()
        assert pcd.shape == (0, 3)

    def test_static_map_downsampled(self):
        # Create a dense static map with many duplicate voxels
        rng = np.random.default_rng(0)
        pts = rng.uniform(0, 0.01, (1000, 3)).astype(np.float32)
        scene = PointCloudScene(
            PointCloudSceneConfig(downsample_voxel_size=0.05), static_map=pts
        )
        pcd = scene.get_pointcloud()
        # All points are in one voxel, so should collapse
        assert pcd.shape[0] < 100


class TestStaticMode:
    def test_update_does_nothing(self):
        static_pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        scene = PointCloudScene(
            PointCloudSceneConfig(mode="static"), static_map=static_pts
        )

        snap = _make_snapshot()
        cam = _identity_cam_pose()
        scene.update(snap, cam)

        pcd = scene.get_pointcloud()
        np.testing.assert_array_equal(pcd, static_pts)

    def test_clear_resets_dynamic_not_static(self):
        static_pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        scene = PointCloudScene(
            PointCloudSceneConfig(mode="static"), static_map=static_pts
        )
        scene.clear()
        pcd = scene.get_pointcloud()
        np.testing.assert_array_equal(pcd, static_pts)


class TestLatestMode:
    def test_update_replaces_previous(self):
        scene = PointCloudScene(PointCloudSceneConfig(enable_ground_filter=False))

        snap1 = _make_snapshot(depth=np.full((10, 10), 1.0, dtype=np.float32))
        cam1 = _identity_cam_pose()
        scene.update(snap1, cam1)
        obs1 = scene.current_observations()

        snap2 = _make_snapshot(depth=np.full((10, 10), 2.0, dtype=np.float32))
        scene.update(snap2, cam1)
        obs2 = scene.current_observations()

        # obs2 should have different points (depth 2m vs 1m)
        assert obs1.shape[0] > 0
        assert obs2.shape[0] > 0
        # Depths differ, so points should differ
        assert not np.allclose(obs1, obs2)

    def test_clear_empties_observations(self):
        scene = PointCloudScene(PointCloudSceneConfig(enable_ground_filter=False))
        snap = _make_snapshot()
        scene.update(snap, _identity_cam_pose())
        assert scene.current_observations().shape[0] > 0

        scene.clear()
        assert scene.current_observations().shape[0] == 0

    def test_get_pointcloud_merges_static_and_dynamic(self):
        static_pts = np.array([[10.0, 10.0, 10.0]], dtype=np.float32)
        scene = PointCloudScene(
            PointCloudSceneConfig(enable_ground_filter=False), static_map=static_pts
        )

        snap = _make_snapshot()
        scene.update(snap, _identity_cam_pose())

        pcd = scene.get_pointcloud()
        # Should have static + dynamic points
        assert pcd.shape[0] > 1


class TestAccumulatedMode:
    def test_accumulates_across_updates(self):
        scene = PointCloudScene(
            PointCloudSceneConfig(
                mode="accumulated",
                enable_ground_filter=False,
                crop_radius=100.0,
            )
        )

        # Two updates with cameras pointing in different directions
        snap = _make_snapshot(depth=np.full((10, 10), 1.0, dtype=np.float32))

        cam1 = np.eye(4, dtype=np.float64)
        cam1[0, 3] = 0.0  # camera at (0,0,0)
        scene.update(snap, cam1)
        count1 = scene.current_observations().shape[0]

        cam2 = np.eye(4, dtype=np.float64)
        cam2[0, 3] = 5.0  # camera at (5,0,0)
        scene.update(snap, cam2)
        count2 = scene.current_observations().shape[0]

        # Second update from a different position should add new points
        assert count2 >= count1


class TestCombineMode:
    def test_downsample_prevents_unbounded_growth(self):
        scene = PointCloudScene(
            PointCloudSceneConfig(
                mode="combine",
                enable_ground_filter=False,
                downsample_voxel_size=0.1,
                crop_radius=100.0,
            )
        )

        snap = _make_snapshot(depth=np.full((10, 10), 1.0, dtype=np.float32))
        cam = _identity_cam_pose()

        # Multiple updates with same view should not grow without bound
        for _ in range(10):
            scene.update(snap, cam)

        obs = scene.current_observations()
        # With a 10x10 depth at 1m, points span a small volume
        # After 10 updates with dedup, count should be bounded
        assert obs.shape[0] < 5000


class TestRayCastingMode:
    def test_removes_free_space_points(self):
        """Ray casting removes points that are in front of the measured surface.

        Free-space logic: if an existing point's camera-frame depth is LESS than
        the measured depth minus a margin, it lies in free space and is removed.
        Points behind the surface (occluded) are kept.
        """
        scene = PointCloudScene(
            PointCloudSceneConfig(
                mode="ray_casting",
                enable_ground_filter=False,
                crop_radius=100.0,
            )
        )

        # First update: add points at depth=1.0 (near)
        snap_near = _make_snapshot(depth=np.full((10, 10), 1.0, dtype=np.float32))
        cam = _identity_cam_pose()
        scene.update(snap_near, cam)
        count_after_first = scene.current_observations().shape[0]
        assert count_after_first > 0

        # Second update with depth=2.0 (far).
        # Existing points at z=1.0 are in front of the surface at z=2.0,
        # so they should be cleared as free space, then new points at z=2.0 added.
        snap_far = _make_snapshot(depth=np.full((10, 10), 2.0, dtype=np.float32))
        scene.update(snap_far, cam)
        count_after_second = scene.current_observations().shape[0]

        # The near points should have been removed (free space), replaced by far.
        # Count should not roughly double.
        assert count_after_second <= count_after_first * 1.5


class TestGoalCloud:
    def test_set_and_get(self):
        scene = PointCloudScene()
        goal = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        scene.set_goal_pcd(goal)

        retrieved = scene.get_goal_pcd()
        np.testing.assert_array_equal(retrieved, goal)

    def test_goal_is_independent_of_clear(self):
        scene = PointCloudScene()
        goal = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        scene.set_goal_pcd(goal)
        scene.clear()

        retrieved = scene.get_goal_pcd()
        np.testing.assert_array_equal(retrieved, goal)

    def test_get_goal_returns_copy(self):
        scene = PointCloudScene()
        goal = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        scene.set_goal_pcd(goal)

        retrieved = scene.get_goal_pcd()
        retrieved[0, 0] = 999.0
        # Original should be unchanged
        np.testing.assert_array_equal(scene.get_goal_pcd()[0, 0], 1.0)


class TestRobotFilter:
    def test_robot_filter_fn_is_called(self):
        call_count = [0]

        def mock_filter(pts, snapshot):
            call_count[0] += 1
            # Remove half the points
            return pts[: pts.shape[0] // 2]

        scene = PointCloudScene(
            PointCloudSceneConfig(enable_ground_filter=False),
            robot_filter_fn=mock_filter,
        )
        snap = _make_snapshot()
        scene.update(snap, _identity_cam_pose())

        assert call_count[0] == 1

    def test_robot_filter_reduces_points(self):
        def halve(pts, snapshot):
            return pts[: pts.shape[0] // 2]

        cfg = PointCloudSceneConfig(enable_ground_filter=False)
        scene_no_filter = PointCloudScene(cfg)
        scene_with_filter = PointCloudScene(cfg, robot_filter_fn=halve)

        snap = _make_snapshot()
        cam = _identity_cam_pose()

        scene_no_filter.update(snap, cam)
        scene_with_filter.update(snap, cam)

        assert (
            scene_with_filter.current_observations().shape[0]
            < scene_no_filter.current_observations().shape[0]
        )


class TestGroundFilter:
    def test_ground_filtered_by_default(self):
        scene = PointCloudScene(PointCloudSceneConfig(ground_z_threshold=0.3))
        # Depth image at 0.5m through identity camera → Z values are ~0.5m in world
        # Points have Z ≈ world Y in camera coords... actually identity camera
        # puts depth along camera Z which is world Z in identity pose.
        # With identity cam, depth=0.5 → world Z=0.5 → passes ground filter (>0.3)
        snap = _make_snapshot(depth=np.full((10, 10), 0.5, dtype=np.float32))
        scene.update(snap, _identity_cam_pose())
        count_pass = scene.current_observations().shape[0]

        # Depth=0.25 → world Z=0.25 → filtered by ground filter (<=0.3)
        scene.clear()
        snap_low = _make_snapshot(depth=np.full((10, 10), 0.25, dtype=np.float32))
        scene.update(snap_low, _identity_cam_pose())
        count_filtered = scene.current_observations().shape[0]

        assert count_pass > 0
        assert count_filtered == 0

    def test_ground_filter_disabled(self):
        scene = PointCloudScene(PointCloudSceneConfig(enable_ground_filter=False))
        snap = _make_snapshot(depth=np.full((10, 10), 0.25, dtype=np.float32))
        scene.update(snap, _identity_cam_pose())
        assert scene.current_observations().shape[0] > 0
