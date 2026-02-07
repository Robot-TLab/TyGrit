"""Tests for TyGrit.utils (math, transforms, pointcloud, depth)."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from TyGrit.utils.depth import (
    depth_to_pointcloud,
    depth_to_world_pointcloud,
    pointcloud_from_mask,
    project_points_to_image,
)
from TyGrit.utils.math import (
    angle_wrap,
    matrix_to_quaternion,
    quaternion_to_matrix,
    translation_from_matrix,
)
from TyGrit.utils.pointcloud import (
    crop_sphere,
    filter_ground,
    merge_dedup,
    points_in_frustum_mask,
    voxel_downsample,
)
from TyGrit.utils.transforms import (
    create_pose_matrix,
    create_transform_matrix,
    pose_to_base,
    pose_to_world,
    se2_to_matrix,
)

# ── math ─────────────────────────────────────────────────────────────────────


class TestAngleWrap:
    def test_zero(self):
        assert angle_wrap(0.0) == pytest.approx(0.0)

    def test_positive_over_pi(self):
        # 3*pi wraps to -pi (boundary case of (-pi, pi])
        result = angle_wrap(3 * np.pi)
        assert abs(abs(result) - np.pi) < 1e-10

    def test_negative(self):
        result = angle_wrap(-3 * np.pi)
        assert abs(abs(result) - np.pi) < 1e-10

    def test_two_pi(self):
        result = angle_wrap(2 * np.pi)
        assert result == pytest.approx(0.0, abs=1e-10)


class TestQuaternionMatrix:
    def test_identity_quaternion(self):
        q = np.array([0, 0, 0, 1], dtype=np.float64)  # xyzw
        T = quaternion_to_matrix(q)
        np.testing.assert_allclose(T, np.eye(4), atol=1e-12)

    def test_roundtrip(self):
        q_in = R.random().as_quat()  # xyzw
        T = quaternion_to_matrix(q_in)
        q_out = matrix_to_quaternion(T)
        # Quaternions are equivalent up to sign
        if np.dot(q_in, q_out) < 0:
            q_out = -q_out
        np.testing.assert_allclose(q_in, q_out, atol=1e-10)

    def test_90_deg_z(self):
        q = R.from_euler("z", 90, degrees=True).as_quat()
        T = quaternion_to_matrix(q)
        expected = np.eye(4)
        expected[:3, :3] = R.from_euler("z", 90, degrees=True).as_matrix()
        np.testing.assert_allclose(T, expected, atol=1e-12)


class TestTranslationFromMatrix:
    def test_extract(self):
        T = np.eye(4)
        T[:3, 3] = [1, 2, 3]
        t = translation_from_matrix(T)
        np.testing.assert_allclose(t, [1, 2, 3])


# ── transforms ───────────────────────────────────────────────────────────────


class TestCreatePoseMatrix:
    def test_identity(self):
        T = create_pose_matrix(np.zeros(3), np.array([0, 0, 0, 1], dtype=np.float64))
        np.testing.assert_allclose(T, np.eye(4), atol=1e-12)

    def test_translation(self):
        T = create_pose_matrix(
            np.array([1.0, 2.0, 3.0]),
            np.array([0, 0, 0, 1], dtype=np.float64),
        )
        np.testing.assert_allclose(T[:3, 3], [1, 2, 3])


class TestCreateTransformMatrix:
    def test_identity(self):
        T = create_transform_matrix(np.zeros(3), np.eye(3))
        np.testing.assert_allclose(T, np.eye(4))


class TestPoseToWorldAndBack:
    def test_roundtrip(self):
        base_pos = np.array([1.0, 2.0, 0.0])
        base_yaw = 0.5
        ee_pos_orig = np.array([0.5, 0.1, 0.8])
        ee_quat_orig = R.from_euler("z", 0.3).as_quat()

        world_pos, world_quat = pose_to_world(
            base_pos, base_yaw, ee_pos_orig, ee_quat_orig
        )
        ee_pos_back, ee_quat_back = pose_to_base(
            world_pos, world_quat, base_pos, base_yaw
        )
        np.testing.assert_allclose(ee_pos_back, ee_pos_orig, atol=1e-10)

        # Quaternion sign ambiguity
        if np.dot(ee_quat_orig, ee_quat_back) < 0:
            ee_quat_back = -ee_quat_back
        np.testing.assert_allclose(ee_quat_back, ee_quat_orig, atol=1e-10)

    def test_zero_base(self):
        base_pos = np.array([0.0, 0.0, 0.0])
        ee_pos = np.array([1.0, 0.0, 0.0])
        ee_quat = np.array([0.0, 0.0, 0.0, 1.0])
        world_pos, world_quat = pose_to_world(base_pos, 0.0, ee_pos, ee_quat)
        np.testing.assert_allclose(world_pos, [1.0, 0.0, 0.0], atol=1e-12)


class TestSE2ToMatrix:
    def test_identity(self):
        T = se2_to_matrix(0, 0, 0)
        np.testing.assert_allclose(T, np.eye(4), atol=1e-12)

    def test_translation_only(self):
        T = se2_to_matrix(3.0, 4.0, 0)
        np.testing.assert_allclose(T[:3, 3], [3, 4, 0])

    def test_rotation(self):
        T = se2_to_matrix(0, 0, np.pi / 2)
        # Should rotate x-axis to y-axis
        np.testing.assert_allclose(T[:2, 0], [0, 1], atol=1e-12)


# ── pointcloud ───────────────────────────────────────────────────────────────


class TestVoxelDownsample:
    def test_empty(self):
        pts = np.empty((0, 3), dtype=np.float32)
        result = voxel_downsample(pts, 0.1)
        assert result.shape == (0, 3)

    def test_duplicates_removed(self):
        pts = np.array([[0.0, 0.0, 0.0], [0.001, 0.001, 0.001]], dtype=np.float32)
        result = voxel_downsample(pts, 0.01)
        assert result.shape[0] == 1

    def test_separate_points_kept(self):
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
        result = voxel_downsample(pts, 0.1)
        assert result.shape[0] == 2


class TestMergeDedup:
    def test_empty_add(self):
        base = np.array([[0, 0, 0]], dtype=np.float32)
        add = np.empty((0, 3), dtype=np.float32)
        result = merge_dedup(base, add, 0.1)
        assert result.shape == (1, 3)

    def test_empty_base(self):
        base = np.empty((0, 3), dtype=np.float32)
        add = np.array([[0, 0, 0]], dtype=np.float32)
        result = merge_dedup(base, add, 0.1)
        assert result.shape == (1, 3)

    def test_dedup(self):
        base = np.array([[0, 0, 0]], dtype=np.float32)
        add = np.array([[0.01, 0.01, 0.01]], dtype=np.float32)
        result = merge_dedup(base, add, 0.1)
        # Both fall in the same voxel, base wins
        assert result.shape[0] == 1

    def test_new_points_added(self):
        base = np.array([[0, 0, 0]], dtype=np.float32)
        add = np.array([[1, 1, 1]], dtype=np.float32)
        result = merge_dedup(base, add, 0.1)
        assert result.shape[0] == 2


class TestCropSphere:
    def test_all_inside(self):
        pts = np.array([[0, 0, 0], [0.5, 0.5, 0]], dtype=np.float32)
        center = np.array([0, 0, 0], dtype=np.float32)
        result = crop_sphere(pts, center, 2.0)
        assert result.shape[0] == 2

    def test_one_outside(self):
        pts = np.array([[0, 0, 0], [10, 10, 0]], dtype=np.float32)
        center = np.array([0, 0, 0], dtype=np.float32)
        result = crop_sphere(pts, center, 2.0)
        assert result.shape[0] == 1


class TestFilterGround:
    def test_removes_low_points(self):
        pts = np.array([[0, 0, 0.1], [0, 0, 0.5], [0, 0, 1.0]], dtype=np.float32)
        result = filter_ground(pts, 0.3)
        assert result.shape[0] == 2

    def test_empty_input(self):
        pts = np.empty((0, 3), dtype=np.float32)
        result = filter_ground(pts, 0.3)
        assert result.shape == (0, 3)


class TestPointsInFrustumMask:
    def test_basic(self):
        # Create a camera at origin looking along +Z
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        T = np.eye(4, dtype=np.float64)

        # Point at [0, 0, 1] in world -- should project to image center
        pts = np.array([[0, 0, 1]], dtype=np.float32)
        mask = points_in_frustum_mask(pts, K, T, (0.1, 10.0))
        assert mask[0]

    def test_behind_camera(self):
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        T = np.eye(4, dtype=np.float64)

        pts = np.array([[0, 0, -1]], dtype=np.float32)
        mask = points_in_frustum_mask(pts, K, T, (0.1, 10.0))
        assert not mask[0]


# ── depth ────────────────────────────────────────────────────────────────────


class TestDepthToPointcloud:
    def test_empty_depth(self):
        depth = np.zeros((10, 10), dtype=np.float32)
        K = np.eye(3, dtype=np.float64)
        result = depth_to_pointcloud(depth, K)
        assert result.shape == (0, 3)

    def test_single_pixel(self):
        depth = np.zeros((5, 5), dtype=np.float32)
        depth[2, 2] = 1.0  # 1 metre depth at the principal point
        K = np.array([[1, 0, 2], [0, 1, 2], [0, 0, 1]], dtype=np.float64)
        result = depth_to_pointcloud(depth, K)
        assert result.shape == (1, 3)
        # At the principal point, x=y=0, z=depth
        np.testing.assert_allclose(result[0], [0, 0, 1], atol=1e-5)

    def test_stride(self):
        depth = np.ones((10, 10), dtype=np.float32)
        K = np.array([[10, 0, 5], [0, 10, 5], [0, 0, 1]], dtype=np.float64)
        result_1 = depth_to_pointcloud(depth, K, stride=1)
        result_2 = depth_to_pointcloud(depth, K, stride=2)
        assert result_2.shape[0] < result_1.shape[0]


class TestDepthToWorldPointcloud:
    def test_transforms_to_world(self):
        depth = np.zeros((5, 5), dtype=np.float32)
        depth[2, 2] = 1.0
        K = np.array([[1, 0, 2], [0, 1, 2], [0, 0, 1]], dtype=np.float64)

        # Camera shifted 10m along X
        T_wc = np.eye(4, dtype=np.float64)
        T_wc[0, 3] = 10.0

        result = depth_to_world_pointcloud(depth, K, T_wc, z_range=(0.1, 5.0), stride=1)
        assert result.shape[0] == 1
        # Point was at (0, 0, 1) in camera; camera at x=10
        np.testing.assert_allclose(result[0, 0], 10.0, atol=1e-3)


class TestPointcloudFromMask:
    def test_no_valid_depth(self):
        depth = np.zeros((5, 5), dtype=np.float32)
        mask = np.ones((5, 5), dtype=bool)
        K = np.eye(3, dtype=np.float64)
        result = pointcloud_from_mask(depth, mask, K)
        assert result.shape == (0, 3)

    def test_single_valid(self):
        depth = np.zeros((5, 5), dtype=np.float32)
        depth[2, 3] = 2.0
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 3] = True
        K = np.array([[1, 0, 2.5], [0, 1, 2.5], [0, 0, 1]], dtype=np.float64)
        result = pointcloud_from_mask(depth, mask, K)
        assert result.shape == (1, 3)
        # x = (3 - 2.5) * 2 / 1 = 1.0, y = (2 - 2.5) * 2 / 1 = -1.0, z = 2
        np.testing.assert_allclose(result[0], [1.0, -1.0, 2.0], atol=1e-5)


class TestProjectPointsToImage:
    def test_center_projection(self):
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        T = np.eye(4, dtype=np.float64)
        pts = np.array([[0, 0, 1]], dtype=np.float32)
        u, v, z = project_points_to_image(pts, K, T)
        np.testing.assert_allclose(u[0], 320, atol=1e-3)
        np.testing.assert_allclose(v[0], 240, atol=1e-3)
        np.testing.assert_allclose(z[0], 1.0, atol=1e-3)
