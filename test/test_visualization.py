"""Tests for TyGrit.visualization module."""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for CI

import numpy as np  # noqa: E402

from TyGrit.visualization.image import (  # noqa: E402
    colorize_depth,
    draw_points_on_image,
    overlay_heatmap,
    overlay_mask,
)
from TyGrit.visualization.pointcloud_viz import (  # noqa: E402
    render_gripper_poses,
    render_pointcloud_3view,
    render_pointcloud_topdown,
)
from TyGrit.visualization.save import save_image, save_pointcloud_ply  # noqa: E402

# ── image ────────────────────────────────────────────────────────────────────


class TestColorizeDepth:
    def test_shape_and_dtype(self):
        depth = np.random.rand(48, 64).astype(np.float32) * 2.0
        img = colorize_depth(depth)
        assert img.shape == (48, 64, 3)
        assert img.dtype == np.uint8

    def test_invalid_pixels_are_black(self):
        depth = np.zeros((10, 10), dtype=np.float32)
        img = colorize_depth(depth)
        assert np.all(img == 0)


class TestOverlayMask:
    def test_shape_preserved(self):
        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:7, 3:7] = True
        out = overlay_mask(rgb, mask)
        assert out.shape == rgb.shape

    def test_unmasked_unchanged(self):
        rgb = np.full((10, 10, 3), 128, dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=bool)
        out = overlay_mask(rgb, mask)
        np.testing.assert_array_equal(out, rgb)


class TestOverlayHeatmap:
    def test_shape_preserved(self):
        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        heatmap = np.random.rand(10, 10).astype(np.float32)
        out = overlay_heatmap(rgb, heatmap)
        assert out.shape == rgb.shape


class TestDrawPointsOnImage:
    def test_no_points(self):
        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        uv = np.empty((0, 2), dtype=np.float64)
        out = draw_points_on_image(rgb, uv)
        np.testing.assert_array_equal(out, rgb)

    def test_out_of_bounds_ignored(self):
        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        uv = np.array([[-5, -5], [100, 100]], dtype=np.float64)
        out = draw_points_on_image(rgb, uv)
        np.testing.assert_array_equal(out, rgb)


# ── pointcloud_viz ───────────────────────────────────────────────────────────


class TestRenderPointcloudTopdown:
    def test_returns_rgb_image(self):
        pts = np.random.randn(100, 3).astype(np.float32)
        img = render_pointcloud_topdown(pts, figsize=(4, 4))
        assert img.ndim == 3
        assert img.shape[2] == 3
        assert img.dtype == np.uint8


class TestRenderPointcloud3View:
    def test_returns_rgb_image(self):
        pts = np.random.randn(50, 3).astype(np.float32)
        img = render_pointcloud_3view(pts, figsize=(12, 4))
        assert img.ndim == 3
        assert img.dtype == np.uint8


class TestRenderGripperPoses:
    def test_no_grasps(self):
        pts = np.random.randn(50, 3).astype(np.float32)
        img = render_gripper_poses(pts, grasps=[], figsize=(4, 4))
        assert img.ndim == 3

    def test_with_grasps(self):
        pts = np.random.randn(50, 3).astype(np.float32)
        grasps = [np.eye(4), np.eye(4)]
        scores = np.array([0.5, 0.9])
        img = render_gripper_poses(pts, grasps=grasps, scores=scores, figsize=(4, 4))
        assert img.ndim == 3


# ── save ─────────────────────────────────────────────────────────────────────


class TestSaveImage:
    def test_roundtrip(self, tmp_path):
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        p = save_image(img, tmp_path / "test.png")
        assert p.exists()


class TestSavePointcloudPly:
    def test_creates_file(self, tmp_path):
        pts = np.random.randn(20, 3).astype(np.float32)
        p = save_pointcloud_ply(pts, tmp_path / "test.ply")
        assert p.exists()
        content = p.read_text()
        assert "element vertex 20" in content

    def test_with_colors(self, tmp_path):
        pts = np.random.randn(10, 3).astype(np.float32)
        colors = np.random.randint(0, 255, (10, 3), dtype=np.uint8)
        p = save_pointcloud_ply(pts, tmp_path / "color.ply", colors=colors)
        content = p.read_text()
        assert "property uchar red" in content
