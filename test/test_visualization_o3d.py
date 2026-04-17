"""Tests for TyGrit.visualization.o3d geometry primitives.

Only the numpy-side geometry construction is covered — no window is opened.
Skipped in envs without ``open3d`` installed (e.g. the ``default`` env).
"""

import numpy as np
import pytest

o3d = pytest.importorskip("open3d")

from TyGrit.types.grasp import GraspPose  # noqa: E402
from TyGrit.visualization.o3d import (  # noqa: E402
    make_frame,
    make_gripper_lineset,
    make_pointcloud,
)


class TestMakePointcloud:
    def test_points_round_trip(self):
        pts = np.random.RandomState(0).rand(50, 3).astype(np.float32)
        pcd = make_pointcloud(pts)
        out = np.asarray(pcd.points)
        assert out.shape == (50, 3)
        np.testing.assert_allclose(out, pts, atol=1e-6)

    def test_uniform_color(self):
        pts = np.zeros((10, 3), dtype=np.float32)
        pcd = make_pointcloud(pts, colors=(1.0, 0.0, 0.0))
        colors = np.asarray(pcd.colors)
        assert colors.shape == (10, 3)
        np.testing.assert_allclose(colors[:, 0], 1.0)
        np.testing.assert_allclose(colors[:, 1:], 0.0)

    def test_per_point_colors_rescaled_from_uint8(self):
        pts = np.zeros((3, 3), dtype=np.float32)
        colors_u8 = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.float64)
        pcd = make_pointcloud(pts, colors=colors_u8)
        out = np.asarray(pcd.colors)
        np.testing.assert_allclose(out, colors_u8 / 255.0)


class TestMakeFrame:
    def test_default_origin(self):
        frame = make_frame(size=0.5)
        verts = np.asarray(frame.vertices)
        # coordinate-frame mesh is centered at the origin
        assert np.isclose(np.abs(verts).min(), 0.0, atol=1e-6)

    def test_transform_translates_origin(self):
        T = np.eye(4)
        T[:3, 3] = [1.0, 2.0, 3.0]
        frame = make_frame(size=0.1, transform=T)
        verts = np.asarray(frame.vertices)
        center = verts.mean(axis=0)
        # mesh center shifts with the translation even though it's a mesh of
        # several arrows — a coarse bbox center is sufficient here
        bbox_center = (verts.max(0) + verts.min(0)) / 2.0
        assert bbox_center[0] > 1.0 - 0.5
        assert bbox_center[1] > 2.0 - 0.5
        assert bbox_center[2] > 3.0 - 0.5
        assert center.shape == (3,)


class TestMakeGripperLineset:
    def test_identity_transform_geometry(self):
        ls = make_gripper_lineset(
            np.eye(4),
            color=(1.0, 0.0, 0.0),
            width=0.12,
            depth=0.10,
            wrist_length=0.06,
        )
        pts = np.asarray(ls.points)
        lines = np.asarray(ls.lines)
        colors = np.asarray(ls.colors)

        assert pts.shape == (6, 3)
        assert lines.shape == (5, 2)
        assert colors.shape == (5, 3)
        np.testing.assert_allclose(colors[0], [1.0, 0.0, 0.0])

        # Control-point layout (grasp-local frame, identity transform)
        np.testing.assert_allclose(pts[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(pts[1], [0.06, 0.0, 0.0])
        np.testing.assert_allclose(pts[2], [-0.06, 0.0, 0.0])
        np.testing.assert_allclose(pts[3], [0.0, 0.0, -0.06])
        np.testing.assert_allclose(pts[4], [0.06, 0.0, 0.10])
        np.testing.assert_allclose(pts[5], [-0.06, 0.0, 0.10])

    def test_translation_applied_to_all_points(self):
        T = np.eye(4)
        T[:3, 3] = [1.0, 2.0, 3.0]
        ls = make_gripper_lineset(T, color=(0, 0, 1))
        pts = np.asarray(ls.points)
        # wrist center lands at the translation
        np.testing.assert_allclose(pts[0], [1.0, 2.0, 3.0])

    def test_rotation_applied(self):
        # 90° rotation around Z: +x → +y
        T = np.eye(4)
        T[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        ls = make_gripper_lineset(T, color=(0, 1, 0), width=0.2, depth=0.1)
        pts = np.asarray(ls.points)
        # finger base at local +x ends up at world +y
        np.testing.assert_allclose(pts[1], [0.0, 0.1, 0.0], atol=1e-9)


class TestShowGraspsContract:
    """``show_grasps`` opens a window, so we can't run it under pytest.

    We can still verify it accepts a ``list[GraspPose]`` without argument-
    shape errors by constructing the geometries separately via the same
    primitives — which is what the function does internally.
    """

    def test_graspposes_consumable(self):
        grasps = [
            GraspPose(transform=np.eye(4), score=0.1),
            GraspPose(transform=np.eye(4), score=0.9),
        ]
        best_idx = int(np.argmax([g.score for g in grasps]))
        assert best_idx == 1

        # Build the geometry the same way show_grasps does internally.
        pcd = make_pointcloud(np.zeros((4, 3), dtype=np.float32))
        assert len(np.asarray(pcd.points)) == 4
        for i, g in enumerate(grasps):
            color = (0.0, 0.0, 1.0) if i == best_idx else (1.0, 0.2, 0.2)
            ls = make_gripper_lineset(g.transform, color=color)
            assert len(np.asarray(ls.points)) == 6
