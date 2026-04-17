"""Interactive Open3D visualizations for development and debugging.

Unlike the sibling modules in this package (``image``, ``pointcloud_viz``,
``save``) which rasterize to numpy arrays for headless pipelines, the helpers
here open a live window via :func:`open3d.visualization.draw_geometries`.
They are intended for interactive inspection during development and are
only available in the ``thirdparty`` pixi env, which has ``open3d`` installed.

Design:

* Small primitive builders (``make_pointcloud``, ``make_gripper_lineset``,
  ``make_frame``) produce standalone Open3D geometries that can be freely
  composed into a scene.
* ``show`` is a thin wrapper over ``draw_geometries`` that also adds a
  world-frame triad by default.
* ``show_grasps`` is a higher-level recipe matching grasp_anywhere v1's
  ``visualize_grasps_pcd``: top-scoring grasp drawn in ``best_color``, the
  rest in ``other_color``.

Open3D is imported lazily inside each function so that merely importing this
module in environments without ``open3d`` (e.g. ``default``) does not fail.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from TyGrit.types.grasp import GraspPose

__all__ = [
    "make_pointcloud",
    "make_frame",
    "make_gripper_lineset",
    "show",
    "show_grasps",
]


# ── primitives ──────────────────────────────────────────────────────────────


def make_pointcloud(
    points: npt.NDArray[np.floating],
    colors: npt.NDArray[np.floating] | tuple[float, float, float] | None = None,
) -> Any:
    """Build an ``open3d.geometry.PointCloud`` from numpy arrays.

    Args:
        points: ``(N, 3)`` xyz.
        colors: either ``(N, 3)`` per-point RGB in [0, 1] or [0, 255], or a
            single ``(r, g, b)`` triple used to paint the whole cloud. ``None``
            leaves the cloud uncolored (Open3D picks a default shade).
    """
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if colors is None:
        return pcd

    arr = np.asarray(colors, dtype=np.float64)
    if arr.shape == (3,):
        pcd.paint_uniform_color(arr.tolist())
    else:
        if arr.max() > 1.0:
            arr = arr / 255.0
        pcd.colors = o3d.utility.Vector3dVector(arr)
    return pcd


def make_frame(
    size: float = 0.1,
    transform: npt.NDArray[np.floating] | None = None,
) -> Any:
    """Build an RGB coordinate triad at the origin (or at ``transform``)."""
    import open3d as o3d

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    if transform is not None:
        frame.transform(np.asarray(transform, dtype=np.float64))
    return frame


def make_gripper_lineset(
    transform: npt.NDArray[np.floating],
    color: tuple[float, float, float] = (1.0, 0.2, 0.2),
    width: float = 0.12,
    depth: float = 0.10,
    wrist_length: float = 0.06,
) -> Any:
    """Build a 6-point / 5-edge parallel-jaw gripper outline at ``transform``.

    Control points (in the grasp-local frame):

    * 0 — wrist center (grasp origin)
    * 1, 2 — finger bases (±width/2 along X)
    * 3 — wrist tail (−wrist_length along Z)
    * 4, 5 — fingertips (±width/2 along X, +depth along Z)

    Lines: wrist→base ×2, wrist→tail, base→tip ×2. Convention matches
    grasp_anywhere v1's ``visualize_grasps_pcd`` so existing grasp data
    renders identically.
    """
    import open3d as o3d

    local = np.array(
        [
            [0.0, 0.0, 0.0],
            [width / 2, 0.0, 0.0],
            [-width / 2, 0.0, 0.0],
            [0.0, 0.0, -wrist_length],
            [width / 2, 0.0, depth],
            [-width / 2, 0.0, depth],
        ],
        dtype=np.float64,
    )
    lines = np.array(
        [[0, 1], [0, 2], [0, 3], [1, 4], [2, 5]],
        dtype=np.int32,
    )

    t = np.asarray(transform, dtype=np.float64)
    pts_h = np.hstack([local, np.ones((local.shape[0], 1))])
    world = (t @ pts_h.T).T[:, :3]

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(world)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(
        np.tile(np.asarray(color, dtype=np.float64), (lines.shape[0], 1))
    )
    return ls


# ── scene entry points ──────────────────────────────────────────────────────


def show(
    geometries: Sequence[Any],
    *,
    window_name: str = "TyGrit viz",
    include_world_frame: bool = True,
    world_frame_size: float = 0.1,
) -> None:
    """Open an interactive Open3D window showing ``geometries``.

    Blocks until the user closes the window.
    """
    import open3d as o3d

    geoms = list(geometries)
    if include_world_frame:
        geoms.append(make_frame(size=world_frame_size))
    o3d.visualization.draw_geometries(geoms, window_name=window_name)


def show_grasps(
    points: npt.NDArray[np.floating],
    grasps: Sequence[GraspPose],
    *,
    cloud_color: npt.NDArray[np.floating] | tuple[float, float, float] | None = (
        0.4,
        0.6,
        0.85,
    ),
    best_color: tuple[float, float, float] = (0.0, 0.0, 1.0),
    other_color: tuple[float, float, float] = (1.0, 0.2, 0.2),
    window_name: str = "Grasps",
    include_world_frame: bool = True,
) -> None:
    """Show point cloud + gripper outlines for a list of ``GraspPose``.

    The highest-scoring grasp is drawn in ``best_color``; the rest in
    ``other_color``. Matches grasp_anywhere v1's ``visualize_grasps_pcd``.
    """
    geometries: list[Any] = [make_pointcloud(points, colors=cloud_color)]
    if grasps:
        best_idx = int(np.argmax([g.score for g in grasps]))
        for i, g in enumerate(grasps):
            color = best_color if i == best_idx else other_color
            geometries.append(
                make_gripper_lineset(np.asarray(g.transform), color=color)
            )
    show(
        geometries,
        window_name=window_name,
        include_world_frame=include_world_frame,
    )
