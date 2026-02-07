"""Visualization utilities for development and debugging.

All rendering functions produce numpy arrays (no interactive windows),
making them safe for headless servers and easy to pipe into the VIZ
logging level.
"""

from TyGrit.visualization.image import (
    colorize_depth,
    draw_points_on_image,
    overlay_heatmap,
    overlay_mask,
)
from TyGrit.visualization.pointcloud_viz import (
    render_gripper_poses,
    render_pointcloud_3view,
    render_pointcloud_topdown,
)
from TyGrit.visualization.save import save_image, save_pointcloud_ply

__all__ = [
    # image
    "colorize_depth",
    "overlay_mask",
    "overlay_heatmap",
    "draw_points_on_image",
    # pointcloud
    "render_pointcloud_topdown",
    "render_pointcloud_3view",
    "render_gripper_poses",
    # save
    "save_image",
    "save_pointcloud_ply",
]
