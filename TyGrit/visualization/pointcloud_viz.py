"""Point-cloud and 3-D visualization rendered to images via matplotlib.

No Open3D windows — everything produces numpy arrays or saves to files,
making it safe for headless servers and easy to integrate with the VIZ
logging level.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def render_pointcloud_topdown(
    points: npt.NDArray[np.float32],
    colors: npt.NDArray[np.float32] | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    point_size: float = 0.3,
    figsize: tuple[int, int] = (8, 8),
    title: str = "",
) -> npt.NDArray[np.uint8]:
    """Render a top-down (XY) view of a point cloud to an RGB image.

    Args:
        points: (N, 3) world-frame points.
        colors: (N, 3) RGB colours in [0, 1] (optional).
        xlim: X axis limits. Auto-computed if *None*.
        ylim: Y axis limits. Auto-computed if *None*.
        point_size: Matplotlib scatter size.
        figsize: Figure dimensions in inches.
        title: Plot title.

    Returns:
        (H, W, 3) uint8 RGB image.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        points[:, 0],
        points[:, 1],
        c=colors if colors is not None else points[:, 2],
        s=point_size,
        cmap="viridis" if colors is None else None,
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return _fig_to_array(fig)


def render_pointcloud_3view(
    points: npt.NDArray[np.float32],
    colors: npt.NDArray[np.float32] | None = None,
    point_size: float = 0.3,
    figsize: tuple[int, int] = (18, 6),
    title: str = "",
) -> npt.NDArray[np.uint8]:
    """Render XY (top), XZ (side), YZ (front) views of a point cloud.

    Returns:
        (H, W, 3) uint8 RGB image with all three panels.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    c = colors if colors is not None else points[:, 2]
    cmap = "viridis" if colors is None else None

    views = [
        (0, 1, "Top-down (XY)", "X (m)", "Y (m)"),
        (0, 2, "Side (XZ)", "X (m)", "Z (m)"),
        (1, 2, "Front (YZ)", "Y (m)", "Z (m)"),
    ]

    for ax, (i, j, vtitle, xlabel, ylabel) in zip(axes, views):
        ax.scatter(points[:, i], points[:, j], c=c, s=point_size, cmap=cmap)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect("equal")
        ax.set_title(vtitle)
        ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    return _fig_to_array(fig)


def render_gripper_poses(
    points: npt.NDArray[np.float32],
    grasps: Sequence[npt.NDArray[np.float64]],
    scores: npt.NDArray[np.float64] | None = None,
    best_color: tuple[float, float, float] = (0.0, 0.0, 1.0),
    other_color: tuple[float, float, float] = (1.0, 0.0, 0.0),
    width: float = 0.12,
    depth: float = 0.10,
    figsize: tuple[int, int] = (10, 10),
    title: str = "Grasp Poses",
) -> npt.NDArray[np.uint8]:
    """Render point cloud + line-based gripper poses in a top-down view.

    Args:
        points: (N, 3) point cloud.
        grasps: Sequence of (4, 4) grasp pose matrices.
        scores: (G,) scores — the highest is drawn in *best_color*.
        best_color: RGB [0, 1] for the top-scoring grasp.
        other_color: RGB [0, 1] for all other grasps.
        width: Gripper opening width (m).
        depth: Gripper finger depth (m).
        figsize: Figure size.
        title: Plot title.

    Returns:
        (H, W, 3) uint8 RGB image.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Draw point cloud
    ax.scatter(points[:, 0], points[:, 1], c="gray", s=0.3, alpha=0.4)

    best_idx = int(np.argmax(scores)) if scores is not None and len(scores) > 0 else -1

    # Gripper template (local frame)
    template = np.array(
        [
            [0, 0, 0],
            [width / 2, 0, 0],
            [-width / 2, 0, 0],
            [0, 0, -0.06],
            [width / 2, 0, depth],
            [-width / 2, 0, depth],
        ]
    )
    lines = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5)]

    for i, g in enumerate(grasps):
        mat = np.asarray(g).reshape(4, 4)
        pts_h = np.hstack([template, np.ones((6, 1))])
        pts_w = (mat @ pts_h.T).T[:, :3]

        color = best_color if i == best_idx else other_color
        for a, b in lines:
            ax.plot(
                [pts_w[a, 0], pts_w[b, 0]],
                [pts_w[a, 1], pts_w[b, 1]],
                color=color,
                linewidth=2 if i == best_idx else 1,
            )

    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return _fig_to_array(fig)


# ── helpers ──────────────────────────────────────────────────────────────────


def _fig_to_array(fig: plt.Figure) -> npt.NDArray[np.uint8]:
    """Rasterize a matplotlib figure to an RGB numpy array, then close it."""
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3].copy()
    plt.close(fig)
    return img
