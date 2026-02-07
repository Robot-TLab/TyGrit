"""Helpers for saving visualizations to disk.

Integrates with the VIZ logging level — call these inside
``log.log("VIZ", ...)`` guards so they are automatically silenced in
production.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt


def save_image(
    image: npt.NDArray[np.uint8],
    path: str | Path,
) -> Path:
    """Write an RGB uint8 image to *path* (png/jpg).

    Returns the resolved path for logging.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return p


def save_pointcloud_ply(
    points: npt.NDArray[np.float32],
    path: str | Path,
    colors: npt.NDArray[np.uint8] | None = None,
) -> Path:
    """Write an (N, 3) point cloud to a PLY file (ASCII).

    Args:
        points: (N, 3) xyz.
        path: Output .ply path.
        colors: (N, 3) uint8 RGB (optional).

    Returns the resolved path for logging.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    n = points.shape[0]
    if colors is not None and colors.shape[0] != n:
        colors = None

    with open(p, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")

        for i in range(n):
            line = f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f}"
            if colors is not None:
                line += f" {colors[i, 0]} {colors[i, 1]} {colors[i, 2]}"
            f.write(line + "\n")

    return p
