"""Image-based visualization: overlays, depth coloring, segmentation masks.

All functions are **pure** — they take arrays in and return arrays out.
No windows, no side effects.
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt


def colorize_depth(
    depth: npt.NDArray[np.float32],
    min_depth: float = 0.0,
    max_depth: float = 3.0,
    colormap: int = cv2.COLORMAP_TURBO,
    invalid_color: tuple[int, int, int] = (0, 0, 0),
) -> npt.NDArray[np.uint8]:
    """Convert a depth map to a coloured RGB image.

    Args:
        depth: (H, W) depth in metres.
        min_depth: Depth mapped to the low end of the colormap.
        max_depth: Depth mapped to the high end of the colormap.
        colormap: OpenCV colormap constant.
        invalid_color: RGB colour for pixels with depth <= 0.

    Returns:
        (H, W, 3) uint8 RGB image.
    """
    valid = depth > 0
    normalised = np.clip((depth - min_depth) / (max_depth - min_depth), 0, 1)
    gray = (normalised * 255).astype(np.uint8)

    coloured_bgr = cv2.applyColorMap(gray, colormap)
    coloured_rgb = cv2.cvtColor(coloured_bgr, cv2.COLOR_BGR2RGB)

    coloured_rgb[~valid] = invalid_color
    return coloured_rgb


def overlay_mask(
    rgb: npt.NDArray[np.uint8],
    mask: npt.NDArray[np.bool_],
    color: tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.4,
) -> npt.NDArray[np.uint8]:
    """Blend a segmentation mask onto an RGB image.

    Args:
        rgb: (H, W, 3) base image.
        mask: (H, W) boolean mask.
        color: RGB colour for masked region.
        alpha: Blending factor (0 = only rgb, 1 = only color).

    Returns:
        (H, W, 3) uint8 blended image.
    """
    out = rgb.copy()
    overlay = np.full_like(rgb, color, dtype=np.uint8)
    out[mask] = cv2.addWeighted(rgb, 1 - alpha, overlay, alpha, 0)[mask]
    return out


def overlay_heatmap(
    rgb: npt.NDArray[np.uint8],
    heatmap: npt.NDArray[np.float32],
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> npt.NDArray[np.uint8]:
    """Blend a [0, 1] heatmap onto an RGB image.

    Args:
        rgb: (H, W, 3) base image.
        heatmap: (H, W) values in [0, 1].
        alpha: Blending factor.
        colormap: OpenCV colormap constant.

    Returns:
        (H, W, 3) uint8 blended image.
    """
    gray = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(gray, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb, 1 - alpha, heatmap_rgb, alpha, 0)


def draw_points_on_image(
    rgb: npt.NDArray[np.uint8],
    uv: npt.NDArray[np.float64],
    color: tuple[int, int, int] = (255, 0, 0),
    radius: int = 3,
) -> npt.NDArray[np.uint8]:
    """Draw projected 3-D points onto an image.

    Args:
        rgb: (H, W, 3) base image.
        uv: (N, 2) pixel coordinates (u, v).
        color: BGR colour for the circles.
        radius: Circle radius in pixels.

    Returns:
        (H, W, 3) uint8 image with circles drawn.
    """
    out = rgb.copy()
    h, w = out.shape[:2]
    for u, v in uv.astype(int):
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(out, (u, v), radius, color, -1)
    return out
