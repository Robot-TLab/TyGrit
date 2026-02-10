"""Sim ground-truth segmenter — uses ManiSkill per-pixel instance IDs."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


class SimSegmenter:
    """Segmenter that uses ground-truth instance segmentation from simulation.

    Expects ``segmentation`` arrays with shape ``(H, W, 1)`` as produced by
    ManiSkill's ``get_sensor_snapshot()``.
    """

    def segment(
        self,
        rgb: npt.NDArray[np.uint8],
        point: tuple[int, int],
        segmentation: npt.NDArray[np.int32] | None = None,
        *,
        background_id: int = 0,
    ) -> npt.NDArray[np.uint8] | None:
        """Return a binary mask for the object at *point*.

        Parameters
        ----------
        rgb:
            ``(H, W, 3)`` RGB image (unused — kept for protocol compatibility).
        point:
            ``(u, v)`` pixel coordinate.
        segmentation:
            ``(H, W, 1)`` instance segmentation IDs.  Required for this
            backend; returns ``None`` when not provided.
        background_id:
            Instance ID treated as background (default ``0``).

        Returns
        -------
        (H, W) uint8 mask with values in {0, 1}, or ``None`` if *point* is
        out-of-bounds, hits background, or *segmentation* is not provided.
        """
        if segmentation is None:
            return None

        seg = np.asarray(segmentation)
        if seg.ndim != 3 or seg.shape[-1] != 1:
            raise ValueError(
                f"Expected segmentation shape (H, W, 1), got shape={seg.shape}"
            )

        seg_ids = seg[..., 0].astype(np.int32, copy=False)
        h, w = seg_ids.shape
        u, v = int(point[0]), int(point[1])

        if not (0 <= u < w and 0 <= v < h):
            return None

        instance_id = int(seg_ids[v, u])
        if instance_id == int(background_id):
            return None

        mask = seg_ids == instance_id
        if not np.any(mask):
            return None

        return mask.astype(np.uint8)

    # ------------------------------------------------------------------
    # Sim-specific helpers (not part of the Segmenter protocol)
    # ------------------------------------------------------------------

    @staticmethod
    def segment_by_id(
        segmentation: npt.NDArray[np.int32],
        object_id: int,
    ) -> npt.NDArray[np.uint8]:
        """Return a binary mask for a specific instance *object_id*.

        Parameters
        ----------
        segmentation:
            ``(H, W, 1)`` instance segmentation IDs.
        object_id:
            The instance ID to extract.

        Returns
        -------
        ``(H, W)`` uint8 mask.
        """
        seg_ids = segmentation[..., 0].astype(np.int32, copy=False)
        return (seg_ids == object_id).astype(np.uint8)
