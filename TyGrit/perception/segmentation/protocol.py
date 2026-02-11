"""Protocol for segmentation backends."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt


class Segmenter(Protocol):
    """Segments an object at a given pixel coordinate."""

    def segment(
        self,
        rgb: npt.NDArray[np.uint8],
        point: tuple[int, int],
        segmentation: npt.NDArray[np.int32] | None = None,
    ) -> npt.NDArray[np.uint8] | None:
        """Return a binary mask for the object at *point*.

        Parameters
        ----------
        rgb:
            ``(H, W, 3)`` RGB image.
        point:
            ``(u, v)`` pixel coordinate.
        segmentation:
            ``(H, W, 1)`` instance segmentation IDs (optional, used by
            sim backends).

        Returns
        -------
        ``(H, W)`` uint8 mask with values in {0, 1}, or ``None`` if no
        object is found.
        """
        ...
