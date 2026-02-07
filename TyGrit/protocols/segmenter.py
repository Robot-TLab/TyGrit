"""Protocol for image segmentation services."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt


class Segmenter(Protocol):
    """Segments an RGB image into instance masks."""

    def segment(
        self,
        rgb: npt.NDArray[np.uint8],
    ) -> npt.NDArray[np.int32]: ...
