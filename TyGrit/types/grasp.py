"""Grasp-related data types."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class GraspPose:
    """A candidate grasp with a quality score."""

    transform: npt.NDArray[np.float64]  # (4, 4)
    score: float
