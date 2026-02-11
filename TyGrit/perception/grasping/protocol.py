"""Protocol for grasp prediction backends."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt

from TyGrit.types.grasp import GraspPose


class GraspPredictor(Protocol):
    """Generates grasp candidates from a point cloud."""

    def predict(self, cloud: npt.NDArray[np.float32]) -> list[GraspPose]:
        """Generate grasp candidates for *cloud* (N, 3)."""
        ...
