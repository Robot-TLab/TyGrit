"""Protocol for grasp prediction services."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt

from TyGrit.types.planning import GraspPose


class GraspPredictor(Protocol):
    """Predicts candidate grasps from a point cloud."""

    def predict(self, cloud: npt.NDArray[np.float32]) -> list[GraspPose]: ...
