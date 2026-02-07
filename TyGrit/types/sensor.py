"""Sensor data types for TyGrit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from TyGrit.types.robot import RobotState


@dataclass(frozen=True)
class SensorSnapshot:
    """A single synchronised capture from the robot's sensors."""

    rgb: npt.NDArray[np.uint8]  # (H, W, 3)
    depth: npt.NDArray[np.float32]  # (H, W) in metres
    intrinsics: npt.NDArray[np.float64]  # (3, 3)
    robot_state: RobotState
    segmentation: npt.NDArray[np.int32] | None = None
