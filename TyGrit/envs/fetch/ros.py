"""ROS-backed Fetch robot environment (placeholder).

Will connect to a real Fetch robot via ROS 2. The class satisfies the
:class:`~TyGrit.envs.base.RobotBase` Protocol; trajectory tracking,
gaze, and other higher-level concerns live in their own modules and
consume this placeholder via duck typing.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from TyGrit.envs.fetch import FetchRobot
from TyGrit.envs.fetch.config import FetchEnvConfig
from TyGrit.types.robots import RobotState
from TyGrit.types.sensors import SensorSnapshot


class ROSFetchRobot(FetchRobot):
    """Fetch robot driven by ROS 2 (not yet implemented)."""

    def __init__(self, config: FetchEnvConfig | None = None) -> None:
        raise NotImplementedError("ROS backend is not yet implemented")

    @property
    def camera_ids(self) -> list[str]:
        raise NotImplementedError

    def get_sensor_snapshot(self, camera_id: str) -> SensorSnapshot:
        raise NotImplementedError

    def get_robot_state(self) -> RobotState:
        raise NotImplementedError

    def get_observation(self) -> SensorSnapshot:
        raise NotImplementedError

    def step(self, action: npt.NDArray[np.float32]) -> SensorSnapshot:
        raise NotImplementedError

    def control_gripper(self, position: float) -> None:
        raise NotImplementedError

    def reset(self) -> SensorSnapshot:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
