"""ROS-backed Fetch robot environment (placeholder).

Will connect to a real Fetch robot via ROS 2.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from TyGrit.envs.fetch.config import FetchEnvConfig
from TyGrit.envs.fetch.fetch import FetchRobot
from TyGrit.types.planning import Trajectory
from TyGrit.types.robot import RobotState
from TyGrit.types.sensor import SensorSnapshot


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

    def look_at(self, target: npt.NDArray[np.float64], camera_id: str) -> None:
        raise NotImplementedError

    def step(self, action: npt.NDArray[np.float32]) -> SensorSnapshot:
        raise NotImplementedError

    def execute_trajectory(self, trajectory: Trajectory) -> bool:
        raise NotImplementedError

    def start_trajectory(self, trajectory: Trajectory) -> None:
        raise NotImplementedError

    def stop_motion(self) -> None:
        raise NotImplementedError

    def is_motion_done(self) -> bool:
        raise NotImplementedError

    def control_gripper(self, position: float) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
