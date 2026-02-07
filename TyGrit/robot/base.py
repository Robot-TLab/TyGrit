"""Base class for all robot environment implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from TyGrit.types.planning import Trajectory
from TyGrit.types.robot import RobotState
from TyGrit.types.sensor import SensorSnapshot


class RobotBase(ABC):
    """Abstract base every robot environment (sim / real) must subclass.

    Robot-specific capabilities (torso, specific joint control, MPC,
    gaze optimisation) belong in concrete subclasses (e.g. ``FetchRobot``),
    not here.
    """

    # ── sensing ──────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def camera_ids(self) -> list[str]:
        """Available camera identifiers (e.g. ``["head", "wrist"]``)."""
        ...

    @abstractmethod
    def get_sensor_snapshot(self, camera_id: str) -> SensorSnapshot:
        """Capture RGB-D + robot state from the specified camera."""
        ...

    @abstractmethod
    def get_robot_state(self) -> RobotState: ...

    # ── active perception ────────────────────────────────────────────────

    @abstractmethod
    def look_at(self, target: npt.NDArray[np.float64], camera_id: str) -> None:
        """Point *camera_id* at a 3-D world-frame target ``[x, y, z]``.

        The implementation resolves this to whatever mechanism the robot
        uses (pan-tilt head, arm IK for wrist camera, etc.).
        Not all cameras are steerable — implementations should raise
        ``NotImplementedError`` for fixed cameras.
        """
        ...

    # ── stepping (synchronous control) ──────────────────────────────────

    @abstractmethod
    def step(self, action: npt.NDArray[np.float32]) -> SensorSnapshot:
        """Apply *action* for one simulation/control step and return new obs.

        The action vector is robot-specific (e.g. joint velocities).
        No background thread — the caller controls the loop rate.
        """
        ...

    @abstractmethod
    def get_observation(self) -> SensorSnapshot:
        """Return the latest observation without stepping."""
        ...

    # ── motion ───────────────────────────────────────────────────────────

    @abstractmethod
    def execute_trajectory(self, trajectory: Trajectory) -> bool: ...

    @abstractmethod
    def start_trajectory(self, trajectory: Trajectory) -> None: ...

    @abstractmethod
    def stop_motion(self) -> None: ...

    @abstractmethod
    def is_motion_done(self) -> bool: ...

    # ── end-effector ─────────────────────────────────────────────────────

    @abstractmethod
    def control_gripper(self, position: float) -> None: ...

    # ── lifecycle ────────────────────────────────────────────────────────

    @abstractmethod
    def close(self) -> None: ...
