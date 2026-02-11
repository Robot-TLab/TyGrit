"""Protocol for robot environment implementations."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt

from TyGrit.types.planning import Trajectory
from TyGrit.types.robot import RobotState
from TyGrit.types.sensor import SensorSnapshot


class RobotBase(Protocol):
    """Protocol every robot environment (sim / real) must satisfy.

    Robot-specific capabilities (torso, specific joint control, MPC,
    gaze optimisation) belong in concrete subclasses (e.g. ``FetchRobot``),
    not here.
    """

    # ── sensing ──────────────────────────────────────────────────────────

    @property
    def camera_ids(self) -> list[str]:
        """Available camera identifiers (e.g. ``["head", "wrist"]``)."""
        ...

    def get_sensor_snapshot(self, camera_id: str) -> SensorSnapshot:
        """Capture RGB-D + robot state from the specified camera."""
        ...

    def get_robot_state(self) -> RobotState: ...

    # ── active perception ────────────────────────────────────────────────

    def look_at(self, target: npt.NDArray[np.float64], camera_id: str) -> None:
        """Point *camera_id* at a 3-D world-frame target ``[x, y, z]``.

        The implementation resolves this to whatever mechanism the robot
        uses (pan-tilt head, arm IK for wrist camera, etc.).
        Not all cameras are steerable — implementations should raise
        ``NotImplementedError`` for fixed cameras.
        """
        ...

    # ── stepping (synchronous control) ──────────────────────────────────

    def step(self, action: npt.NDArray[np.float32]) -> SensorSnapshot:
        """Apply *action* for one simulation/control step and return new obs.

        The action vector is robot-specific (e.g. joint velocities).
        No background thread — the caller controls the loop rate.
        """
        ...

    def get_observation(self) -> SensorSnapshot:
        """Return the latest observation without stepping."""
        ...

    # ── motion ───────────────────────────────────────────────────────────

    def execute_trajectory(self, trajectory: Trajectory) -> bool: ...

    def start_trajectory(self, trajectory: Trajectory) -> None: ...

    def stop_motion(self) -> None: ...

    def is_motion_done(self) -> bool: ...

    # ── end-effector ─────────────────────────────────────────────────────

    def control_gripper(self, position: float) -> None: ...

    # ── lifecycle ────────────────────────────────────────────────────────

    def reset(self) -> SensorSnapshot:
        """Reset the environment and return a fresh observation."""
        ...

    def close(self) -> None: ...
