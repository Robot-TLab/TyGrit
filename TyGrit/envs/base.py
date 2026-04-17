"""Protocol for robot sensor/actuation adapters.

The ``RobotBase`` protocol is the *thin* contract a robot core (e.g.
``FetchRobotCore``) must satisfy: cameras + state read, low-level step,
gripper command, reset, and close. Trajectory tracking, active-
perception, MPC, and IK live above this layer in
:mod:`TyGrit.controller`, :mod:`TyGrit.gaze`, and the scheduler — they
consume a :class:`RobotBase` rather than extending it.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt

from TyGrit.types.robots import RobotState
from TyGrit.types.sensors import SensorSnapshot


class RobotBase(Protocol):
    """Protocol every robot environment (sim / real) must satisfy."""

    # ── sensing ──────────────────────────────────────────────────────────

    @property
    def camera_ids(self) -> list[str]:
        """Available camera identifiers (e.g. ``["head", "wrist"]``)."""
        ...

    def get_sensor_snapshot(self, camera_id: str) -> SensorSnapshot:
        """Capture RGB-D + robot state from the specified camera."""
        ...

    def get_robot_state(self) -> RobotState: ...

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

    # ── end-effector ─────────────────────────────────────────────────────

    def control_gripper(self, position: float) -> None: ...

    # ── lifecycle ────────────────────────────────────────────────────────

    def reset(self) -> SensorSnapshot:
        """Reset the environment and return a fresh observation."""
        ...

    def close(self) -> None: ...
