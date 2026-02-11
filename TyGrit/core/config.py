"""Configuration and callback types for the receding-horizon scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt

from TyGrit.scene.scene import Scene
from TyGrit.types.planning import Trajectory
from TyGrit.types.robot import RobotState


@dataclass(frozen=True)
class SchedulerConfig:
    """Parameters for the receding horizon scheduler."""

    steps_per_iteration: int = 10
    waypoint_lookahead: int = 2


# ── Callable type aliases for the scheduler's pluggable components ────────

ControllerFn = Callable[[RobotState, Trajectory, int], npt.NDArray[np.float32]]
CheckFn = Callable[[Trajectory, Scene], tuple[bool, bool]]
GazeFn = Callable[[Trajectory, int], npt.NDArray[np.float64] | None]
