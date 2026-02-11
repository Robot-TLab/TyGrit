"""Planning-related data types for TyGrit."""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from TyGrit.types.geometry import SE2Pose


@dataclass(frozen=True)
class SchedulerFeedback:
    """Information the scheduler passes up to the subgoal generator.

    Encapsulates what the low-level execution loop knows so the high-level
    decision-maker can react without reaching into the planner directly.
    """

    is_path_valid: bool | None = None
    is_goal_valid: bool | None = None
    trajectory_exhausted: bool = False


@dataclass(frozen=True)
class Trajectory:
    """A planned trajectory: sequences of arm configs and base poses."""

    arm_path: tuple[npt.NDArray[np.float64], ...]  # tuple of 8-DOF configs
    base_configs: tuple[SE2Pose, ...]


class PlanningMode(enum.Enum):
    """Which planning strategy the scheduler should use for a subgoal."""

    ARM = "arm"  # collision-aware arm-only (e.g. RRT)
    WHOLE_BODY = "whole_body"  # collision-aware arm + base (e.g. multilayer RRT)
    INTERPOLATION = (
        "interpolation"  # linear joint-space interpolation (no collision check)
    )


@dataclass(frozen=True)
class Subgoal:
    """A subgoal returned by the subgoal generator.

    Carries the goal configuration **and** the planning mode so the
    scheduler knows which planner method to call.
    """

    mode: PlanningMode
    arm_joints: npt.NDArray[np.float64]  # (8,) goal arm config
    base_pose: SE2Pose | None = None  # required for WHOLE_BODY
