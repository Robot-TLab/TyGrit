"""Planning-related data types for TyGrit."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import numpy.typing as npt

from TyGrit.types.failures import PlannerFailure
from TyGrit.types.geometry import SE2Pose


@dataclass(frozen=True)
class Trajectory:
    """A planned trajectory: sequences of arm configs and base poses."""

    arm_path: tuple[npt.NDArray[np.float64], ...]  # tuple of 8-DOF configs
    base_configs: tuple[SE2Pose, ...]


@dataclass(frozen=True)
class PlanResult:
    """Result of a motion-planning query."""

    success: bool
    trajectory: Trajectory | None = None
    failure: PlannerFailure | None = None
    stats: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class GraspPose:
    """A candidate grasp with a quality score."""

    transform: npt.NDArray[np.float64]  # (4, 4)
    score: float


@dataclass(frozen=True)
class StageResult:
    """Outcome of a pipeline stage (grasp / prepose / observe / place).

    The ``failure`` field accepts any subsystem failure enum — the stage
    doesn't flatten the type, it just forwards whatever the subsystem
    reported.
    """

    success: bool
    failure: Enum | None = None
    message: str = ""
