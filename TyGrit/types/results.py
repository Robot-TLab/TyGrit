"""Result and outcome types for TyGrit subsystems.

Every subsystem that can succeed or fail returns a typed result
dataclass.  Grouping them here makes it easy to find and extend
the vocabulary of outcomes without hunting through logic modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from TyGrit.types.failures import PlannerFailure
from TyGrit.types.planning import Trajectory


@dataclass(frozen=True)
class PlanResult:
    """Result of a motion-planning query."""

    success: bool
    trajectory: Trajectory | None = None
    failure: PlannerFailure | None = None
    stats: dict[str, float] = field(default_factory=dict)


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


class SchedulerOutcome(Enum):
    """Possible outcomes of a scheduler run."""

    SUCCESS = "success"
    MAX_ITERATIONS = "max_iterations"
    PLAN_FAILURE = "plan_failure"


@dataclass
class SchedulerResult:
    """Result returned by :meth:`Scheduler.run`."""

    outcome: SchedulerOutcome
    iterations: int = 0
    total_steps: int = 0
    stats: dict[str, float] = field(default_factory=dict)
