"""Protocol for motion planning."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt

from TyGrit.types.results import PlanResult
from TyGrit.types.robot import WholeBodyConfig


class MotionPlanner(Protocol):
    """Plans collision-free trajectories."""

    def plan_arm(
        self,
        start: npt.NDArray[np.float64],
        goal: npt.NDArray[np.float64],
    ) -> PlanResult: ...

    def plan_whole_body(
        self,
        start: WholeBodyConfig,
        goal: WholeBodyConfig,
    ) -> PlanResult: ...

    def validate_config(self, config: WholeBodyConfig) -> bool: ...
