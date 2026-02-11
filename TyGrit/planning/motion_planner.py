"""Protocol for motion planning."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt

from TyGrit.types.geometry import SE2Pose
from TyGrit.types.results import PlanResult
from TyGrit.types.robot import WholeBodyConfig


class MotionPlanner(Protocol):
    """Plans collision-free trajectories."""

    def update_environment(
        self,
        points: npt.NDArray[np.float32],
        base_pose: SE2Pose,
    ) -> None:
        """Sync the planner's collision world.

        Called once per scheduler iteration, after the scene is updated.

        Args:
            points: (N, 3) filtered world-frame point cloud.
            base_pose: Current base pose — needed so arm-only planning
                knows where the base sits.
        """
        ...

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

    def plan_interpolation(
        self,
        start: npt.NDArray[np.float64],
        goal: npt.NDArray[np.float64],
        base_pose: SE2Pose,
    ) -> PlanResult:
        """Linear joint-space interpolation (no collision checking).

        Used for short, contact-rich motions (grasp, lift) where
        collision-aware planners would reject the goal.

        Args:
            start: (8,) current arm joint config.
            goal: (8,) target arm joint config.
            base_pose: Current base pose (held constant throughout).
        """
        ...

    def validate_config(self, config: WholeBodyConfig) -> bool: ...
