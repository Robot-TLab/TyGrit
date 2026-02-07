"""Protocol for subgoal generators.

A subgoal generator is the *high-level* decision-maker: given the current
world model and robot state, it produces the next joint-space goal for the
low-level motion planner.

Implementations may be behaviour trees, finite-state machines, learned
policies, or simple hard-coded sequences.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt

from TyGrit.scene.representation import SceneRepresentation
from TyGrit.types.robot import RobotState


class SubGoalGenerator(Protocol):
    """Produces the next subgoal for the motion planner."""

    def generate(
        self,
        scene: SceneRepresentation,
        robot_state: RobotState,
    ) -> npt.NDArray[np.float64] | None:
        """Return the next goal configuration, or ``None`` if no goal available.

        Parameters
        ----------
        scene : SceneRepresentation
            Current world model (point cloud, etc.).
        robot_state : RobotState
            Current robot state.

        Returns
        -------
        goal : (N,) joint configuration to plan toward, or ``None``.
        """
        ...
