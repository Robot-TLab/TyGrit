"""Protocol for subgoal generators.

A subgoal generator is the *high-level* decision-maker: given the current
world model and robot state, it produces the next joint-space goal for the
low-level motion planner.
"""

from __future__ import annotations

from typing import Protocol

from TyGrit.scene.scene import Scene
from TyGrit.types.planning import SchedulerFeedback, Subgoal
from TyGrit.types.robot import RobotState


class SubGoalGenerator(Protocol):
    """Produces the next subgoal for the motion planner."""

    def generate(
        self,
        scene: Scene,
        robot_state: RobotState,
        feedback: SchedulerFeedback,
    ) -> Subgoal | None:
        """Return the next subgoal, or ``None`` if no goal available.

        Parameters
        ----------
        scene : Scene
            Current world model (point cloud, etc.).
        robot_state : RobotState
            Current robot state.
        feedback : SchedulerFeedback
            Low-level execution feedback from the scheduler (path/goal
            validity, trajectory status).

        Returns
        -------
        subgoal : Subgoal or ``None``.
        """
        ...

    def goal_predicate(self, state: RobotState) -> bool:
        """Return ``True`` when the task is complete."""
        ...
