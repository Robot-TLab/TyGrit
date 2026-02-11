"""Subgoal generation — high-level decision-making for the scheduler.

This module defines the ``SubGoalGenerator`` protocol, the
``create_subgoal_generator`` factory, and concrete implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from TyGrit.subgoal_generator.config import GraspGeneratorConfig, SubgoalGeneratorConfig
from TyGrit.subgoal_generator.protocol import SubGoalGenerator
from TyGrit.subgoal_generator.tasks.grasp import GraspSubgoalGenerator

if TYPE_CHECKING:
    from TyGrit.checker.collision import CollisionCheckFn
    from TyGrit.config import SystemConfig
    from TyGrit.envs.base import RobotBase

_TASK_DISPATCH: dict[str, type[SubgoalGeneratorConfig]] = {
    "grasp": GraspGeneratorConfig,
}


def create_subgoal_generator(
    config: SystemConfig,
    robot: RobotBase,
    collision_check: CollisionCheckFn,
) -> SubGoalGenerator:
    """Create a subgoal generator from the system config.

    Dispatches on ``config.subgoal.task``.
    """
    task = config.subgoal.task
    if task == "grasp":
        if not isinstance(config.subgoal, GraspGeneratorConfig):
            gen_config = GraspGeneratorConfig()
        else:
            gen_config = config.subgoal
        return GraspSubgoalGenerator(robot, config, collision_check, gen_config)
    msg = f"Unknown subgoal task: {task!r}"
    raise ValueError(msg)


__all__ = [
    "GraspGeneratorConfig",
    "GraspSubgoalGenerator",
    "SubGoalGenerator",
    "SubgoalGeneratorConfig",
    "create_subgoal_generator",
]
