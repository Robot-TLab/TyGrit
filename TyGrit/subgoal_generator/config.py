"""Configuration for subgoal generators."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SubgoalGeneratorConfig:
    """Base subgoal-generator configuration.

    Attributes
    ----------
    task:
        Which task to use: ``"grasp"``.
    """

    task: str = "grasp"


@dataclass(frozen=True)
class GraspGeneratorConfig(SubgoalGeneratorConfig):
    """Parameters for the grasp subgoal generator."""

    task: str = "grasp"
    approach_threshold: float = 0.15
    lift_height: float = 0.15
    lift_threshold: float = 0.15
