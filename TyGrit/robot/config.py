"""Configuration for robot parameters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RobotConfig:
    """Robot-specific configuration."""

    name: str = "fetch"
    planning_dof: int = 8  # torso + 7 arm
    head_dof: int = 2  # pan + tilt
