"""Configuration for motion planning."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlannerConfig:
    """Parameters for motion planning."""

    timeout: float = 5.0
    point_radius: float = 0.03
