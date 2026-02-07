"""Geometric data types for TyGrit."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SE2Pose:
    """2D pose in the plane (position + heading)."""

    x: float
    y: float
    theta: float
