"""Tracking event dataclass.

A ``TrackingEvent`` is a single recorded observation from a subsystem.
It carries two tiers of context:

- ``info``: small metadata (always kept at any recording level).
- ``data``: large context — observations, point clouds, configs
  (only kept at DATA levels, stripped at INFO levels).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass(frozen=True)
class TrackingEvent:
    """A single recorded event from a subsystem.

    Attributes:
        subsystem: Which subsystem produced this event
                   (e.g. ``"ik"``, ``"planner"``, ``"perception"``).
        stage: Which pipeline stage was active
               (e.g. ``"grasp"``, ``"prepose"``, ``"observe"``).
               Empty string if not inside a stage.
        success: Whether the operation succeeded.
        failure: The specific failure enum value, or *None* if success.
        step: Episode step counter (from ``Tracker.tick()``).
        info: Small metadata — always recorded.
        data: Large context — only recorded at DATA levels.
    """

    subsystem: str
    stage: str
    success: bool
    failure: Enum | None
    step: int
    info: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
