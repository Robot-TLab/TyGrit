"""Configuration for the gaze controller."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GazeConfig:
    """Parameters for the gaze optimiser."""

    lookahead_window: int = 80
    decay_rate: float = 0.99
    velocity_weight: float = 1.0
    joint_priorities: dict[str, float] = field(
        default_factory=lambda: {
            "base": 3.0,
            "torso": 2.0,
            "shoulder_lift": 1.3,
            "elbow": 1.1,
            "wrist_flex": 1.0,
            "gripper": 1.0,
        }
    )
