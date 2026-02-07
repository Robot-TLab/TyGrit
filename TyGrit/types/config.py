"""Configuration data types for TyGrit."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SceneConfig:
    """Parameters for scene / belief-state maintenance."""

    ground_z_threshold: float = 0.3
    depth_range: tuple[float, float] = (0.2, 3.0)
    enable_ground_filter: bool = True
    merge_radius: float = 0.03
    downsample_voxel_size: float = 0.05
    crop_radius: float = 2.5


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


@dataclass(frozen=True)
class PlannerConfig:
    """Parameters for motion planning."""

    timeout: float = 5.0
    point_radius: float = 0.03


@dataclass(frozen=True)
class RobotConfig:
    """Robot-specific configuration."""

    name: str = "fetch"
    planning_dof: int = 8  # torso + 7 arm
    head_dof: int = 2  # pan + tilt


@dataclass(frozen=True)
class SystemConfig:
    """Top-level configuration aggregating all sub-configs."""

    robot: RobotConfig = field(default_factory=RobotConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    gaze: GazeConfig = field(default_factory=GazeConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
