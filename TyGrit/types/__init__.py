"""Public re-exports for TyGrit type definitions."""

from TyGrit.types.config import (
    GazeConfig,
    MPCConfig,
    PlannerConfig,
    RobotConfig,
    SceneConfig,
    SchedulerConfig,
    SystemConfig,
)
from TyGrit.types.failures import (
    ExecutionFailure,
    GraspFailure,
    IKFailure,
    PerceptionFailure,
    PlannerFailure,
)
from TyGrit.types.geometry import SE2Pose
from TyGrit.types.planning import (
    GraspPose,
    PlanResult,
    StageResult,
    Trajectory,
)
from TyGrit.types.robot import IKSolution, JointState, RobotState, WholeBodyConfig
from TyGrit.types.sensor import SensorSnapshot

__all__ = [
    # geometry
    "SE2Pose",
    # robot
    "IKSolution",
    "JointState",
    "RobotState",
    "WholeBodyConfig",
    # sensor
    "SensorSnapshot",
    # failures (per-subsystem)
    "PlannerFailure",
    "IKFailure",
    "GraspFailure",
    "PerceptionFailure",
    "ExecutionFailure",
    # planning
    "Trajectory",
    "PlanResult",
    "GraspPose",
    "StageResult",
    # config
    "SceneConfig",
    "GazeConfig",
    "MPCConfig",
    "PlannerConfig",
    "RobotConfig",
    "SchedulerConfig",
    "SystemConfig",
]
