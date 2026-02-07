"""Robot state data types for TyGrit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from TyGrit.types.geometry import SE2Pose


@dataclass(frozen=True)
class JointState:
    """Named joint positions."""

    names: tuple[str, ...]
    positions: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.names) != len(self.positions):
            raise ValueError(
                f"names ({len(self.names)}) and positions ({len(self.positions)}) "
                "must have the same length"
            )


@dataclass(frozen=True)
class RobotState:
    """Full robot state: base + arm + head."""

    base_pose: SE2Pose
    planning_joints: tuple[float, ...]  # 8-DOF: torso + 7 arm
    head_joints: tuple[float, ...]  # 2-DOF: pan + tilt


@dataclass(frozen=True)
class WholeBodyConfig:
    """Whole-body configuration for planning: arm + base."""

    arm_joints: npt.NDArray[np.float64]  # (8,) torso + 7 arm
    base_pose: SE2Pose


@dataclass(frozen=True)
class IKSolution:
    """A single IK solution, tagged with the joint names it corresponds to."""

    joint_names: tuple[str, ...]
    positions: npt.NDArray[np.float64]  # (dof,)

    def __post_init__(self) -> None:
        if len(self.joint_names) != self.positions.shape[0]:
            raise ValueError(
                f"joint_names ({len(self.joint_names)}) and positions "
                f"({self.positions.shape[0]}) must have the same length"
            )
