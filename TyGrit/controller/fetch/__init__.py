"""Fetch-specific controllers."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from TyGrit.controller.fetch.gripper import GRIPPER_CLOSED, GRIPPER_OPEN
from TyGrit.controller.fetch.mpc import (
    MPCConfig,
    compute_mpc_action,
    robot_state_to_mpc_state,
)
from TyGrit.core.config import ControllerFn
from TyGrit.types.planning import Trajectory
from TyGrit.types.robot import RobotState


def make_mpc_controller(config: MPCConfig | None = None) -> ControllerFn:
    """Create a scheduler-compatible MPC controller function.

    Returns a callable with signature
    ``(state, trajectory, waypoint_idx) → action`` matching
    :data:`TyGrit.core.scheduler.ControllerFn`.

    Parameters
    ----------
    config : MPCConfig, optional
        MPC tuning parameters.  Uses defaults if ``None``.
    """
    cfg = config or MPCConfig()

    def controller_fn(
        state: RobotState,
        trajectory: Trajectory,
        waypoint_idx: int,
    ) -> npt.NDArray[np.float32]:
        x = robot_state_to_mpc_state(state)
        wp = trajectory.arm_path[waypoint_idx]
        bp = trajectory.base_configs[waypoint_idx]
        x_ref = np.array([bp.x, bp.y, bp.theta, *wp], dtype=np.float64)
        return compute_mpc_action(x, x_ref, cfg)

    return controller_fn


__all__ = [
    "compute_mpc_action",
    "make_mpc_controller",
    "GRIPPER_OPEN",
    "GRIPPER_CLOSED",
]
