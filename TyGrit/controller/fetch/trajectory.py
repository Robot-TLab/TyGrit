"""Synchronous trajectory tracking for a Fetch robot.

Drives a :class:`FetchRobotCore` through a :class:`Trajectory` waypoint
by waypoint using :func:`compute_mpc_action` — the MPC loop lives here
(not in ``TyGrit/envs/fetch/core.py``) because trajectory tracking is
strictly above the sensor/actuation adapter layer.

Callers that want the legacy "block until the trajectory finishes"
behaviour call :func:`execute_trajectory`. The online-control path
(`scheduler.py`) composes a different controller_fn — this module is
the batch executor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from TyGrit.controller.fetch.mpc import (
    MPCConfig,
    compute_mpc_action,
    robot_state_to_mpc_state,
)

if TYPE_CHECKING:
    from TyGrit.envs.fetch.core import FetchRobotCore
    from TyGrit.types.planning import Trajectory


def execute_trajectory(
    robot: "FetchRobotCore",
    trajectory: "Trajectory",
    mpc_config: MPCConfig | None = None,
) -> bool:
    """Drive ``robot`` through every waypoint of ``trajectory``.

    Per waypoint, runs a bounded loop of one-step MPC corrections
    until either the state error drops below
    :attr:`FetchEnvConfig.convergence_threshold` or
    :attr:`FetchEnvConfig.max_steps_per_waypoint` iterations elapse.

    Parameters
    ----------
    robot
        :class:`FetchRobotCore` (or duck-typed equivalent with
        ``get_robot_state`` + ``step`` + a ``_config`` holding
        ``convergence_threshold`` and ``max_steps_per_waypoint``).
    trajectory
        Arm + base waypoint lists.
    mpc_config
        MPC tuning; defaults to :class:`MPCConfig()` when ``None``.

    Returns
    -------
    bool
        Always ``True`` today — future versions may return ``False``
        on convergence failure. Callers should not rely on the exact
        post-loop state.
    """
    cfg = robot._config  # noqa: SLF001 — trajectory is Fetch-aware
    mpc = mpc_config or MPCConfig()

    for arm_wp, base_wp in zip(trajectory.arm_path, trajectory.base_configs):
        x_ref = np.array(
            [base_wp.x, base_wp.y, base_wp.theta, *arm_wp],
            dtype=np.float64,
        )
        for _ in range(cfg.max_steps_per_waypoint):
            state = robot.get_robot_state()
            x = robot_state_to_mpc_state(state)
            error = float(np.linalg.norm(x_ref - x))
            if error < cfg.convergence_threshold:
                break
            u = compute_mpc_action(x, x_ref, mpc)
            robot.step(u)
    return True


__all__ = ["execute_trajectory"]
