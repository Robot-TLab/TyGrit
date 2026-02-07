"""One-step MPC path tracker for the Fetch robot.

Ported from grasp_anywhere/envs/maniskill/maniskill_env_mpc.py (_solve_one_step_mpc).

Pure function: state + reference → velocity command.  No env, no thread.

Fetch-specific: 11-DOF state [x, y, θ, torso, 7 arm], 10-DOF control.

Linearised kinematics:  x_next ≈ x + B·u  where
    B = [[cos θ, 0, 0, …],
         [sin θ, 0, 0, …],
         [0,     1, 0, …],
         [0,     0, 1, …],
         [0,     0, 0, I₇]]

Solve: (Bᵀ Q B + R) u = Bᵀ Q (x_ref − x)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class MPCConfig:
    """Parameters for the one-step MPC path tracker."""

    state_weights: tuple[float, ...] = (20.0, 20.0, 15.0) + (12.0,) * 8
    control_weights: tuple[float, ...] = (0.5, 0.8) + (1.0,) * 8
    gain: float = 2.5
    v_max: float = 5.0
    w_max: float = 5.0
    joint_vel_max: tuple[float, ...] = (2.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0)


def compute_mpc_action(
    x: npt.NDArray[np.float64],
    x_ref: npt.NDArray[np.float64],
    config: MPCConfig | None = None,
) -> npt.NDArray[np.float32]:
    """Compute a single-step velocity command to track *x_ref* from *x*.

    Parameters
    ----------
    x : (11,) current whole-body state [x, y, θ, torso, 7 arm]
    x_ref : (11,) reference state
    config : MPC tuning parameters (uses defaults if ``None``)

    Returns
    -------
    u : (10,) velocity command [v, ω, torso_vel, 7 arm_vels]
    """
    if config is None:
        config = MPCConfig()

    Q = np.diag(np.array(config.state_weights, dtype=np.float32))
    R = np.diag(np.array(config.control_weights, dtype=np.float32))

    nx = 11
    nu = 10

    th = float(x[2])
    cth = np.cos(th)
    sth = np.sin(th)

    B = np.zeros((nx, nu), dtype=np.float32)
    B[0, 0] = cth  # dx/dv
    B[1, 0] = sth  # dy/dv
    B[2, 1] = 1.0  # dθ/dω
    B[3, 2] = 1.0  # dtorso/dtorso_vel
    for j in range(7):
        B[4 + j, 3 + j] = 1.0  # darm_j/darm_vel_j

    dx = (x_ref - x).astype(np.float32)

    BtQ = B.T @ Q
    H = BtQ @ B + R
    g = BtQ @ dx

    try:
        u = np.linalg.solve(H, g)
    except np.linalg.LinAlgError:
        u = np.linalg.lstsq(H, g, rcond=None)[0]

    u = (config.gain * u).astype(np.float32)

    # Clamp velocities
    u[0] = np.clip(u[0], -config.v_max, config.v_max)
    u[1] = np.clip(u[1], -config.w_max, config.w_max)
    joint_max = np.array(config.joint_vel_max, dtype=np.float32)
    u[2:] = np.clip(u[2:], -joint_max, joint_max)

    return u
