"""Gaze controller — compute where the head should look.

Ported from grasp_anywhere/observation/gaze_optimizer.py as a pure function.

Given link position trajectories and the current waypoint index, compute a
weighted-average 3-D target that emphasises fast-moving joints in the near future.

    weight = (decay_rate ^ distance) * velocity_in_world_frame

Zero velocity → zero weight (stationary parts ignored).

This is robot-agnostic: any robot with a steerable head can use this.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


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


def compute_gaze_target(
    link_positions: npt.NDArray[np.float64],
    current_idx: int,
    config: GazeConfig | None = None,
) -> npt.NDArray[np.float64]:
    """Compute the weighted gaze target from link position trajectories.

    Parameters
    ----------
    link_positions : (T, K, 3)
        World-frame positions of K tracked links at each of T waypoints.
    current_idx : int
        Current waypoint index in the trajectory.
    config : GazeConfig
        Gaze tuning parameters (uses defaults if ``None``).

    Returns
    -------
    target : (3,) world-frame gaze target position.
        Falls back to the mean position of all links at *current_idx*
        if all weights are zero.
    """
    if config is None:
        config = GazeConfig()

    T, K, _ = link_positions.shape
    window = config.lookahead_window
    end = min(current_idx + window, T)
    start = current_idx

    if start >= end:
        return link_positions[min(current_idx, T - 1)].mean(axis=0)

    future = link_positions[start:end]  # (W, K, 3)
    W = future.shape[0]

    # Distance decay: (decay_rate ^ [0, 1, 2, ...])
    distances = np.arange(W, dtype=np.float64)
    decay = config.decay_rate**distances  # (W,)

    # Velocity: finite difference of positions (magnitude)
    if W > 1:
        velocities = np.linalg.norm(np.diff(future, axis=0), axis=2)  # (W-1, K)
        # Pad first step with zero velocity
        velocities = np.concatenate(
            [np.zeros((1, K), dtype=np.float64), velocities], axis=0
        )
    else:
        velocities = np.zeros((W, K), dtype=np.float64)

    # Combined weights: (W, K) = decay[:, None] * velocity * velocity_weight
    weights = decay[:, None] * velocities * config.velocity_weight  # (W, K)

    total = weights.sum()
    if total < 1e-12:
        return link_positions[min(current_idx, T - 1)].mean(axis=0)

    # Weighted average over all waypoints and links
    # future: (W, K, 3), weights: (W, K)
    target = (future * weights[:, :, None]).sum(axis=(0, 1)) / total
    return target
