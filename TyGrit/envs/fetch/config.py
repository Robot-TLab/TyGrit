"""Fetch robot environment configuration."""

from __future__ import annotations

from dataclasses import dataclass

from TyGrit.envs.config import EnvConfig


@dataclass(frozen=True)
class FetchEnvConfig(EnvConfig):
    """Configuration for Fetch robot environments.

    The ``backend`` field (inherited from ``EnvConfig``) selects the
    simulation or hardware driver.  Backend-specific fields are ignored
    by other backends.
    """

    robot: str = "fetch"

    # ── trajectory execution ──────────────────────────────────────────────
    convergence_threshold: float = 0.05
    max_steps_per_waypoint: int = 200

    # ── head PD controller ────────────────────────────────────────────────
    gaze_kp: float = 3.0
    gaze_max_vel: float = 2.0

    # ── ManiSkill-specific (ignored when backend != "maniskill") ──────────
    env_id: str = "ReplicaCAD_SceneManipulation-v1"
    obs_mode: str = "rgb+depth+state+segmentation"
    control_mode: str = "pd_joint_vel"
    render_mode: str | None = "human"
