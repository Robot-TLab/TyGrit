"""Fetch robot environment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from TyGrit.envs.config import EnvConfig
from TyGrit.types.worlds import SceneSamplerConfig


def _default_scene_sampler() -> SceneSamplerConfig:
    """Default to the hand-crafted 6-apt ReplicaCAD baseline manifest.

    Step 6 will generate a full 90-scene version at the same path; all
    existing callers will transparently pick up the expanded pool on
    their next config construction.
    """
    return SceneSamplerConfig(manifest_path="resources/worlds/replicacad.json")


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

    # ── scene/world selection ────────────────────────────────────────────
    # Replaces the old hardcoded env_id string. The sampler deterministically
    # picks a SceneSpec per (env_idx, reset_count) from the manifest — see
    # TyGrit.worlds.sampler for the v1 repeating-scene fix.
    scene_sampler: SceneSamplerConfig = field(default_factory=_default_scene_sampler)

    # ── ManiSkill-specific (ignored when backend != "maniskill") ──────────
    obs_mode: str = "rgb+depth+state+segmentation"
    control_mode: str = "pd_joint_vel"
    render_mode: str | None = "human"
    num_envs: int = 1
