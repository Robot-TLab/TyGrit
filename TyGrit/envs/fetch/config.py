"""Fetch robot environment configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from TyGrit.envs.config import EnvConfig
from TyGrit.types.worlds import SceneSamplerConfig


def _default_scene_sampler() -> SceneSamplerConfig:
    """Default to the hand-crafted 6-apt ReplicaCAD baseline manifest.

    Step 6 will generate a full 90-scene version at the same path; all
    existing callers will transparently pick up the expanded pool on
    their next config construction.
    """
    return SceneSamplerConfig(manifest_path="resources/worlds/replicacad.json")


def _default_sim_opts() -> Mapping[str, Any]:
    """Sim-specific defaults routed verbatim into the handler constructor.

    The default values here match the previous hard-wired ManiSkill
    behaviour (``obs_mode="rgb+depth+state+segmentation"``,
    ``control_mode="pd_joint_vel"``, ``render_mode="human"``). Using an
    opaque :class:`Mapping` keeps :class:`FetchEnvConfig` free of any
    per-simulator field — a new Genesis or Isaac Sim backend pass its
    own opts without forcing edits here.
    """
    return {
        "obs_mode": "rgb+depth+state+segmentation",
        "control_mode": "pd_joint_vel",
        "render_mode": "human",
    }


@dataclass(frozen=True)
class FetchEnvConfig(EnvConfig):
    """Configuration for Fetch robot environments.

    The ``backend`` field (inherited from ``EnvConfig``) selects the
    simulation or hardware driver.  Simulator-specific options travel
    in :attr:`sim_opts` so this dataclass stays simulator-agnostic.
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

    # ── simulator options (opaque, per-backend) ───────────────────────────
    # Passed verbatim as kwargs to the backend's ``SimHandler`` constructor.
    # Keeping this opaque means a new simulator (Genesis, Isaac Sim, …) can
    # ship its own opt keys without touching ``FetchEnvConfig``.
    sim_opts: Mapping[str, Any] = field(default_factory=_default_sim_opts)

    num_envs: int = 1
