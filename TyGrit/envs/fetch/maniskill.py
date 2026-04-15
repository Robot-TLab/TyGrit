"""ManiSkill-backed Fetch robot — thin composition over the new sim layer.

After the 2026-04-15 sim refactor this file is a 30-line glue layer:

* :class:`ManiSkillFetchRobot` constructs a
  :class:`~TyGrit.sim.maniskill.ManiSkillSimHandler` (robot-agnostic,
  reusable for any :class:`~TyGrit.types.robots.RobotCfg`) and composes
  it with :class:`~TyGrit.envs.fetch.core.FetchRobotCore` (Fetch-specific
  controller logic).

The old Fetch-specific :class:`ManiSkillFetchSimBackend` and the
``FetchSimBackend`` Protocol have been deleted; their functionality
folded into the unified :class:`SimHandler` family in :mod:`TyGrit.sim`.
"""

from __future__ import annotations

from TyGrit.controller.fetch.mpc import MPCConfig
from TyGrit.envs.fetch import FetchRobot
from TyGrit.envs.fetch.config import FetchEnvConfig
from TyGrit.envs.fetch.core import FetchRobotCore
from TyGrit.robots.fetch import FETCH_CFG
from TyGrit.sim.maniskill import ManiSkillSimHandler
from TyGrit.worlds.sampler import create_sampler


class ManiSkillFetchRobot(FetchRobotCore, FetchRobot):
    """Fetch robot driven by ManiSkill3.

    Composes :class:`~TyGrit.envs.fetch.core.FetchRobotCore`
    (sim-agnostic Fetch controller logic) with
    :class:`~TyGrit.sim.maniskill.ManiSkillSimHandler` (robot-agnostic
    ManiSkill adapter). Inherits :class:`FetchRobot` for the public
    factory dispatch.

    Parameters
    ----------
    config
        Fetch env config; defaults to ``FetchEnvConfig()``.
    mpc_config
        Optional MPC tuning override.
    """

    def __init__(
        self,
        config: FetchEnvConfig | None = None,
        mpc_config: MPCConfig | None = None,
    ) -> None:
        cfg = config or FetchEnvConfig()

        # Resolve the scene pool here and pass it to both layers — the
        # handler needs the SceneSpec list at gym.make time, the core
        # needs it for deterministic per-reset sampling. Both consume
        # the same SceneSamplerConfig so the pool definition lives in
        # one place (``cfg.scene_sampler``).
        sampler = create_sampler(cfg.scene_sampler)
        initial_idx = sampler.sample_idx(env_idx=0, reset_count=0)

        handler = ManiSkillSimHandler(
            FETCH_CFG,
            sampler.scenes,
            initial_scene_idx=initial_idx,
            camera_resolution=(cfg.camera_width, cfg.camera_height),
            obs_mode=cfg.obs_mode,
            control_mode=cfg.control_mode,
            render_mode=cfg.render_mode,
        )

        FetchRobotCore.__init__(self, cfg, handler, mpc_config)

        # Initial render kicks the viewer once after the
        # construction-time reset so the first frame is visible.
        if cfg.render_mode == "human":
            handler.render()


__all__ = ["ManiSkillFetchRobot"]
