"""Fetch robot environment: factory + config.

Public API:
    * :class:`FetchEnvConfig` — robot/sim configuration dataclass.
    * :class:`FetchRobot` — the abstract Fetch interface; call
      ``FetchRobot.create(config)`` to get the right backend
      (ManiSkill / Genesis / Isaac Sim / ROS).

The factory composes :class:`~TyGrit.envs.fetch.core.FetchRobotCore`
(robot-specific, sim-agnostic) with the appropriate
:class:`~TyGrit.sim.base.SimHandler` selected via
:func:`TyGrit.sim.create_sim_handler`. Adding a new simulator requires
**zero** edits to this module — the dispatcher lives in
:mod:`TyGrit.sim`.

``config.num_envs > 1`` temporarily still routes through the legacy
:class:`~TyGrit.envs.fetch.maniskill_vec.ManiSkillFetchRobotVec` glue
until the vec handlers land in :mod:`TyGrit.sim` (§7.5 / §7.6).
"""

from __future__ import annotations

from TyGrit.envs.fetch.config import FetchEnvConfig


class FetchRobot:
    """Fetch mobile manipulator.

    Use :meth:`FetchRobot.create` to instantiate the correct backend.
    """

    @staticmethod
    def create(config: FetchEnvConfig | None = None):
        """Factory: create a Fetch robot driven by the configured sim.

        Parameters
        ----------
        config
            Environment configuration. ``config.backend`` selects the
            driver (``"maniskill"``, ``"genesis"``, ``"isaac_sim"``,
            ``"ros"``); ``config.num_envs`` selects single-env vs
            vectorised. Defaults to ``FetchEnvConfig()`` (single-env
            ManiSkill) when ``None``.
        """
        cfg = config or FetchEnvConfig()

        if cfg.backend == "ros":
            from TyGrit.envs.fetch.ros import ROSFetchRobot

            return ROSFetchRobot(cfg)

        if cfg.num_envs > 1:
            # Temporary: the vec handler has not yet been lifted into
            # TyGrit.sim, so vec paths still go through the legacy
            # Fetch-specific wrapper. Lands with §7.5 / §7.6.
            if cfg.backend != "maniskill":
                raise NotImplementedError(
                    f"FetchRobot.create: backend={cfg.backend!r} with "
                    f"num_envs={cfg.num_envs} — only 'maniskill' has a vec "
                    f"path today. Genesis / Isaac Sim vec handlers land with "
                    f"§7.5 in prompts/multi_sim_mobile_manip_refactor.md."
                )
            from TyGrit.envs.fetch.maniskill_vec import ManiSkillFetchRobotVec

            return ManiSkillFetchRobotVec(cfg, None)

        from TyGrit.envs.fetch.core import FetchRobotCore
        from TyGrit.robots.fetch import FETCH_CFG
        from TyGrit.sim import create_sim_handler
        from TyGrit.worlds.sampler import create_sampler

        sampler = create_sampler(cfg.scene_sampler)
        initial_idx = sampler.sample_idx(env_idx=0, reset_count=0)

        handler_opts = dict(cfg.sim_opts)
        if cfg.backend == "maniskill":
            # ManiSkillSimHandler accepts a single resolution for all
            # cameras; wire the env-level (w, h) to it.
            handler_opts.setdefault(
                "camera_resolution", (cfg.camera_width, cfg.camera_height)
            )

        handler = create_sim_handler(
            cfg.backend,
            FETCH_CFG,
            sampler.scenes,
            num_envs=cfg.num_envs,
            initial_scene_idx=initial_idx,
            **handler_opts,
        )

        core = FetchRobotCore(cfg, handler)

        # Kick the viewer once after construction so the first frame
        # is visible; no-op when render_mode is not "human".
        if handler_opts.get("render_mode") == "human":
            handler.render()

        return core


__all__ = ["FetchEnvConfig", "FetchRobot"]
