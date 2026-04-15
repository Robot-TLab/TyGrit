"""Fetch robot environment: factory + config.

Public API:
    * :class:`FetchEnvConfig` — robot/sim configuration dataclass.
    * :class:`FetchRobot` — the abstract Fetch interface; call
      ``FetchRobot.create(config)`` to get the right backend
      (ManiSkill / Genesis / Isaac Sim / ROS).

The factory composes the appropriate sim-agnostic Fetch core
(:class:`~TyGrit.envs.fetch.core.FetchRobotCore` for single-env,
:class:`~TyGrit.envs.fetch.core_vec.FetchRobotCoreVec` for vec) with
the matching :class:`~TyGrit.sim.base.SimHandler` /
:class:`~TyGrit.sim.base.SimHandlerVec` selected via
:func:`TyGrit.sim.create_sim_handler`. Adding a new simulator requires
**zero** edits to this module — the dispatcher lives in
:mod:`TyGrit.sim`.
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

        Returns a :class:`~TyGrit.envs.fetch.core.FetchRobotCore` for
        ``num_envs == 1`` and a
        :class:`~TyGrit.envs.fetch.core_vec.FetchRobotCoreVec` for
        ``num_envs > 1``.

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

        from TyGrit.robots.fetch import FETCH_CFG
        from TyGrit.sim import create_sim_handler
        from TyGrit.worlds.sampler import create_sampler

        sampler = create_sampler(cfg.scene_sampler)
        initial_idx = sampler.sample_idx(env_idx=0, reset_count=0)

        handler_opts = dict(cfg.sim_opts)
        if cfg.backend == "maniskill":
            # ManiSkillSimHandler / *Vec accept one resolution for every
            # camera in the robot's CameraSpec list; wire the env-level
            # (w, h) to it so callers don't need to know.
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

        if cfg.num_envs > 1:
            from TyGrit.envs.fetch.core_vec import FetchRobotCoreVec

            core = FetchRobotCoreVec(cfg, handler)
        else:
            from TyGrit.envs.fetch.core import FetchRobotCore

            core = FetchRobotCore(cfg, handler)

        # Kick the viewer once after construction so the first frame
        # is visible; no-op when render_mode is not "human".
        if handler_opts.get("render_mode") == "human":
            handler.render()

        return core


__all__ = ["FetchEnvConfig", "FetchRobot"]
