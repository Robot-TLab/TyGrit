"""Fetch robot environment: factory + config.

Public API:
    * :class:`FetchEnvConfig` — robot/sim configuration dataclass.
    * :class:`FetchRobot` — the abstract Fetch interface; call
      ``FetchRobot.create(config)`` to get the right backend
      (ManiSkill / ROS / future Genesis or Isaac Sim).

Per-backend implementations live as siblings:

* ``maniskill.py`` — single-env ManiSkill via the new
  :class:`~TyGrit.sim.maniskill.ManiSkillSimHandler`.
* ``maniskill_vec.py`` — vectorised RL path (legacy; pending
  ``SimHandlerVec`` parallel design).
* ``ros.py`` — placeholder for real-robot work.

Fetch-specific *controller logic* (MPC, gaze, action assembly) lives
in :mod:`TyGrit.envs.fetch.core` and is sim-agnostic — it consumes a
:class:`~TyGrit.sim.base.SimHandler` and reads
:data:`~TyGrit.robots.fetch.FETCH_CFG` for joint metadata.
"""

from __future__ import annotations

from TyGrit.controller.fetch.mpc import MPCConfig
from TyGrit.envs.fetch.config import FetchEnvConfig


class FetchRobot:
    """Fetch mobile manipulator.

    Satisfies the :class:`~TyGrit.envs.base.RobotBase` protocol via
    duck typing. Use :meth:`FetchRobot.create` to instantiate the
    correct backend.
    """

    @staticmethod
    def create(
        config: FetchEnvConfig | None = None,
        mpc_config: MPCConfig | None = None,
    ) -> "FetchRobot":
        """Factory: create a FetchRobot backed by the configured backend.

        Parameters
        ----------
        config
            Environment configuration. ``config.backend`` selects the
            driver (``"maniskill"`` or ``"ros"``); ``config.num_envs``
            selects single-env vs vectorised. Defaults to
            ``FetchEnvConfig()`` (single-env ManiSkill) when ``None``.
        mpc_config
            MPC controller tuning, used by ManiSkill backends.
        """
        if config is None:
            config = FetchEnvConfig()

        if config.backend == "maniskill":
            if config.num_envs > 1:
                from TyGrit.envs.fetch.maniskill_vec import ManiSkillFetchRobotVec

                return ManiSkillFetchRobotVec(config, mpc_config)

            from TyGrit.envs.fetch.maniskill import ManiSkillFetchRobot

            return ManiSkillFetchRobot(config, mpc_config)

        if config.backend == "ros":
            from TyGrit.envs.fetch.ros import ROSFetchRobot

            return ROSFetchRobot(config)

        raise ValueError(
            f"FetchRobot.create: unknown backend {config.backend!r}; "
            f"expected 'maniskill' or 'ros'"
        )


__all__ = ["FetchEnvConfig", "FetchRobot"]
