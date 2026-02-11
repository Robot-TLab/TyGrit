"""Fetch robot — abstract parent for all Fetch environment backends.

Downstream code imports ``FetchRobot`` and uses ``FetchRobot.create(config)``
to get the right backend (ManiSkill, ROS, etc.) based on ``config.backend``.
"""

from __future__ import annotations

from TyGrit.controller.fetch.mpc import MPCConfig
from TyGrit.envs.fetch.config import FetchEnvConfig


class FetchRobot:
    """Fetch mobile manipulator.

    Satisfies the :class:`~TyGrit.envs.base.RobotBase` protocol.
    Use ``FetchRobot.create(config)`` to instantiate the correct backend.
    Fetch-specific controllers live in ``controller.fetch``.
    """

    @staticmethod
    def create(
        config: FetchEnvConfig | None = None,
        mpc_config: MPCConfig | None = None,
    ) -> FetchRobot:
        """Factory: create a FetchRobot backed by the configured backend.

        Parameters
        ----------
        config : FetchEnvConfig
            Environment configuration. ``config.backend`` selects the driver.
            Uses default ``FetchEnvConfig()`` (ManiSkill) if ``None``.
        mpc_config : MPCConfig, optional
            MPC controller tuning (used by ManiSkill backend).
        """
        if config is None:
            config = FetchEnvConfig()

        if config.backend == "maniskill":
            from TyGrit.envs.fetch.maniskill import ManiSkillFetchRobot

            return ManiSkillFetchRobot(config, mpc_config)

        if config.backend == "ros":
            from TyGrit.envs.fetch.ros import ROSFetchRobot

            return ROSFetchRobot(config)

        raise ValueError(
            f"Unknown backend {config.backend!r}; expected 'maniskill' or 'ros'"
        )
