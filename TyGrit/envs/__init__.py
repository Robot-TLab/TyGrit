"""Robot environment base and factory."""

from __future__ import annotations

from TyGrit.controller.fetch.mpc import MPCConfig
from TyGrit.envs.base import RobotBase
from TyGrit.envs.config import EnvConfig


def create_env(
    config: EnvConfig | None = None,
    mpc_config: MPCConfig | None = None,
) -> RobotBase:
    """Create a robot environment from *config*.

    Dispatches on ``config.robot`` to the correct robot factory,
    which then dispatches on ``config.backend`` to the correct driver.

    Parameters
    ----------
    config : EnvConfig, optional
        Environment configuration.  Defaults to ``FetchEnvConfig()``.
    mpc_config : MPCConfig, optional
        MPC controller tuning (passed to backends that use it).
    """
    if config is None:
        from TyGrit.envs.fetch.config import FetchEnvConfig

        config = FetchEnvConfig()

    if config.robot == "fetch":
        from TyGrit.envs.fetch.fetch import FetchRobot

        return FetchRobot.create(config, mpc_config=mpc_config)

    raise ValueError(f"Unknown robot {config.robot!r}; expected 'fetch'")


__all__ = ["RobotBase", "EnvConfig", "create_env"]
