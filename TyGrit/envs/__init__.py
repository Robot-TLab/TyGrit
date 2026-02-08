"""Robot environment base and factory."""

from TyGrit.envs.base import RobotBase
from TyGrit.envs.config import EnvConfig


def create_env(config: EnvConfig | None = None) -> RobotBase:
    """Create a robot environment from *config*.

    Dispatches on ``config.robot`` to the correct robot factory,
    which then dispatches on ``config.backend`` to the correct driver.
    """
    if config is None:
        from TyGrit.envs.fetch.config import FetchEnvConfig

        config = FetchEnvConfig()

    if config.robot == "fetch":
        from TyGrit.envs.fetch.fetch import FetchRobot

        return FetchRobot.create(config)

    raise ValueError(f"Unknown robot {config.robot!r}; expected 'fetch'")


__all__ = ["RobotBase", "EnvConfig", "create_env"]
