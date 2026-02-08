"""Top-level configuration: SystemConfig and TOML loader."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from TyGrit.controller.fetch.mpc import MPCConfig
from TyGrit.core.scheduler import SchedulerConfig
from TyGrit.envs.fetch.config import FetchEnvConfig
from TyGrit.gaze.gaze import GazeConfig
from TyGrit.planning.config import PlannerConfig
from TyGrit.scene.config import SceneConfig


@dataclass(frozen=True)
class SystemConfig:
    """Top-level configuration aggregating all sub-configs."""

    env: FetchEnvConfig = field(default_factory=FetchEnvConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    gaze: GazeConfig = field(default_factory=GazeConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    mpc: MPCConfig = field(default_factory=MPCConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


def load_config(path: str | Path) -> SystemConfig:
    """Load a SystemConfig from a TOML file.

    Sections in the TOML file correspond to sub-config names (``env``,
    ``scene``, ``gaze``, ``planner``, ``mpc``, ``scheduler``).  Only the
    sections present are overridden; missing sections use defaults.

    Parameters
    ----------
    path : str or Path
        Path to the TOML file.

    Returns
    -------
    SystemConfig
    """
    path = Path(path)
    with path.open("rb") as f:
        raw = tomllib.load(f)

    section_map: dict[str, type] = {
        "env": FetchEnvConfig,
        "scene": SceneConfig,
        "gaze": GazeConfig,
        "planner": PlannerConfig,
        "mpc": MPCConfig,
        "scheduler": SchedulerConfig,
    }

    kwargs: dict[str, object] = {}
    for key, cls in section_map.items():
        section = raw.get(key, {})
        kwargs[key] = cls(**section)

    return SystemConfig(**kwargs)  # type: ignore[arg-type]
