"""Top-level configuration: SystemConfig and TOML loader."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from TyGrit.controller.fetch.mpc import MPCConfig
from TyGrit.core.config import SchedulerConfig
from TyGrit.envs.fetch.config import FetchEnvConfig
from TyGrit.gaze.config import GazeConfig
from TyGrit.perception.grasping.config import GraspGenConfig, GraspPredictorConfig
from TyGrit.perception.segmentation.config import (
    SAM3SegmenterConfig,
    SegmenterConfig,
)
from TyGrit.planning.config import PlannerConfig, VampPlannerConfig
from TyGrit.scene.config import PointCloudSceneConfig, SceneConfig
from TyGrit.subgoal_generator.config import GraspGeneratorConfig, SubgoalGeneratorConfig


@dataclass(frozen=True)
class SystemConfig:
    """Top-level configuration aggregating all sub-configs."""

    env: FetchEnvConfig = field(default_factory=FetchEnvConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    gaze: GazeConfig = field(default_factory=GazeConfig)
    grasping: GraspPredictorConfig = field(default_factory=GraspPredictorConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    segmentation: SegmenterConfig = field(default_factory=SegmenterConfig)
    mpc: MPCConfig = field(default_factory=MPCConfig)
    subgoal: SubgoalGeneratorConfig = field(default_factory=SubgoalGeneratorConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


# ── Config dispatch ──────────────────────────────────────────────────────
# Sections with a discriminator field (e.g. ``backend``) dispatch to the
# concrete subclass that matches.

_PLANNER_DISPATCH: dict[str, type] = {
    "vamp_preview": VampPlannerConfig,
}

_SCENE_DISPATCH: dict[str, type] = {
    "latest": PointCloudSceneConfig,
    "accumulated": PointCloudSceneConfig,
    "combine": PointCloudSceneConfig,
    "ray_casting": PointCloudSceneConfig,
    "static": PointCloudSceneConfig,
}

_GRASPING_DISPATCH: dict[str, type] = {
    "graspgen": GraspGenConfig,
}

_SEGMENTATION_DISPATCH: dict[str, type] = {
    "sam3": SAM3SegmenterConfig,
}

_SUBGOAL_DISPATCH: dict[str, type] = {
    "grasp": GraspGeneratorConfig,
}


def _resolve_class(
    section: dict, default_cls: type, dispatch: dict[str, type], key: str
) -> type:
    """Pick the concrete subclass from *dispatch* based on *section[key]*."""
    discriminator = section.get(key)
    if discriminator is not None and discriminator in dispatch:
        return dispatch[discriminator]
    return default_cls


def load_config(path: str | Path) -> SystemConfig:
    """Load a SystemConfig from a TOML file.

    Sections in the TOML file correspond to sub-config names (``env``,
    ``scene``, ``gaze``, ``grasping``, ``segmentation``, ``planner``,
    ``subgoal``, ``mpc``, ``scheduler``).  Only the sections present are overridden; missing sections
    use defaults.

    Sections with a discriminator field (``algorithm`` for planner, ``mode``
    for scene, ``backend`` for grasping/segmentation) automatically select the
    concrete config subclass.

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

    # Simple sections — always the same class
    simple_map: dict[str, type] = {
        "env": FetchEnvConfig,
        "gaze": GazeConfig,
        "mpc": MPCConfig,
        "scheduler": SchedulerConfig,
    }

    kwargs: dict[str, object] = {}
    for key, cls in simple_map.items():
        section = raw.get(key, {})
        kwargs[key] = cls(**section)

    # Dispatched sections
    planner_section = raw.get("planner", {})
    planner_cls = _resolve_class(
        planner_section, PlannerConfig, _PLANNER_DISPATCH, "algorithm"
    )
    kwargs["planner"] = planner_cls(**planner_section)

    scene_section = raw.get("scene", {})
    scene_cls = _resolve_class(scene_section, SceneConfig, _SCENE_DISPATCH, "mode")
    kwargs["scene"] = scene_cls(**scene_section)

    grasping_section = raw.get("grasping", {})
    grasping_cls = _resolve_class(
        grasping_section, GraspPredictorConfig, _GRASPING_DISPATCH, "backend"
    )
    kwargs["grasping"] = grasping_cls(**grasping_section)

    seg_section = raw.get("segmentation", {})
    seg_cls = _resolve_class(
        seg_section, SegmenterConfig, _SEGMENTATION_DISPATCH, "backend"
    )
    kwargs["segmentation"] = seg_cls(**seg_section)

    subgoal_section = raw.get("subgoal", {})
    subgoal_cls = _resolve_class(
        subgoal_section, SubgoalGeneratorConfig, _SUBGOAL_DISPATCH, "task"
    )
    kwargs["subgoal"] = subgoal_cls(**subgoal_section)

    return SystemConfig(**kwargs)  # type: ignore[arg-type]
