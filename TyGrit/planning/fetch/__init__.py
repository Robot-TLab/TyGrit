"""Fetch-specific motion-planner factory."""

from __future__ import annotations

from TyGrit.planning.config import PlannerConfig, VampPlannerConfig
from TyGrit.planning.motion_planner import MotionPlanner


def create_fetch_planner(
    planner: str,
    config: PlannerConfig | None = None,
) -> MotionPlanner:
    """Create a Fetch motion planner by name.

    Args:
        planner: One of ``"vamp_preview"``.
        config: Optional :class:`VampPlannerConfig`.  A default is created
            when *None*.
    """
    if planner == "vamp_preview":
        from TyGrit.planning.fetch.vamp_preview import VampPreviewPlanner

        if config is None:
            config = VampPlannerConfig()
        if not isinstance(config, VampPlannerConfig):
            raise TypeError(
                f"Expected VampPlannerConfig for vamp_preview planner, "
                f"got {type(config).__name__}"
            )
        return VampPreviewPlanner(config)

    raise ValueError(f"Unknown Fetch planner: {planner!r}. Available: vamp_preview")
