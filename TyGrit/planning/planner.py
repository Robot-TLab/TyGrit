"""Robot-agnostic motion-planner factory.

Usage::

    from TyGrit.planning.planner import create_planner

    planner = create_planner("fetch", "vamp_preview")
"""

from __future__ import annotations

from TyGrit.planning.config import PlannerConfig
from TyGrit.planning.motion_planner import MotionPlanner


def create_planner(
    robot: str,
    planner: str,
    config: PlannerConfig | None = None,
) -> MotionPlanner:
    """Create a motion planner for *robot*.

    Args:
        robot: Robot name (``"fetch"``).
        planner: Planner name.  Available planners per robot:

            **Fetch:**

            - ``"vamp_preview"`` — VAMP-based planner (arm + whole-body).
              Requires the ``vamp`` package.

        config: Optional planner-specific configuration.  When *None*,
            a default config is created.
    """
    if robot == "fetch":
        from TyGrit.planning.fetch import create_fetch_planner

        return create_fetch_planner(planner, config)
    raise ValueError(f"Unknown robot: {robot!r}")
