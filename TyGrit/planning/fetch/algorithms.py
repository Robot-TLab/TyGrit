"""Pure-function planning algorithms wrapping VAMP.

All vamp objects are passed as arguments — no module-level state.
Adapted from ``grasp_anywhere/robot/utils/whole_body_planners.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from TyGrit.types.failures import PlannerFailure
from TyGrit.types.results import PlanResult
from TyGrit.utils.planning import lists_to_trajectory, vamp_path_to_lists

# ---------------------------------------------------------------------------
# Arm-only planning (RRT-Connect)
# ---------------------------------------------------------------------------


def plan_arm_rrtc(
    start: npt.NDArray[np.float64],
    goal: npt.NDArray[np.float64],
    env: Any,
    planner_func: Any,
    plan_settings: Any,
    simp_settings: Any,
    sampler: Any,
    module: Any,
    *,
    simplify: bool = True,
    interpolation_steps: int = 16,
) -> PlanResult:
    """Plan an arm-only path using the configured VAMP planner function."""
    result = planner_func(start, goal, env, plan_settings, sampler)

    if not result.solved:
        return PlanResult(
            success=False,
            failure=PlannerFailure.NO_PATH_FOUND,
            stats={
                "planning_time_ms": result.nanoseconds / 1e6,
                "planning_iterations": result.iterations,
            },
        )

    path = result.path
    if simplify:
        simple = module.simplify(path, env, simp_settings, sampler)
        path = simple.path

    path.interpolate(interpolation_steps)

    arm_path = vamp_path_to_lists(path)
    # Arm-only: base stays at origin
    base_configs = [[0.0, 0.0, 0.0]] * len(arm_path)

    stats: dict[str, float] = {
        "planning_time_ms": result.nanoseconds / 1e6,
        "planning_iterations": float(result.iterations),
    }

    return PlanResult(
        success=True,
        trajectory=lists_to_trajectory(arm_path, base_configs),
        stats=stats,
    )


# ---------------------------------------------------------------------------
# Whole-body planning: multilayer RRT-Connect
# ---------------------------------------------------------------------------


def plan_whole_body_multilayer_rrtc(
    start_joints: list[float],
    goal_joints: list[float],
    start_base: list[float],
    goal_base: list[float],
    env: Any,
    module: Any,
    plan_settings: Any,
    simp_settings: Any,
    sampler: Any,
    *,
    interpolation_density: float = 0.08,
) -> PlanResult:
    """Whole-body planning via multilayer RRT-Connect."""
    import vamp_preview

    hybrid_astar_cls = getattr(vamp_preview, "HybridAStarConfig", None)
    if hybrid_astar_cls is not None:
        config = hybrid_astar_cls()
        config.reverse_penalty = 50
        plan_settings.hybrid_astar_config = config

    result = module.multilayer_rrtc(
        start_joints,
        goal_joints,
        start_base,
        goal_base,
        env,
        plan_settings,
        sampler,
    )

    if not result.is_successful():
        stats: dict[str, float] = {
            "arm_planning_time_ms": result.arm_result.nanoseconds / 1e6,
            "base_planning_time_ms": result.base_result.nanoseconds / 1e6,
            "total_planning_time_ms": result.nanoseconds / 1e6,
        }
        return PlanResult(
            success=False,
            failure=PlannerFailure.NO_PATH_FOUND,
            stats=stats,
        )

    arm_path_list = vamp_path_to_lists(result.arm_result.path)
    base_path = result.base_path

    wb = module.whole_body_simplify(
        arm_path_list,
        base_path,
        env,
        simp_settings,
        sampler,
    )
    wb.interpolate(interpolation_density)

    arm_path = vamp_path_to_lists(wb.arm_result.path)
    base_configs = vamp_path_to_lists(wb.base_path)

    stats = {
        "arm_planning_time_ms": result.arm_result.nanoseconds / 1e6,
        "base_planning_time_ms": result.base_result.nanoseconds / 1e6,
        "total_planning_time_ms": result.nanoseconds / 1e6,
        "simplification_time_ms": wb.arm_result.nanoseconds / 1e6,
    }

    return PlanResult(
        success=True,
        trajectory=lists_to_trajectory(arm_path, base_configs),
        stats=stats,
    )


# ---------------------------------------------------------------------------
# Whole-body planning: FCIT*
# ---------------------------------------------------------------------------


def plan_whole_body_fcit(
    start_joints: list[float],
    goal_joints: list[float],
    start_base: list[float],
    goal_base: list[float],
    env: Any,
    module: Any,
    bounds_xy: tuple[float, float, float, float],
    sampler: Any,
    *,
    interpolation_density: float = 0.08,
    max_iterations: int = 200,
    max_samples: int = 8192,
    batch_size: int = 4096,
    reverse_weight: float = 10.0,
    optimize: bool = True,
) -> PlanResult:
    """Whole-body planning via FCIT*."""
    import vamp_preview

    fcit_neighbor_cls = getattr(vamp_preview, "FCITNeighborParams")
    fcit_settings_cls = getattr(vamp_preview, "FCITSettings")
    neighbor_params = fcit_neighbor_cls(
        module.dimension(),
        module.space_measure(),
    )
    settings = fcit_settings_cls(neighbor_params)
    settings.max_iterations = max_iterations
    settings.max_samples = max_samples
    settings.batch_size = batch_size
    settings.reverse_weight = reverse_weight
    settings.optimize = optimize

    x_min, x_max, y_min, y_max = bounds_xy
    res = module.fcit_wb(
        start_joints,
        goal_joints,
        start_base,
        goal_base,
        env,
        settings,
        sampler,
        x_min,
        x_max,
        y_min,
        y_max,
    )

    if not (res.validate_paths() and len(res.arm_result.path) >= 2):
        stats: dict[str, float] = {}
        if hasattr(res.arm_result, "nanoseconds"):
            stats["arm_planning_time_ms"] = res.arm_result.nanoseconds / 1e6
        return PlanResult(
            success=False,
            failure=PlannerFailure.NO_PATH_FOUND,
            stats=stats,
        )

    if interpolation_density is not None:
        try:
            res.interpolate(interpolation_density)
        except Exception:
            pass

    arm_path = vamp_path_to_lists(res.arm_result.path)
    base_configs = vamp_path_to_lists(res.base_path)

    stats = {}
    if hasattr(res, "nanoseconds"):
        stats["total_planning_time_ms"] = res.nanoseconds / 1e6
    if hasattr(res.arm_result, "nanoseconds"):
        stats["arm_planning_time_ms"] = res.arm_result.nanoseconds / 1e6

    return PlanResult(
        success=True,
        trajectory=lists_to_trajectory(arm_path, base_configs),
        stats=stats,
    )
