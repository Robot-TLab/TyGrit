"""Configuration for motion planning."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlannerConfig:
    """Parameters for motion planning."""

    timeout: float = 5.0
    point_radius: float = 0.03


@dataclass(frozen=True)
class VampPlannerConfig(PlannerConfig):
    """Configuration for the VAMP motion planner.

    Attributes:
        algorithm: Arm-only planner algorithm (e.g. ``"rrtc"``).
        whole_body_algorithm: Whole-body planner — ``"multilayer_rrtc"``
            or ``"fcit_wb"``.
        sampler: Sampler name passed to VAMP (e.g. ``"halton"``).
        simplify: Whether to simplify planned paths.
        interpolation_steps: Number of interpolation steps for arm-only plans.
        interpolation_density: Density parameter for whole-body interpolation.
        bounds_padding: Padding added to point-cloud XY bounds for FCIT*.
        fcit_max_iterations: Maximum iterations for FCIT* planner.
        fcit_max_samples: Maximum samples for FCIT* planner.
        fcit_batch_size: Batch size for FCIT* planner.
        fcit_reverse_weight: Reverse weight for FCIT* planner.
        fcit_optimize: Whether FCIT* should optimise the path.
    """

    algorithm: str = "rrtc"
    whole_body_algorithm: str = "multilayer_rrtc"
    sampler: str = "halton"
    simplify: bool = True
    interpolation_steps: int = 16
    interpolation_density: float = 0.08
    bounds_padding: float = 0.1
    fcit_max_iterations: int = 200
    fcit_max_samples: int = 8192
    fcit_batch_size: int = 4096
    fcit_reverse_weight: float = 10.0
    fcit_optimize: bool = True
