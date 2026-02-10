"""Grasp prediction backends."""

from TyGrit.perception.grasping.config import GraspGenConfig, GraspPredictorConfig
from TyGrit.perception.grasping.graspgen import GraspGenPredictor

__all__ = [
    "GraspGenConfig",
    "GraspGenPredictor",
    "GraspPredictorConfig",
    "create_grasp_predictor",
]


def create_grasp_predictor(
    config: GraspPredictorConfig,
) -> GraspGenPredictor:
    """Create a grasp predictor from *config*.

    Dispatches on ``config.backend``:

    - ``"graspgen"`` → :class:`GraspGenPredictor` (requires :class:`GraspGenConfig`)
    """
    if config.backend == "graspgen":
        if not isinstance(config, GraspGenConfig):
            raise TypeError(
                "graspgen backend requires GraspGenConfig, "
                f"got {type(config).__name__}"
            )
        return GraspGenPredictor(config)

    raise ValueError(f"Unknown grasp predictor backend: {config.backend!r}")
