"""Configuration for grasp prediction backends."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GraspPredictorConfig:
    """Base grasp-predictor configuration.

    Attributes
    ----------
    backend:
        Which backend to use: ``"graspgen"``.
    """

    backend: str = "graspgen"


@dataclass(frozen=True)
class GraspGenConfig(GraspPredictorConfig):
    """Parameters for the GraspGen diffusion-based grasp predictor."""

    backend: str = "graspgen"
    checkpoint_config_path: str = ""
    num_grasps: int = 200
    topk_num_grasps: int = 100
    min_grasps: int = 40
    max_tries: int = 6
    remove_outliers: bool = True
    score_threshold: float = 0.0
    max_grasps: int = 50
    device: str = "cuda"
