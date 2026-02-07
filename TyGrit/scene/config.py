"""Configuration for the scene / belief-state module."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SceneConfig:
    """Parameters for scene / belief-state maintenance."""

    ground_z_threshold: float = 0.3
    depth_range: tuple[float, float] = (0.2, 3.0)
    enable_ground_filter: bool = True
    merge_radius: float = 0.03
    downsample_voxel_size: float = 0.05
    crop_radius: float = 2.5
