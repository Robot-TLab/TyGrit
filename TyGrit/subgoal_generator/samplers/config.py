"""Configuration for subgoal samplers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GraspSamplerConfig:
    """Parameters for the grasp sampler."""

    target_object_id: int = 1
    grasp_depth_offset: float = 0.11  # retract along approach axis (metres)
    max_ik_attempts: int = 20  # per grasp pose
    max_grasps_to_try: int = 10
