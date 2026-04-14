"""The Fetch mobile manipulator as pure data. Consumed by per-sim backends in TyGrit/envs/."""

from __future__ import annotations

from TyGrit.kinematics.fetch.constants import (
    HEAD_JOINT_NAMES,
    JOINT_LIMITS_LOWER,
    JOINT_LIMITS_UPPER,
    PLANNING_JOINT_NAMES,
)
from TyGrit.types.robots import RobotSpec

FETCH_SPEC = RobotSpec(
    name="fetch",
    sim_uids={"maniskill": "fetch"},
    planning_joint_names=PLANNING_JOINT_NAMES,
    head_joint_names=HEAD_JOINT_NAMES,
    # Source of truth: ManiSkill's installed Fetch agent defines
    # Fetch.base_joint_names in mani_skill.agents.robots.fetch.fetch.Fetch.__init__.
    base_joint_names=(
        "root_x_axis_joint",
        "root_y_axis_joint",
        "root_z_rotation_joint",
    ),
    is_mobile=True,
    # Source of truth: TyGrit.envs.fetch.maniskill_setup._CONTROLLER_ORDER.
    controller_order=("arm", "gripper", "body", "base"),
    camera_ids=("head",),
    # Source of truth: TyGrit.envs.fetch.maniskill reads sensor_data["fetch_head"].
    camera_sensor_ids={"head": "fetch_head"},
    joint_limits_lower=tuple(float(limit) for limit in JOINT_LIMITS_LOWER),
    joint_limits_upper=tuple(float(limit) for limit in JOINT_LIMITS_UPPER),
    # Source of truth: TyGrit.envs.fetch.core._randomize_robot_pose fallback.
    default_spawn_pose=(-1.0, 0.0, 0.0),
)

__all__ = ["FETCH_SPEC"]
