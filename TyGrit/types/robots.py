"""Pure data types for the robot layer.

``RobotSpec`` is the robot-side counterpart to
:class:`TyGrit.types.worlds.SceneSpec`: one frozen, sim-agnostic value
that backends consume to build their native robot handle.

Example
-------
>>> spec.sim_uids["maniskill"]
'fetch'
>>> spec.controller_order
('arm', 'gripper', 'body', 'base')
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RobotSpec:
    """Full specification of a robot platform.

    Parameters
    ----------
    name
        Stable TyGrit robot identifier, for example ``"fetch"``.
    sim_uids
        Mapping from backend name to that simulator's robot uid.
    planning_joint_names
        Joint order used by the planning stack.
    head_joint_names
        Joint order for a pan-tilt head. Empty when the robot has no
        steerable head.
    base_joint_names
        The ``(x, y, theta)`` joint triplet for a mobile base. Empty
        when the robot is fixed-base.
    is_mobile
        Whether the robot has a mobile base.
    controller_order
        Order in which per-controller actions are concatenated by the
        simulator adapter.
    camera_ids
        Public camera ids exposed by TyGrit.
    camera_sensor_ids
        Mapping from public camera id to the simulator-facing sensor id.
    joint_limits_lower, joint_limits_upper
        Lower and upper bounds for ``planning_joint_names``.
    default_spawn_pose
        Fallback ``(x, y, theta)`` spawn pose when the world provides no
        navigable-position data.
    """

    name: str
    sim_uids: dict[str, str]
    planning_joint_names: tuple[str, ...]
    head_joint_names: tuple[str, ...]
    base_joint_names: tuple[str, ...]
    is_mobile: bool
    controller_order: tuple[str, ...]
    camera_ids: tuple[str, ...]
    camera_sensor_ids: dict[str, str]
    joint_limits_lower: tuple[float, ...]
    joint_limits_upper: tuple[float, ...]
    default_spawn_pose: tuple[float, float, float] | None

    def __post_init__(self) -> None:
        num_planning_joints = len(self.planning_joint_names)
        if len(self.joint_limits_lower) != num_planning_joints:
            raise ValueError(
                "RobotSpec joint_limits_lower must match planning_joint_names length"
            )
        if len(self.joint_limits_upper) != num_planning_joints:
            raise ValueError(
                "RobotSpec joint_limits_upper must match planning_joint_names length"
            )
        if any(
            lo > hi
            for lo, hi in zip(
                self.joint_limits_lower,
                self.joint_limits_upper,
            )
        ):
            raise ValueError("RobotSpec joint_limits_lower entries must be <= upper")
        if self.is_mobile:
            if len(self.base_joint_names) != 3:
                raise ValueError(
                    "RobotSpec mobile robots must define exactly 3 base_joint_names"
                )
            if len(set(self.base_joint_names)) != len(self.base_joint_names):
                raise ValueError(
                    "RobotSpec base_joint_names must not contain duplicates"
                )
            return
        if self.base_joint_names != ():
            raise ValueError(
                "RobotSpec fixed-base robots must use empty base_joint_names"
            )
        if self.default_spawn_pose is not None:
            raise ValueError(
                "RobotSpec fixed-base robots must not define default_spawn_pose"
            )
