"""Pure data types for the robot layer.

:class:`RobotCfg` is the robot-side counterpart to
:class:`TyGrit.types.worlds.SceneSpec`: one frozen, sim-agnostic value
that per-sim handlers consume to build the native robot handle. Adding
a new robot (e.g. AutoLife) is a matter of writing a single new
``RobotCfg`` — no new backend code. Adding a new simulator is a matter
of writing a single new ``SimHandler`` in :mod:`TyGrit.sim` — no
per-robot code.

Design notes
------------
* **Frozen dataclasses**. Instances are hashable and safe to share
  across parallel envs. ``Mapping`` fields are wrapped in
  :class:`types.MappingProxyType` post-construction so consumers can't
  silently mutate them.
* **No simulator imports**. Every field is a plain Python value.
  Per-sim tuning that would need to mention a sim SDK lives in a
  nested ``sim_params: Mapping[sim_name, Mapping[str, float]]`` dict
  so the *type* stays sim-free.
* **Per-sim asset paths**. ManiSkill has a built-in Fetch registry and
  consumes :attr:`sim_uids`. Genesis / MuJoCo want a URDF / MJCF file.
  Isaac Sim prefers a USD stage. A single ``RobotCfg`` carries all
  four; each handler picks the one native to its sim.
* **Quaternion convention** for all cameras / mounts: ``(x, y, z, w)``
  (SciPy / ROS). Sim handlers convert at the boundary.

Example
-------
>>> from TyGrit.robots.fetch import FETCH_CFG
>>> FETCH_CFG.sim_uids["maniskill"]
'fetch'
>>> FETCH_CFG.actuators["arm"].control_mode
'velocity'
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt

from TyGrit.types.geometry import SE2Pose

if TYPE_CHECKING:
    # Avoid a circular import at runtime: sensors.py imports RobotState
    # from this module. CameraSpec is referenced only in annotations on
    # RobotCfg which use string-form via `from __future__ import
    # annotations`, so the import never needs to resolve eagerly.
    from TyGrit.types.sensors import CameraSpec

#: Supported joint control modes. Sim handlers translate to their
#: native controller type (velocity → ``pd_joint_vel`` in ManiSkill,
#: ``gs.control_dofs_velocity`` in Genesis, ``JointEffortActionCfg``
#: with a PID in Isaac Lab, etc.).
ControlMode = Literal["velocity", "position", "effort"]


#: Command-semantics hint for an :class:`ActuatorCfg`.
#:
#: - ``"joint"`` (default): the controller's action vector is a
#:   direct joint command (velocity / position / effort). When
#:   ``action_dim == len(joint_names)``, there is a 1-to-1 mapping.
#:   When ``action_dim < len(joint_names)``,
#:   :attr:`ActuatorCfg.command_to_joint_mapping` must provide the
#:   explicit broadcast (e.g. Fetch's 1-scalar gripper driving two
#:   finger joints).
#: - ``"base_twist"``: action is a Cartesian twist ``(v, w)``
#:   (2 scalars) that the sim's own holonomic base controller
#:   translates into base-joint motion. Sims without a native twist
#:   controller raise :class:`NotImplementedError` — there is no
#:   generic joint-level fallback because the twist→joint math
#:   depends on the base's kinematics and the sim's integration
#:   scheme.
ActuatorKind = Literal["joint", "base_twist"]


@dataclass(frozen=True)
class ActuatorCfg:
    """One controller group on the robot.

    A controller is a slice of the low-level action space mapping to a
    set of joints under a single control mode. Sim handlers read
    :attr:`joint_names` + :attr:`control_mode` to configure the native
    controller (e.g. ManiSkill's ``PDJointVelControllerConfig``,
    Genesis's ``control_dofs_velocity``, Isaac Lab's ``ActionTermCfg``).

    Parameters
    ----------
    name
        Controller identifier, e.g. ``"arm"``, ``"gripper"``, ``"body"``,
        ``"base"``. Must be unique within a :class:`RobotCfg`.
    joint_names
        Joints this controller drives. Ordering matches the action slice
        when ``command_to_joint_mapping`` is ``None``.
    control_mode
        How the action vector is interpreted for these joints.
    action_dim
        Length of this controller's slice of the flat action vector.
        Equal to ``len(joint_names)`` for 1-to-1 joint control;
        smaller with ``command_to_joint_mapping`` (Fetch's 1-scalar
        gripper) or when ``kind="base_twist"`` (2-scalar holonomic
        base).
    kind
        Command semantics — see :data:`ActuatorKind`. Defaults to
        ``"joint"`` which is correct for every arm / gripper / body
        controller. Mobile bases where a sim's built-in twist
        controller is expected use ``"base_twist"``.
    command_to_joint_mapping
        Required when ``kind="joint"`` and
        ``action_dim != len(joint_names)``. A tuple of length
        ``len(joint_names)``; entry ``i`` is the action-vector index
        whose value drives ``joint_names[i]``. Example: Fetch's
        gripper with ``action_dim=1`` and
        ``joint_names=("r_finger", "l_finger")`` sets
        ``command_to_joint_mapping=(0, 0)`` so one scalar drives
        both fingers symmetrically.
    sim_params
        Optional per-sim tuning. Keys are sim identifiers (``"maniskill"``,
        ``"genesis"``, ``"isaac_sim"``); values are parameter dicts
        consumed by that sim's handler (e.g. ``{"stiffness": 1e5,
        "damping": 50}``). Empty dict is the common case — sim handlers
        fall back to sensible defaults.
    """

    name: str
    joint_names: tuple[str, ...]
    control_mode: ControlMode
    action_dim: int
    kind: ActuatorKind = "joint"
    command_to_joint_mapping: tuple[int, ...] | None = None
    sim_params: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ActuatorCfg: name must be non-empty")
        if self.action_dim <= 0:
            raise ValueError(
                f"ActuatorCfg {self.name!r}: action_dim must be positive, "
                f"got {self.action_dim}"
            )
        if len(self.joint_names) == 0:
            raise ValueError(
                f"ActuatorCfg {self.name!r}: joint_names must be non-empty"
            )
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError(f"ActuatorCfg {self.name!r}: joint_names must be unique")

        # Validate command_to_joint_mapping against kind + sizes.
        if self.kind == "joint":
            if self.action_dim != len(self.joint_names):
                if self.command_to_joint_mapping is None:
                    raise ValueError(
                        f"ActuatorCfg {self.name!r}: action_dim "
                        f"({self.action_dim}) != len(joint_names) "
                        f"({len(self.joint_names)}) requires an explicit "
                        f"command_to_joint_mapping tuple of length "
                        f"{len(self.joint_names)} (one action-vector index "
                        f"per joint)"
                    )
                if len(self.command_to_joint_mapping) != len(self.joint_names):
                    raise ValueError(
                        f"ActuatorCfg {self.name!r}: command_to_joint_mapping "
                        f"length {len(self.command_to_joint_mapping)} must "
                        f"equal len(joint_names) {len(self.joint_names)}"
                    )
                if any(
                    not 0 <= i < self.action_dim for i in self.command_to_joint_mapping
                ):
                    raise ValueError(
                        f"ActuatorCfg {self.name!r}: command_to_joint_mapping "
                        f"entries must be in [0, action_dim={self.action_dim})"
                    )
            elif self.command_to_joint_mapping is not None:
                raise ValueError(
                    f"ActuatorCfg {self.name!r}: command_to_joint_mapping is "
                    f"only legal when action_dim != len(joint_names)"
                )
        elif self.kind == "base_twist":
            if self.command_to_joint_mapping is not None:
                raise ValueError(
                    f"ActuatorCfg {self.name!r}: command_to_joint_mapping is "
                    f"meaningless for kind='base_twist' (commands are cartesian "
                    f"v/w, not joint-indexed)"
                )

        # Freeze nested mapping fields so downstream mutations can't
        # desync an instance from what it was constructed with.
        frozen_sim_params = {
            sim: MappingProxyType(dict(params))
            for sim, params in self.sim_params.items()
        }
        object.__setattr__(self, "sim_params", MappingProxyType(frozen_sim_params))


@dataclass(frozen=True)
class RobotCfg:
    """Full robot descriptor — sim-agnostic, no sim imports.

    Consumed by per-sim :class:`~TyGrit.sim.base.SimHandler` implementations
    to instantiate the robot. Per-sim handlers pick the native asset
    entry-point (``sim_uids``, ``urdf_path``, ``usd_path``, or
    ``mjcf_path``) and honour the actuator / camera / joint-limit
    configuration uniformly.

    Parameters
    ----------
    name
        Stable TyGrit robot identifier. Shorthand used in configs, logs,
        and manifest files (``"fetch"``, ``"autolife"``, …).
    sim_uids
        Mapping ``sim_name → simulator-registered robot uid``. Populated
        for sims that ship a built-in robot library (e.g. ManiSkill's
        ``"fetch"`` agent). Sims without a registry use the asset-path
        fields below.
    urdf_path
        URDF asset path used by Genesis and MuJoCo fallback loaders.
    usd_path
        USD stage path preferred by Isaac Sim / Isaac Lab.
    mjcf_path
        MuJoCo MJCF path. Genesis can load this too.
    base_link_name
        Name of the robot's base / root link. Handlers expose
        :meth:`get_link_pose` keyed by link name; callers typically
        read this link to compute the world-frame base pose.
    is_mobile
        Whether the robot has a mobile (holonomic or non-holonomic)
        base. Affects the ``base_joint_names`` + ``default_spawn_pose``
        validation and whether :meth:`SimHandler.set_base_pose` is
        legal.
    base_joint_names
        For mobile robots, the three joints that encode base pose as
        ``(x, y, θ)`` in qpos. Empty tuple for fixed-base robots.
    actuators
        Mapping ``controller_name → ActuatorCfg``. Keys must match
        :attr:`controller_order`.
    controller_order
        Order in which per-controller actions are concatenated into
        the flat low-level action vector. The sim handler computes
        :attr:`action_slices` from this.
    planning_joint_names
        Joints exposed to the planning layer. Subset of the robot's
        joints, in the order used by kinematics / IK / trajectory
        planning.
    head_joint_names
        Joints exposed to the active-perception layer (pan-tilt head).
        Empty tuple for robots without a steerable head.
    joint_limits_lower, joint_limits_upper
        Planning-joint limits, aligned with :attr:`planning_joint_names`.
    cameras
        Camera mounts on the robot (see :class:`CameraSpec`). Each has
        a unique ``camera_id`` consumed by :meth:`SimHandler.get_camera`.
    default_spawn_pose
        Fallback ``(x, y, θ)`` when a scene has no navmesh. Required
        for mobile robots, forbidden for fixed-base.
    default_joint_positions
        Mapping ``joint_name → qpos`` applied at reset before any
        randomisation. Keys outside the robot's joint set raise.
    """

    name: str
    sim_uids: Mapping[str, str]
    base_link_name: str
    is_mobile: bool
    urdf_path: str | None = None
    usd_path: str | None = None
    mjcf_path: str | None = None
    base_joint_names: tuple[str, ...] = ()
    actuators: Mapping[str, ActuatorCfg] = field(default_factory=dict)
    controller_order: tuple[str, ...] = ()
    planning_joint_names: tuple[str, ...] = ()
    head_joint_names: tuple[str, ...] = ()
    joint_limits_lower: tuple[float, ...] = ()
    joint_limits_upper: tuple[float, ...] = ()
    cameras: tuple[CameraSpec, ...] = ()
    default_spawn_pose: tuple[float, float, float] | None = None
    default_joint_positions: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Freeze mapping fields.
        object.__setattr__(self, "sim_uids", MappingProxyType(dict(self.sim_uids)))
        object.__setattr__(
            self,
            "actuators",
            MappingProxyType({name: cfg for name, cfg in self.actuators.items()}),
        )
        object.__setattr__(
            self,
            "default_joint_positions",
            MappingProxyType(dict(self.default_joint_positions)),
        )

        # --- joint limits / planning joints alignment ---
        n = len(self.planning_joint_names)
        if len(self.joint_limits_lower) != n:
            raise ValueError(
                f"RobotCfg {self.name!r}: joint_limits_lower len "
                f"{len(self.joint_limits_lower)} != planning_joint_names len {n}"
            )
        if len(self.joint_limits_upper) != n:
            raise ValueError(
                f"RobotCfg {self.name!r}: joint_limits_upper len "
                f"{len(self.joint_limits_upper)} != planning_joint_names len {n}"
            )
        if any(
            lo > hi for lo, hi in zip(self.joint_limits_lower, self.joint_limits_upper)
        ):
            raise ValueError(
                f"RobotCfg {self.name!r}: joint_limits_lower entries must be <= upper"
            )

        # --- mobile vs fixed-base contract ---
        if self.is_mobile:
            if len(self.base_joint_names) != 3:
                raise ValueError(
                    f"RobotCfg {self.name!r}: mobile robots must define exactly 3 "
                    f"base_joint_names (x, y, θ); got {self.base_joint_names!r}"
                )
            if len(set(self.base_joint_names)) != 3:
                raise ValueError(
                    f"RobotCfg {self.name!r}: base_joint_names must not contain duplicates"
                )
            if self.default_spawn_pose is None:
                raise ValueError(
                    f"RobotCfg {self.name!r}: mobile robots must set "
                    f"default_spawn_pose for the no-navmesh fallback"
                )
        else:
            if self.base_joint_names:
                raise ValueError(
                    f"RobotCfg {self.name!r}: fixed-base robots must have "
                    f"empty base_joint_names, got {self.base_joint_names!r}"
                )
            if self.default_spawn_pose is not None:
                raise ValueError(
                    f"RobotCfg {self.name!r}: fixed-base robots must not set "
                    f"default_spawn_pose"
                )

        # --- asset path: at least one route to instantiate the robot ---
        has_sim_uid = len(self.sim_uids) > 0
        has_asset_path = any(
            p is not None for p in (self.urdf_path, self.usd_path, self.mjcf_path)
        )
        if not (has_sim_uid or has_asset_path):
            raise ValueError(
                f"RobotCfg {self.name!r}: must set at least one of sim_uids, "
                f"urdf_path, usd_path, or mjcf_path so sim handlers can "
                f"instantiate the robot"
            )

        # --- actuators / controller_order consistency ---
        if set(self.actuators) != set(self.controller_order):
            raise ValueError(
                f"RobotCfg {self.name!r}: controller_order keys "
                f"{sorted(self.controller_order)} must match actuators keys "
                f"{sorted(self.actuators)}"
            )
        if len(set(self.controller_order)) != len(self.controller_order):
            raise ValueError(
                f"RobotCfg {self.name!r}: controller_order must not contain duplicates"
            )

        # --- cameras: unique ids ---
        cam_ids = [c.camera_id for c in self.cameras]
        if len(set(cam_ids)) != len(cam_ids):
            raise ValueError(
                f"RobotCfg {self.name!r}: camera_id values must be unique, "
                f"got {cam_ids!r}"
            )

        # --- default_joint_positions keys must name real joints ---
        known_joints = (
            set(self.planning_joint_names)
            | set(self.head_joint_names)
            | set(self.base_joint_names)
            | {j for cfg in self.actuators.values() for j in cfg.joint_names}
        )
        unknown = set(self.default_joint_positions) - known_joints
        if unknown:
            raise ValueError(
                f"RobotCfg {self.name!r}: default_joint_positions references "
                f"unknown joints {sorted(unknown)}"
            )

    # ── derived helpers (read-only views on the data) ──────────────

    def total_action_dim(self) -> int:
        """Sum of action dims across :attr:`controller_order`.

        The flat action vector the sim handler accepts has this length;
        per-controller slices are :attr:`action_slices_from_order`.
        """
        return sum(self.actuators[name].action_dim for name in self.controller_order)

    def action_slices_from_order(self) -> Mapping[str, slice]:
        """Per-controller slices into the flat action vector.

        Sim handlers with the flexibility to order controllers natively
        may expose a different layout; when they do, they override their
        own :attr:`SimHandler.action_slices`. The default — and ManiSkill's
        layout — follows ``controller_order``.
        """
        out: dict[str, slice] = {}
        start = 0
        for name in self.controller_order:
            stop = start + self.actuators[name].action_dim
            out[name] = slice(start, stop)
            start = stop
        return MappingProxyType(out)

    def camera_by_id(self, camera_id: str) -> CameraSpec:
        """Look up a :class:`CameraSpec` by id. Raises :class:`KeyError`."""
        for cam in self.cameras:
            if cam.camera_id == camera_id:
                return cam
        raise KeyError(
            f"RobotCfg {self.name!r}: no camera with id {camera_id!r}; "
            f"available: {[c.camera_id for c in self.cameras]}"
        )


__all__ = ["ActuatorCfg", "ControlMode", "RobotCfg"]


# ────────────────────────────────────────────────────────────────
# Runtime robot-state types (formerly types/robot.py — merged here
# on 2026-04-15 to retire the parallel singular/plural hierarchy
# CLAUDE.md Rule 4 forbids).
# ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class JointState:
    """Named joint positions."""

    names: tuple[str, ...]
    positions: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.names) != len(self.positions):
            raise ValueError(
                f"names ({len(self.names)}) and positions ({len(self.positions)}) "
                "must have the same length"
            )


@dataclass(frozen=True)
class RobotState:
    """Full robot state: base + arm + head."""

    base_pose: SE2Pose
    planning_joints: tuple[float, ...]  # 8-DOF: torso + 7 arm
    head_joints: tuple[float, ...]  # 2-DOF: pan + tilt


@dataclass(frozen=True)
class WholeBodyConfig:
    """Whole-body configuration for planning: arm + base."""

    arm_joints: npt.NDArray[np.float64]  # (8,) torso + 7 arm
    base_pose: SE2Pose


@dataclass(frozen=True)
class IKSolution:
    """A single IK solution, tagged with the joint names it corresponds to."""

    joint_names: tuple[str, ...]
    positions: npt.NDArray[np.float64]  # (dof,)

    def __post_init__(self) -> None:
        if len(self.joint_names) != self.positions.shape[0]:
            raise ValueError(
                f"joint_names ({len(self.joint_names)}) and positions "
                f"({self.positions.shape[0]}) must have the same length"
            )


__all__.extend(["IKSolution", "JointState", "RobotState", "WholeBodyConfig"])

# Note: the legacy flat ``RobotSpec`` dataclass and ``FETCH_SPEC``
# alias were deleted on 2026-04-15 once every call site (sim/,
# envs/, maniskill_helpers, maniskill_vec) migrated to RobotCfg /
# FETCH_CFG. See git history if you need the old definition.
