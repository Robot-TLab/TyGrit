"""The Fetch mobile manipulator as pure data.

Exports :data:`FETCH_CFG`: the canonical :class:`RobotCfg` descriptor
carrying URDF, per-sim uids, actuator specs, camera mounts, and joint
limits. Every :mod:`TyGrit.sim` handler consumes it.

Source of truth for each field is referenced in the trailing comment.
"""

from __future__ import annotations

from pathlib import Path

from TyGrit.robots.fetch.kinematics.constants import (
    HEAD_JOINT_NAMES,
    JOINT_LIMITS_LOWER,
    JOINT_LIMITS_UPPER,
    PLANNING_JOINT_NAMES,
)
from TyGrit.types.robots import ActuatorCfg, RobotCfg
from TyGrit.types.sensors import CameraSpec

# Resolve the sim-loadable Fetch URDF path once. Kept as a module-level
# string so FETCH_CFG stays pure data (no :class:`Path` objects — the
# frozen dataclass wants plain types for trivial pickling).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FETCH_SIM_URDF = str(_PROJECT_ROOT / "resources" / "fetch" / "sim" / "fetch.urdf")

# ── actuator specs ────────────────────────────────────────────────────
# Source of truth: TyGrit.sim.maniskill_helpers.build_action_slices
# iterates controller_order and reads each controller's action_space
# from ManiSkill's Fetch agent. The dims below mirror that layout.
#
# Per-sim tuning (stiffness, damping, …) is intentionally sparse: we
# let ManiSkill's defaults ride unless a sim-specific number clearly
# needs to be overridden. Add sim_params only when a specific sim's
# default is wrong.

_ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "upperarm_roll_joint",
    "elbow_flex_joint",
    "forearm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
)

_BODY_JOINT_NAMES = (
    "head_pan_joint",
    "head_tilt_joint",
    "torso_lift_joint",
)

_BASE_JOINT_NAMES = (
    "root_x_axis_joint",
    "root_y_axis_joint",
    "root_z_rotation_joint",
)

_GRIPPER_JOINT_NAMES = (
    "r_gripper_finger_joint",
    "l_gripper_finger_joint",
)

_ACTUATORS = {
    "arm": ActuatorCfg(
        name="arm",
        joint_names=_ARM_JOINT_NAMES,
        control_mode="velocity",
        action_dim=7,
    ),
    "gripper": ActuatorCfg(
        name="gripper",
        # Two finger joints driven symmetrically by one scalar. The
        # explicit command_to_joint_mapping=(0, 0) says "both fingers
        # read command[0]". Prior blind broadcast was Fetch-correct
        # but Genesis's handler couldn't tell it apart from the base
        # case, which is semantically different.
        joint_names=_GRIPPER_JOINT_NAMES,
        control_mode="position",
        action_dim=1,
        command_to_joint_mapping=(0, 0),
    ),
    "body": ActuatorCfg(
        name="body",
        joint_names=_BODY_JOINT_NAMES,
        control_mode="velocity",
        action_dim=3,
    ),
    "base": ActuatorCfg(
        name="base",
        # Holonomic base: action is a cartesian twist (v, w), NOT a
        # joint command. ManiSkill's base controller integrates it into
        # the 3 base joints internally. Sims without a native twist
        # controller (Genesis today) raise when asked to drive a
        # base_twist actuator — generic joint-level fallback is unsafe
        # because the twist→joint math depends on base kinematics.
        joint_names=_BASE_JOINT_NAMES,
        control_mode="velocity",
        action_dim=2,
        kind="base_twist",
    ),
}

# ── head camera ──────────────────────────────────────────────────────
# Mount pose (head_camera_link relative to head_tilt_link) from the
# Fetch URDF; see ``HEAD_CAMERA_OFFSET`` in kinematics/fetch/constants.py.
# The camera frame follows SAPIEN's +x forward / +y left / +z up
# convention — sim handlers apply the CV→link rotation at the boundary
# when returning rgb/depth.

_HEAD_CAMERA = CameraSpec(
    camera_id="head",
    parent_link="head_camera_link",
    # The URDF already bakes HEAD_CAMERA_OFFSET into the head_camera_link
    # placement, so from head_camera_link itself the camera sits at the
    # origin with identity orientation.
    position=(0.0, 0.0, 0.0),
    orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    width=640,
    height=480,
    fovy_degrees=45.0,
    near=0.01,
    far=10.0,
    # ManiSkill's Fetch agent pre-registers a head camera sensor under
    # the id ``"fetch_head"`` inside its agent class. TyGrit exposes it
    # as ``"head"`` publicly — this mapping lets the ManiSkill handler
    # translate one to the other without a Fetch-specific shim.
    sim_sensor_ids={"maniskill": "fetch_head"},
)

# ── FETCH_CFG (canonical) ────────────────────────────────────────────

FETCH_CFG = RobotCfg(
    name="fetch",
    # ManiSkill ships a ``"fetch"`` agent; Genesis / Isaac Sim / MuJoCo
    # load from URDF / USD / MJCF (we only have the URDF today — USD
    # and MJCF can be added when those sims are wired up).
    #
    sim_uids={"maniskill": "fetch"},
    # Genesis / Isaac Sim load the Fetch URDF verbatim, so they need
    # the sim-loadable ManiSkill asset materialized under
    # ``resources/fetch/sim/`` via ``pixi run setup-fetch-sim``.
    urdf_path=_FETCH_SIM_URDF,
    usd_path=None,
    mjcf_path=None,
    # Base link queries for get_link_pose — FetchRobotCore uses this
    # for the qpos↔world offset calibration on every reset.
    base_link_name="base_link",
    is_mobile=True,
    base_joint_names=_BASE_JOINT_NAMES,
    actuators=_ACTUATORS,
    # Order determines concatenation layout of the flat action vector.
    # FetchRobotCore._assemble_action reads this order.
    controller_order=("arm", "gripper", "body", "base"),
    planning_joint_names=PLANNING_JOINT_NAMES,
    head_joint_names=HEAD_JOINT_NAMES,
    joint_limits_lower=tuple(float(x) for x in JOINT_LIMITS_LOWER),
    joint_limits_upper=tuple(float(x) for x in JOINT_LIMITS_UPPER),
    cameras=(_HEAD_CAMERA,),
    # No-navmesh fallback used by FetchRobotCore._randomize_robot_pose
    # (ManiSkill's default Fetch spawn).
    default_spawn_pose=(-1.0, 0.0, 0.0),
    default_joint_positions={},
)


__all__ = ["FETCH_CFG"]
