"""Base-pose reachability sampling for the Fetch robot.

Provides pure functions for generating candidate robot base poses around a
target object and filtering them by IK reachability of a top-down grasp.
Used by world generators to produce (base_pose, init_qpos, grasp_hint) tuples
for the mobile-grasping dataset.

No simulator imports; the IKFast C extension is deferred to call time.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

# ARM_JOINT_NAMES: the 7 arm joints in IKFast solution order (indices 1-7 of
# PLANNING_JOINT_NAMES from constants.py — index 0 is torso_lift_joint).
_ARM_JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "upperarm_roll_joint",
    "elbow_flex_joint",
    "forearm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
)

# Shoulder-lift free-parameter sweep range (radians), from JOINT_LIMITS in constants.py.
_SHOULDER_LIFT_MIN: float = -1.221
_SHOULDER_LIFT_MAX: float = 1.518
_SHOULDER_LIFT_SAMPLES: int = 8


def sample_base_poses(
    object_position: tuple[float, float, float],
    num_candidates: int = 32,
    min_dist: float = 0.45,
    max_dist: float = 0.90,
    rng: np.random.Generator | None = None,
) -> list[tuple[float, float, float]]:
    """Sample candidate robot base poses on a ring around *object_position*.

    Each returned pose ``(bx, by, btheta)`` places the robot at a horizontal
    distance in ``[min_dist, max_dist]`` from the object, facing toward it.

    Args:
        object_position: Target object world-frame position ``(x, y, z)``.
        num_candidates: Number of poses to sample.
        min_dist: Minimum XY distance from base origin to object (metres).
        max_dist: Maximum XY distance from base origin to object (metres).
        rng: Optional numpy random generator; uses the global default if None.

    Returns:
        List of ``(x, y, theta)`` base poses.
    """
    if rng is None:
        rng = np.random.default_rng()

    ox, oy, _ = object_position
    angles = rng.uniform(0.0, 2.0 * math.pi, size=num_candidates)
    dists = rng.uniform(min_dist, max_dist, size=num_candidates)

    poses: list[tuple[float, float, float]] = []
    for angle, dist in zip(angles, dists):
        bx = ox - dist * math.cos(angle)
        by = oy - dist * math.sin(angle)
        btheta = angle  # robot faces toward the object
        poses.append((float(bx), float(by), float(btheta)))
    return poses


def _build_T_world_base(bx: float, by: float, btheta: float) -> npt.NDArray[np.float64]:
    """4×4 homogeneous transform: base_link in world frame."""
    T = np.eye(4, dtype=np.float64)
    c, s = math.cos(btheta), math.sin(btheta)
    T[0, 0], T[0, 1] = c, -s
    T[1, 0], T[1, 1] = s, c
    T[0, 3], T[1, 3] = bx, by
    return T


def check_ik_reachability(
    base_pose: tuple[float, float, float],
    object_position: tuple[float, float, float],
    torso_lift: float = 0.2,
) -> tuple[bool, dict[str, float] | None, tuple[float, ...] | None]:
    """Check whether a top-down grasp above *object_position* is IK-reachable.

    The grasp target is 0.15 m above the object with the gripper pointing
    straight down and the gripper X-axis pointing from the object toward
    the robot base.

    IKFast import is deferred here because the C extension requires the
    ``build/`` directory to be on ``PYTHONPATH`` and must not be imported
    at module load time.

    Args:
        base_pose: Robot base world pose ``(bx, by, btheta)``.
        object_position: Object world-frame position ``(ox, oy, oz)``.
        torso_lift: Torso lift joint value in metres (default 0.2).

    Returns:
        ``(True, init_qpos_dict, grasp_hint_tuple)`` on success, or
        ``(False, None, None)`` when no IK solution exists.
    """
    # Deferred import: ikfast_fetch C extension is only importable when
    # the project has been built and ext/ is on sys.path.
    from TyGrit.robots.fetch.kinematics.ikfast import IKFastSolver  # noqa: PLC0415

    bx, by, btheta = base_pose
    ox, oy, oz = object_position

    # ── Grasp pose in world frame ────────────────────────────────────────────
    grasp_pos_world = np.array([ox, oy, oz + 0.15], dtype=np.float64)

    # Gripper Z points down (-Z world).  Gripper X points from object toward
    # robot base (approach direction).  Gripper Y = Z × X (right-hand rule).
    approach = np.array([bx - ox, by - oy, 0.0], dtype=np.float64)
    approach_norm = np.linalg.norm(approach)
    if approach_norm < 1e-6:
        # Base is directly above the object; choose an arbitrary X direction.
        approach = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        approach /= approach_norm

    gripper_x = approach  # +X toward robot base
    gripper_z = np.array([0.0, 0.0, -1.0], dtype=np.float64)  # pointing down
    gripper_y = np.cross(gripper_z, gripper_x)  # right-hand rule
    gripper_y /= np.linalg.norm(gripper_y)

    R_world = np.column_stack([gripper_x, gripper_y, gripper_z])  # 3×3

    T_grasp_world = np.eye(4, dtype=np.float64)
    T_grasp_world[:3, :3] = R_world
    T_grasp_world[:3, 3] = grasp_pos_world

    # ── Transform grasp to base_link frame ──────────────────────────────────
    T_world_base = _build_T_world_base(bx, by, btheta)
    T_base_world = np.linalg.inv(T_world_base)
    T_grasp_base = T_base_world @ T_grasp_world

    # ── IK sweep over shoulder_lift free parameter ──────────────────────────
    solver = IKFastSolver()
    shoulder_lifts = np.linspace(
        _SHOULDER_LIFT_MIN, _SHOULDER_LIFT_MAX, _SHOULDER_LIFT_SAMPLES
    )

    best_solution: npt.NDArray[np.float64] | None = None
    for sl in shoulder_lifts:
        solutions = solver.solve_all(T_grasp_base, free_params=[torso_lift, float(sl)])
        if solutions:
            best_solution = solutions[0]
            break

    if best_solution is None:
        return (False, None, None)

    # ── Build init_qpos_dict ─────────────────────────────────────────────────
    # best_solution is length-8: [torso_lift, shoulder_pan, shoulder_lift,
    #   upperarm_roll, elbow_flex, forearm_roll, wrist_flex, wrist_roll]
    # Head pan: angle from base_link X-axis to object in base XY plane.
    obj_base = T_base_world @ np.array([ox, oy, oz, 1.0], dtype=np.float64)
    head_pan = float(math.atan2(obj_base[1], obj_base[0]))

    init_qpos: dict[str, float] = {
        "torso_lift_joint": float(best_solution[0]),
        "head_pan_joint": head_pan,
        "head_tilt_joint": 0.5,
    }
    for name, val in zip(_ARM_JOINT_NAMES, best_solution[1:]):
        init_qpos[name] = float(val)

    # ── Grasp hint in world frame (x, y, z, qx, qy, qz, qw) ────────────────
    r_world = Rotation.from_matrix(R_world)
    quat_xyzw = r_world.as_quat()  # SciPy returns [x, y, z, w]
    grasp_hint: tuple[float, ...] = (
        float(grasp_pos_world[0]),
        float(grasp_pos_world[1]),
        float(grasp_pos_world[2]),
        float(quat_xyzw[0]),
        float(quat_xyzw[1]),
        float(quat_xyzw[2]),
        float(quat_xyzw[3]),
    )

    return (True, init_qpos, grasp_hint)


def filter_reachable_base_poses(
    candidates: list[tuple[float, float, float]],
    object_position: tuple[float, float, float],
    torso_lifts: tuple[float, ...] = (0.1, 0.2, 0.3),
) -> list[
    tuple[tuple[float, float, float], dict[str, float], tuple[float, ...] | None]
]:
    """Filter candidate base poses to those from which the object is IK-reachable.

    For each candidate, the torso-lift values are tried in order and the first
    successful IK result is kept.

    Args:
        candidates: List of ``(x, y, theta)`` candidate base poses.
        object_position: Target object world-frame position ``(ox, oy, oz)``.
        torso_lifts: Torso-lift values to try for each candidate.

    Returns:
        List of ``(base_pose, init_qpos_dict, grasp_hint_tuple)`` for every
        candidate that has at least one valid IK solution.
    """
    results: list[
        tuple[tuple[float, float, float], dict[str, float], tuple[float, ...] | None]
    ] = []

    for base_pose in candidates:
        for torso_lift in torso_lifts:
            reachable, init_qpos, grasp_hint = check_ik_reachability(
                base_pose, object_position, torso_lift=torso_lift
            )
            if reachable:
                assert init_qpos is not None  # guaranteed by check_ik_reachability
                results.append((base_pose, init_qpos, grasp_hint))
                break  # first successful torso_lift is sufficient

    return results
