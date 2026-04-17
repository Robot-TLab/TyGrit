"""Robot reachability checks for mobile-grasping dataset generation.

Given a target object pose (world-frame) and a candidate robot base pose
``(x, y, theta)``, the core question is: *can the Fetch arm reach the
object from this base placement?*

The check is purely kinematic — no simulation, no collision checking,
no motion planning. It answers whether an IK solution *exists* for the
given configuration; downstream consumers (the generator CLI or a live
policy) handle obstacles.

Design
------
* **Pure functions, no class state.** IK solvers are passed in so callers
  control construction lifetime (the IKFast solver is cheap to create;
  TRAC-IK needs a URDF string).
* **World-frame in, bool out.** The function handles the world-to-base
  coordinate transform internally — callers never need to think about
  kinematic frames.
* **Robot-agnostic API shape, Fetch-specific internals.** The public
  function is ``check_reachability(...)``; the body currently calls
  IKFast because that's the only robot TyGrit has. When a second robot
  lands, generalize the dispatch.

Coordinate convention
---------------------
All poses are world-frame Z-up.  Base pose is ``(x, y, theta)`` where
``theta = 0`` points along world +X (CCW positive).  Object pose is a
4x4 homogeneous transform.  Internally, the function converts the
object pose into the IK solver's ``base_link`` frame by undoing the
base SE(2) transform and the fixed torso-to-base offset.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from TyGrit.robots.fetch.kinematics.constants import (
    JOINT_LIMITS_LOWER,
    JOINT_LIMITS_UPPER,
)


def _se2_to_4x4(x: float, y: float, theta: float) -> npt.NDArray[np.float64]:
    """Build a 4x4 world-frame transform from an SE(2) base pose.

    The robot base sits at ``(x, y, 0)`` with heading ``theta`` about
    world +Z. The Fetch base_link is at floor level (z = 0).
    """
    c, s = math.cos(theta), math.sin(theta)
    return np.array(
        [
            [c, -s, 0.0, x],
            [s, c, 0.0, y],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def world_to_base_link(
    target_world: npt.NDArray[np.float64],
    base_x: float,
    base_y: float,
    base_theta: float,
) -> npt.NDArray[np.float64]:
    """Transform a world-frame 4x4 pose into Fetch's base_link frame.

    Parameters
    ----------
    target_world
        4x4 homogeneous transform of the target in world frame.
    base_x, base_y, base_theta
        Robot base SE(2) pose in world frame.

    Returns
    -------
    NDArray
        4x4 homogeneous transform of the target in base_link frame.
    """
    T_world_base = _se2_to_4x4(base_x, base_y, base_theta)
    T_base_world = np.linalg.inv(T_world_base)
    return T_base_world @ target_world


def check_reachability(
    object_pose_world: npt.NDArray[np.float64],
    base_x: float,
    base_y: float,
    base_theta: float,
    *,
    ik_solver=None,
    torso_samples: int = 5,
    shoulder_samples: int = 5,
) -> tuple[bool, npt.NDArray[np.float64] | None]:
    """Check whether Fetch can reach ``object_pose_world`` from a base pose.

    Tries multiple free-parameter combinations (torso height, shoulder
    lift) across the joint-limit range. Returns the first valid IK
    solution found, or ``(False, None)`` if none exists.

    Parameters
    ----------
    object_pose_world
        4x4 homogeneous transform of the target grasp frame in world
        frame.
    base_x, base_y, base_theta
        Robot base SE(2) pose in world frame.
    ik_solver
        An :class:`~TyGrit.robots.fetch.kinematics.ikfast.IKFastSolver`
        instance (or compatible). If ``None``, one is created internally.
    torso_samples
        Number of torso heights to try uniformly across the joint-limit
        range.
    shoulder_samples
        Number of shoulder-lift values to try per torso height.

    Returns
    -------
    tuple[bool, NDArray | None]
        ``(True, joint_positions)`` if a valid IK solution exists, where
        ``joint_positions`` is an 8-element array
        ``[torso, s_pan, s_lift, u_roll, e_flex, f_roll, w_flex, w_roll]``.
        ``(False, None)`` otherwise.
    """
    if ik_solver is None:
        from TyGrit.robots.fetch.kinematics.ikfast import IKFastSolver

        ik_solver = IKFastSolver()

    # Transform target into base_link frame for IKFast.
    target_base = world_to_base_link(
        object_pose_world,
        base_x,
        base_y,
        base_theta,
    )

    # Sweep free parameters: torso_lift, shoulder_lift.
    torso_lo, torso_hi = float(JOINT_LIMITS_LOWER[0]), float(JOINT_LIMITS_UPPER[0])
    shoulder_lo, shoulder_hi = float(JOINT_LIMITS_LOWER[2]), float(
        JOINT_LIMITS_UPPER[2]
    )

    torso_vals = np.linspace(torso_lo, torso_hi, torso_samples)
    shoulder_vals = np.linspace(shoulder_lo, shoulder_hi, shoulder_samples)

    for torso in torso_vals:
        for shoulder in shoulder_vals:
            solutions = ik_solver.solve_all(
                target_base, free_params=[float(torso), float(shoulder)]
            )
            if solutions:
                return True, solutions[0]

    return False, None


def sample_base_poses_around_object(
    object_position: tuple[float, float, float],
    *,
    min_distance: float = 0.6,
    max_distance: float = 1.2,
    num_distances: int = 3,
    num_angles: int = 8,
    face_object: bool = True,
) -> list[tuple[float, float, float]]:
    """Generate candidate base poses arranged in rings around an object.

    The robot base is placed at various distances and angles in the XY
    plane, with theta oriented to face the object (if ``face_object``
    is True). This is the search space the generator CLI sweeps to find
    reachable configurations.

    Parameters
    ----------
    object_position
        ``(x, y, z)`` world-frame position of the target object.
    min_distance, max_distance
        Inner and outer radius of the annular search region (metres).
    num_distances
        Number of radial steps between ``min_distance`` and
        ``max_distance``.
    num_angles
        Number of angular steps around the full 2-pi circle.
    face_object
        If True, theta is set so the robot faces the object. If False,
        theta is set to 0 (for caller-controlled orientation).

    Returns
    -------
    list[tuple[float, float, float]]
        Candidate ``(x, y, theta)`` base poses.
    """
    ox, oy = object_position[0], object_position[1]
    distances = np.linspace(min_distance, max_distance, num_distances)
    angles = np.linspace(0, 2 * math.pi, num_angles, endpoint=False)

    poses: list[tuple[float, float, float]] = []
    for d in distances:
        for a in angles:
            bx = ox + d * math.cos(a)
            by = oy + d * math.sin(a)
            if face_object:
                # Theta points the robot at the object.
                theta = math.atan2(oy - by, ox - bx)
            else:
                theta = 0.0
            poses.append((float(bx), float(by), float(theta)))

    return poses
