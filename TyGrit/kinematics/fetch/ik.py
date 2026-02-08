"""Fetch IK solver factory.

Each solver expects the target EE pose (4x4 homogeneous transform) in the
frame of the kinematic chain's base link.  Pick the solver that matches the
frame you already have the target in — no extra transforms needed.

Available solvers:

- ``"ikfast_base"`` — IKFast analytical, 8-DOF (torso + 7 arm).
    Target in **base_link** frame.  Fast, no URDF needed.

- ``"trac_arm"`` — TRAC-IK numerical, 7-DOF (arm only).
    Target in **torso_lift_link** frame.  Use when the torso is deliberately
    fixed (e.g. to avoid camera shake).  You choose the torso height, then
    transform the target into torso_lift_link frame before calling solve().

- ``"trac_base"`` — TRAC-IK numerical, 8-DOF (torso + 7 arm).
    Target in **base_link** frame.  Like ikfast_base but numerical; handles
    the torso joint internally so no manual torso transform needed.

- ``"trac_whole_body"`` — TRAC-IK numerical, 11-DOF (base x,y,theta + torso + 7 arm).
    Target in **world** frame.  Solves for base placement too — useful for
    navigation-aware planning.
"""

from __future__ import annotations

from TyGrit.kinematics.ik import IKSolverBase

# solver name -> (base_link, ee_link) for TRAC-IK solvers
_TRAC_CHAINS: dict[str, tuple[str, str]] = {
    "trac_arm": ("torso_lift_link", "gripper_link"),  # 7-DOF, target in torso frame
    "trac_base": ("base_link", "gripper_link"),  # 8-DOF, target in base frame
    "trac_whole_body": ("world_link", "gripper_link"),  # 11-DOF, target in world frame
}


def create_fetch_ik_solver(
    solver: str,
    urdf_string: str = "",
    timeout: float = 0.2,
    epsilon: float = 1e-6,
) -> IKSolverBase:
    """Create a Fetch IK solver by name.

    Args:
        solver: One of ``"ikfast_base"``, ``"trac_arm"``, ``"trac_base"``,
            ``"trac_whole_body"``.
        urdf_string: URDF XML string (required for ``trac_*`` solvers).
        timeout: TRAC-IK timeout in seconds.
        epsilon: TRAC-IK convergence tolerance.
    """
    if solver == "ikfast_base":
        from TyGrit.kinematics.fetch.ikfast import IKFastSolver

        return IKFastSolver()

    if solver in _TRAC_CHAINS:
        from TyGrit.kinematics.fetch.trac_ik import TracIKSolver

        base_link, ee_link = _TRAC_CHAINS[solver]
        return TracIKSolver(
            base_link=base_link,
            ee_link=ee_link,
            urdf_string=urdf_string,
            timeout=timeout,
            epsilon=epsilon,
        )

    raise ValueError(
        f"Unknown Fetch IK solver: {solver!r}. "
        f"Available: {', '.join(['ikfast_base', *_TRAC_CHAINS])}"
    )
