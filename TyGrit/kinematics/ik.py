"""Inverse-kinematics base class and robot-agnostic factory.

Usage::

    from TyGrit.kinematics.ik import create_ik_solver

    solver = create_ik_solver("fetch", "ikfast_base")
    joints = solver.solve(target_pose_4x4)  # pose in base_link frame
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt


class IKSolverBase(Protocol):
    """Protocol for all IK solvers.

    ``solve()`` takes a 4x4 homogeneous target pose expressed in the
    solver's base frame (determined at construction time) and returns
    joint angles.
    """

    @property
    def base_frame(self) -> str:
        """Frame ID that target poses must be expressed in."""
        ...

    def solve(
        self,
        target_pose: npt.NDArray[np.float64],
        seed: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Return joint angles for *target_pose* (4x4 homogeneous transform).

        Args:
            target_pose: 4x4 homogeneous transform of the desired EE pose,
                expressed in the solver's base frame.
            seed: Optional initial joint-angle guess.  If *None*, the solver
                picks a reasonable default.

        Raises:
            ValueError: If no solution can be found.
        """
        ...


def create_ik_solver(robot: str, solver: str, **kwargs) -> IKSolverBase:
    """Create an IK solver for *robot*.

    Each solver expects the target EE pose in a specific frame — pick the
    solver that matches the frame you already have the target in.

    Args:
        robot: Robot name (``"fetch"``).
        solver: Solver name.  Available solvers per robot:

            **Fetch:**

            - ``"ikfast_base"`` — analytical, 8-DOF, target in **base_link** frame.
              Fast, no URDF needed.
            - ``"trac_base"`` — numerical, 8-DOF, target in **base_link** frame.
              Handles torso internally.
            - ``"trac_arm"`` — numerical, 7-DOF, target in **torso_lift_link** frame.
              Use when torso is fixed; transform target to torso frame first.
            - ``"trac_whole_body"`` — numerical, 11-DOF, target in **world** frame.
              Also solves for base placement.

        **kwargs: Forwarded to the robot-specific factory.
            For Fetch TRAC-IK solvers: ``urdf_string``, ``timeout``, ``epsilon``.
    """
    if robot == "fetch":
        from TyGrit.kinematics.fetch.ik import create_fetch_ik_solver

        return create_fetch_ik_solver(solver, **kwargs)
    raise ValueError(f"Unknown robot: {robot}")
