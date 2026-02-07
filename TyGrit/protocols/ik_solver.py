"""Protocol for inverse-kinematics solvers."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt

from TyGrit.types.robot import IKSolution


class IKSolver(Protocol):
    """Inverse-kinematics solver bound to a specific kinematic chain.

    Each concrete implementation is constructed with whatever robot-specific
    context it needs (URDF, base pose, chain definition).  The protocol
    exposes only the chain metadata and a uniform ``solve()`` interface.
    """

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Ordered joint names this solver produces solutions for."""
        ...

    @property
    def dof(self) -> int:
        """Degrees of freedom of the kinematic chain."""
        ...

    def solve(
        self,
        target_pose: npt.NDArray[np.float64],
        seed: npt.NDArray[np.float64] | None = None,
    ) -> list[IKSolution]:
        """Return zero or more IK solutions for *target_pose* (4x4 matrix).

        Each :class:`IKSolution` carries the joint names and values so the
        caller never has to guess the ordering.
        """
        ...
