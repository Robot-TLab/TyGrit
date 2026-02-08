"""Inverse-kinematics solver protocol — standard types only."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt


class IKSolver(Protocol):
    """Inverse-kinematics solver bound to a specific kinematic chain.

    Interface uses only standard types (ndarray, str).  Raises
    ``ValueError`` when no solution is found.
    """

    def solve(
        self,
        target_pose: npt.NDArray[np.float64],
        seed: npt.NDArray[np.float64] | None = None,
        joint_limits_lower: npt.NDArray[np.float64] | None = None,
        joint_limits_upper: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Return joint angles for *target_pose* (4x4 matrix).

        Raises:
            ValueError: If no solution can be found.
        """
        ...
