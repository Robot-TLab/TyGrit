"""IKFast analytical IK solver for the Fetch robot.

Wraps the vendored ``ikfast_fetch`` C extension (``ext/ikfast_fetch/``).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

import ikfast_fetch
from TyGrit.kinematics.fetch.constants import JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER


def _is_valid(solution: list[float]) -> bool:
    sol = np.array(solution)
    return bool(np.all(sol >= JOINT_LIMITS_LOWER) and np.all(sol <= JOINT_LIMITS_UPPER))


class IKFastSolver:
    """Analytical IK for the Fetch arm (8-DOF: torso + 7 arm joints).

    Target pose must be in **base_link frame** (4x4 homogeneous transform).
    Also provides :meth:`solve_all` for multi-solution enumeration.
    """

    @property
    def base_frame(self) -> str:
        return "base_link"

    def solve(
        self,
        target_pose: npt.NDArray[np.float64],
        seed: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Return the joint solution nearest to *seed*.

        Raises:
            ValueError: If no valid solution exists.
        """
        solutions = self.solve_all(target_pose)
        if not solutions:
            raise ValueError("IKFast: no valid solution found for target pose.")

        if seed is None:
            return solutions[0]

        # Return the solution closest to seed (L2 norm).
        dists = [np.linalg.norm(s - seed) for s in solutions]
        return solutions[int(np.argmin(dists))]

    def solve_all(
        self,
        target_pose: npt.NDArray[np.float64],
        free_params: list[float] | None = None,
    ) -> list[npt.NDArray[np.float64]]:
        """Return all valid joint solutions within joint limits.

        Args:
            target_pose: 4x4 homogeneous transform.
            free_params: ``[torso_lift, shoulder_lift]``.  Randomly sampled
                within joint limits if *None*.
        """
        if free_params is None:
            torso_lift = float(
                np.random.uniform(JOINT_LIMITS_LOWER[0], JOINT_LIMITS_UPPER[0])
            )
            shoulder_lift = float(
                np.random.uniform(JOINT_LIMITS_LOWER[2], JOINT_LIMITS_UPPER[2])
            )
            free_params = [torso_lift, shoulder_lift]

        position = target_pose[:3, 3].tolist()
        rotation = target_pose[:3, :3].tolist()

        raw = ikfast_fetch.get_ik(rotation, position, free_params)
        if not raw:
            return []

        return [np.array(sol, dtype=np.float64) for sol in raw if _is_valid(sol)]
