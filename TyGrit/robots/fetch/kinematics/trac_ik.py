"""TRAC-IK solver wrapper.

Wraps the vendored ``pytracik`` C++ extension (``ext/trac_ik/``).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R

import pytracik


class TracIKSolver:
    """Numerical IK solver using TRAC-IK (dual Newton-Raphson + SQP).

    Target pose must be in the frame of ``base_link`` passed to the constructor
    (4x4 homogeneous transform).
    """

    def __init__(
        self,
        base_link: str,
        ee_link: str,
        urdf_string: str,
        timeout: float = 0.2,
        epsilon: float = 1e-6,
    ) -> None:
        self._base_link = base_link
        self._solver = pytracik.TRAC_IK(
            base_link,
            ee_link,
            urdf_string,
            timeout,
            epsilon,
            pytracik.SolveType.Distance,
        )
        self._lower = np.array(pytracik.get_joint_lower_bounds(self._solver))
        self._upper = np.array(pytracik.get_joint_upper_bounds(self._solver))

    @property
    def base_frame(self) -> str:
        return self._base_link

    @property
    def num_joints(self) -> int:
        return len(self._lower)

    @property
    def joint_limits(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        return self._lower, self._upper

    def solve(
        self,
        target_pose: npt.NDArray[np.float64],
        seed: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Return joint angles for *target_pose* (4x4 matrix).

        Uses midpoint of joint limits as default seed.

        Raises:
            ValueError: If no solution can be found.
        """
        pos = target_pose[:3, 3]
        rot = target_pose[:3, :3]
        quat = R.from_matrix(rot).as_quat()  # [x, y, z, w]

        if seed is None:
            seed = (self._lower + self._upper) / 2.0

        # pytracik.ik returns array where r[0] >= 0 means success, r[1:] is solution
        r = pytracik.ik(
            self._solver,
            np.asarray(seed, dtype=np.float64),
            pos[0],
            pos[1],
            pos[2],
            quat[0],
            quat[1],
            quat[2],
            quat[3],
        )

        if r[0] < 0:
            raise ValueError("TRAC-IK: no solution found for target pose.")

        return np.array(r[1:], dtype=np.float64)
