"""Forward-kinematics base classes and robot-agnostic factory.

Three solver ABCs for different use cases:

- :class:`SkeletonFKSolver` — all link poses as ``dict[str, NDArray]``.
- :class:`EEFKSolver` — single end-effector 4×4 as ``NDArray``.
- :class:`BatchFKSolver` — batched poses as ``dict[str, Tensor]`` (GPU-ready).

Usage::

    from TyGrit.kinematics.fk import create_fk_solver

    fk = create_fk_solver("fetch", "fk")
    link_poses = fk.solve(joint_angles)  # {link_name: 4x4} in base_link frame

    ee_fk = create_fk_solver("fetch", "fk_ikfast")
    T = ee_fk.solve(joint_angles)  # 4x4 in base_link frame

    batch_fk = create_fk_solver("fetch", "fk_batch")
    poses = batch_fk.solve(joint_angles_BxN)  # {link_name: (B, 4, 4)} tensors
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from torch import Tensor


class SkeletonFKSolver(ABC):
    """All link poses via forward kinematics.

    ``solve()`` returns a dict mapping link names to 4×4 homogeneous
    transforms in ``base_frame``.
    """

    @property
    @abstractmethod
    def base_frame(self) -> str:
        """Frame ID that output poses are expressed in."""
        ...

    @abstractmethod
    def solve(
        self, joint_angles: npt.NDArray[np.float64]
    ) -> dict[str, npt.NDArray[np.float64]]:
        """Return all link poses for *joint_angles*.

        Args:
            joint_angles: Joint values in radians (length depends on robot).

        Returns:
            Dict mapping link names to 4×4 pose matrices in ``base_frame``.
        """
        ...


class EEFKSolver(ABC):
    """End-effector-only forward kinematics.

    ``solve()`` returns a single 4×4 homogeneous transform for the
    end-effector in ``base_frame``.
    """

    @property
    @abstractmethod
    def base_frame(self) -> str:
        """Frame ID that the output pose is expressed in."""
        ...

    @abstractmethod
    def solve(self, joint_angles: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return the end-effector pose for *joint_angles*.

        Args:
            joint_angles: Joint values in radians (length depends on robot).

        Returns:
            4×4 homogeneous transform of the end-effector in ``base_frame``.
        """
        ...


class BatchFKSolver(ABC):
    """Batched forward kinematics (GPU-ready, torch tensors).

    ``solve()`` returns a dict mapping link names to ``(B, 4, 4)`` pose
    tensors in ``base_frame``.
    """

    @property
    @abstractmethod
    def base_frame(self) -> str:
        """Frame ID that output poses are expressed in."""
        ...

    @abstractmethod
    def solve(self, joint_angles: Tensor) -> dict[str, Tensor]:
        """Return all link poses for a batch of joint configurations.

        Args:
            joint_angles: ``(B, N)`` tensor of joint values in radians.

        Returns:
            Dict mapping link names to ``(B, 4, 4)`` pose tensors in
            ``base_frame``.
        """
        ...


def create_fk_solver(
    robot: str, solver: str
) -> Union[SkeletonFKSolver, EEFKSolver, BatchFKSolver]:
    """Create an FK solver for *robot*.

    All output poses are in the solver's ``base_frame``.

    Args:
        robot: Robot name (``"fetch"``).
        solver: Solver name.  Available solvers per robot:

            **Fetch** (all output in **base_link** frame):

            - ``"fk"`` → :class:`SkeletonFKSolver` — all link 4×4 poses,
              10 joints (torso + 7 arm + 2 head).  Pure NumPy.
            - ``"fk_ikfast"`` → :class:`EEFKSolver` — gripper 4×4 only,
              8 joints (torso + 7 arm).  Fast IKFast C extension.
            - ``"fk_batch"`` → :class:`BatchFKSolver` — all link ``(B, 4, 4)``
              poses, 10 joints.  PyTorch, GPU-ready.
    """
    if robot == "fetch":
        from TyGrit.kinematics.fetch.fk import create_fetch_fk_solver

        return create_fetch_fk_solver(solver)
    raise ValueError(f"Unknown robot: {robot}")
