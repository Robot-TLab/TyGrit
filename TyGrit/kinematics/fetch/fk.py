"""Fetch FK solver factory.

Available solvers:

- ``"fk"`` — :class:`~TyGrit.kinematics.fetch.fk_numpy.FetchSkeletonFK`
    All 15 link 4×4 poses, 10 joints (torso + 7 arm + 2 head).
    Pure NumPy, no C extension needed.

- ``"fk_ikfast"`` — :class:`~TyGrit.kinematics.fetch.fk_ikfast.FetchIKFastFK`
    Gripper 4×4 only, 8 joints (torso + 7 arm).
    Uses IKFast C extension for speed.

- ``"fk_batch"`` — :class:`~TyGrit.kinematics.fetch.fk_torch.FetchBatchFK`
    All 15 link ``(B, 4, 4)`` poses, 10 joints.
    PyTorch, GPU-ready.
"""

from __future__ import annotations

from typing import Union

from TyGrit.kinematics.fk import BatchFKSolver, EEFKSolver, SkeletonFKSolver


def create_fetch_fk_solver(
    solver: str,
) -> Union[SkeletonFKSolver, EEFKSolver, BatchFKSolver]:
    """Create a Fetch FK solver by name.

    Args:
        solver: One of ``"fk"``, ``"fk_ikfast"``, ``"fk_batch"``.
    """
    if solver == "fk":
        from TyGrit.kinematics.fetch.fk_numpy import FetchSkeletonFK

        return FetchSkeletonFK()

    if solver == "fk_ikfast":
        from TyGrit.kinematics.fetch.fk_ikfast import FetchIKFastFK

        return FetchIKFastFK()

    if solver == "fk_batch":
        from TyGrit.kinematics.fetch.fk_torch import FetchBatchFK

        return FetchBatchFK()

    raise ValueError(
        f"Unknown Fetch FK solver: {solver!r}. Available: 'fk', 'fk_ikfast', 'fk_batch'"
    )
