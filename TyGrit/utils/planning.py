"""Planning utility functions — VAMP path conversion helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from TyGrit.types.geometry import SE2Pose
from TyGrit.types.planning import Trajectory


def vamp_path_to_lists(path_like: Any) -> list[list[float]]:
    """Normalise a VAMP path object into a plain list-of-lists.

    Handles VAMP bound/config types that expose ``to_list()``,
    numpy arrays, and plain lists.
    """
    sequence: list[list[float]] = []
    for elem in path_like:
        if isinstance(elem, list):
            sequence.append(elem)
        elif isinstance(elem, np.ndarray):
            sequence.append(elem.tolist())
        elif hasattr(elem, "to_list"):
            sequence.append(elem.to_list())
        elif hasattr(elem, "config"):
            sequence.append(list(elem.config))
        else:
            sequence.append(list(elem))
    return sequence


def lists_to_trajectory(
    arm_path: list[list[float]],
    base_configs: list[list[float]],
) -> Trajectory:
    """Convert raw arm/base lists into a :class:`Trajectory`."""
    return Trajectory(
        arm_path=tuple(np.asarray(c, dtype=np.float64) for c in arm_path),
        base_configs=tuple(SE2Pose(x=b[0], y=b[1], theta=b[2]) for b in base_configs),
    )
