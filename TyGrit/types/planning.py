"""Planning-related data types for TyGrit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from TyGrit.types.geometry import SE2Pose


@dataclass(frozen=True)
class Trajectory:
    """A planned trajectory: sequences of arm configs and base poses."""

    arm_path: tuple[npt.NDArray[np.float64], ...]  # tuple of 8-DOF configs
    base_configs: tuple[SE2Pose, ...]
