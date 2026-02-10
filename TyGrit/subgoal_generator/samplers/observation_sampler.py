"""Observation sampler — points the robot's camera at a world target.

Thin wrapper around :meth:`RobotBase.look_at` so that the observation
step can be composed as a sampler alongside grasp/place samplers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy.typing as npt

if TYPE_CHECKING:
    import numpy as np

    from TyGrit.envs.base import RobotBase


class ObservationSampler:
    """Direct the robot's head camera toward a world-frame target."""

    def __init__(self, robot: RobotBase) -> None:
        self._robot = robot

    def look_at_target(
        self,
        target_world: npt.NDArray[np.float64],
        camera_id: str = "head",
    ) -> None:
        """Point the camera at *target_world* ``[x, y, z]``."""
        self._robot.look_at(target_world, camera_id)
