"""Protocol for collision checking.

Pure functional: takes geometry (extracted from the scene) and a robot
configuration, returns whether it is collision-free.  Does not own any
scene data.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt


class CollisionChecker(Protocol):
    """Checks robot configurations against point-cloud geometry."""

    def set_environment(
        self,
        cloud: npt.NDArray[np.float32],
        point_radius: float,
    ) -> None:
        """Load environment geometry for subsequent checks.

        Args:
            cloud: (N, 3) world-frame obstacle point cloud.
            point_radius: Inflation radius per point.
        """
        ...

    def validate(self, config: npt.NDArray[np.float64]) -> bool:
        """Return ``True`` if *config* is collision-free."""
        ...
