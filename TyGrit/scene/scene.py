"""Protocol for the scene — the central world model / belief state.

Today the scene is point-cloud based; it will evolve into an object-centric
scene graph.  All filtering (ground, robot self-filter) happens inside the
scene so consumers get clean geometry.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt

from TyGrit.types.sensor import SensorSnapshot


class Scene(Protocol):
    """Abstract world model that maintains the current belief state.

    Implementations own the data, handle updates from sensors, and provide
    query methods for downstream consumers (collision checking, grasp
    planning, visualization).
    """

    def update(
        self,
        snapshot: SensorSnapshot,
        camera_pose: npt.NDArray[np.float64],
    ) -> None:
        """Integrate a new sensor observation into the belief state.

        Args:
            snapshot: Synchronised RGB-D capture + robot state.
            camera_pose: (4, 4) camera-to-world extrinsic matrix.
        """
        ...

    def get_pointcloud(self) -> npt.NDArray[np.float32]:
        """Return the current belief as a world-frame point cloud (N, 3).

        The returned cloud is already filtered (ground removed, robot
        self-points removed, etc.).
        """
        ...

    def clear(self) -> None:
        """Reset the dynamic observations, keeping any static map."""
        ...
