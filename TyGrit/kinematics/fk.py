"""Forward-kinematics protocol."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt


class ForwardKinematics(Protocol):
    """Callable that maps joint angles to link poses.

    Any bare function with signature ``(ndarray) -> dict[str, ndarray]``
    satisfies this protocol automatically via ``__call__``.
    """

    def __call__(
        self, joint_angles: npt.NDArray[np.float64]
    ) -> dict[str, npt.NDArray[np.float64]]: ...
