"""Low-level math helpers: angle wrapping, quaternion / matrix conversions.

All quaternions use **[x, y, z, w]** convention (SciPy / ROS).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def angle_wrap(angle: float) -> float:
    """Wrap *angle* (radians) into (-pi, pi]."""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def quaternion_to_matrix(
    quaternion: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Convert quaternion [x, y, z, w] to a 4x4 homogeneous matrix.

    Ported from ``grasp_anywhere.robot.utils.transform_utils.quaternion_matrix``.
    """
    x, y, z, w = quaternion
    n = w * w + x * x + y * y + z * z
    s = 2.0 / n if n > 0 else 0.0

    wx = w * s * x
    wy = w * s * y
    wz = w * s * z
    xx = x * s * x
    xy = x * s * y
    xz = x * s * z
    yy = y * s * y
    yz = y * s * z
    zz = z * s * z

    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy, 0.0],
            [xy + wz, 1.0 - (xx + zz), yz - wx, 0.0],
            [xz - wy, yz + wx, 1.0 - (xx + yy), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Extract [x, y, z, w] quaternion from the rotation part of a 4x4 matrix.

    Ported from ``grasp_anywhere.robot.utils.transform_utils.quaternion_from_matrix``.
    """
    M = matrix[:3, :3]
    t = np.trace(M)

    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (M[2, 1] - M[1, 2]) * s
        y = (M[0, 2] - M[2, 0]) * s
        z = (M[1, 0] - M[0, 1]) * s
    elif M[0, 0] > M[1, 1] and M[0, 0] > M[2, 2]:
        s = 2.0 * np.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2])
        w = (M[2, 1] - M[1, 2]) / s
        x = 0.25 * s
        y = (M[0, 1] + M[1, 0]) / s
        z = (M[0, 2] + M[2, 0]) / s
    elif M[1, 1] > M[2, 2]:
        s = 2.0 * np.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2])
        w = (M[0, 2] - M[2, 0]) / s
        x = (M[0, 1] + M[1, 0]) / s
        y = 0.25 * s
        z = (M[1, 2] + M[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1])
        w = (M[1, 0] - M[0, 1]) / s
        x = (M[0, 2] + M[2, 0]) / s
        y = (M[1, 2] + M[2, 1]) / s
        z = 0.25 * s

    return np.array([x, y, z, w], dtype=np.float64)


def translation_from_matrix(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Extract the translation vector from a 4x4 homogeneous matrix."""
    return matrix[:3, 3].copy()
