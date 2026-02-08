"""General robot self-filter: remove points inside collision spheres.

Robot-agnostic — takes pre-computed world-frame link poses and sphere
definitions as arguments instead of calling FK internally.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def filter_robot_points(
    points: npt.NDArray[np.float32],
    link_poses: dict[str, npt.NDArray[np.float64]],
    spheres: dict[str, list[list[float]]],
    sphere_radius: float = 0.08,
) -> npt.NDArray[np.float32]:
    """Remove points that lie inside the robot's collision volume.

    Args:
        points: (N, 3) world-frame point cloud.
        link_poses: Dict mapping link names to 4x4 world-frame poses.
        spheres: Dict mapping link names to lists of [x, y, z] sphere
            centres in the link's local frame.
        sphere_radius: Collision sphere radius in metres.

    Returns:
        Filtered (M, 3) point cloud with robot points removed.
    """
    if points.shape[0] == 0:
        return points

    # Collect all sphere centres in world frame
    centres = []
    for link_name, local_spheres in spheres.items():
        if link_name not in link_poses:
            continue
        T_world_link = link_poses[link_name]
        rot = T_world_link[:3, :3]
        t = T_world_link[:3, 3]
        local_arr = np.asarray(local_spheres, dtype=np.float64)  # (K, 3)
        world_spheres = (local_arr @ rot.T) + t  # (K, 3)
        centres.append(world_spheres)

    if not centres:
        return points

    all_centres = np.vstack(centres).astype(np.float32)  # (S, 3)

    # Vectorised distance check: keep points outside all spheres
    r2 = sphere_radius * sphere_radius
    keep = np.ones(points.shape[0], dtype=bool)

    batch_size = 64
    for i in range(0, len(all_centres), batch_size):
        batch = all_centres[i : i + batch_size]  # (B, 3)
        # (N, 1, 3) - (1, B, 3) → (N, B, 3)
        diffs = points[:, np.newaxis, :] - batch[np.newaxis, :, :]
        d2 = np.sum(diffs * diffs, axis=2)  # (N, B)
        inside_any = np.any(d2 < r2, axis=1)  # (N,)
        keep &= ~inside_any

    return points[keep]
