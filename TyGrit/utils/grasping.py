"""Grasp-selection utilities — pure functions operating on GraspPose lists."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R

from TyGrit.types.grasp import GraspPose

# ── GraspGen → Fetch gripper convention ──────────────────────────────────────
# GraspGen outputs: Z = approach, Y = closing, X = normal.
# Fetch gripper_link: X = approach, Y = closing, Z = normal.
# So: X_ee = Z_grasp, Y_ee = Y_grasp, Z_ee = X_grasp.
T_GRASPGEN_TO_FETCH_EE: npt.NDArray[np.float64] = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def select_diverse_grasps(
    grasps: list[GraspPose],
    num_select: int,
) -> list[GraspPose]:
    """Select diverse grasps via greedy farthest-point sampling on rotation angle.

    Parameters
    ----------
    grasps : list[GraspPose]
        Grasps sorted by score (best first).
    num_select : int
        Number of diverse grasps to return.

    Returns
    -------
    list[GraspPose]
        The selected subset, preserving relative order.
    """
    if len(grasps) <= num_select:
        return list(grasps)

    rotations = [R.from_matrix(g.transform[:3, :3]) for g in grasps]

    selected = [0]
    for _ in range(num_select - 1):
        best_idx = -1
        best_min_dist = -1.0

        for i in range(len(grasps)):
            if i in selected:
                continue
            min_dist = min(
                (rotations[i].inv() * rotations[j]).magnitude() for j in selected
            )
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = i

        if best_idx >= 0:
            selected.append(best_idx)

    selected.sort()
    return [grasps[i] for i in selected]


def filter_by_score(
    grasps: list[GraspPose],
    threshold: float,
) -> list[GraspPose]:
    """Return grasps whose score meets *threshold*."""
    return [g for g in grasps if g.score >= threshold]
