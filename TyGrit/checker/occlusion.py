"""Self-occlusion check: does the robot body block the camera's view of a target?

Pure function — takes pre-computed FK link poses, target sphere, and camera
intrinsics; returns True when the projected robot silhouette overlaps the
projected target sphere on the image plane.
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from TyGrit.kinematics.fetch.constants import (
    FETCH_SPHERES,
    HEAD_CAMERA_OFFSET,
    R_CV_TO_CAMERA_LINK,
)

# camera_link → OpenCV rotation (inverse of the CV-to-link transform).
_R_LINK_TO_CV: npt.NDArray[np.float64] = np.linalg.inv(R_CV_TO_CAMERA_LINK)


def _spheres_to_mask(
    centers_cam: npt.NDArray[np.float64],
    radii: npt.NDArray[np.float64],
    K: npt.NDArray[np.float64],
    h: int,
    w: int,
) -> npt.NDArray[np.bool_]:
    """Project 3-D spheres onto a 2-D boolean mask via pinhole model."""
    mask = np.zeros((h, w), dtype=np.uint8)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    for center, radius in zip(centers_cam, radii):
        if center[2] <= 0:
            continue
        u = int(fx * center[0] / center[2] + cx)
        v = int(fy * center[1] / center[2] + cy)
        pixel_radius = max(1, int(fx * radius / center[2]))
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(mask, (u, v), pixel_radius, 1, -1)

    return mask.astype(bool)


def check_self_occlusion(
    link_poses: dict[str, npt.NDArray[np.float64]],
    target_center: npt.NDArray[np.float64],
    target_radius: float,
    intrinsics: npt.NDArray[np.float64],
    image_shape: tuple[int, int],
) -> bool:
    """Check whether the robot's body occludes a target sphere in the camera image.

    All inputs are in the **base_link** frame (the FK output frame).

    Args:
        link_poses: FK output — dict mapping link names to 4x4 poses.
        target_center: (3,) target sphere centre in base_link frame.
        target_radius: Target sphere radius in metres.
        intrinsics: (3, 3) camera intrinsic matrix K.
        image_shape: (height, width) of the camera image.

    Returns:
        True if the projected robot silhouette overlaps the projected target.
    """
    # ── Camera extrinsics ────────────────────────────────────────────────
    T_head_tilt = link_poses.get("head_tilt_link")
    if T_head_tilt is None:
        return False

    # T_base_cam_link = T_base_head_tilt @ translate(HEAD_CAMERA_OFFSET)
    T_base_cam_link = T_head_tilt.copy()
    T_base_cam_link[:3, 3] += T_head_tilt[:3, :3] @ HEAD_CAMERA_OFFSET

    # T_cv_base maps base-frame points into OpenCV camera coordinates.
    T_cam_link_base = np.linalg.inv(T_base_cam_link)
    T_cv_base = _R_LINK_TO_CV @ T_cam_link_base

    R_cv_base = T_cv_base[:3, :3]
    t_cv_base = T_cv_base[:3, 3]

    # ── Collect robot spheres in camera (OpenCV) frame ───────────────────
    robot_centers_cam: list[npt.NDArray[np.float64]] = []
    robot_radii: list[npt.NDArray[np.float64]] = []

    for link_name, sphere_list in FETCH_SPHERES.items():
        T = link_poses.get(link_name)
        if T is None:
            continue
        rot = T[:3, :3]
        t = T[:3, 3]
        arr = np.asarray(sphere_list, dtype=np.float64)  # (K, 4)
        centers_base = (arr[:, :3] @ rot.T) + t  # (K, 3)
        centers_cam = (centers_base @ R_cv_base.T) + t_cv_base  # (K, 3)
        robot_centers_cam.append(centers_cam)
        robot_radii.append(arr[:, 3])

    if not robot_centers_cam:
        return False

    all_centers = np.vstack(robot_centers_cam)
    all_radii = np.concatenate(robot_radii)

    # ── Target sphere in camera frame ────────────────────────────────────
    target_cam = R_cv_base @ target_center + t_cv_base

    h, w = image_shape
    robot_mask = _spheres_to_mask(all_centers, all_radii, intrinsics, h, w)
    target_mask = _spheres_to_mask(
        target_cam[np.newaxis], np.array([target_radius]), intrinsics, h, w
    )

    return bool(np.any(robot_mask & target_mask))
