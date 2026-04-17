"""Fetch robot static data: kinematic offsets, collision spheres, joint names/limits.

Extracted from:
- grasp_anywhere/robot/kinematics.py (kinematic offsets)
- grasp_anywhere/observation/gaze_optimizer.py (collision spheres)
- grasp_anywhere/robot/fetch.py (planning joint names)
- grasp_anywhere/robot/ik/ikfast_api.py (joint limits)
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# ── Kinematic offsets (from URDF, metres) ────────────────────────────────────

TORSO_BASE_OFFSET: npt.NDArray[np.float64] = np.array([-0.086875, 0.0, 0.37743])
SHOULDER_PAN_OFFSET: npt.NDArray[np.float64] = np.array([0.119525, 0.0, 0.34858])
SHOULDER_LIFT_OFFSET: npt.NDArray[np.float64] = np.array([0.117, 0.0, 0.06])
UPPERARM_ROLL_OFFSET: npt.NDArray[np.float64] = np.array([0.219, 0.0, 0.0])
ELBOW_FLEX_OFFSET: npt.NDArray[np.float64] = np.array([0.133, 0.0, 0.0])
FOREARM_ROLL_OFFSET: npt.NDArray[np.float64] = np.array([0.197, 0.0, 0.0])
WRIST_FLEX_OFFSET: npt.NDArray[np.float64] = np.array([0.1245, 0.0, 0.0])
WRIST_ROLL_OFFSET: npt.NDArray[np.float64] = np.array([0.1385, 0.0, 0.0])
GRIPPER_OFFSET: npt.NDArray[np.float64] = np.array([0.16645, 0.0, 0.0])

TORSO_FIXED_OFFSET: npt.NDArray[np.float64] = np.array([-0.086875, 0.0, 0.377425])
HEAD_PAN_OFFSET: npt.NDArray[np.float64] = np.array([0.053125, 0.0, 0.603001])
HEAD_TILT_OFFSET: npt.NDArray[np.float64] = np.array([0.14253, 0.0, 0.057999])
R_GRIPPER_FINGER_OFFSET: npt.NDArray[np.float64] = np.array([0.0, 0.065425, 0.0])
L_GRIPPER_FINGER_OFFSET: npt.NDArray[np.float64] = np.array([0.0, -0.065425, 0.0])

# head_tilt_link → head_camera_link (from Fetch URDF, verified against SAPIEN)
HEAD_CAMERA_OFFSET: npt.NDArray[np.float64] = np.array([0.055, 0.0, 0.0225])

# Rotation from OpenCV camera coords → head_camera_link coords.
# SAPIEN mounts the camera at head_camera_link with its own convention:
#   forward = +X_link,  right = -Y_link,  up = +Z_link
# OpenCV camera convention:
#   forward = +Z_cv,  right = +X_cv,  down = +Y_cv
# Mapping: +Z_cv (fwd) → +X_link,  +X_cv (right) → -Y_link,  +Y_cv (down) → -Z_link
R_CV_TO_CAMERA_LINK: npt.NDArray[np.float64] = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# ── Planning joint names (8-DOF: torso + 7 arm) ─────────────────────────────

PLANNING_JOINT_NAMES: tuple[str, ...] = (
    "torso_lift_joint",
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "upperarm_roll_joint",
    "elbow_flex_joint",
    "forearm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
)

# ── Head joint names (2-DOF: pan + tilt) ──────────────────────────────────────

HEAD_JOINT_NAMES: tuple[str, ...] = (
    "head_pan_joint",
    "head_tilt_joint",
)

# ── Joint limits (8-DOF) ────────────────────────────────────────────────────

JOINT_LIMITS_LOWER: npt.NDArray[np.float64] = np.array(
    [0.0, -1.6056, -1.221, -np.pi, -2.251, -np.pi, -2.16, -np.pi]
)
JOINT_LIMITS_UPPER: npt.NDArray[np.float64] = np.array(
    [0.38615, 1.6056, 1.518, np.pi, 2.251, np.pi, 2.16, np.pi]
)

# ── Collision spheres (from fetch_spherized.urdf) ───────────────────────────
# dict[link_name, list[list[float]]]  —  each entry is [x, y, z, radius] in link frame

FETCH_SPHERES: dict[str, list[list[float]]] = {
    "base_link": [
        [-0.12, 0.0, 0.182, 0.24],
        [0.225, 0.0, 0.31, 0.066],
        [0.08, -0.06, 0.16, 0.22],
        [0.215, -0.07, 0.31, 0.066],
        [0.185, -0.135, 0.31, 0.066],
        [0.13, -0.185, 0.31, 0.066],
        [0.065, -0.2, 0.31, 0.066],
        [0.01, -0.2, 0.31, 0.066],
        [0.08, 0.06, 0.16, 0.22],
        [0.215, 0.07, 0.31, 0.066],
        [0.185, 0.135, 0.31, 0.066],
        [0.13, 0.185, 0.31, 0.066],
        [0.065, 0.2, 0.31, 0.066],
        [0.01, 0.2, 0.31, 0.066],
    ],
    "torso_lift_link": [
        [-0.1, -0.05, 0.15, 0.15],
        [-0.1, 0.05, 0.15, 0.15],
        [-0.1, 0.05, 0.3, 0.15],
        [-0.1, 0.05, 0.45, 0.15],
        [-0.1, -0.05, 0.45, 0.15],
        [-0.1, -0.05, 0.3, 0.15],
        # torso_lift_link_collision_2 (fixed child, same FK pose)
        [0.1, 0.0, 0.24, 0.07],
    ],
    "torso_fixed_link": [
        [-0.1, -0.07, 0.35, 0.12],
        [-0.1, 0.07, 0.35, 0.12],
        [-0.1, -0.07, 0.2, 0.12],
        [-0.1, 0.07, 0.2, 0.12],
        [-0.1, 0.07, 0.07, 0.12],
        [-0.1, -0.07, 0.07, 0.12],
    ],
    "head_pan_link": [
        [0.0, 0.0, 0.06, 0.15],
        [0.145, 0.0, 0.058, 0.05],
        [0.145, -0.0425, 0.058, 0.05],
        [0.145, 0.0425, 0.058, 0.05],
        [0.145, 0.085, 0.058, 0.05],
        [0.145, -0.085, 0.058, 0.05],
        [0.0625, -0.115, 0.03, 0.03],
        [0.088, -0.115, 0.03, 0.03],
        [0.1135, -0.115, 0.03, 0.03],
        [0.139, -0.115, 0.03, 0.03],
        [0.0625, -0.115, 0.085, 0.03],
        [0.088, -0.115, 0.085, 0.03],
        [0.1135, -0.115, 0.085, 0.03],
        [0.139, -0.115, 0.085, 0.03],
        [0.16, -0.115, 0.075, 0.03],
        [0.168, -0.115, 0.0575, 0.03],
        [0.16, -0.115, 0.04, 0.03],
        [0.0625, 0.115, 0.03, 0.03],
        [0.088, 0.115, 0.03, 0.03],
        [0.1135, 0.115, 0.03, 0.03],
        [0.139, 0.115, 0.03, 0.03],
        [0.0625, 0.115, 0.085, 0.03],
        [0.088, 0.115, 0.085, 0.03],
        [0.1135, 0.115, 0.085, 0.03],
        [0.139, 0.115, 0.085, 0.03],
        [0.16, 0.115, 0.075, 0.03],
        [0.168, 0.115, 0.0575, 0.03],
        [0.16, 0.115, 0.04, 0.03],
    ],
    "shoulder_pan_link": [
        [0.0, 0.0, 0.0, 0.055],
        [0.025, -0.015, 0.035, 0.055],
        [0.05, -0.03, 0.06, 0.055],
        [0.12, -0.03, 0.06, 0.055],
    ],
    "shoulder_lift_link": [
        [0.025, 0.04, 0.025, 0.04],
        [-0.025, 0.04, -0.025, 0.04],
        [0.025, 0.04, -0.025, 0.04],
        [-0.025, 0.04, 0.025, 0.04],
        [0.08, 0.0, 0.0, 0.055],
        [0.11, 0.0, 0.0, 0.055],
        [0.14, 0.0, 0.0, 0.055],
    ],
    "upperarm_roll_link": [
        [-0.02, 0.0, 0.0, 0.055],
        [0.03, 0.0, 0.0, 0.055],
        [0.08, 0.0, 0.0, 0.055],
        [0.11, -0.045, 0.02, 0.03],
        [0.11, -0.045, -0.02, 0.03],
        [0.155, -0.045, 0.02, 0.03],
        [0.155, -0.045, -0.02, 0.03],
        [0.13, 0.0, 0.0, 0.055],
    ],
    "elbow_flex_link": [
        [0.02, 0.045, 0.02, 0.03],
        [0.02, 0.045, -0.02, 0.03],
        [-0.02, 0.045, 0.02, 0.03],
        [-0.02, 0.045, -0.02, 0.03],
        [0.08, 0.0, 0.0, 0.055],
        [0.14, 0.0, 0.0, 0.055],
    ],
    "forearm_roll_link": [
        [0.0, 0.0, 0.0, 0.055],
        [0.05, -0.06, 0.02, 0.03],
        [0.05, -0.06, -0.02, 0.03],
        [0.1, -0.06, 0.02, 0.03],
        [0.1, -0.06, -0.02, 0.03],
        [0.15, -0.06, 0.02, 0.03],
        [0.15, -0.06, -0.02, 0.03],
    ],
    "wrist_flex_link": [
        [0.0, 0.0, 0.0, 0.055],
        [0.06, 0.0, 0.0, 0.055],
        [0.02, 0.045, 0.02, 0.03],
        [0.02, 0.045, -0.02, 0.03],
        [-0.02, 0.045, 0.02, 0.03],
        [-0.02, 0.045, -0.02, 0.03],
    ],
    "wrist_roll_link": [
        [-0.03, 0.0, 0.0, 0.055],
        [0.0, 0.0, 0.0, 0.055],
    ],
    "gripper_link": [
        [-0.07, 0.02, 0.0, 0.05],
        [-0.07, -0.02, 0.0, 0.05],
        [-0.1, 0.02, 0.0, 0.05],
        [-0.1, -0.02, 0.0, 0.05],
    ],
    "r_gripper_finger_link": [
        [0.017, -0.0085, -0.005, 0.012],
        [0.017, -0.0085, 0.005, 0.012],
        [0.0, -0.0085, -0.005, 0.012],
        [0.0, -0.0085, 0.005, 0.012],
        [-0.017, -0.0085, -0.005, 0.012],
        [-0.017, -0.0085, 0.005, 0.012],
    ],
    "l_gripper_finger_link": [
        [0.017, 0.0085, -0.005, 0.012],
        [0.017, 0.0085, 0.005, 0.012],
        [0.0, 0.0085, -0.005, 0.012],
        [0.0, 0.0085, 0.005, 0.012],
        [-0.017, 0.0085, -0.005, 0.012],
        [-0.017, 0.0085, 0.005, 0.012],
    ],
}
