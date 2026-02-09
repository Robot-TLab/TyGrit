from pathlib import Path

import numpy as np
import torch
import trimesh
from grasp_gen.robot import load_control_points_core, load_default_gripper_config

# Fetch gripper dimensions (metres, from fetch_description URDF)
_FINGER_LENGTH = 0.07  # along approach (X in Fetch frame)
_FINGER_THICKNESS = 0.014  # along closing (Y in Fetch frame)
_FINGER_WIDTH = 0.038  # lateral (Z in Fetch frame)
_FINGER_OFFSET_Y = 0.065425  # centre-to-finger-link offset
_BASE_LENGTH = 0.04
_BASE_WIDTH = 0.12
_BASE_HEIGHT = 0.025


class GripperModel:
    def __init__(self, data_root_dir=None):
        # Programmatic box mesh — no external STL/OBJ required.
        base = trimesh.creation.box(
            extents=[_BASE_LENGTH, _BASE_WIDTH, _BASE_HEIGHT],
        )
        base.apply_translation([-_BASE_LENGTH / 2, 0.0, 0.0])

        finger_r = trimesh.creation.box(
            extents=[_FINGER_LENGTH, _FINGER_THICKNESS, _FINGER_WIDTH],
        )
        finger_r.apply_translation([0.0, -_FINGER_OFFSET_Y, 0.0])

        finger_l = trimesh.creation.box(
            extents=[_FINGER_LENGTH, _FINGER_THICKNESS, _FINGER_WIDTH],
        )
        finger_l.apply_translation([0.0, _FINGER_OFFSET_Y, 0.0])

        self.mesh = trimesh.util.concatenate([base, finger_r, finger_l])

    def get_gripper_collision_mesh(self):
        return self.mesh

    def get_gripper_visual_mesh(self):
        return self.mesh


def load_control_points() -> torch.Tensor:
    gripper_config = load_default_gripper_config(Path(__file__).stem)
    control_points = load_control_points_core(gripper_config)
    control_points = np.vstack([control_points, np.zeros(3)])
    control_points = np.hstack([control_points, np.ones([len(control_points), 1])])
    control_points = torch.from_numpy(control_points).float()
    return control_points.T


def load_control_points_for_visualization():
    gripper_config = load_default_gripper_config(Path(__file__).stem)
    control_points = load_control_points_core(gripper_config)

    mid_point = (control_points[0] + control_points[1]) / 2

    control_points = [
        control_points[-2],
        control_points[0],
        mid_point,
        [0, 0, 0],
        mid_point,
        control_points[1],
        control_points[-1],
    ]
    return [control_points]
