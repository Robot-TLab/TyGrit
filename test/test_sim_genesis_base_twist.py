"""Default-env tests for Genesis holonomic base twist handling."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from TyGrit.types.robots import ActuatorCfg


@pytest.fixture
def genesis_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "genesis", SimpleNamespace())
    sys.modules.pop("TyGrit.sim.genesis", None)
    return importlib.import_module("TyGrit.sim.genesis")


def _make_handler(genesis_module, theta: float):
    handler = genesis_module.GenesisSimHandler.__new__(genesis_module.GenesisSimHandler)
    handler._scene = SimpleNamespace(dt=0.05)
    handler._joint_name_to_idx = {
        "root_x_axis_joint": 0,
        "root_y_axis_joint": 1,
        "root_z_rotation_joint": 2,
    }
    handler._robot_entity = Mock()
    handler._robot_entity.get_dofs_position.return_value = np.array(
        [0.0, 0.0, theta], dtype=np.float64
    )
    return handler


def _base_actuator() -> ActuatorCfg:
    return ActuatorCfg(
        name="base",
        joint_names=(
            "root_x_axis_joint",
            "root_y_axis_joint",
            "root_z_rotation_joint",
        ),
        control_mode="velocity",
        action_dim=2,
        kind="base_twist",
    )


def test_apply_base_twist_world_x_at_zero_yaw(genesis_module) -> None:
    handler = _make_handler(genesis_module, theta=0.0)
    handler._apply_base_twist(_base_actuator(), np.array([1.0, 0.0], dtype=np.float64))

    velocities, dof_indices = handler._robot_entity.control_dofs_velocity.call_args.args
    np.testing.assert_allclose(velocities, [1.0, 0.0, 0.0], atol=1e-7)
    assert dof_indices == [0, 1, 2]


def test_apply_base_twist_world_y_at_pi_over_two(genesis_module) -> None:
    handler = _make_handler(genesis_module, theta=np.pi / 2.0)
    handler._apply_base_twist(_base_actuator(), np.array([1.0, 0.0], dtype=np.float64))

    velocities, _ = handler._robot_entity.control_dofs_velocity.call_args.args
    np.testing.assert_allclose(velocities, [0.0, 1.0, 0.0], atol=1e-7)


def test_apply_base_twist_rotation_only(genesis_module) -> None:
    handler = _make_handler(genesis_module, theta=0.3)
    handler._apply_base_twist(_base_actuator(), np.array([0.0, 1.0], dtype=np.float64))

    velocities, _ = handler._robot_entity.control_dofs_velocity.call_args.args
    np.testing.assert_allclose(velocities, [0.0, 0.0, 1.0], atol=1e-7)


def test_apply_action_dispatches_base_twist(genesis_module) -> None:
    handler = genesis_module.GenesisSimHandler.__new__(genesis_module.GenesisSimHandler)
    handler._total_action_dim = 2
    handler._action_slices = {"base": slice(0, 2)}
    handler._robot_cfg = SimpleNamespace(
        actuators={"base": _base_actuator()}, cameras=()
    )
    handler._scene = Mock()
    handler._apply_base_twist = Mock()

    handler.apply_action(np.array([0.25, -0.5], dtype=np.float32))

    handler._apply_base_twist.assert_called_once()
    actuator, command = handler._apply_base_twist.call_args.args
    assert actuator.name == "base"
    np.testing.assert_allclose(command, [0.25, -0.5], atol=1e-7)
    handler._scene.step.assert_called_once_with()


def _gripper_actuator() -> ActuatorCfg:
    """Fetch's gripper: 1 scalar command driving 2 finger joints
    symmetrically via command_to_joint_mapping=(0, 0)."""
    return ActuatorCfg(
        name="gripper",
        joint_names=("r_finger", "l_finger"),
        control_mode="position",
        action_dim=1,
        command_to_joint_mapping=(0, 0),
    )


def _arm_actuator() -> ActuatorCfg:
    """Direct 1:1 joint command, no mapping needed."""
    return ActuatorCfg(
        name="arm",
        joint_names=("j1", "j2"),
        control_mode="velocity",
        action_dim=2,
    )


def test_apply_action_command_to_joint_mapping_broadcast(genesis_module) -> None:
    """When ``command_to_joint_mapping`` is set, the command index is fanned
    across joints exactly as the mapping says — not blindly broadcast."""
    handler = genesis_module.GenesisSimHandler.__new__(genesis_module.GenesisSimHandler)
    handler._total_action_dim = 1
    handler._action_slices = {"gripper": slice(0, 1)}
    handler._joint_name_to_idx = {"r_finger": 7, "l_finger": 8}
    handler._robot_cfg = SimpleNamespace(
        actuators={"gripper": _gripper_actuator()}, cameras=()
    )
    handler._scene = Mock()
    handler._robot_entity = Mock()

    handler.apply_action(np.array([0.42], dtype=np.float32))

    # Both fingers must receive the same scalar (0.42), via the
    # gripper actuator's position controller (control_dofs_position).
    handler._robot_entity.control_dofs_position.assert_called_once()
    values, dof_idxs = handler._robot_entity.control_dofs_position.call_args.args
    np.testing.assert_allclose(values, [0.42, 0.42], atol=1e-7)
    assert dof_idxs == [7, 8]


def _camera_spec_at_origin():
    """A CameraSpec with identity offset relative to its parent link."""
    from TyGrit.types.sensors import CameraSpec

    return CameraSpec(
        camera_id="head",
        parent_link="head_camera_link",
        position=(0.0, 0.0, 0.0),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )


def test_update_attached_cameras_follows_link(genesis_module) -> None:
    """Per-step camera pose update must use link world pose composed
    with the CameraSpec offset, not the construction-time identity
    we pass to ``scene.add_camera``."""
    handler = genesis_module.GenesisSimHandler.__new__(genesis_module.GenesisSimHandler)
    cam_spec = _camera_spec_at_origin()
    handler._robot_cfg = SimpleNamespace(actuators={}, cameras=(cam_spec,))
    cam_handle = Mock()
    handler._cameras = {"head": cam_handle}
    handler._robot_entity = Mock()
    head_link = Mock()
    head_link.get_pos.return_value = np.array([1.0, 2.0, 0.5])
    head_link.get_quat.return_value = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz identity
    handler._robot_entity.get_link.return_value = head_link

    handler._update_attached_cameras()

    handler._robot_entity.get_link.assert_called_once_with("head_camera_link")
    cam_handle.set_pose.assert_called_once()
    pos, quat = (
        cam_handle.set_pose.call_args.kwargs.values()
        if (cam_handle.set_pose.call_args.kwargs)
        else cam_handle.set_pose.call_args.args
    )
    np.testing.assert_allclose(pos, [1.0, 2.0, 0.5], atol=1e-7)
    # Identity link rotation × identity offset = identity (wxyz).
    np.testing.assert_allclose(quat, [1.0, 0.0, 0.0, 0.0], atol=1e-7)


def test_reset_to_scene_idx_same_idx_uses_fast_path(genesis_module) -> None:
    """Repeating the same scene_idx must call scene.reset(), not rebuild."""
    handler = genesis_module.GenesisSimHandler.__new__(genesis_module.GenesisSimHandler)
    handler._scenes = ("a", "b")  # only len() is used
    handler._active_scene_idx = 0
    handler._scene = Mock()
    # _build_scene shouldn't be called on the fast path; intercept it
    # by patching the bound method.
    handler._build_scene = Mock()

    handler.reset_to_scene_idx(0)

    handler._scene.reset.assert_called_once_with()
    handler._build_scene.assert_not_called()


def test_reset_to_scene_idx_different_idx_full_rebuild(genesis_module) -> None:
    """A different scene_idx must trigger _build_scene, not the fast path."""
    handler = genesis_module.GenesisSimHandler.__new__(genesis_module.GenesisSimHandler)
    handler._scenes = ("a", "b")
    handler._active_scene_idx = 0
    handler._scene = Mock()
    handler._build_scene = Mock()

    handler.reset_to_scene_idx(1)

    handler._build_scene.assert_called_once_with(1)
    handler._scene.reset.assert_not_called()


def test_update_attached_cameras_no_op_without_cameras(genesis_module) -> None:
    """A robot with no cameras must not query its links per step."""
    handler = genesis_module.GenesisSimHandler.__new__(genesis_module.GenesisSimHandler)
    handler._robot_cfg = SimpleNamespace(actuators={}, cameras=())
    handler._cameras = {}
    handler._robot_entity = Mock()

    handler._update_attached_cameras()

    handler._robot_entity.get_link.assert_not_called()


def test_apply_action_one_to_one_joint_command(genesis_module) -> None:
    """Without ``command_to_joint_mapping``, action[i] drives joint[i] directly."""
    handler = genesis_module.GenesisSimHandler.__new__(genesis_module.GenesisSimHandler)
    handler._total_action_dim = 2
    handler._action_slices = {"arm": slice(0, 2)}
    handler._joint_name_to_idx = {"j1": 0, "j2": 1}
    handler._robot_cfg = SimpleNamespace(actuators={"arm": _arm_actuator()}, cameras=())
    handler._scene = Mock()
    handler._robot_entity = Mock()

    handler.apply_action(np.array([0.1, -0.2], dtype=np.float32))

    handler._robot_entity.control_dofs_velocity.assert_called_once()
    values, dof_idxs = handler._robot_entity.control_dofs_velocity.call_args.args
    np.testing.assert_allclose(values, [0.1, -0.2], atol=1e-7)
    assert dof_idxs == [0, 1]
