"""Structural tests for :class:`TyGrit.sim.base.SimHandler`.

These verify the Protocol shape *without* importing any sim SDK, so
they run in the default pixi env. Concrete per-sim conformance tests
live alongside their sim's env (world / genesis / isaacsim) and are
skipped when that sim isn't available.
"""

from __future__ import annotations

import inspect

import numpy as np

from TyGrit.sim.base import SimHandler


class _DummyHandler:
    """Minimal structural implementation used only to confirm the
    Protocol shape is what we think it is."""

    def __init__(self, robot_cfg) -> None:
        self._cfg = robot_cfg

    @property
    def robot_cfg(self):
        return self._cfg

    @property
    def num_envs(self) -> int:
        return 1

    @property
    def total_action_dim(self) -> int:
        return 0

    @property
    def action_slices(self):
        return {}

    @property
    def joint_name_to_idx(self):
        return {}

    def get_qpos(self, env_idx: int = 0):
        return np.zeros(0, dtype=np.float64)

    def get_link_pose(self, link_name: str, env_idx: int = 0):
        return np.eye(4, dtype=np.float64)

    def get_camera(self, camera_id: str, env_idx: int = 0):
        return (
            np.zeros((0, 0, 3), dtype=np.uint8),
            np.zeros((0, 0), dtype=np.float32),
            None,
        )

    def get_intrinsics(self, camera_id: str):
        return np.eye(3, dtype=np.float64)

    def apply_action(self, action):
        return None

    def reset_to_scene_idx(self, idx: int, *, seed=None):
        return None

    def set_joint_positions(self, positions, *, env_idx: int = 0):
        return None

    def set_base_pose(self, x, y, theta, *, env_idx: int = 0):
        return None

    def get_navigable_positions(self):
        return []

    def render(self):
        return None

    def close(self):
        return None


def test_dummy_conforms_to_protocol() -> None:
    from TyGrit.robots.fetch import FETCH_CFG

    h = _DummyHandler(FETCH_CFG)
    assert isinstance(h, SimHandler)


def test_protocol_method_surface() -> None:
    # Defensive: if someone deletes a Protocol member, this test
    # catches the drift before a concrete handler silently diverges.
    expected = {
        "robot_cfg",
        "num_envs",
        "total_action_dim",
        "action_slices",
        "joint_name_to_idx",
        "get_qpos",
        "get_link_pose",
        "get_camera",
        "get_intrinsics",
        "apply_action",
        "reset_to_scene_idx",
        "set_joint_positions",
        "set_base_pose",
        "get_navigable_positions",
        "render",
        "close",
    }
    members = {
        name for name, _ in inspect.getmembers(SimHandler) if not name.startswith("_")
    }
    missing = expected - members
    unexpected = members - expected
    assert not missing, f"SimHandler is missing required members: {missing}"
    assert not unexpected, f"SimHandler has unexpected members: {unexpected}"


def test_isaac_sim_skeleton_conforms() -> None:
    # Importable in default env; no Isaac SDK needed to check shape.
    from TyGrit.sim.isaac_sim import IsaacSimSimHandler

    expected_methods = {
        "get_qpos",
        "get_link_pose",
        "get_camera",
        "get_intrinsics",
        "apply_action",
        "reset_to_scene_idx",
        "set_joint_positions",
        "set_base_pose",
        "get_navigable_positions",
        "render",
        "close",
    }
    for name in expected_methods:
        assert callable(
            getattr(IsaacSimSimHandler, name)
        ), f"IsaacSimSimHandler missing method {name}"
