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


def test_isaac_sim_no_notimplementederror() -> None:
    """§7.4 — every NotImplementedError raise removed from the
    isaac_sim module body. Real Isaac Lab calls now live there."""
    from pathlib import Path

    src = Path("TyGrit/sim/isaac_sim.py").read_text()
    assert "raise NotImplementedError" not in src, (
        "TyGrit/sim/isaac_sim.py still has `raise NotImplementedError` "
        "stubs — every method must call the real Isaac Lab API."
    )


# ── SimHandlerVec Protocol conformance ────────────────────────────────


class _DummyHandlerVec:
    """Structural conformance check for :class:`SimHandlerVec`.

    The Protocol requires torch tensors at runtime, but the structural
    membership check goes through method names + signatures only — torch
    is never imported here. That keeps the test runnable in the default
    pixi env.
    """

    def __init__(self, robot_cfg, num_envs: int, device: str) -> None:
        self._cfg = robot_cfg
        self._num_envs = num_envs
        self._device = device

    @property
    def robot_cfg(self):
        return self._cfg

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def device(self) -> str:
        return self._device

    @property
    def total_action_dim(self) -> int:
        return 0

    @property
    def action_slices(self):
        return {}

    @property
    def joint_name_to_idx(self):
        return {}

    def get_qpos(self):
        return None  # would be torch.Tensor at runtime

    def get_link_pose(self, link_name: str):
        return None

    def get_camera(self, camera_id: str):
        return (None, None, None)

    def get_intrinsics(self, camera_id: str):
        return np.eye(3, dtype=np.float64)

    def apply_action(self, action):
        return None

    def reset_to_scene_idx(self, idxs, *, seed=None):
        return None

    def set_joint_positions(self, positions, *, env_ids=None):
        return None

    def set_base_pose(self, xy_theta, *, env_ids=None):
        return None

    def get_navigable_positions(self):
        return []

    def render(self):
        return None

    def close(self):
        return None


def test_dummy_vec_conforms_to_vec_protocol() -> None:
    from TyGrit.robots.fetch import FETCH_CFG
    from TyGrit.sim.base import SimHandlerVec

    h = _DummyHandlerVec(FETCH_CFG, num_envs=4, device="cpu")
    assert isinstance(h, SimHandlerVec)


def test_vec_protocol_method_surface() -> None:
    from TyGrit.sim.base import SimHandlerVec

    expected = {
        "robot_cfg",
        "num_envs",
        "device",
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
        name
        for name, _ in inspect.getmembers(SimHandlerVec)
        if not name.startswith("_")
    }
    missing = expected - members
    unexpected = members - expected
    assert not missing, f"SimHandlerVec is missing required members: {missing}"
    assert not unexpected, f"SimHandlerVec has unexpected members: {unexpected}"


# ── create_sim_handler factory ────────────────────────────────────────


def test_create_sim_handler_unknown_sim_raises() -> None:
    """Factory rejects unknown sim names with a clear message."""
    import pytest

    from TyGrit.robots.fetch import FETCH_CFG
    from TyGrit.sim import create_sim_handler

    with pytest.raises(ValueError, match="unknown sim_name"):
        create_sim_handler("not_a_real_sim", FETCH_CFG, [])


def test_create_sim_handler_exposed_from_package() -> None:
    """``from TyGrit.sim import create_sim_handler`` must work in the
    default pixi env (no sim SDK present)."""
    from TyGrit.sim import SimHandler, SimHandlerVec, create_sim_handler

    assert callable(create_sim_handler)
    assert SimHandler is not None
    assert SimHandlerVec is not None
