"""Tests for the envs/fetch factory + create_sim_handler routing.

Covers the cleanup work landed under §7.1 / §7.2 / §7.3 of the
multi-sim refactor:

* ``FetchEnvConfig`` no longer carries ManiSkill-specific fields
  (``obs_mode`` / ``control_mode`` / ``render_mode``); they live in
  the opaque ``sim_opts`` mapping.
* ``FetchRobot.create`` routes through
  :func:`TyGrit.sim.create_sim_handler` — no per-sim glue file.
* ``envs/fetch/`` ships no module whose filename contains a sim name
  on the single-env path (after maniskill.py was deleted); the only
  remaining exception is ``maniskill_vec.py`` which lands with §7.6.

These checks run in the default pixi env (no sim SDK required).
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest


def test_fetch_env_config_has_no_maniskill_fields() -> None:
    """§7.3 — obs_mode/control_mode/render_mode moved to sim_opts."""
    from TyGrit.envs.fetch.config import FetchEnvConfig

    field_names = {f.name for f in FetchEnvConfig.__dataclass_fields__.values()}
    forbidden = {"obs_mode", "control_mode", "render_mode"}
    leaked = field_names & forbidden
    assert not leaked, (
        f"FetchEnvConfig still carries ManiSkill-specific fields {leaked}; "
        "they should live in sim_opts (§7.3)."
    )
    assert "sim_opts" in field_names


def test_fetch_env_config_default_sim_opts() -> None:
    """Defaults preserve the previous ManiSkill behaviour."""
    from TyGrit.envs.fetch.config import FetchEnvConfig

    cfg = FetchEnvConfig()
    assert cfg.sim_opts.get("obs_mode") == "rgb+depth+state+segmentation"
    assert cfg.sim_opts.get("control_mode") == "pd_joint_vel"
    assert cfg.sim_opts.get("render_mode") == "human"


def test_fetch_env_config_custom_sim_opts() -> None:
    """Callers can override sim_opts wholesale (per-sim config sink)."""
    from TyGrit.envs.fetch.config import FetchEnvConfig

    cfg = FetchEnvConfig(
        sim_opts={
            "obs_mode": "rgbd",
            "control_mode": "pd_joint_delta_pos",
            "render_mode": None,
        }
    )
    assert cfg.sim_opts["obs_mode"] == "rgbd"
    assert cfg.sim_opts["render_mode"] is None


def test_fetch_robot_create_unknown_backend_raises() -> None:
    """Factory rejects unknown backends (cf. create_sim_handler)."""
    from TyGrit.envs.fetch import FetchEnvConfig, FetchRobot

    cfg = FetchEnvConfig(backend="not_a_real_backend")
    # Either ValueError ("unknown backend") or NotImplementedError
    # ("vec path missing") is acceptable; both are explicit failures.
    with pytest.raises((ValueError, NotImplementedError)):
        FetchRobot.create(cfg)


def test_fetch_robot_create_signature_drops_mpc_config() -> None:
    """§7.2 — MPC tuning moved to TyGrit.controller.fetch.trajectory.

    FetchRobot.create no longer accepts mpc_config; the trajectory
    executor takes it directly.
    """
    from TyGrit.envs.fetch import FetchRobot

    sig = inspect.signature(FetchRobot.create)
    assert "mpc_config" not in sig.parameters


def test_envs_fetch_no_maniskill_top_level_module() -> None:
    """§7.1 — TyGrit/envs/fetch/maniskill.py was deleted; the legacy
    glue is now routed by FetchRobot.create + create_sim_handler.
    """
    p = Path("TyGrit/envs/fetch/maniskill.py")
    assert not p.exists(), (
        f"{p} still exists; FetchRobot.create should route via "
        "TyGrit.sim.create_sim_handler instead of importing per-sim glue."
    )


def test_envs_fetch_core_no_mpc_or_ik_imports() -> None:
    """§7.2 — envs/fetch/core.py is a sensor/actuation adapter only."""
    src = Path("TyGrit/envs/fetch/core.py").read_text()
    forbidden = ("compute_mpc_action", "forward_kinematics", "MPCConfig")
    for token in forbidden:
        # Allow mention in module docstrings only — checked by absence
        # in the import block.
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")) and token in stripped:
                pytest.fail(
                    f"core.py still imports {token!r}: {line}; "
                    "trajectory tracking + IK should live in "
                    "TyGrit.controller.fetch.trajectory / TyGrit.gaze.fetch_head."
                )


def test_robot_base_protocol_is_thin() -> None:
    """RobotBase is a sensor/actuation contract — no trajectory or
    look_at methods (those live above the env layer)."""
    from TyGrit.envs.base import RobotBase

    members = {
        name for name, _ in inspect.getmembers(RobotBase) if not name.startswith("_")
    }
    forbidden = {
        "execute_trajectory",
        "start_trajectory",
        "stop_motion",
        "is_motion_done",
        "look_at",
    }
    leaked = members & forbidden
    assert not leaked, (
        f"RobotBase Protocol still declares {leaked}; trajectory + active-"
        "perception methods belong above the env layer."
    )
