"""Tests for TyGrit.config — SystemConfig and TOML loader."""

from __future__ import annotations

from pathlib import Path

from TyGrit.config import SystemConfig, load_config
from TyGrit.scene.config import SceneConfig


class TestSystemConfig:
    def test_defaults(self):
        cfg = SystemConfig()
        assert cfg.env.robot == "fetch"
        assert cfg.env.backend == "maniskill"
        assert cfg.scene.ground_z_threshold == 0.3
        assert cfg.gaze.lookahead_window == 80
        assert cfg.planner.timeout == 5.0
        assert cfg.mpc.gain == 2.5
        assert cfg.scheduler.steps_per_iteration == 10

    def test_scene_config_custom(self):
        sc = SceneConfig(ground_z_threshold=0.5, merge_radius=0.05)
        assert sc.ground_z_threshold == 0.5
        assert sc.merge_radius == 0.05


class TestLoadConfig:
    def test_load_full(self, tmp_path: Path):
        toml = tmp_path / "cfg.toml"
        toml.write_text("""\
[env]
robot = "fetch"
backend = "ros"

[scene]
ground_z_threshold = 0.5

[gaze]
lookahead_window = 40

[mpc]
gain = 3.0

[scheduler]
steps_per_iteration = 20

[planner]
timeout = 10.0
""")
        cfg = load_config(toml)
        assert cfg.env.robot == "fetch"
        assert cfg.env.backend == "ros"
        assert cfg.scene.ground_z_threshold == 0.5
        assert cfg.gaze.lookahead_window == 40
        assert cfg.mpc.gain == 3.0
        assert cfg.scheduler.steps_per_iteration == 20
        assert cfg.planner.timeout == 10.0

    def test_partial_override(self, tmp_path: Path):
        """Only override one section; others keep defaults."""
        toml = tmp_path / "partial.toml"
        toml.write_text("""\
[mpc]
gain = 1.0
""")
        cfg = load_config(toml)
        assert cfg.mpc.gain == 1.0
        # Defaults preserved
        assert cfg.env.robot == "fetch"
        assert cfg.scene.ground_z_threshold == 0.3
        assert cfg.scheduler.steps_per_iteration == 10

    def test_empty_toml(self, tmp_path: Path):
        """An empty TOML file produces all defaults."""
        toml = tmp_path / "empty.toml"
        toml.write_text("")
        cfg = load_config(toml)
        default = SystemConfig()
        assert cfg.env == default.env
        assert cfg.scene == default.scene
        assert cfg.gaze == default.gaze
        assert cfg.planner == default.planner
        assert cfg.mpc == default.mpc
        assert cfg.scheduler == default.scheduler
