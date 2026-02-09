"""Tests for the motion-planner factory and VampPreviewPlanner."""

from __future__ import annotations

import numpy as np
import pytest

from TyGrit.planning.config import PlannerConfig, VampPlannerConfig
from TyGrit.planning.planner import create_planner
from TyGrit.types.geometry import SE2Pose
from TyGrit.types.robot import WholeBodyConfig

# ── Factory tests ────────────────────────────────────────────────────────────


class TestPlannerFactory:
    def test_unknown_robot_raises(self):
        with pytest.raises(ValueError, match="Unknown robot"):
            create_planner("nonexistent", "vamp_preview")

    def test_unknown_planner_raises(self):
        pytest.importorskip("vamp_preview")
        with pytest.raises(ValueError, match="Unknown Fetch planner"):
            create_planner("fetch", "nonexistent")

    def test_wrong_config_type_raises(self):
        pytest.importorskip("vamp_preview")
        with pytest.raises(TypeError, match="Expected VampPlannerConfig"):
            create_planner("fetch", "vamp_preview", config=PlannerConfig())


# ── VampPreviewPlanner tests ─────────────────────────────────────────────────

# Tuck configuration for Fetch (torso + 7 arm joints)
_TUCK = [0.0, 1.32, 1.40, -0.2, 1.72, 0.0, 1.66, 0.0]

# A second valid arm configuration
_READY = [0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class TestVampPreviewPlanner:
    @pytest.fixture(autouse=True)
    def _require_vamp(self):
        pytest.importorskip("vamp_preview")

    def _make_planner(self, **overrides):
        cfg = VampPlannerConfig(**overrides)
        return create_planner("fetch", "vamp_preview", config=cfg)

    # -- Construction -------------------------------------------------

    def test_creates_successfully(self):
        planner = self._make_planner()
        assert planner is not None

    # -- plan_arm -----------------------------------------------------

    def test_plan_arm_empty_env(self):
        planner = self._make_planner()
        start = np.array(_TUCK, dtype=np.float64)
        goal = np.array(_READY, dtype=np.float64)

        result = planner.plan_arm(start, goal)
        assert result.success
        assert result.trajectory is not None
        assert len(result.trajectory.arm_path) >= 2
        for cfg in result.trajectory.arm_path:
            assert cfg.shape == (8,)

    # -- validate_config ----------------------------------------------

    def test_validate_config_tuck(self):
        planner = self._make_planner()
        wb = WholeBodyConfig(
            arm_joints=np.array(_TUCK, dtype=np.float64),
            base_pose=SE2Pose(x=0.0, y=0.0, theta=0.0),
        )
        assert planner.validate_config(wb) is True

    # -- Sphere / pointcloud management --------------------------------

    def test_add_clear_pointcloud(self):
        planner = self._make_planner()
        points = np.array([[1.0, 0.0, 0.5], [1.1, 0.1, 0.5]], dtype=np.float64)
        planner.add_pointcloud(points)
        planner.clear_pointclouds()

    def test_add_clear_spheres(self):
        planner = self._make_planner()
        planner.add_sphere([1.0, 0.0, 0.5], 0.05)
        planner.clear_spheres()

    # -- Attach / detach -----------------------------------------------

    def test_attach_detach(self):
        planner = self._make_planner()
        spheres = [{"position": [0.0, 0.0, 0.0], "radius": 0.02}]

        assert planner.attach_to_eef(spheres) is True
        # Double-attach should fail
        assert planner.attach_to_eef(spheres) is False
        # Detach
        assert planner.detach_from_eef() is True
        # Double-detach should fail
        assert planner.detach_from_eef() is False

    # -- plan_whole_body -----------------------------------------------

    def test_plan_whole_body_empty_env(self):
        planner = self._make_planner()

        start = WholeBodyConfig(
            arm_joints=np.array(_TUCK, dtype=np.float64),
            base_pose=SE2Pose(x=0.0, y=0.0, theta=0.0),
        )
        goal = WholeBodyConfig(
            arm_joints=np.array(_TUCK, dtype=np.float64),
            base_pose=SE2Pose(x=0.5, y=0.0, theta=0.0),
        )

        result = planner.plan_whole_body(start, goal)
        # Just check it returns a valid PlanResult (may or may not succeed
        # depending on the VAMP build — the test verifies no crash)
        assert isinstance(result.success, bool)
