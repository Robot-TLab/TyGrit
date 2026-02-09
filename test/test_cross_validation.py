"""Cross-validation tests for FK and IK solvers.

Independent implementations that agree on the same inputs are very likely correct.
These tests compare:
- NumPy FK vs IKFast FK (both compute gripper_link pose in base_link frame)
- IK→FK round-trips (IKFast and TracIK)
- IKFast vs TracIK reaching the same target
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from TyGrit.kinematics.fetch.constants import JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER
from TyGrit.kinematics.fetch.fk_numpy import forward_kinematics


def _random_arm_configs(rng: np.random.Generator, n: int) -> np.ndarray:
    """Sample n random 8-DOF configs within joint limits."""
    return rng.uniform(JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER, size=(n, 8))


def _load_urdf() -> str:
    """Load Fetch URDF string, or skip if not found."""
    paths = [
        os.path.join(
            os.path.dirname(__file__), "..", "resources", "fetch", "fetch.urdf"
        ),
        os.environ.get("FETCH_URDF_PATH", ""),
    ]
    for p in paths:
        if p and os.path.exists(p):
            with open(p) as f:
                return f.read()
    pytest.skip("Fetch URDF not found")


# ── FK cross-validation: NumPy vs IKFast ─────────────────────────────────────


class TestFKCrossValidation:
    """NumPy forward_kinematics(10-DOF) vs IKFast ee_forward_kinematics(8-DOF).

    Both should produce the same gripper_link pose in base_link frame.
    """

    @pytest.fixture(autouse=True)
    def _require_ikfast(self):
        pytest.importorskip("ikfast_fetch")

    def _compare(self, arm_config: np.ndarray, atol: float = 1e-6) -> None:
        from TyGrit.kinematics.fetch.fk_ikfast import ee_forward_kinematics

        # NumPy FK needs 10-DOF: 8 arm + 2 head (zeros)
        full_config = np.append(arm_config, [0.0, 0.0])
        T_numpy = forward_kinematics(full_config)["gripper_link"]
        T_ikfast = ee_forward_kinematics(arm_config)
        np.testing.assert_allclose(T_numpy, T_ikfast, atol=atol)

    def test_zero_config(self):
        self._compare(np.zeros(8))

    def test_random_configs(self):
        rng = np.random.default_rng(42)
        configs = _random_arm_configs(rng, 32)
        for i, cfg in enumerate(configs):
            self._compare(cfg)


# ── IK→FK round-trip ─────────────────────────────────────────────────────────


class TestIKFKRoundTrip:
    """IK(FK(q)) should recover a pose that matches the original EE pose."""

    @pytest.fixture(autouse=True)
    def _require_ikfast(self):
        pytest.importorskip("ikfast_fetch")

    def test_ikfast_round_trip(self):
        from TyGrit.kinematics.fetch.fk_ikfast import ee_forward_kinematics
        from TyGrit.kinematics.fetch.ikfast import IKFastSolver

        solver = IKFastSolver()
        rng = np.random.default_rng(42)
        configs = _random_arm_configs(rng, 32)
        solved = 0

        for cfg in configs:
            target = ee_forward_kinematics(cfg)
            solutions = solver.solve_all(target, free_params=[cfg[0], cfg[2]])
            if not solutions:
                continue
            # Pick solution nearest to original config
            dists = [np.linalg.norm(s - cfg) for s in solutions]
            recovered = solutions[int(np.argmin(dists))]
            solved += 1
            T_recovered = ee_forward_kinematics(recovered)
            np.testing.assert_allclose(T_recovered, target, atol=1e-6)

        assert solved > 0, "No IKFast round-trips succeeded"

    def test_trac_ik_round_trip(self):
        pytest.importorskip("pytracik")
        urdf = _load_urdf()

        from TyGrit.kinematics.fetch.fk_ikfast import ee_forward_kinematics
        from TyGrit.kinematics.fetch.ik import create_fetch_ik_solver

        solver = create_fetch_ik_solver("trac_base", urdf_string=urdf, epsilon=1e-6)
        rng = np.random.default_rng(42)
        configs = _random_arm_configs(rng, 32)
        solved = 0

        for cfg in configs:
            target = ee_forward_kinematics(cfg)
            try:
                recovered = solver.solve(target, seed=cfg)
            except ValueError:
                continue
            solved += 1
            T_recovered = ee_forward_kinematics(recovered)
            np.testing.assert_allclose(
                T_recovered[:3, 3],
                target[:3, 3],
                atol=1e-4,
            )
            np.testing.assert_allclose(
                T_recovered[:3, :3],
                target[:3, :3],
                atol=1e-4,
            )

        assert solved > 0, "No TracIK round-trips succeeded"


# ── IK cross-validation: IKFast vs TracIK ────────────────────────────────────


class TestIKCrossValidation:
    """Both IK solvers should reach the same target pose."""

    @pytest.fixture(autouse=True)
    def _require_both(self):
        pytest.importorskip("ikfast_fetch")
        pytest.importorskip("pytracik")

    def test_both_reach_target(self):
        urdf = _load_urdf()

        from TyGrit.kinematics.fetch.fk_ikfast import ee_forward_kinematics
        from TyGrit.kinematics.fetch.ik import create_fetch_ik_solver
        from TyGrit.kinematics.fetch.ikfast import IKFastSolver

        ikfast_solver = IKFastSolver()
        trac_solver = create_fetch_ik_solver(
            "trac_base", urdf_string=urdf, epsilon=1e-6
        )

        rng = np.random.default_rng(42)
        configs = _random_arm_configs(rng, 32)
        compared = 0

        for cfg in configs:
            target = ee_forward_kinematics(cfg)
            solutions = ikfast_solver.solve_all(target, free_params=[cfg[0], cfg[2]])
            if not solutions:
                continue
            dists = [np.linalg.norm(s - cfg) for s in solutions]
            ik_result = solutions[int(np.argmin(dists))]
            try:
                trac_result = trac_solver.solve(target, seed=cfg)
            except ValueError:
                continue
            compared += 1

            # IKFast (analytical) — should be very precise
            T_ikfast = ee_forward_kinematics(ik_result)
            np.testing.assert_allclose(T_ikfast, target, atol=1e-6)

            # TracIK (numerical) — slightly looser tolerance
            T_trac = ee_forward_kinematics(trac_result)
            np.testing.assert_allclose(T_trac[:3, 3], target[:3, 3], atol=1e-4)
            np.testing.assert_allclose(T_trac[:3, :3], target[:3, :3], atol=1e-4)

        assert compared > 0, "No configs solved by both IKFast and TracIK"
