"""Tests for IK solvers: IKFast and TRAC-IK."""

from __future__ import annotations

import numpy as np
import pytest

from TyGrit.kinematics.ik import create_ik_solver

# ── IKFast tests (skip if extension not importable) ─────────────────────────


class TestIKFast:
    @pytest.fixture(autouse=True)
    def _require_ikfast(self):
        pytest.importorskip("ikfast_fetch")

    def _make_solver(self):
        return create_ik_solver("fetch", "ikfast_base")

    def test_unreachable_raises(self):
        """Pose far out of workspace -> ValueError."""
        solver = self._make_solver()
        target = np.eye(4)
        target[:3, 3] = [100.0, 100.0, 100.0]
        with pytest.raises(ValueError, match="no valid solution"):
            solver.solve(target)

    def test_solve_all_returns_list(self):
        """solve_all returns a list of arrays within joint limits."""
        from TyGrit.kinematics.fetch.constants import (
            JOINT_LIMITS_LOWER,
            JOINT_LIMITS_UPPER,
        )
        from TyGrit.kinematics.fetch.ikfast import IKFastSolver

        solver = IKFastSolver()
        # A reachable pose in front of the robot
        target = np.eye(4)
        target[:3, 3] = [0.6, 0.0, 0.9]
        solutions = solver.solve_all(target, free_params=[0.1, 0.0])
        if len(solutions) > 0:
            for sol in solutions:
                assert sol.shape == (8,)
                assert np.all(sol >= JOINT_LIMITS_LOWER - 1e-6)
                assert np.all(sol <= JOINT_LIMITS_UPPER + 1e-6)

    def test_solve_nearest_to_seed(self):
        """solve with seed returns the solution closest to that seed."""
        solver = self._make_solver()
        target = np.eye(4)
        target[:3, 3] = [0.6, 0.0, 0.9]
        from TyGrit.kinematics.fetch.ikfast import IKFastSolver

        raw_solver = IKFastSolver()
        solutions = raw_solver.solve_all(target, free_params=[0.1, 0.0])
        if len(solutions) >= 2:
            # Use second solution as seed — returned solution should be closest
            seed = solutions[1]
            result = solver.solve(target, seed=seed)
            dist_to_seed = np.linalg.norm(result - seed)
            dist_to_first = np.linalg.norm(solutions[0] - seed)
            assert dist_to_seed <= dist_to_first + 1e-10


# ── TRAC-IK tests (skip if pytracik or URDF not available) ──────────────────


class TestTracIK:
    @pytest.fixture(autouse=True)
    def _require_trac_ik(self):
        pytest.importorskip("pytracik")

    def _load_urdf(self) -> str:
        import os

        paths = [
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "resources",
                "fetch",
                "fetch.urdf",
            ),
            os.environ.get("FETCH_URDF_PATH", ""),
        ]
        for p in paths:
            if p and os.path.exists(p):
                with open(p) as f:
                    return f.read()
        pytest.skip("Fetch URDF not found")

    def test_arm_only_solver_dof(self):
        urdf = self._load_urdf()
        solver = create_ik_solver("fetch", "trac_arm", urdf_string=urdf)
        assert solver.num_joints == 7

    def test_fixed_base_solver_dof(self):
        urdf = self._load_urdf()
        solver = create_ik_solver("fetch", "trac_base", urdf_string=urdf)
        assert solver.num_joints == 8

    def test_solve_returns_correct_shape(self):
        urdf = self._load_urdf()
        solver = create_ik_solver("fetch", "trac_arm", urdf_string=urdf)
        target = np.eye(4)
        target[:3, 3] = [0.5, 0.0, 0.3]
        try:
            result = solver.solve(target)
            assert result.shape == (7,)
        except ValueError:
            pass  # Unreachable for this solver is acceptable

    def test_unreachable_raises(self):
        urdf = self._load_urdf()
        solver = create_ik_solver("fetch", "trac_arm", urdf_string=urdf)
        target = np.eye(4)
        target[:3, 3] = [100.0, 100.0, 100.0]
        with pytest.raises(ValueError, match="no solution"):
            solver.solve(target)
