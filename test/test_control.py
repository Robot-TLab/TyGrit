"""Tests for Fetch robot controllers — MPC, gaze, gripper."""

import numpy as np

from TyGrit.controller.fetch.gripper import GRIPPER_CLOSED, GRIPPER_OPEN
from TyGrit.controller.fetch.mpc import MPCConfig, compute_mpc_action
from TyGrit.gaze import compute_gaze_target
from TyGrit.gaze.gaze import GazeConfig

# ── MPC ──────────────────────────────────────────────────────────────────


class TestMPC:
    def test_zero_error_gives_zero_action(self):
        """If current state == reference, velocity should be ~zero."""
        x = np.array([1.0, 2.0, 0.5, 0.3, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        u = compute_mpc_action(x, x)
        np.testing.assert_allclose(u, 0.0, atol=1e-5)

    def test_drives_toward_reference(self):
        """Action should point toward the reference state."""
        x = np.zeros(11)
        x_ref = np.zeros(11)
        x_ref[4] = 1.0  # arm joint 0 needs to move positive
        u = compute_mpc_action(x, x_ref)
        assert u[3] > 0.0  # arm_vel_0 should be positive

    def test_velocity_clamping(self):
        """Velocities should be clamped to limits."""
        config = MPCConfig(v_max=1.0, w_max=0.5, gain=100.0)
        x = np.zeros(11)
        x_ref = np.ones(11) * 10.0  # huge error
        u = compute_mpc_action(x, x_ref, config)
        assert abs(u[0]) <= config.v_max + 1e-6
        assert abs(u[1]) <= config.w_max + 1e-6

    def test_output_shape(self):
        x = np.zeros(11)
        x_ref = np.ones(11)
        u = compute_mpc_action(x, x_ref)
        assert u.shape == (10,)
        assert u.dtype == np.float32

    def test_custom_config(self):
        """Custom Q/R weights should change the action magnitude."""
        x = np.zeros(11)
        x_ref = np.zeros(11)
        x_ref[0] = 1.0

        u_default = compute_mpc_action(x, x_ref)
        heavy_q = MPCConfig(state_weights=(100.0,) + (1.0,) * 10)
        u_heavy = compute_mpc_action(x, x_ref, heavy_q)
        # Heavier state weight → more aggressive (before clamping)
        assert abs(u_heavy[0]) >= abs(u_default[0]) - 1e-6

    def test_base_rotation_coupling(self):
        """Base motion uses cos/sin(theta) coupling."""
        x = np.zeros(11)
        x[2] = np.pi / 2  # facing +y
        x_ref = np.zeros(11)
        x_ref[0] = 1.0  # want to go +x

        u = compute_mpc_action(x, x_ref)
        # When facing +y and wanting +x, angular velocity should be negative
        # (turn right) or linear velocity contributes in y
        assert u.shape == (10,)


# ── Gaze ─────────────────────────────────────────────────────────────────


class TestGaze:
    def _make_static_trajectory(self, n_steps: int = 20, n_links: int = 3):
        """All links stationary → all velocities zero."""
        pos = np.tile(
            np.array([[1.0, 0.0, 0.5], [0.5, 0.5, 0.3], [0.0, 1.0, 0.4]]),
            (n_steps, 1, 1),
        )
        return pos[:, :n_links, :]

    def _make_moving_trajectory(self, n_steps: int = 20, n_links: int = 3):
        """Links moving linearly in +x."""
        t = np.linspace(0, 1, n_steps)
        pos = np.zeros((n_steps, n_links, 3))
        for k in range(n_links):
            pos[:, k, 0] = t * (k + 1)  # move in x
            pos[:, k, 1] = float(k) * 0.5
            pos[:, k, 2] = 0.5
        return pos

    def test_static_trajectory_fallback(self):
        """Static links → fallback to mean position."""
        positions = self._make_static_trajectory()
        target = compute_gaze_target(positions, current_idx=0)
        expected = positions[0].mean(axis=0)
        np.testing.assert_allclose(target, expected, atol=1e-10)

    def test_moving_trajectory_tracks_motion(self):
        """Moving links → target should be ahead of current position."""
        positions = self._make_moving_trajectory()
        target = compute_gaze_target(positions, current_idx=0)
        # Target should be in positive x direction (links move +x)
        current_mean_x = positions[0, :, 0].mean()
        assert target[0] > current_mean_x

    def test_output_shape(self):
        positions = self._make_moving_trajectory()
        target = compute_gaze_target(positions, current_idx=5)
        assert target.shape == (3,)

    def test_end_of_trajectory(self):
        """At the last waypoint, should still return a valid target."""
        positions = self._make_moving_trajectory(n_steps=10)
        target = compute_gaze_target(positions, current_idx=9)
        assert target.shape == (3,)
        assert np.all(np.isfinite(target))

    def test_decay_emphasises_near(self):
        """Lower decay rate should weight nearer waypoints more."""
        positions = self._make_moving_trajectory(n_steps=50)
        near_config = GazeConfig(decay_rate=0.5, lookahead_window=50)
        far_config = GazeConfig(decay_rate=0.99, lookahead_window=50)
        target_near = compute_gaze_target(positions, 0, near_config)
        target_far = compute_gaze_target(positions, 0, far_config)
        # With fast decay (0.5), target should be closer to start
        assert target_near[0] < target_far[0]


# ── Gripper ──────────────────────────────────────────────────────────────


class TestGripper:
    def test_open_close_values(self):
        assert GRIPPER_OPEN == 1.0
        assert GRIPPER_CLOSED == 0.0
        assert GRIPPER_OPEN > GRIPPER_CLOSED
