"""Tests for CausalMoMa 8-channel RL reward functions."""

import pytest

torch = pytest.importorskip("torch")

from TyGrit.rl.rewards import (  # noqa: E402
    collision_reward,
    ee_local_position_reward,
    ee_orientation_reward,
    gaze_reward,
    grasp_reward,
    reach_reward,
)


class TestReachReward:
    def test_no_progress(self):
        pos = torch.tensor([[1.0, 2.0, 3.0]])
        target = torch.tensor([[2.0, 2.0, 3.0]])
        prev_dist = torch.tensor([1.0])
        r, d = reach_reward(pos, target, prev_dist)
        assert r.item() == pytest.approx(0.0)
        assert d.item() == pytest.approx(1.0)

    def test_getting_closer(self):
        ee = torch.tensor([[0.5, 0.0, 0.0]])
        target = torch.tensor([[1.0, 0.0, 0.0]])
        prev_dist = torch.tensor([1.0])  # was 1m away
        # dist=0.5 crosses goal_dist_tol=0.55 → potential 0.5 + bonus 10
        r, d = reach_reward(ee, target, prev_dist)
        assert r.item() == pytest.approx(10.5)
        assert d.item() == pytest.approx(0.5)

    def test_moving_away(self):
        ee = torch.tensor([[0.0, 0.0, 0.0]])
        target = torch.tensor([[1.0, 0.0, 0.0]])
        prev_dist = torch.tensor([0.5])  # was 0.5m away
        r, d = reach_reward(ee, target, prev_dist)
        assert r.item() == pytest.approx(-0.5)  # moved 0.5m away

    def test_goal_bonus_triggered(self):
        """Sparse +10 bonus when crossing goal_dist_tol threshold."""
        ee = torch.tensor([[0.8, 0.0, 0.0]])
        target = torch.tensor([[1.0, 0.0, 0.0]])  # dist = 0.2 < 0.55
        prev_dist = torch.tensor([0.6])  # was >= 0.55, now < 0.55
        r, d = reach_reward(ee, target, prev_dist, goal_bonus=10.0, goal_dist_tol=0.55)
        # potential = 0.6 - 0.2 = 0.4, plus bonus = 10.0
        assert r.item() == pytest.approx(10.4)
        assert d.item() == pytest.approx(0.2)

    def test_goal_bonus_not_retriggered(self):
        """Bonus only fires on threshold crossing, not every step below."""
        ee = torch.tensor([[0.9, 0.0, 0.0]])
        target = torch.tensor([[1.0, 0.0, 0.0]])  # dist = 0.1
        prev_dist = torch.tensor([0.2])  # already below 0.55
        r, d = reach_reward(ee, target, prev_dist, goal_bonus=10.0, goal_dist_tol=0.55)
        # potential = 0.2 - 0.1 = 0.1, no bonus (was already inside)
        assert r.item() == pytest.approx(0.1)

    def test_goal_bonus_default(self):
        """Default goal bonus fires with default parameters."""
        ee = torch.tensor([[0.9, 0.0, 0.0]])
        target = torch.tensor([[1.0, 0.0, 0.0]])  # dist = 0.1 < 0.55
        prev_dist = torch.tensor([0.6])  # was >= 0.55
        r, d = reach_reward(ee, target, prev_dist)
        # potential = 0.6 - 0.1 = 0.5, plus default bonus = 10.0
        assert r.item() == pytest.approx(10.5)


class TestEEOrientationReward:
    def test_aligned(self):
        fwd = torch.tensor([[0.0, 0.0, -1.0]])
        target = torch.tensor([[0.0, 0.0, -1.0]])
        r = ee_orientation_reward(fwd, target)
        assert r.item() == pytest.approx(0.0)

    def test_opposite(self):
        fwd = torch.tensor([[0.0, 0.0, 1.0]])
        target = torch.tensor([[0.0, 0.0, -1.0]])
        r = ee_orientation_reward(fwd, target)
        assert r.item() == pytest.approx(-1.0)

    def test_range(self):
        fwd = torch.tensor([[1.0, 0.0, 0.0]])
        target = torch.tensor([[0.0, 0.0, -1.0]])
        r = ee_orientation_reward(fwd, target)
        assert -1.0 <= r.item() <= 0.0


class TestEELocalPositionReward:
    def test_same_height(self):
        ee = torch.tensor([[0.0, 0.0, 1.0]])
        target = torch.tensor([[1.0, 1.0, 1.0]])
        r = ee_local_position_reward(ee, target)
        assert r.item() == pytest.approx(0.2)  # offset

    def test_different_height(self):
        ee = torch.tensor([[0.0, 0.0, 0.0]])
        target = torch.tensor([[0.0, 0.0, 1.0]])
        r = ee_local_position_reward(ee, target)
        assert r.item() < 0.2


class TestCollisionReward:
    def test_no_collision(self):
        in_collision = torch.tensor([False, False, False])
        r = collision_reward(in_collision)
        assert r.shape == (3,)
        assert (r == 0.0).all()

    def test_collision(self):
        in_collision = torch.tensor([False, True, True])
        r = collision_reward(in_collision)
        assert r[0].item() == pytest.approx(0.0)
        assert r[1].item() == pytest.approx(-1.0)
        assert r[2].item() == pytest.approx(-1.0)

    def test_all_colliding(self):
        in_collision = torch.tensor([True, True])
        r = collision_reward(in_collision)
        assert (r == -1.0).all()


class TestGazeReward:
    def test_target_ahead_and_close(self):
        target = torch.tensor([[0.5, 0.0, 0.0]])
        cam_pos = torch.tensor([[0.0, 0.0, 0.0]])
        cam_fwd = torch.tensor([[1.0, 0.0, 0.0]])
        ee_pos = torch.tensor([[0.3, 0.0, 0.0]])  # 0.2m from target
        r = gaze_reward(target, cam_pos, cam_fwd, ee_pos)
        assert r.item() == pytest.approx(0.2 * 0.8)  # weight = 1 - 0.2/1.0

    def test_target_ahead_but_far(self):
        target = torch.tensor([[5.0, 0.0, 0.0]])
        cam_pos = torch.tensor([[0.0, 0.0, 0.0]])
        cam_fwd = torch.tensor([[1.0, 0.0, 0.0]])
        ee_pos = torch.tensor([[0.0, 0.0, 0.0]])  # 5m from target
        r = gaze_reward(target, cam_pos, cam_fwd, ee_pos)
        assert r.item() == pytest.approx(0.0)  # too far, weight=0

    def test_target_behind(self):
        target = torch.tensor([[-1.0, 0.0, 0.0]])
        cam_pos = torch.tensor([[0.0, 0.0, 0.0]])
        cam_fwd = torch.tensor([[1.0, 0.0, 0.0]])
        ee_pos = torch.tensor([[-0.9, 0.0, 0.0]])  # close but behind camera
        r = gaze_reward(target, cam_pos, cam_fwd, ee_pos)
        assert r.item() == pytest.approx(0.0)  # not in FOV


class TestGraspReward:
    def test_closing_near_target(self):
        gripper_action = torch.tensor([0.5])
        ee = torch.tensor([[0.0, 0.0, 0.0]])
        target = torch.tensor([[0.0, 0.0, 0.0]])
        r = grasp_reward(gripper_action, ee, target, height_threshold=0.1)
        assert r.item() > 0  # should be positive

    def test_closing_far_from_target(self):
        gripper_action = torch.tensor([0.5])
        ee = torch.tensor([[0.0, 0.0, 0.0]])
        target = torch.tensor([[1.0, 0.0, 0.0]])
        r = grasp_reward(gripper_action, ee, target, height_threshold=0.1)
        assert r.item() < 0  # should be negative

    def test_opening_far_from_target(self):
        gripper_action = torch.tensor([-0.5])
        ee = torch.tensor([[0.0, 0.0, 0.0]])
        target = torch.tensor([[1.0, 0.0, 0.0]])
        r = grasp_reward(gripper_action, ee, target, height_threshold=0.1)
        assert r.item() > 0  # opening far away is good

    def test_range(self):
        gripper_action = torch.tensor([1.0])
        ee = torch.tensor([[0.0, 0.0, 0.0]])
        target = torch.tensor([[0.0, 0.0, 0.0]])
        r = grasp_reward(gripper_action, ee, target, height_threshold=0.1)
        assert -0.2 - 1e-6 <= r.item() <= 0.2 + 1e-6
