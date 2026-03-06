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
    def test_zero_at_target(self):
        pos = torch.tensor([[1.0, 2.0, 3.0]])
        assert reach_reward(pos, pos).item() == pytest.approx(0.0)

    def test_negative_when_distant(self):
        ee = torch.tensor([[0.0, 0.0, 0.0]])
        target = torch.tensor([[1.0, 0.0, 0.0]])
        r = reach_reward(ee, target)
        assert r.item() == pytest.approx(-1.0)

    def test_batched(self):
        ee = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        target = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        r = reach_reward(ee, target)
        assert r.shape == (2,)
        assert r[0].item() == pytest.approx(-1.0)
        assert r[1].item() == pytest.approx(0.0)


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
        has_collision = torch.tensor([False, False, False])
        r = collision_reward(has_collision)
        assert r.shape == (3,)
        assert (r == 0.0).all()

    def test_collision(self):
        has_collision = torch.tensor([True, False, True])
        r = collision_reward(has_collision)
        assert r[0].item() == pytest.approx(-1.0)
        assert r[1].item() == pytest.approx(0.0)
        assert r[2].item() == pytest.approx(-1.0)


class TestGazeReward:
    def test_target_directly_ahead(self):
        target = torch.tensor([[1.0, 0.0, 0.0]])
        cam_pos = torch.tensor([[0.0, 0.0, 0.0]])
        cam_fwd = torch.tensor([[1.0, 0.0, 0.0]])
        r = gaze_reward(target, cam_pos, cam_fwd)
        assert r.item() == pytest.approx(0.2)

    def test_target_behind(self):
        target = torch.tensor([[-1.0, 0.0, 0.0]])
        cam_pos = torch.tensor([[0.0, 0.0, 0.0]])
        cam_fwd = torch.tensor([[1.0, 0.0, 0.0]])
        r = gaze_reward(target, cam_pos, cam_fwd)
        assert r.item() == pytest.approx(0.0)


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
