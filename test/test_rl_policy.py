"""Tests for RL policy, value network, and causal discovery."""

import pytest

torch = pytest.importorskip("torch")

from TyGrit.rl.config import TrainConfig, default_causal_matrix  # noqa: E402
from TyGrit.rl.policy import (  # noqa: E402
    FactoredPolicy,
    MultiChannelValue,
    NatureCNN,
    layer_init,
)


def _make_sample_obs(batch: int = 4, h: int = 128, w: int = 128):
    """Create a sample observation dict for network construction."""
    return {
        "rgb": torch.randint(0, 255, (batch, h, w, 6), dtype=torch.uint8),
        "depth": torch.randint(0, 1000, (batch, h, w, 2), dtype=torch.int16),
        "state": torch.randn(batch, 36),
    }


class TestLayerInit:
    def test_linear(self):
        layer = layer_init(torch.nn.Linear(10, 5))
        assert layer.weight.shape == (5, 10)

    def test_conv2d(self):
        layer = layer_init(torch.nn.Conv2d(3, 32, 8, 4))
        assert layer.weight.shape == (32, 3, 8, 8)


class TestNatureCNN:
    def test_output_shape(self):
        sample = _make_sample_obs(2)
        cnn = NatureCNN(sample, cnn_feature_dim=256, state_feature_dim=256)
        out = cnn(sample)
        assert out.shape == (2, 512)

    def test_output_dim_attribute(self):
        sample = _make_sample_obs(1)
        cnn = NatureCNN(sample)
        assert cnn.output_dim == 512


class TestFactoredPolicy:
    def test_get_action_shape(self):
        sample = _make_sample_obs(4)
        policy = FactoredPolicy(sample_obs=sample, action_dim=11)
        action, log_prob = policy.get_action(sample)
        assert action.shape == (4, 11)
        assert log_prob.shape == (4, 11)

    def test_per_dim_log_prob_not_summed(self):
        sample = _make_sample_obs(2)
        policy = FactoredPolicy(sample_obs=sample, action_dim=11)
        action, log_prob = policy.get_action(sample)
        assert log_prob.shape == (2, 11)

    def test_evaluate_actions(self):
        sample = _make_sample_obs(4)
        policy = FactoredPolicy(sample_obs=sample, action_dim=11)
        actions = torch.randn(4, 11)
        log_prob, entropy = policy.evaluate_actions(sample, actions)
        assert log_prob.shape == (4, 11)
        assert entropy.shape == (4, 11)
        assert (entropy > 0).all()

    def test_deterministic_action(self):
        sample = _make_sample_obs(1)
        policy = FactoredPolicy(sample_obs=sample, action_dim=11)
        a1, _ = policy.get_action(sample, deterministic=True)
        a2, _ = policy.get_action(sample, deterministic=True)
        assert torch.allclose(a1, a2)


class TestMultiChannelValue:
    def test_output_shape(self):
        sample = _make_sample_obs(8)
        vf = MultiChannelValue(sample_obs=sample, reward_channels=8)
        values = vf(sample)
        assert values.shape == (8, 8)

    def test_eight_channels(self):
        sample = _make_sample_obs(4)
        vf = MultiChannelValue(sample_obs=sample, reward_channels=8)
        values = vf(sample)
        assert values.shape[1] == 8


class TestCausalMatrix:
    def test_default_shape(self):
        B = default_causal_matrix()
        assert B.shape == (8, 13)

    def test_binary(self):
        B = default_causal_matrix()
        assert ((B == 0) | (B == 1)).all()

    def test_grasp_only_gripper(self):
        B = default_causal_matrix()
        # Row 7 (grasp) should only connect to gripper (dim 10)
        assert B[7, 10] == 1
        assert B[7, :10].sum() == 0

    def test_reach_includes_base_and_arm(self):
        B = default_causal_matrix()
        # Row 0 (reach): base (0,1) + some arm joints
        assert B[0, 0] == 1  # v
        assert B[0, 1] == 1  # w
        assert B[0, 3:6].sum() >= 2  # shoulder + upperarm

    def test_gaze_base_and_head(self):
        B = default_causal_matrix()
        # Row 6 (gaze): base v + head_pan + head_tilt (free head control)
        assert B[6, 0] == 1  # v
        assert B[6, 11] == 1  # head_pan
        assert B[6, 12] == 1  # head_tilt
        assert B[6, 2:11].sum() == 0  # no arm or gripper

    def test_base_col_base_and_head(self):
        B = default_causal_matrix()
        # Row 3 (base_col): base v,w + head_pan,head_tilt (head links can collide)
        assert B[3, 0] == 1  # v
        assert B[3, 1] == 1  # w
        assert B[3, 11] == 1  # head_pan
        assert B[3, 12] == 1  # head_tilt
        assert B[3, 2:11].sum() == 0  # no arm or gripper


class TestCausalAdvantage:
    def test_matmul_shape(self):
        """Core CausalMoMa operation: advantages @ B -> causal advantages."""
        B = default_causal_matrix()
        advantages = torch.randn(32, 8)  # (batch, reward_channels)
        causal_adv = advantages @ B  # (batch, action_dim)
        assert causal_adv.shape == (32, 13)

    def test_zero_advantage_zero_output(self):
        B = default_causal_matrix()
        advantages = torch.zeros(1, 8)
        causal_adv = advantages @ B
        assert torch.allclose(causal_adv, torch.zeros(1, 13))

    def test_sparsity_effect(self):
        """Grasp advantage should only affect gripper action."""
        B = default_causal_matrix()
        # Only grasp advantage is nonzero (channel 7)
        advantages = torch.zeros(1, 8)
        advantages[0, 7] = 1.0
        causal_adv = advantages @ B
        # Only gripper dim (10) should be nonzero
        assert causal_adv[0, 10] == 1.0
        assert causal_adv[0, :10].sum() == 0.0


class TestTrainConfig:
    def test_defaults(self):
        cfg = TrainConfig()
        assert cfg.num_envs == 64
        assert cfg.reward_channels == 8
        assert cfg.action_dim == 13

    def test_obs_mode_rgbd(self):
        cfg = TrainConfig()
        assert cfg.obs_mode == "rgbd"

    def test_cnn_dims(self):
        cfg = TrainConfig()
        assert cfg.cnn_feature_dim == 256
        assert cfg.state_feature_dim == 256
