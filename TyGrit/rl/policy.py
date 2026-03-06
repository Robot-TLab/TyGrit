"""Factored policy and value networks for CausalMoMa with visual input.

Architecture based on ManiSkill's CleanRL PPO-RGB baseline:
- NatureCNN encodes RGB+depth images
- MLP encodes proprioceptive state
- Features concatenated for actor and critic heads

The policy outputs per-dimension log probabilities (not summed) so that
FPPO can compute per-action-dimension importance ratios.  The value
network outputs one value per reward channel.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal


def layer_init(
    layer: nn.Linear | nn.Conv2d,
    std: float = float(np.sqrt(2)),
    bias_const: float = 0.0,
) -> nn.Linear | nn.Conv2d:
    """Orthogonal weight init (from ManiSkill/CleanRL baseline)."""
    nn.init.orthogonal_(layer.weight, gain=float(std))
    nn.init.constant_(layer.bias, bias_const)
    return layer


class NatureCNN(nn.Module):
    """Visual + proprioceptive encoder.

    Encodes RGB-D images with a 3-layer CNN and proprioceptive state
    with a single linear layer, then concatenates features.

    Parameters
    ----------
    sample_obs : dict[str, Tensor]
        Sample observation dict with keys ``"rgb"``, ``"depth"``, ``"state"``.
        Used to infer input dimensions.
    cnn_feature_dim : int
        Output dimension of the CNN branch.
    state_feature_dim : int
        Output dimension of the state MLP branch.
    """

    def __init__(
        self,
        sample_obs: dict[str, Tensor],
        cnn_feature_dim: int = 256,
        state_feature_dim: int = 256,
    ) -> None:
        super().__init__()

        # Infer image channels: rgb (H,W,C_rgb) + depth (H,W,C_depth)
        rgb = sample_obs["rgb"]  # (N, H, W, C_rgb)
        depth = sample_obs["depth"]  # (N, H, W, C_depth)
        in_channels = rgb.shape[-1] + depth.shape[-1]

        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute flattened CNN output size
        with torch.no_grad():
            dummy_img = torch.zeros(1, in_channels, rgb.shape[1], rgb.shape[2])
            n_flat = self.cnn(dummy_img).shape[1]

        self.cnn_linear = nn.Sequential(
            layer_init(nn.Linear(n_flat, cnn_feature_dim)),
            nn.ReLU(),
        )

        # State encoder
        state_dim = sample_obs["state"].shape[-1]
        self.state_linear = nn.Sequential(
            layer_init(nn.Linear(state_dim, state_feature_dim)),
            nn.ReLU(),
        )

        self.output_dim = cnn_feature_dim + state_feature_dim

    def forward(self, obs: dict[str, Tensor]) -> Tensor:
        """Encode dict observation to feature vector.

        Args:
            obs: Dict with ``"rgb"`` (N,H,W,C), ``"depth"`` (N,H,W,C), ``"state"`` (N,D).

        Returns:
            ``(N, output_dim)`` feature vector.
        """
        rgb = obs["rgb"].float() / 255.0  # (N, H, W, C_rgb)
        depth = obs["depth"].float() / 1000.0  # (N, H, W, C_depth) — mm to m

        # Concat along channel dim, then permute to (N, C, H, W) for conv
        img = torch.cat([rgb, depth], dim=-1)  # (N, H, W, C_total)
        img = img.permute(0, 3, 1, 2)  # (N, C_total, H, W)

        vis_features = self.cnn_linear(self.cnn(img))  # (N, cnn_dim)
        state_features = self.state_linear(obs["state"].float())  # (N, state_dim)

        return torch.cat([vis_features, state_features], dim=1)


class FactoredPolicy(nn.Module):
    """Gaussian policy with per-dimension log probabilities.

    Uses NatureCNN encoder for visual+state input.

    Parameters
    ----------
    sample_obs : dict[str, Tensor]
        Sample observation dict (for NatureCNN dimension inference).
    action_dim : int
        Action dimensionality.
    cnn_feature_dim : int
        CNN branch output dimension.
    state_feature_dim : int
        State branch output dimension.
    """

    def __init__(
        self,
        sample_obs: dict[str, Tensor],
        action_dim: int,
        cnn_feature_dim: int = 256,
        state_feature_dim: int = 256,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim

        self.feature_net = NatureCNN(
            sample_obs,
            cnn_feature_dim=cnn_feature_dim,
            state_feature_dim=state_feature_dim,
        )

        feat_dim = self.feature_net.output_dim
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(feat_dim, feat_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(feat_dim, action_dim), std=0.01 * np.sqrt(2)),
        )
        # Per-dimension learnable log_std (ManiSkill default: -0.5)
        self.actor_logstd = nn.Parameter(-0.5 * torch.ones(action_dim))

    def forward(self, obs: dict[str, Tensor]) -> Normal:
        """Return action distribution.

        Args:
            obs: Dict observation with rgb, depth, state.

        Returns:
            ``Normal`` distribution with batch shape ``(B, action_dim)``.
        """
        features = self.feature_net(obs)
        mean = self.actor_mean(features)
        std = torch.exp(self.actor_logstd).expand_as(mean)
        return Normal(mean, std)

    def get_action(
        self, obs: dict[str, Tensor], deterministic: bool = False
    ) -> tuple[Tensor, Tensor]:
        """Sample action and return ``(action, per_dim_log_prob)``.

        Args:
            obs: Dict observation.
            deterministic: Use mean instead of sampling.

        Returns:
            action: ``(B, action_dim)``
            log_prob: ``(B, action_dim)`` -- per-dimension, NOT summed.
        """
        dist = self.forward(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action)  # (B, action_dim)
        return action, log_prob

    def evaluate_actions(
        self, obs: dict[str, Tensor], actions: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Evaluate given actions under current policy.

        Args:
            obs: Dict observation.
            actions: ``(B, action_dim)``

        Returns:
            log_prob: ``(B, action_dim)`` per-dimension log probabilities.
            entropy: ``(B, action_dim)`` per-dimension entropy.
        """
        dist = self.forward(obs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy


class MultiChannelValue(nn.Module):
    """Value network outputting one value per reward channel.

    Uses a separate NatureCNN encoder (not shared with policy).

    Parameters
    ----------
    sample_obs : dict[str, Tensor]
        Sample observation dict (for NatureCNN dimension inference).
    reward_channels : int
        Number of reward channels.
    cnn_feature_dim : int
        CNN branch output dimension.
    state_feature_dim : int
        State branch output dimension.
    """

    def __init__(
        self,
        sample_obs: dict[str, Tensor],
        reward_channels: int,
        cnn_feature_dim: int = 256,
        state_feature_dim: int = 256,
    ) -> None:
        super().__init__()

        self.feature_net = NatureCNN(
            sample_obs,
            cnn_feature_dim=cnn_feature_dim,
            state_feature_dim=state_feature_dim,
        )

        feat_dim = self.feature_net.output_dim
        self.value_head = nn.Sequential(
            layer_init(nn.Linear(feat_dim, feat_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(feat_dim, reward_channels), std=1.0),
        )

    def forward(self, obs: dict[str, Tensor]) -> Tensor:
        """Predict per-channel values.

        Args:
            obs: Dict observation with rgb, depth, state.

        Returns:
            ``(B, reward_channels)`` value estimates.
        """
        features = self.feature_net(obs)
        return self.value_head(features)
