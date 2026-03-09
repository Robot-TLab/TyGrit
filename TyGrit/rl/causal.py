"""Causal discovery via Conditional Mutual Information (CMI).

Learns the causal matrix ``B`` of shape ``(reward_channels, action_dim)``
by estimating how much each action dimension contributes to predicting
each reward term.

Implements the CMI estimation from CausalMoMa (Hu et al., RSS 2023):
1. Train a generative model to predict reward terms from (obs, action).
2. For each action dimension, measure prediction degradation when that
   dimension is masked out.
3. Threshold the CMI estimates to get a binary causal matrix.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class CMIEstimator(nn.Module):
    """Generative model for estimating CMI between actions and rewards.

    Architecture: per-reward-term prediction networks with maskable
    action inputs.  CMI is estimated as the increase in prediction loss
    when an action dimension is dropped.

    Parameters
    ----------
    obs_dim : int
        Observation vector dimensionality.
    action_dim : int
        Action vector dimensionality.
    reward_dim : int
        Number of reward channels.
    hidden_dims : tuple[int, ...]
        Hidden layer sizes for the prediction MLPs.
    cmi_threshold : float
        Binarisation threshold for the causal matrix.
    ema_tau : float
        Exponential moving average smoothing for CMI estimates.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        reward_dim: int,
        hidden_dims: tuple[int, ...] = (128, 128),
        cmi_threshold: float = 0.12,
        ema_tau: float = 0.999,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.cmi_threshold = cmi_threshold
        self.ema_tau = ema_tau

        # Per-reward prediction networks
        # Input: obs + action (maskable) → predicted reward_j
        input_dim = obs_dim + action_dim
        self.predictors = nn.ModuleList()
        for _ in range(reward_dim):
            layers: list[nn.Module] = []
            in_d = input_dim
            for h in hidden_dims:
                layers.extend([nn.Linear(in_d, h), nn.ReLU()])
                in_d = h
            # Output: (mu, log_sigma) for Gaussian prediction
            layers.append(nn.Linear(in_d, 2))
            self.predictors.append(nn.Sequential(*layers))

        # Running CMI estimates: (reward_dim, action_dim)
        self.register_buffer("cmi_running", torch.zeros(reward_dim, action_dim))
        # Binary causal matrix
        self.register_buffer("causal_matrix", torch.ones(reward_dim, action_dim))

    def predict(
        self, obs: Tensor, action: Tensor, mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Predict reward terms from (obs, action).

        Args:
            obs: ``(B, obs_dim)`` observations.
            action: ``(B, action_dim)`` actions.
            mask: ``(B, action_dim)`` binary mask (1 = keep, 0 = drop).
                  None means keep all.

        Returns:
            mu: ``(B, reward_dim)`` predicted means.
            log_sigma: ``(B, reward_dim)`` predicted log-std.
        """
        if mask is not None:
            action = action * mask

        x = torch.cat([obs, action], dim=1)
        mus = []
        log_sigmas = []
        for predictor in self.predictors:
            out = predictor(x)  # (B, 2)
            mus.append(out[:, 0])
            log_sigmas.append(out[:, 1])

        return torch.stack(mus, dim=1), torch.stack(log_sigmas, dim=1)

    def prediction_loss(
        self,
        obs: Tensor,
        action: Tensor,
        rewards: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Gaussian negative log-likelihood per reward channel.

        Args:
            obs: ``(B, obs_dim)``
            action: ``(B, action_dim)``
            rewards: ``(B, reward_dim)`` actual reward values.
            mask: Optional ``(B, action_dim)`` binary mask.

        Returns:
            ``(B, reward_dim)`` per-sample, per-channel NLL.
        """
        mu, log_sigma = self.predict(obs, action, mask)
        sigma = torch.exp(log_sigma).clamp(min=1e-6)
        nll = (
            0.5 * ((rewards - mu) / sigma) ** 2
            + log_sigma
            + 0.5 * torch.log(torch.tensor(2 * torch.pi, device=mu.device))
        )
        return nll

    def update_causal_matrix(
        self, obs: Tensor, action: Tensor, rewards: Tensor
    ) -> Tensor:
        """Estimate CMI and update the binary causal matrix.

        For each action dimension i, measures how much the prediction
        degrades when a_i is masked out.

        Args:
            obs: ``(B, obs_dim)``
            action: ``(B, action_dim)``
            rewards: ``(B, reward_dim)``

        Returns:
            Updated causal matrix ``(reward_dim, action_dim)``.
        """
        with torch.no_grad():
            # Full prediction loss
            full_loss = self.prediction_loss(obs, action, rewards)  # (B, R)
            full_loss_mean = full_loss.mean(dim=0)  # (R,)

            # Masked prediction loss per action dimension
            cmi = torch.zeros(self.reward_dim, self.action_dim, device=obs.device)
            for i in range(self.action_dim):
                mask = torch.ones_like(action)
                mask[:, i] = 0.0
                masked_loss = self.prediction_loss(
                    obs, action, rewards, mask=mask
                )  # (B, R)
                # CMI = increase in loss when dropping action_i
                cmi[:, i] = masked_loss.mean(dim=0) - full_loss_mean

            # EMA update
            self.cmi_running = (
                self.ema_tau * self.cmi_running + (1 - self.ema_tau) * cmi
            )

            # Binarise
            self.causal_matrix = (self.cmi_running >= self.cmi_threshold).float()

        return self.causal_matrix

    def get_causal_matrix(self) -> Tensor:
        """Return the current binary causal matrix ``(R, A)``."""
        return self.causal_matrix


def discover_causal_structure(
    obs: Tensor,
    actions: Tensor,
    rewards: Tensor,
    action_dim: int,
    reward_dim: int,
    n_steps: int = 1000,
    lr: float = 1e-3,
    cmi_threshold: float = 0.12,
    eval_freq: int = 10,
    device: str = "cuda",
) -> Tensor:
    """Run causal discovery from collected rollout data.

    Trains the CMI estimator and returns the discovered causal matrix.

    Args:
        obs: ``(N, obs_dim)`` observations.
        actions: ``(N, action_dim)`` actions.
        rewards: ``(N, reward_dim)`` per-channel rewards.
        action_dim: Number of action dimensions.
        reward_dim: Number of reward channels.
        n_steps: Training steps for the generative model.
        lr: Learning rate.
        cmi_threshold: Binarisation threshold.
        eval_freq: How often to update the causal matrix estimate.
        device: Device to train on.

    Returns:
        Binary causal matrix ``(reward_dim, action_dim)``.
    """
    obs_dim = obs.shape[1]
    model = CMIEstimator(
        obs_dim=obs_dim,
        action_dim=action_dim,
        reward_dim=reward_dim,
        cmi_threshold=cmi_threshold,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset_size = obs.shape[0]

    obs = obs.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)

    for step in range(n_steps):
        # Sample batch
        idx = torch.randint(0, dataset_size, (256,))
        batch_obs = obs[idx]
        batch_act = actions[idx]
        batch_rew = rewards[idx]

        # Train with full input
        loss_full = model.prediction_loss(batch_obs, batch_act, batch_rew).mean()

        # Train with one random action masked
        mask = torch.ones_like(batch_act)
        drop_idx = torch.randint(0, action_dim, (batch_act.shape[0],))
        mask[torch.arange(batch_act.shape[0]), drop_idx] = 0.0
        loss_masked = model.prediction_loss(
            batch_obs, batch_act, batch_rew, mask=mask
        ).mean()

        loss = loss_full + loss_masked
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Periodically update causal matrix
        if (step + 1) % eval_freq == 0:
            eval_idx = torch.randint(0, dataset_size, (512,))
            model.update_causal_matrix(
                obs[eval_idx], actions[eval_idx], rewards[eval_idx]
            )

    return model.get_causal_matrix()
