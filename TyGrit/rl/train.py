"""FPPO (Factored PPO) training — CausalMoMa baseline reproduction.

All rollout buffers, GAE, and training computation stay on GPU.
Uses the ManiSkill obs dict directly for qpos/qvel/camera, with one
TCP pose read for ee_pos/ee_forward — no redundant sim queries.

Reward channels (8, CausalMoMa):
    reach, ee_orient, ee_local_pos, base_col, arm_col, self_col, gaze, grasp

Usage::

    pixi run python -m TyGrit.rl.train
    pixi run python -m TyGrit.rl.train --num-envs 256 --total-timesteps 5000000
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
from loguru import logger
from torch import Tensor

from TyGrit.envs.fetch.config import FetchEnvConfig
from TyGrit.envs.fetch.maniskill_vec import ManiSkillFetchRobotVec
from TyGrit.rl.config import TrainConfig, default_causal_matrix
from TyGrit.rl.obs import DictArray, build_obs_dict
from TyGrit.rl.policy import FactoredPolicy, MultiChannelValue
from TyGrit.rl.rewards import (
    collision_reward,
    ee_local_position_reward,
    ee_orientation_reward,
    gaze_reward,
    grasp_reward,
    reach_reward,
)
from TyGrit.tasks.loader import load_tasks

CHANNEL_NAMES = (
    "reach",
    "ee_orient",
    "ee_local_pos",
    "base_col",
    "arm_col",
    "self_col",
    "gaze",
    "grasp",
)

# Link groups for collision classification (matching CausalMoMa)
BASE_COLLISION_LINKS = frozenset(
    [
        "base_link",
        "head_pan_link",
        "head_tilt_link",
        "torso_lift_link",
    ]
)
ARM_COLLISION_LINKS = frozenset(
    [
        "shoulder_pan_link",
        "shoulder_lift_link",
        "upperarm_roll_link",
        "elbow_flex_link",
        "forearm_roll_link",
        "wrist_flex_link",
        "wrist_roll_link",
        "gripper_link",
        "l_gripper_finger_link",
        "r_gripper_finger_link",
    ]
)

# Non-adjacent robot link pairs that can physically self-collide.
# Verified via scene.get_pairwise_contact_forces() in GPU sim.
SELF_COLLISION_PAIRS: list[tuple[str, str]] = [
    # Head vs arm (confirmed: head_pan <-> elbow_flex in debug)
    ("head_pan_link", "upperarm_roll_link"),
    ("head_pan_link", "elbow_flex_link"),
    ("head_pan_link", "forearm_roll_link"),
    ("head_pan_link", "wrist_flex_link"),
    ("head_pan_link", "wrist_roll_link"),
    ("head_pan_link", "gripper_link"),
    ("head_pan_link", "l_gripper_finger_link"),
    ("head_pan_link", "r_gripper_finger_link"),
    ("head_tilt_link", "upperarm_roll_link"),
    ("head_tilt_link", "elbow_flex_link"),
    ("head_tilt_link", "forearm_roll_link"),
    ("head_tilt_link", "wrist_flex_link"),
    ("head_tilt_link", "wrist_roll_link"),
    ("head_tilt_link", "gripper_link"),
    ("head_tilt_link", "l_gripper_finger_link"),
    ("head_tilt_link", "r_gripper_finger_link"),
    # Base/torso vs distal arm
    ("base_link", "wrist_flex_link"),
    ("base_link", "wrist_roll_link"),
    ("base_link", "gripper_link"),
    ("base_link", "l_gripper_finger_link"),
    ("base_link", "r_gripper_finger_link"),
    ("torso_lift_link", "wrist_flex_link"),
    ("torso_lift_link", "wrist_roll_link"),
    ("torso_lift_link", "gripper_link"),
    ("torso_lift_link", "l_gripper_finger_link"),
    ("torso_lift_link", "r_gripper_finger_link"),
]

# Desired EE approach direction for top-down grasps (in world frame)
_GRASP_APPROACH_DIR = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)


# ── Collision detection ───────────────────────────────────────────────────────


def _cache_link_groups(robot: ManiSkillFetchRobotVec) -> dict:
    """Pre-filter and cache collision link objects by group.

    Also registers self-collision link pairs with
    ``scene.get_pairwise_contact_forces`` so that the GPU query is
    ready at training time.
    """
    all_links = robot._env.unwrapped.agent.robot.get_links()
    link_map = {link.name: link for link in all_links}
    scene = robot._env.unwrapped.scene

    # Register self-collision pairs (warmup call)
    self_pairs = []
    for la, lb in SELF_COLLISION_PAIRS:
        if la in link_map and lb in link_map:
            pair = (link_map[la], link_map[lb])
            scene.get_pairwise_contact_forces(*pair)  # registers the query
            self_pairs.append(pair)

    return {
        "base": [link_map[n] for n in BASE_COLLISION_LINKS if n in link_map],
        "arm": [link_map[n] for n in ARM_COLLISION_LINKS if n in link_map],
        "self_pairs": self_pairs,
        "scene": scene,
    }


def _detect_collision_group(links: list) -> Tensor:
    """Returns ``(N,)`` max contact force norm across links in group."""
    norms = torch.stack(
        [torch.linalg.norm(link.get_net_contact_forces(), dim=-1) for link in links]
    )
    return norms.max(dim=0).values


def _detect_self_collision(link_groups: dict) -> Tensor:
    """Returns ``(N,)`` max pairwise contact force norm across self-collision pairs.

    Uses ``scene.get_pairwise_contact_forces(link_a, link_b)`` which
    returns ``(N, 3)`` per-env forces for each registered pair.
    """
    scene = link_groups["scene"]
    if not link_groups["self_pairs"]:
        base_links = link_groups["base"]
        N = base_links[0].get_net_contact_forces().shape[0]
        dev = base_links[0].get_net_contact_forces().device
        return torch.zeros(N, dtype=torch.float32, device=dev)
    norms = torch.stack(
        [
            torch.linalg.norm(scene.get_pairwise_contact_forces(la, lb), dim=-1)
            for la, lb in link_groups["self_pairs"]
        ]
    )
    return norms.max(dim=0).values


# ── Sim-pose extraction from step/reset result ────────────────────────────────


def _sim_poses_from_result(result: dict) -> dict[str, Tensor]:
    """Build sim_poses dict from a step/reset result.

    Camera pose comes from the ManiSkill obs dict (``sensor_param``).
    TCP pose comes from the ``ee_pos``/``ee_forward`` keys added by
    ``ManiSkillFetchRobotVec``.
    """
    cam_mat = result["obs"]["sensor_param"]["fetch_head"][
        "cam2world_gl"
    ].float()  # (N, 4, 4)
    return {
        "ee_pos": result["ee_pos"],
        "ee_forward": result["ee_forward"],
        "cam_pos": cam_mat[:, :3, 3],
        "cam_forward": cam_mat[:, :3, 2],
    }


# ── Target object helpers ─────────────────────────────────────────────────────


def _make_target_pos(position: tuple, num_envs: int, device: torch.device) -> Tensor:
    """Broadcast a task target position to ``(N, 3)`` tensor."""
    return (
        torch.tensor(position, dtype=torch.float32, device=device)
        .unsqueeze(0)
        .expand(num_envs, 3)
    )


# ── Reward ────────────────────────────────────────────────────────────────────


def _compute_factored_reward(
    target_pos: Tensor,
    action: Tensor,
    cfg: TrainConfig,
    link_groups: dict,
    sim_poses: dict[str, Tensor],
    prev_dist: Tensor,
) -> tuple[Tensor, dict[str, Tensor], Tensor]:
    """Compute 8-channel CausalMoMa rewards. All tensors stay on GPU.

    Returns ``(total_reward, channel_dict, current_dist)``.
    """
    dev = sim_poses["ee_pos"].device
    target = target_pos.to(dev)

    # Desired approach direction (broadcast to batch)
    target_fwd = (
        _GRASP_APPROACH_DIR.to(dev).unsqueeze(0).expand_as(sim_poses["ee_forward"])
    )

    # 0. Reach (potential-based + sparse goal bonus, matching CausalMoMa)
    r_reach, cur_dist = reach_reward(
        sim_poses["ee_pos"],
        target,
        prev_dist,
        goal_bonus=cfg.reach_goal_bonus,
        goal_dist_tol=cfg.reach_goal_dist_tol,
    )

    # 1. EE orientation
    r_ee_orient = ee_orientation_reward(sim_poses["ee_forward"], target_fwd)

    # 2. EE local position (height)
    r_ee_local_pos = ee_local_position_reward(sim_poses["ee_pos"], target)

    # 3. Base collision (binary penalty, matching CausalMoMa)
    r_base_col = collision_reward(
        _detect_collision_group(link_groups["base"]) > cfg.collision_force_threshold,
    )
    # 4. Arm collision (binary penalty)
    r_arm_col = collision_reward(
        _detect_collision_group(link_groups["arm"]) > cfg.collision_force_threshold,
    )
    # 5. Self-collision (binary penalty)
    r_self_col = collision_reward(
        _detect_self_collision(link_groups) > cfg.collision_force_threshold,
    )

    # 6. Gaze — binary CausalMoMa reward (target in FOV → 0.2, else 0)
    if cfg.encourage_gaze:
        r_gaze = gaze_reward(
            target,
            sim_poses["cam_pos"],
            sim_poses["cam_forward"],
            cfg.fov_threshold,
        )
    else:
        r_gaze = torch.zeros(sim_poses["ee_pos"].shape[0], device=dev)

    # 7. Grasp — gripper action is dim 10
    gripper_action = action[:, 10].to(dev)
    r_grasp = grasp_reward(
        gripper_action,
        sim_poses["ee_pos"],
        target,
        cfg.grasp_dist_threshold,
    )

    # Weighted total
    total = (
        cfg.w_reach * r_reach
        + cfg.w_ee_orient * r_ee_orient
        + cfg.w_ee_local_pos * r_ee_local_pos
        + cfg.w_base_col * r_base_col
        + cfg.w_arm_col * r_arm_col
        + cfg.w_self_col * r_self_col
        + cfg.w_gaze * r_gaze
        + cfg.w_grasp * r_grasp
    )

    terms = dict(
        zip(
            CHANNEL_NAMES,
            [
                r_reach,
                r_ee_orient,
                r_ee_local_pos,
                r_base_col,
                r_arm_col,
                r_self_col,
                r_gaze,
                r_grasp,
            ],
        )
    )
    return total, terms, cur_dist


# ── FPPO Trainer ──────────────────────────────────────────────────────────────


class FPPOTrainer:
    """Factored PPO with CausalMoMa causal policy gradient.

    All rollout buffers, GAE, and training stay on GPU.
    """

    def __init__(
        self,
        robot: ManiSkillFetchRobotVec,
        config: TrainConfig | None = None,
        causal_matrix: Tensor | None = None,
    ) -> None:
        self.robot = robot
        self.cfg = config or TrainConfig()
        self.device = torch.device(self.cfg.device)

        self._suite = load_tasks(self.cfg.task_suite_path)
        logger.info(
            "Loaded task suite: {} scenes, {} tasks",
            len(self._suite.scenes),
            self._suite.total_tasks,
        )

        # ``causal_matrix or default_causal_matrix()`` would raise
        # RuntimeError when ``causal_matrix`` is a real Tensor —
        # truth-testing a multi-element tensor is ambiguous in PyTorch.
        # Use an explicit None check so the ``causal_matrix=Tensor(...)``
        # path actually works.
        if causal_matrix is None:
            causal_matrix = default_causal_matrix()
        self.B = causal_matrix.to(self.device)

        # Use first task's target for sample obs shape inference
        first_task = self._suite.scenes[0].grasp_tasks[0]
        init_target_pos = _make_target_pos(
            first_task.object_pose.position,
            self.cfg.num_envs,
            self.device,
        )
        reset_result = self.robot.reset()
        sample_obs = build_obs_dict(reset_result, init_target_pos)
        sample_obs_dev = {
            k: v[:1].clone().to(self.device) for k, v in sample_obs.items()
        }
        del sample_obs

        self.policy = FactoredPolicy(
            sample_obs=sample_obs_dev,
            action_dim=self.cfg.action_dim,
            cnn_feature_dim=self.cfg.cnn_feature_dim,
            state_feature_dim=self.cfg.state_feature_dim,
        ).to(self.device)

        self.value_net = MultiChannelValue(
            sample_obs=sample_obs_dev,
            reward_channels=self.cfg.reward_channels,
            cnn_feature_dim=self.cfg.cnn_feature_dim,
            state_feature_dim=self.cfg.state_feature_dim,
        ).to(self.device)

        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.cfg.policy_lr,
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(),
            lr=self.cfg.value_lr,
        )

        action_space = robot._env.action_space
        self._action_low = torch.tensor(
            action_space.low,
            dtype=torch.float32,
            device=self.device,
        )
        self._action_high = torch.tensor(
            action_space.high,
            dtype=torch.float32,
            device=self.device,
        )

        self.total_steps = 0
        self.rollout_count = 0
        self._link_groups = _cache_link_groups(self.robot)
        self._sample_obs = sample_obs_dev

        self._wandb_run = None
        if self.cfg.wandb_enabled:
            import wandb

            self._wandb_run = wandb.init(
                project=self.cfg.wandb_project,
                config={
                    k: v for k, v in self.cfg.__dict__.items() if not k.startswith("_")
                },
                name=f"fppo_n{self.cfg.num_envs}",
            )
            logger.info("wandb run: {}", self._wandb_run.url)

    # ── Episode state ─────────────────────────────────────────────────────

    def _reset_all(self) -> tuple[dict[str, Tensor], Tensor]:
        """Soft reset: restore robot home pose, pick a task, settle."""
        scene = random.choice(self._suite.scenes)
        task = random.choice(scene.grasp_tasks)

        reset_result = self.robot.reset(settle_steps=self.cfg.settle_steps)

        target_pos = _make_target_pos(
            task.object_pose.position,
            self.cfg.num_envs,
            self.device,
        )
        self.robot.look_at_batched(target_pos)
        obs = build_obs_dict(reset_result, target_pos)
        return obs, target_pos

    # ── Collect rollout ───────────────────────────────────────────────────

    def collect_rollout(
        self,
        obs_buffer: DictArray,
        actions_buf: Tensor,
        logprobs_buf: Tensor,
        rewards_buf: Tensor,
        values_buf: Tensor,
        dones_buf: Tensor,
        final_values_buf: Tensor,
    ) -> tuple[dict[str, float], Tensor]:
        """Collect one rollout. All buffers and computation stay on GPU."""
        T = self.cfg.rollout_steps
        N = self.cfg.num_envs
        A = self.cfg.action_dim
        dev = self.device

        obs, target_pos = self._reset_all()
        step_count = torch.zeros(N, dtype=torch.long, device=dev)
        prev_action = torch.zeros(N, A, device=dev)
        # Initial distance for potential-based reach reward
        prev_dist = torch.linalg.norm(
            obs["state"][:, 30:33].to(dev) - target_pos.to(dev),
            dim=1,
        )

        reward_per_step = torch.zeros(T, device=dev)
        channel_sums = torch.zeros(len(CHANNEL_NAMES), device=dev)
        ep_returns = torch.zeros(N, device=dev)
        completed_returns: list[Tensor] = []
        completed_lengths: list[Tensor] = []
        completed_successes: list[Tensor] = []

        for t in range(T):
            with torch.no_grad():
                raw_action, log_prob = self.policy.get_action(obs)
                value = self.value_net(obs)

            # Action smoothing (EMA)
            alpha = self.cfg.smoothing_alpha
            if t == 0:
                action = raw_action
            else:
                action = alpha * raw_action + (1.0 - alpha) * prev_action

            obs_buffer[t] = obs
            actions_buf[t] = raw_action
            logprobs_buf[t] = log_prob
            values_buf[t] = value

            step_result = self.robot.step(action)
            step_count += 1

            # Build sim poses from step result (camera from obs, TCP from extra read)
            sim_poses = _sim_poses_from_result(step_result)

            total_reward, terms, prev_dist = _compute_factored_reward(
                target_pos,
                action,
                self.cfg,
                link_groups=self._link_groups,
                sim_poses=sim_poses,
                prev_dist=prev_dist,
            )

            rewards_buf[t] = torch.stack([terms[n] for n in CHANNEL_NAMES], dim=1)

            prev_action = raw_action.to(dev)

            # Episode termination: grasp success
            ee_dist = torch.linalg.norm(
                step_result["ee_pos"] - target_pos.to(dev), dim=1
            )
            gripper_closing = action[:, 10].to(dev) > 0
            terminated = (ee_dist < self.cfg.grasp_dist_threshold) & gripper_closing
            truncated = step_count >= self.cfg.max_episode_steps
            done = (terminated | truncated).float()
            dones_buf[t] = done

            reward_per_step[t] = total_reward.mean()
            channel_sums += torch.stack([terms[n].mean() for n in CHANNEL_NAMES])
            ep_returns += total_reward
            self.total_steps += N

            next_obs = build_obs_dict(step_result, target_pos)

            # Final values bootstrap for truncated envs
            trunc_mask = truncated & ~terminated
            if trunc_mask.any():
                with torch.no_grad():
                    final_values_buf[t, trunc_mask] = self.value_net(
                        {k: v[trunc_mask] for k, v in next_obs.items()},
                    )

            done_mask = done.bool()
            if done_mask.any():
                done_idx = done_mask.nonzero(as_tuple=True)[0]
                completed_returns.append(ep_returns[done_idx].clone())
                completed_lengths.append(step_count[done_idx].float())
                completed_successes.append(terminated[done_idx].float())
                ep_returns[done_idx] = 0.0

                next_obs, target_pos = self._reset_all()
                step_count.zero_()
                prev_dist = torch.linalg.norm(
                    next_obs["state"][:, 30:33].to(dev) - target_pos.to(dev),
                    dim=1,
                )

            obs = next_obs

        with torch.no_grad():
            last_values = self.value_net(obs)

        channel_means = (channel_sums / T).cpu()
        rollout_stats: dict[str, float] = {
            "mean_reward": reward_per_step.mean().item(),
        }
        for i, name in enumerate(CHANNEL_NAMES):
            rollout_stats[f"reward/{name}"] = channel_means[i].item()
        if completed_returns:
            all_ret = torch.cat(completed_returns)
            all_len = torch.cat(completed_lengths)
            all_suc = torch.cat(completed_successes)
            rollout_stats["episode/return_mean"] = all_ret.mean().item()
            rollout_stats["episode/return_std"] = all_ret.std().item()
            rollout_stats["episode/length_mean"] = all_len.mean().item()
            rollout_stats["episode/success_rate"] = all_suc.mean().item()
            rollout_stats["episode/count"] = float(len(all_ret))

        return rollout_stats, last_values

    # ── GAE ───────────────────────────────────────────────────────────────

    @staticmethod
    def compute_gae(
        rewards: Tensor,
        values: Tensor,
        dones: Tensor,
        last_values: Tensor,
        final_values: Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[Tensor, Tensor]:
        """GAE with final_values bootstrap. Runs on whatever device inputs are on."""
        T, N, R = rewards.shape
        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros(N, R, device=rewards.device)
        next_values = last_values

        for t in reversed(range(T)):
            adjusted_rewards = rewards[t] + gamma * final_values[t] * dones[
                t
            ].unsqueeze(1)
            next_non_terminal = (1.0 - dones[t]).unsqueeze(1)
            delta = (
                adjusted_rewards + gamma * next_values * next_non_terminal - values[t]
            )
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
            next_values = values[t]

        return advantages, advantages + values

    # ── Train step ────────────────────────────────────────────────────────

    def train_step(
        self,
        obs_buffer: DictArray,
        actions_buf: Tensor,
        logprobs_buf: Tensor,
        advantages_buf: Tensor,
        returns_buf: Tensor,
    ) -> dict[str, float]:
        dataset_size = self.cfg.rollout_steps * self.cfg.num_envs
        BS = self.cfg.batch_size

        flat_obs = obs_buffer.flatten()
        flat_actions = actions_buf.reshape(dataset_size, -1)
        flat_old_logprobs = logprobs_buf.reshape(dataset_size, -1)
        flat_advantages = advantages_buf.reshape(dataset_size, -1)
        flat_returns = returns_buf.reshape(dataset_size, -1)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.cfg.n_epochs):
            indices = torch.randperm(dataset_size, device=self.device)
            for start in range(0, dataset_size, BS):
                idx = indices[start : min(start + BS, dataset_size)]

                batch_obs = {k: v[idx] for k, v in flat_obs.items()}
                batch_actions = flat_actions[idx]
                batch_old_logprobs = flat_old_logprobs[idx]
                batch_advantages = flat_advantages[idx]
                batch_returns = flat_returns[idx]

                causal_advantages = batch_advantages @ self.B
                if self.cfg.normalize_advantage:
                    causal_advantages = (
                        causal_advantages - causal_advantages.mean(dim=0)
                    ) / (causal_advantages.std(dim=0) + 1e-8)

                log_prob, entropy = self.policy.evaluate_actions(
                    batch_obs,
                    batch_actions,
                )
                ratio = torch.exp(log_prob - batch_old_logprobs)
                surr1 = causal_advantages * ratio
                surr2 = causal_advantages * torch.clamp(
                    ratio,
                    1.0 - self.cfg.clip_range,
                    1.0 + self.cfg.clip_range,
                )
                policy_loss = -torch.min(surr1, surr2).sum(dim=1).mean()
                entropy_loss = -entropy.sum(dim=1).mean()

                values_pred = self.value_net(batch_obs)
                value_loss = torch.nn.functional.mse_loss(values_pred, batch_returns)

                loss = (
                    policy_loss
                    + self.cfg.vf_coef * value_loss
                    + self.cfg.ent_coef * entropy_loss
                )

                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
                self.policy_optimizer.step()
                self.value_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

            # Early stopping on KL
            if self.cfg.target_kl is not None:
                with torch.no_grad():
                    kl_sum = 0.0
                    kl_count = 0
                    for kb_start in range(0, dataset_size, BS):
                        kb_end = min(kb_start + BS, dataset_size)
                        kb_obs = {k: v[kb_start:kb_end] for k, v in flat_obs.items()}
                        kb_lp, _ = self.policy.evaluate_actions(
                            kb_obs,
                            flat_actions[kb_start:kb_end],
                        )
                        kl_sum += (
                            (flat_old_logprobs[kb_start:kb_end] - kb_lp).sum().item()
                        )
                        kl_count += kb_end - kb_start
                    if kl_sum / kl_count > self.cfg.target_kl:
                        break

        # Explained variance
        with torch.no_grad():
            all_vpred = []
            for ev_start in range(0, dataset_size, BS):
                ev_end = min(ev_start + BS, dataset_size)
                ev_obs = {k: v[ev_start:ev_end] for k, v in flat_obs.items()}
                all_vpred.append(self.value_net(ev_obs))
            y_pred = torch.cat(all_vpred, dim=0).mean(dim=1)
            y_true = flat_returns.mean(dim=1)
            var_y = torch.var(y_true)
            explained_var = 1 - torch.var(y_true - y_pred) / (var_y + 1e-8)

        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
            "explained_variance": explained_var.item(),
        }

    # ── Main loop ─────────────────────────────────────────────────────────

    def train(self) -> None:
        log_dir = Path(self.cfg.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        T = self.cfg.rollout_steps
        N = self.cfg.num_envs
        R = self.cfg.reward_channels
        A = self.cfg.action_dim
        dev = self.device

        first_task = self._suite.scenes[0].grasp_tasks[0]
        reset_result = self.robot.reset()
        sample_obs = build_obs_dict(
            reset_result,
            _make_target_pos(first_task.object_pose.position, N, dev),
        )
        obs_buffer = DictArray((T, N), sample_obs)
        del sample_obs

        actions_buf = torch.zeros(T, N, A, device=dev)
        logprobs_buf = torch.zeros(T, N, A, device=dev)
        rewards_buf = torch.zeros(T, N, R, device=dev)
        values_buf = torch.zeros(T, N, R, device=dev)
        dones_buf = torch.zeros(T, N, device=dev)
        final_values_buf = torch.zeros(T, N, R, device=dev)

        logger.info(
            "Starting FPPO (CausalMoMa baseline) | envs={} | steps={} | channels={}",
            N,
            self.cfg.total_timesteps,
            R,
        )
        logger.info("Causal matrix B:\n{}", self.B)

        start_time = time.time()

        while self.total_steps < self.cfg.total_timesteps:
            if self.cfg.anneal_lr:
                frac = 1.0 - self.total_steps / self.cfg.total_timesteps
                self.policy_optimizer.param_groups[0]["lr"] = self.cfg.policy_lr * frac
                self.value_optimizer.param_groups[0]["lr"] = self.cfg.value_lr * frac

            final_values_buf.zero_()

            rollout_stats, last_values = self.collect_rollout(
                obs_buffer,
                actions_buf,
                logprobs_buf,
                rewards_buf,
                values_buf,
                dones_buf,
                final_values_buf,
            )

            advantages, returns = self.compute_gae(
                rewards_buf,
                values_buf,
                dones_buf,
                last_values,
                final_values_buf,
                self.cfg.gamma,
                self.cfg.gae_lambda,
            )

            train_stats = self.train_step(
                obs_buffer,
                actions_buf,
                logprobs_buf,
                advantages,
                returns,
            )
            self.rollout_count += 1

            if self.rollout_count % self.cfg.log_interval == 0:
                elapsed = time.time() - start_time
                fps = self.total_steps / elapsed
                mean_reward = rollout_stats["mean_reward"]
                metrics: dict[str, float] = {
                    "rollout": self.rollout_count,
                    "total_steps": self.total_steps,
                    "fps": fps,
                    "mean_reward": mean_reward,
                    "train/policy_loss": train_stats["policy_loss"],
                    "train/value_loss": train_stats["value_loss"],
                    "train/entropy": train_stats["entropy"],
                    "train/explained_variance": train_stats["explained_variance"],
                    "train/policy_lr": self.policy_optimizer.param_groups[0]["lr"],
                    "train/value_lr": self.value_optimizer.param_groups[0]["lr"],
                }
                metrics.update(rollout_stats)

                logger.info(
                    "rollout={} steps={} fps={:.0f} | "
                    "reward={:.3f} pol={:.4f} val={:.4f} ent={:.4f} ev={:.3f}",
                    self.rollout_count,
                    self.total_steps,
                    fps,
                    mean_reward,
                    train_stats["policy_loss"],
                    train_stats["value_loss"],
                    train_stats["entropy"],
                    train_stats["explained_variance"],
                )
                if self._wandb_run is not None:
                    import wandb

                    wandb.log(metrics, step=self.total_steps)

            if self.rollout_count % self.cfg.save_interval == 0:
                self.save(log_dir / f"checkpoint_{self.rollout_count}.pt")

        self.save(log_dir / "final.pt")
        logger.info("Training complete. {} total steps.", self.total_steps)
        if self._wandb_run is not None:
            import wandb

            wandb.finish()

    def save(self, path: Path) -> None:
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "value_state_dict": self.value_net.state_dict(),
                "policy_optimizer": self.policy_optimizer.state_dict(),
                "value_optimizer": self.value_optimizer.state_dict(),
                "total_steps": self.total_steps,
                "rollout_count": self.rollout_count,
                "causal_matrix": self.B,
                "config": self.cfg,
                "sample_obs": self._sample_obs,
            },
            path,
        )
        logger.info("Saved checkpoint: {}", path)

    def load(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.value_net.load_state_dict(ckpt["value_state_dict"])
        self.policy_optimizer.load_state_dict(ckpt["policy_optimizer"])
        self.value_optimizer.load_state_dict(ckpt["value_optimizer"])
        self.total_steps = ckpt["total_steps"]
        self.rollout_count = ckpt["rollout_count"]
        self.B = ckpt["causal_matrix"].to(self.device)
        logger.info("Loaded checkpoint: {} (step {})", path, self.total_steps)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="FPPO training for Fetch grasping")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--log-dir", type=str, default="runs/fppo")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--encourage-gaze", action="store_true")
    args = parser.parse_args()

    train_config = TrainConfig(
        num_envs=args.num_envs,
        total_timesteps=args.total_timesteps,
        log_dir=args.log_dir,
        device=args.device,
        wandb_enabled=not args.no_wandb,
        encourage_gaze=args.encourage_gaze,
    )

    t0 = time.time()
    logger.info("[init] Creating ManiSkill env (num_envs={})...", args.num_envs)
    env_config = FetchEnvConfig(
        num_envs=args.num_envs,
        obs_mode="rgbd",
        render_mode="human" if args.render else None,
        camera_width=128,
        camera_height=128,
    )
    robot = ManiSkillFetchRobotVec(config=env_config)
    logger.info("[init] ManiSkill env created in {:.1f}s", time.time() - t0)

    t1 = time.time()
    logger.info("[init] Creating FPPOTrainer...")
    trainer = FPPOTrainer(robot=robot, config=train_config)
    logger.info("[init] FPPOTrainer created in {:.1f}s", time.time() - t1)

    if args.resume:
        trainer.load(Path(args.resume))

    logger.info(
        "[init] Total init time: {:.1f}s. Starting training...", time.time() - t0
    )
    trainer.train()
    robot.close()


if __name__ == "__main__":
    main()
