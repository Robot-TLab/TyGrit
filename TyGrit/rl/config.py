"""Training hyperparameters for the RL pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters for FPPO (Factored PPO) training.

    Reward channels (8, matching CausalMoMa):
        reach, ee_orient, ee_local_pos, base_col, arm_col, self_col, gaze, grasp

    Action dims (13):
        v, w, torso, shoulder_pan, shoulder_lift, upperarm_roll,
        elbow_flex, forearm_roll, wrist_flex, wrist_roll, gripper,
        head_pan, head_tilt
    """

    # ── Environment ───────────────────────────────────────────────────────
    num_envs: int = 64
    max_episode_steps: int = 500
    obs_mode: str = "rgbd"
    task_suite_path: str = "resources/benchmark/grasp_benchmark.json"
    settle_steps: int = 10

    # ── Reward weights ────────────────────────────────────────────────────
    w_reach: float = 0.7  # CausalMoMa reaching_reward_scale
    w_ee_orient: float = 0.5
    w_ee_local_pos: float = 0.5
    w_base_col: float = 1.0  # CausalMoMa: binary -1
    w_arm_col: float = 1.0
    w_self_col: float = 1.0
    w_gaze: float = 1.0
    w_grasp: float = 1.0
    encourage_gaze: bool = False  # reward head looking at target (active perception)

    grasp_dist_threshold: float = 0.05
    fov_threshold: float = 0.7  # radians, angular distance for gaze

    # ── Reaching goal bonus (CausalMoMa) ──────────────────────────────
    reach_goal_bonus: float = 10.0  # sparse +10 when within dist_tol
    reach_goal_dist_tol: float = 0.1  # metres (CausalMoMa ReachingGoalReward)

    # ── Collision ─────────────────────────────────────────────────────────
    collision_force_threshold: float = 0.5  # N — used for binary detection

    # ── Action smoothing ──────────────────────────────────────────────────
    smoothing_alpha: float = 0.7  # EMA: a_exec = α*a_new + (1-α)*a_prev

    # ── PPO ───────────────────────────────────────────────────────────────
    total_timesteps: int = 5_000_000
    rollout_steps: int = 2048
    n_epochs: int = 10
    batch_size: int = 512
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    target_kl: float | None = 0.15
    normalize_advantage: bool = True
    anneal_lr: bool = False

    # ── Learning rates ────────────────────────────────────────────────────
    policy_lr: float = 5e-5
    value_lr: float = 1e-4

    # ── Network ───────────────────────────────────────────────────────────
    cnn_feature_dim: int = 256
    state_feature_dim: int = 256

    # ── Entropy ───────────────────────────────────────────────────────────
    ent_coef: float = 0.01
    vf_coef: float = 0.5

    # ── CausalMoMa ────────────────────────────────────────────────────────
    reward_channels: int = 8  # match CausalMoMa
    action_dim: int = 13  # v, w, torso, 7 arm, gripper, head_pan, head_tilt

    # ── Logging ───────────────────────────────────────────────────────────
    log_interval: int = 10
    save_interval: int = 100
    log_dir: str = "runs/fppo"
    device: str = "cuda"
    wandb_project: str = "tygrit-rl"
    wandb_enabled: bool = True


def default_causal_matrix() -> Tensor:
    """CausalMoMa sparse causal matrix adapted for our action space.

    Shape ``(8, 13)`` — 8 reward channels × 13 action dims.

    Rows (reward terms): reach, ee_orient, ee_local_pos,
        base_col, arm_col, self_col, gaze, grasp.
    Cols (action dims): v, w, torso, shoulder_pan, shoulder_lift,
        upperarm_roll, elbow_flex, forearm_roll, wrist_flex,
        wrist_roll, gripper, head_pan, head_tilt.

    Columns 0–10 are identical to CausalMoMa CMI (Hu et al., RSS 2023).
    Columns 11–12 (head) extend for free head control: head causally
    affects base_col (head links can collide) and gaze.
    """
    B = torch.zeros(8, 13)

    #                           v  w  ts sp sl ur ef fr wf wr gr hp ht
    B[0] = torch.tensor(
        [1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32
    )  # reach
    B[1] = torch.tensor(
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0], dtype=torch.float32
    )  # ee_orient
    B[2] = torch.tensor(
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32
    )  # ee_local_pos
    B[3] = torch.tensor(
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.float32
    )  # base_col (head links can collide)
    B[4] = torch.tensor(
        [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32
    )  # arm_col
    B[5] = torch.tensor(
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.float32
    )  # self_col
    B[6] = torch.tensor(
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.float32
    )  # gaze (head free: pan + tilt)
    B[7] = torch.tensor(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=torch.float32
    )  # grasp

    return B
