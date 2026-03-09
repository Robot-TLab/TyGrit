# RL Baseline (FPPO)

The RL baseline reproduces [CausalMoMa](https://github.com/JiahengHu/CausalMoMa) (Hu et al., RSS 2023) — Factored PPO with a learned causal matrix that maps reward channels to action dimensions.

## Overview

The policy controls all 13 DOF of the Fetch robot (base, torso, arm, gripper, head) from RGB-D observations and proprioceptive state. Rewards are decomposed into 8 independent channels, each associated with a subset of action dimensions via a sparse causal matrix discovered through Conditional Mutual Information (CMI).

### Reward Channels

| Channel | Description | Scale |
|---------|-------------|-------|
| `reach` | Potential-based L2 distance (EE to target) + sparse goal bonus (+10 at < 0.55 m) | 0.7 |
| `ee_orient` | End-effector orientation error (keep grasp-ready) | 0.5 |
| `ee_local_pos` | EE height relative to target height | 0.5 |
| `base_col` | Binary base/head collision penalty | 1.0 |
| `arm_col` | Binary arm collision penalty | 1.0 |
| `self_col` | Binary self-collision penalty | 1.0 |
| `gaze` | Target visible in head camera FOV | 1.0 |
| `grasp` | Gripper action reward at target proximity | 1.0 |

### Action Space (13-dim continuous)

```
v, w, torso, shoulder_pan, shoulder_lift, upperarm_roll,
elbow_flex, forearm_roll, wrist_flex, wrist_roll, gripper,
head_pan, head_tilt
```

## Setup

The RL baseline uses a dedicated `rl` pixi environment that is lighter than the full `default` environment (no ROS, GraspGen, VAMP, `torch-scatter`, or `spconv`):

```bash
pixi install -e rl
```

## Training

Requires a CUDA GPU. Set `OMP_NUM_THREADS=1` to prevent LAPACK thread deadlocks with ManiSkill's GPU workers.

### Quick Start

```bash
OMP_NUM_THREADS=1 pixi run -e rl python -m TyGrit.rl.train
```

### Common Options

```bash
OMP_NUM_THREADS=1 pixi run -e rl python -m TyGrit.rl.train --num-envs 64 --total-timesteps 100000000 --log-dir runs/my_experiment --no-wandb --render --resume runs/fppo/checkpoint_100.pt
```

### Key Hyperparameters

All hyperparameters are in `TyGrit/rl/config.py` (`TrainConfig`). The defaults follow CausalMoMa:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_envs` | 64 | Parallel ManiSkill environments |
| `rollout_steps` | 2048 | Steps per rollout (covers ~4 episodes) |
| `total_timesteps` | 5,000,000 | Total training steps |
| `batch_size` | 512 | Mini-batch size for PPO updates |
| `n_epochs` | 10 | PPO epochs per rollout |
| `policy_lr` | 5e-5 | Policy learning rate |
| `value_lr` | 1e-4 | Value network learning rate |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda |
| `clip_range` | 0.2 | PPO clip range |
| `target_kl` | 0.15 | KL early stopping threshold |
| `max_episode_steps` | 2048 | Episode truncation length |

### GPU Memory

The rollout buffer for RGB-D observations is the dominant memory consumer. Approximate GPU memory usage:

| `num_envs` | Rollout buffer | Total (approx.) |
|------------|----------------|-----------------|
| 16 | ~2 GB | ~6 GB |
| 32 | ~4 GB | ~10 GB |
| 64 | ~8 GB | ~16 GB |

If you hit OOM, reduce `--num-envs`.

### Logging

Training logs to console and optionally to [Weights & Biases](https://wandb.ai). Tracked metrics include:

- Per-channel reward means (`reward/reach`, `reward/base_col`, etc.)
- Episode return, length, and success rate
- Policy loss, value loss, entropy, explained variance

Checkpoints are saved every 100 rollouts to `runs/fppo/`.

## Evaluation

Load a trained checkpoint and run with `--render` to visualize:

```bash
OMP_NUM_THREADS=1 pixi run -e rl python -m TyGrit.rl.train \
    --resume runs/fppo/final.pt \
    --render \
    --no-wandb \
    --num-envs 1
```

## Reference

> Jiaheng Hu, Peter Stone, Roberto Mart&iacute;n-Mart&iacute;n.
> *CausalMoMa: Real-time Whole-body Mobile Manipulation via Causal Factorization.*
> RSS 2023. [GitHub](https://github.com/JiahengHu/CausalMoMa)
