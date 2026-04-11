"""Evaluation script for trained FPPO policies.

Loads a trained policy checkpoint and evaluates on the task suite,
reporting success rate, episode length, and cumulative reward.

Usage::

    pixi run python -m TyGrit.rl.eval --checkpoint runs/fppo/final.pt
    pixi run python -m TyGrit.rl.eval --checkpoint runs/fppo/final.pt --render
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from TyGrit.envs.fetch.config import FetchEnvConfig
from TyGrit.envs.fetch.maniskill_vec import ManiSkillFetchRobotVec
from TyGrit.rl.obs import build_obs_dict
from TyGrit.rl.policy import FactoredPolicy
from TyGrit.rl.train import (
    _cache_link_groups,
    _compute_factored_reward,
    _make_target_pos,
    _sim_poses_from_result,
)
from TyGrit.tasks.loader import load_tasks


def _place_target_marker(
    robot: ManiSkillFetchRobotVec,
    position: tuple[float, float, float],
    radius: float = 0.03,
    color: tuple[float, float, float, float] = (1.0, 0.2, 0.2, 0.6),
) -> None:
    """Add a translucent sphere at *position* in the rendered scene.

    Uses Sapien's actor builder to create a visual-only (no collision)
    kinematic actor so it doesn't affect physics.
    """
    import sapien

    scene = robot._env.unwrapped.scene  # type: ignore[attr-defined]
    builder = scene.create_actor_builder()
    builder.add_sphere_visual(
        radius=radius,
        material=sapien.render.RenderMaterial(base_color=color),
    )
    marker = builder.build_kinematic(name="target_marker")
    marker.set_pose(sapien.Pose(p=list(position)))


def _run_episode(
    robot: ManiSkillFetchRobotVec,
    policy: FactoredPolicy,
    target_pos: torch.Tensor,
    config,
    device: str,
    link_groups: dict[str, list],
    init_result: dict,
) -> tuple[bool, int, float]:
    """Run a single evaluation episode. Returns (success, length, reward)."""
    robot.look_at_batched(target_pos)

    ep_reward = 0.0
    ep_len = 0
    done = False
    terminated = False
    current_result = init_result

    # Initial distance for potential-based reach reward
    init_ee = init_result["ee_pos"]
    prev_dist = torch.linalg.norm(init_ee - target_pos.to(init_ee.device), dim=1)

    while not done:
        obs = build_obs_dict(current_result, target_pos)
        obs_dev = {k: v.to(device) for k, v in obs.items()}
        with torch.no_grad():
            action, _ = policy.get_action(obs_dev, deterministic=True)

        step_result = robot.step(action.cpu())
        current_result = step_result
        ep_len += 1

        sim_poses = _sim_poses_from_result(step_result)
        robot.look_at_batched(target_pos)

        total_reward, _terms, prev_dist = _compute_factored_reward(
            target_pos,
            action,
            config,
            link_groups,
            sim_poses=sim_poses,
            prev_dist=prev_dist,
        )
        ep_reward += total_reward.sum().item()

        # Success: gripper closing near target
        ee_dist = torch.linalg.norm(
            step_result["ee_pos"] - target_pos.to(step_result["ee_pos"].device),
            dim=1,
        )
        gripper_closing = action[:, -1] > 0
        terminated = bool(
            (ee_dist < config.grasp_dist_threshold).any() and gripper_closing.any()
        )
        truncated = ep_len >= config.max_episode_steps
        done = terminated or truncated

    return terminated, ep_len, ep_reward


def evaluate(
    checkpoint_path: Path,
    render: bool = False,
    device: str = "cuda",
    task_suite_path: str = "resources/benchmark/grasp_benchmark.json",
) -> dict[str, float]:
    """Evaluate a trained policy on the task suite."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    sample_obs = ckpt["sample_obs"]

    policy = FactoredPolicy(
        sample_obs=sample_obs,
        action_dim=config.action_dim,
        cnn_feature_dim=config.cnn_feature_dim,
        state_feature_dim=config.state_feature_dim,
    ).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()

    env_config = FetchEnvConfig(
        num_envs=1,
        obs_mode="rgbd",
        render_mode="human" if render else None,
        camera_width=128,
        camera_height=128,
    )
    robot = ManiSkillFetchRobotVec(config=env_config)
    link_groups = _cache_link_groups(robot)

    suite = load_tasks(task_suite_path)
    logger.info(
        "Evaluating on task suite: {} scenes, {} tasks",
        len(suite.scenes),
        suite.total_tasks,
    )

    successes = []
    episode_lengths = []
    episode_rewards = []

    task_pairs = list(suite.iter_tasks())
    for idx, (scene, task) in enumerate(task_pairs):
        reset_result = robot.reset(seed=scene.seed)
        target_pos = _make_target_pos(
            task.object_pose.position,
            1,
            torch.device(device),
        )

        if render:
            _place_target_marker(robot, task.object_pose.position)

        success, ep_len, ep_reward = _run_episode(
            robot,
            policy,
            target_pos,
            config,
            device,
            link_groups=link_groups,
            init_result=reset_result,
        )

        successes.append(success)
        episode_lengths.append(ep_len)
        episode_rewards.append(ep_reward)

        if (idx + 1) % 10 == 0:
            logger.info(
                "Task {}/{} ({}/{}) | success={} len={} reward={:.2f}",
                idx + 1,
                len(task_pairs),
                scene.scene_id,
                task.task_index,
                success,
                ep_len,
                ep_reward,
            )

    robot.close()

    results = {
        "success_rate": float(np.mean(successes)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
    }

    n_eval = len(successes)
    logger.info("=== Evaluation Results ({} episodes) ===", n_eval)
    logger.info("  Success rate:     {:.1%}", results["success_rate"])
    logger.info("  Mean ep length:   {:.1f}", results["mean_episode_length"])
    logger.info(
        "  Mean reward:      {:.2f} +/- {:.2f}",
        results["mean_reward"],
        results["std_reward"],
    )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained FPPO policy")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--task-suite",
        type=str,
        default="resources/benchmark/grasp_benchmark.json",
        help="Path to task suite JSON for evaluation",
    )
    args = parser.parse_args()

    evaluate(
        checkpoint_path=Path(args.checkpoint),
        render=args.render,
        device=args.device,
        task_suite_path=args.task_suite,
    )


if __name__ == "__main__":
    main()
