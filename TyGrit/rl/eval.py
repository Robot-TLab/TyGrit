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

from TyGrit.envs.fetch import FetchRobot
from TyGrit.envs.fetch.config import FetchEnvConfig
from TyGrit.envs.fetch.core_vec import FetchRobotCoreVec
from TyGrit.gaze.fetch_head import look_at_batched
from TyGrit.rl.obs import build_obs_dict
from TyGrit.rl.policy import FactoredPolicy
from TyGrit.rl.train import (
    _cache_link_groups,
    _compute_factored_reward,
    _make_target_pos,
    _sim_poses_from_robot,
)
from TyGrit.tasks.loader import load_tasks


def _place_target_marker(
    robot: FetchRobotCoreVec,
    position: tuple[float, float, float],
    radius: float = 0.03,
    color: tuple[float, float, float, float] = (1.0, 0.2, 0.2, 0.6),
) -> None:
    """Add a translucent sphere at *position* in the rendered scene.

    Uses Sapien's actor builder to create a visual-only (no collision)
    kinematic actor so it doesn't affect physics.
    """
    import sapien

    scene = robot.handler._env.unwrapped.scene  # noqa: SLF001 — ManiSkill-specific
    builder = scene.create_actor_builder()
    builder.add_sphere_visual(
        radius=radius,
        material=sapien.render.RenderMaterial(base_color=color),
    )
    marker = builder.build_kinematic(name="target_marker")
    marker.set_pose(sapien.Pose(p=list(position)))


def _run_episode(
    robot: FetchRobotCoreVec,
    policy: FactoredPolicy,
    target_pos: torch.Tensor,
    config,
    device: str,
    link_groups: dict[str, list],
) -> tuple[bool, int, float]:
    """Run a single evaluation episode. Returns (success, length, reward).

    Requires the robot to already be reset — caller invokes
    ``robot.reset()`` before entering this loop. Initial TCP pose is
    read from the handler's cached obs.
    """
    look_at_batched(robot, target_pos)

    ep_reward = 0.0
    ep_len = 0
    done = False
    terminated = False

    # Initial distance for potential-based reach reward
    init_poses = _sim_poses_from_robot(robot)
    prev_dist = torch.linalg.norm(
        init_poses["ee_pos"] - target_pos.to(init_poses["ee_pos"].device),
        dim=1,
    )

    while not done:
        obs = build_obs_dict(robot, target_pos)
        obs_dev = {k: v.to(device) for k, v in obs.items()}
        with torch.no_grad():
            action, _ = policy.get_action(obs_dev, deterministic=True)

        robot.step(action.cpu())
        ep_len += 1

        sim_poses = _sim_poses_from_robot(robot)
        look_at_batched(robot, target_pos)

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
            sim_poses["ee_pos"] - target_pos.to(sim_poses["ee_pos"].device),
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

    # num_envs=2 so FetchRobot.create routes through the vec path; eval
    # only uses env 0 logically, the second env is idle. We accept the
    # small overhead because the eval code assumes batched shapes
    # throughout (look_at_batched, build_obs_dict, sim_poses).
    env_config = FetchEnvConfig(
        num_envs=2,
        sim_opts={
            "obs_mode": "rgbd",
            "control_mode": "pd_joint_vel",
            "render_mode": "human" if render else None,
        },
        camera_width=128,
        camera_height=128,
    )
    robot = FetchRobot.create(config=env_config)
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
        robot.reset(seed=scene.seed)
        target_pos = _make_target_pos(
            task.object_pose.position,
            robot.num_envs,
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
