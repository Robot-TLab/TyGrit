"""Factored reward functions matching CausalMoMa's 8-channel structure.

All functions operate on batched GPU tensors. Reward channels:

0. reach         — potential-based L2 distance (EE → target)
1. ee_orient     — end-effector orientation error (keep grasp-ready)
2. ee_local_pos  — EE height relative to target height
3. base_col      — base/head collision with environment
4. arm_col       — arm collision with environment
5. self_col      — self-collision
6. gaze          — target visible in head camera FOV
7. grasp         — gripper action reward at target height
"""

from __future__ import annotations

import torch
from torch import Tensor


def reach_reward(
    ee_pos: Tensor,
    target_pos: Tensor,
    prev_dist: Tensor,
    goal_bonus: float = 10.0,
    goal_dist_tol: float = 0.1,
) -> tuple[Tensor, Tensor]:
    """Potential-based reach reward + dense goal bonus (CausalMoMa).

    Returns ``(reward, current_dist)`` each ``(B,)``.
    The potential component is positive when getting closer, negative when
    moving away.  A bonus of *goal_bonus* is added **every step** while the
    distance is below *goal_dist_tol* (matching CausalMoMa's
    ``ReachingGoalReward``).
    """
    dist = torch.linalg.norm(ee_pos - target_pos, dim=1)
    reward = prev_dist - dist
    # Dense goal bonus: +goal_bonus every step while within tolerance
    within_goal = (dist < goal_dist_tol).float()
    reward = reward + goal_bonus * within_goal
    return reward, dist


def ee_orientation_reward(
    ee_forward: Tensor,
    target_forward: Tensor,
) -> Tensor:
    """Negative orientation error scaled by 0.5.

    Measures how well the EE's approach direction aligns with desired grasp
    direction (typically downward for top grasps).

    Args:
        ee_forward: ``(B, 3)`` EE forward/approach direction (unit).
        target_forward: ``(B, 3)`` desired approach direction (unit).

    Returns:
        ``(B,)`` reward in ``[-0.5, 0]``.
    """
    # Cosine distance: 0 when aligned, 2 when opposite
    cos_sim = (ee_forward * target_forward).sum(dim=1)
    return -0.5 * (1.0 - cos_sim)


def ee_local_position_reward(
    ee_pos: Tensor,
    target_pos: Tensor,
) -> Tensor:
    """Reward for keeping EE height close to target height.

    Matches CausalMoMa's proportional local reward with offset.

    Returns ``(B,)`` reward.
    """
    height_dist = torch.abs(ee_pos[:, 2] - target_pos[:, 2])
    return -0.5 * height_dist + 0.2


def collision_reward(in_collision: Tensor) -> Tensor:
    """Binary collision penalty matching CausalMoMa.

    Args:
        in_collision: ``(B,)`` boolean tensor — True when collision detected.

    Returns:
        ``(B,)`` penalty: ``-1.0`` if colliding, ``0.0`` otherwise.
    """
    return -in_collision.float()


def gaze_reward(
    target_pos: Tensor,
    camera_pos: Tensor,
    camera_forward: Tensor,
    fov_threshold: float = 0.7,
) -> Tensor:
    """Binary gaze reward matching CausalMoMa (Hu et al., RSS 2023).

    Returns 0.2 when the target is within the camera FOV, 0 otherwise.

    Args:
        target_pos: ``(B, 3)`` target positions in world frame.
        camera_pos: ``(B, 3)`` camera positions in world frame.
        camera_forward: ``(B, 3)`` camera forward direction.
        fov_threshold: Angular distance threshold in radians.

    Returns:
        ``(B,)`` reward: ``0.2`` if in FOV, ``0.0`` otherwise.
    """
    direction = target_pos - camera_pos
    direction = direction / (torch.linalg.norm(direction, dim=1, keepdim=True) + 1e-8)
    cos_angle = (direction * camera_forward).sum(dim=1)
    cos_threshold = torch.cos(
        torch.tensor(fov_threshold, device=cos_angle.device, dtype=cos_angle.dtype)
    )
    in_fov = (cos_angle > cos_threshold).float()

    return 0.2 * in_fov


def grasp_reward(
    gripper_action: Tensor,
    ee_pos: Tensor,
    target_pos: Tensor,
    height_threshold: float = 0.05,
) -> Tensor:
    """Action-based grasp reward matching CausalMoMa.

    Rewards closing the gripper when EE is near target height,
    penalizes closing when far away.

    Args:
        gripper_action: ``(B,)`` gripper action in [-1, 1] (positive = close).
        ee_pos: ``(B, 3)`` end-effector position.
        target_pos: ``(B, 3)`` target position.
        height_threshold: Distance threshold for "at target".

    Returns:
        ``(B,)`` reward in ``[-0.2, 0.2]``.
    """
    ee_dist = torch.linalg.norm(ee_pos - target_pos, dim=1)
    base_penalty = torch.abs(gripper_action) * -0.2

    # Near target + closing = reward; far + closing = penalty
    near_target = ee_dist < height_threshold
    closing = gripper_action > 0

    # Flip sign when action is appropriate for the situation
    reward = base_penalty.clone()
    reward[near_target & closing] *= -1  # reward closing near target
    reward[~near_target & ~closing] *= -1  # reward opening far from target

    return reward


def action_rate_penalty(
    action: Tensor,
    prev_action: Tensor,
) -> Tensor:
    """L2 penalty on action change between consecutive timesteps.

    Returns ``(B,)`` negative squared-norm of the action delta.
    """
    return -torch.sum((action - prev_action) ** 2, dim=1)
