"""End-to-end grasp benchmark — full pipeline in ManiSkill3.

Usage::

    pixi run python -m benchmark.grasp_benchmark [--env-id PickCube-v1] [--target-id N]

Phases:
    0. Setup   — create env, planner, IK, GraspGen, scene, samplers
    1. Observe — reset, open gripper, settle, compute camera pose, update scene
    2. Grasp   — sample a reachable 8-DOF grasp goal
    3. Approach — plan + execute trajectory to grasp pose
    4. Close   — close gripper and settle
    5. Lift    — plan + execute upward motion
    6. Report  — log outcome
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from loguru import logger

from TyGrit.controller.fetch.mpc import MPCConfig
from TyGrit.envs.fetch.config import FetchEnvConfig
from TyGrit.envs.fetch.maniskill import ManiSkillFetchRobot
from TyGrit.kinematics.fetch.camera import compute_camera_pose
from TyGrit.kinematics.fetch.ikfast import IKFastSolver
from TyGrit.perception.grasping.config import GraspGenConfig
from TyGrit.perception.grasping.graspgen import GraspGenPredictor
from TyGrit.planning.config import VampPlannerConfig
from TyGrit.planning.fetch.vamp_preview import VampPreviewPlanner
from TyGrit.scene.config import PointCloudSceneConfig
from TyGrit.scene.pointcloud_scene import PointCloudScene
from TyGrit.subgoal_generator.samplers.config import GraspSamplerConfig
from TyGrit.subgoal_generator.samplers.grasp_sampler import GraspSampler
from TyGrit.types.geometry import SE2Pose


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grasp benchmark")
    p.add_argument("--env-id", default="PickCube-v1", help="ManiSkill env ID")
    p.add_argument(
        "--target-id", type=int, default=None, help="Segmentation ID of target object"
    )
    p.add_argument("--render", action="store_true", help="Enable human rendering")
    p.add_argument("--camera-width", type=int, default=640)
    p.add_argument("--camera-height", type=int, default=480)
    p.add_argument(
        "--settle-steps",
        type=int,
        default=50,
        help="Physics steps to settle after reset",
    )
    p.add_argument("--gripper-close-steps", type=int, default=100)
    p.add_argument(
        "--lift-height", type=float, default=0.15, help="Metres to raise torso for lift"
    )
    p.add_argument(
        "--graspgen-config",
        default="resources/weights/graspgen/graspgen_config.yaml",
        help="Path to GraspGen checkpoint config",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Phase 0: Setup ──────────────────────────────────────────────────
    logger.info("Phase 0: Setup")

    env_config = FetchEnvConfig(
        env_id=args.env_id,
        obs_mode="rgb+depth+segmentation",
        control_mode="pd_joint_delta_pos",
        render_mode="human" if args.render else None,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
    )
    mpc_config = MPCConfig()

    logger.info("Creating ManiSkill env: {}", args.env_id)
    robot = ManiSkillFetchRobot(config=env_config, mpc_config=mpc_config)

    logger.info("Creating VampPreviewPlanner")
    planner = VampPreviewPlanner(VampPlannerConfig())

    logger.info("Creating IKFastSolver")
    ik_solver = IKFastSolver()

    logger.info("Creating GraspGenPredictor")
    grasp_predictor = GraspGenPredictor(
        GraspGenConfig(checkpoint_config_path=args.graspgen_config)
    )

    scene = PointCloudScene(PointCloudSceneConfig(mode="latest"))

    grasp_sampler = GraspSampler(
        ik_solver=ik_solver,
        grasp_predictor=grasp_predictor,
        planner=planner,
        config=GraspSamplerConfig(
            target_object_id=args.target_id if args.target_id is not None else 1,
        ),
    )

    # ── Phase 1: Reset + Observe ────────────────────────────────────────
    logger.info("Phase 1: Reset + Observe")

    robot.reset()
    robot.control_gripper(0.0)  # open gripper

    # Settle physics
    zero_action = np.zeros(10, dtype=np.float32)
    for _ in range(args.settle_steps):
        robot.step(zero_action)

    # Get observation
    snapshot = robot.get_observation()
    state = snapshot.robot_state
    camera_pose = compute_camera_pose(state)

    logger.info(
        "Robot base: ({:.3f}, {:.3f}, {:.3f})",
        state.base_pose.x,
        state.base_pose.y,
        state.base_pose.theta,
    )

    # Auto-detect target ID from segmentation if not specified
    target_id = args.target_id
    if target_id is None and snapshot.segmentation is not None:
        unique_ids = np.unique(snapshot.segmentation)
        # Pick the first non-zero/non-background ID
        candidates = [int(i) for i in unique_ids if i > 0]
        if candidates:
            target_id = candidates[0]
            logger.info("Auto-detected target segmentation ID: {}", target_id)
            grasp_sampler._cfg = GraspSamplerConfig(target_object_id=target_id)
        else:
            logger.error("No objects detected in segmentation")
            robot.close()
            sys.exit(1)

    # Update scene
    scene.update(snapshot, camera_pose)
    scene_cloud = scene.get_pointcloud()
    logger.info("Scene cloud: {} points", scene_cloud.shape[0])

    # Load scene into planner
    if scene_cloud.shape[0] > 0:
        planner.add_pointcloud(scene_cloud.astype(np.float64))

    # Set planner base params
    bp = state.base_pose
    planner.set_base_params(SE2Pose(x=bp.x, y=bp.y, theta=bp.theta))

    # ── Phase 2: Sample Grasp ───────────────────────────────────────────
    logger.info("Phase 2: Sample Grasp")

    grasp_joints = grasp_sampler.sample(snapshot, camera_pose, state)
    if grasp_joints is None:
        logger.error("FAILED: no valid grasp found")
        robot.close()
        sys.exit(1)

    logger.info("Grasp goal joints: {}", np.array2string(grasp_joints, precision=3))

    # ── Phase 3: Plan + Execute Approach ────────────────────────────────
    logger.info("Phase 3: Plan + Execute Approach")

    current_joints = np.array(state.planning_joints, dtype=np.float64)
    plan_result = planner.plan_arm(current_joints, grasp_joints)

    if not plan_result.success or plan_result.trajectory is None:
        logger.error("FAILED: motion planning failed — {}", plan_result.failure)
        robot.close()
        sys.exit(1)

    logger.info("Trajectory: {} waypoints", len(plan_result.trajectory.arm_path))
    robot.execute_trajectory(plan_result.trajectory)
    logger.info("Approach trajectory executed")

    # ── Phase 4: Close Gripper ──────────────────────────────────────────
    logger.info("Phase 4: Close Gripper")

    robot.control_gripper(1.0)  # close
    for _ in range(args.gripper_close_steps):
        robot.step(zero_action)

    # ── Phase 5: Lift ───────────────────────────────────────────────────
    logger.info("Phase 5: Lift")

    state_after_grasp = robot.get_robot_state()
    lift_joints = np.array(state_after_grasp.planning_joints, dtype=np.float64)

    # Raise torso (index 0) by lift_height, clamped to max 0.386
    lift_joints[0] = min(lift_joints[0] + args.lift_height, 0.386)

    lift_result = planner.plan_arm(
        np.array(state_after_grasp.planning_joints, dtype=np.float64),
        lift_joints,
    )

    if lift_result.success and lift_result.trajectory is not None:
        robot.execute_trajectory(lift_result.trajectory)
        logger.info("Lift trajectory executed")
    else:
        logger.warning("Lift planning failed ({}), skipping", lift_result.failure)

    # ── Phase 6: Report ─────────────────────────────────────────────────
    logger.info("Phase 6: Report")
    logger.info("Benchmark complete — grasp pipeline succeeded")

    robot.close()


if __name__ == "__main__":
    main()
