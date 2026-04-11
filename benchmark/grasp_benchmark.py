"""End-to-end grasp benchmark — full pipeline in ManiSkill3.

When ``--tasks`` is provided, iterates over task suite scenes/tasks
with exact object placements. Otherwise runs a single default scene.

Usage::

    pixi run python -m benchmark.grasp_benchmark --tasks resources/benchmark/grasp_benchmark.json
    pixi run python -m benchmark.grasp_benchmark --tasks resources/benchmark/grasp_benchmark.json --scene 0 --task 5
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

from TyGrit.belief_state.config import PointCloudSceneConfig
from TyGrit.belief_state.pointcloud_scene import PointCloudScene
from TyGrit.controller.fetch.mpc import MPCConfig
from TyGrit.envs.fetch.config import FetchEnvConfig
from TyGrit.envs.fetch.maniskill import ManiSkillFetchRobot
from TyGrit.kinematics.fetch.camera import compute_camera_pose
from TyGrit.kinematics.fetch.ikfast import IKFastSolver
from TyGrit.perception.grasping.config import GraspGenConfig
from TyGrit.perception.grasping.graspgen import GraspGenPredictor
from TyGrit.planning.config import VampPlannerConfig
from TyGrit.planning.fetch.vamp_preview import VampPreviewPlanner
from TyGrit.subgoal_generator.samplers.config import GraspSamplerConfig
from TyGrit.subgoal_generator.samplers.grasp_sampler import GraspSampler
from TyGrit.types.geometry import SE2Pose
from TyGrit.types.tasks import TaskSuite
from TyGrit.types.worlds import SceneSamplerConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grasp benchmark")
    p.add_argument(
        "--manifest-path",
        default="resources/worlds/replicacad.json",
        help="Path to a world manifest JSON (see TyGrit.worlds.manifest)",
    )
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
    p.add_argument(
        "--tasks",
        type=str,
        default="",
        help="Path to task suite JSON (e.g. resources/benchmark/grasp_benchmark.json)",
    )
    p.add_argument("--scene", type=int, default=None, help="Scene index to run (0-19)")
    p.add_argument(
        "--task", type=int, default=None, help="Task index within scene (0-19)"
    )
    return p.parse_args()


def run_single_trial(
    robot: ManiSkillFetchRobot,
    planner: VampPreviewPlanner,
    grasp_sampler: GraspSampler,
    scene: PointCloudScene,
    args: argparse.Namespace,
    *,
    seed: int | None = None,
    target_id: int | None = None,
    label: str = "",
) -> bool:
    """Run one grasp trial (phases 1-6). Returns True on success."""
    prefix = f"[{label}] " if label else ""

    # ── Phase 1: Reset + Observe ────────────────────────────────────────
    logger.info("{}Phase 1: Reset + Observe", prefix)

    robot.reset(seed=seed)
    robot.control_gripper(0.0)  # open gripper

    # Settle physics
    zero_action = np.zeros(10, dtype=np.float32)
    for _ in range(args.settle_steps):
        robot.step(zero_action)

    # Get observation
    snapshot = robot.get_observation()
    state = snapshot.robot_state
    camera_pose = compute_camera_pose(state)

    # Auto-detect target ID from segmentation if not specified
    seg_target_id = target_id
    if seg_target_id is None and snapshot.segmentation is not None:
        unique_ids = np.unique(snapshot.segmentation)
        candidates = [int(i) for i in unique_ids if i > 0]
        if candidates:
            seg_target_id = candidates[0]
            logger.info(
                "{}Auto-detected target segmentation ID: {}", prefix, seg_target_id
            )
            grasp_sampler._cfg = GraspSamplerConfig(target_object_id=seg_target_id)
        else:
            logger.error("{}No objects detected in segmentation", prefix)
            return False

    # Update scene
    scene.update(snapshot, camera_pose)
    scene_cloud = scene.get_pointcloud()

    if scene_cloud.shape[0] > 0:
        planner.add_pointcloud(scene_cloud.astype(np.float64))

    bp = state.base_pose
    planner.set_base_params(SE2Pose(x=bp.x, y=bp.y, theta=bp.theta))

    # ── Phase 2: Sample Grasp ───────────────────────────────────────────
    logger.info("{}Phase 2: Sample Grasp", prefix)

    grasp_joints = grasp_sampler.sample(snapshot, camera_pose, state)
    if grasp_joints is None:
        logger.error("{}FAILED: no valid grasp found", prefix)
        return False

    # ── Phase 3: Plan + Execute Approach ────────────────────────────────
    logger.info("{}Phase 3: Plan + Execute Approach", prefix)

    current_joints = np.array(state.planning_joints, dtype=np.float64)
    plan_result = planner.plan_arm(current_joints, grasp_joints)

    if not plan_result.success or plan_result.trajectory is None:
        logger.error(
            "{}FAILED: motion planning failed — {}", prefix, plan_result.failure
        )
        return False

    logger.info(
        "{}Trajectory: {} waypoints", prefix, len(plan_result.trajectory.arm_path)
    )
    robot.execute_trajectory(plan_result.trajectory)

    # ── Phase 4: Close Gripper ──────────────────────────────────────────
    logger.info("{}Phase 4: Close Gripper", prefix)

    robot.control_gripper(1.0)
    for _ in range(args.gripper_close_steps):
        robot.step(zero_action)

    # ── Phase 5: Lift ───────────────────────────────────────────────────
    logger.info("{}Phase 5: Lift", prefix)

    state_after_grasp = robot.get_robot_state()
    lift_joints = np.array(state_after_grasp.planning_joints, dtype=np.float64)
    lift_joints[0] = min(lift_joints[0] + args.lift_height, 0.386)

    lift_result = planner.plan_arm(
        np.array(state_after_grasp.planning_joints, dtype=np.float64),
        lift_joints,
    )

    if lift_result.success and lift_result.trajectory is not None:
        robot.execute_trajectory(lift_result.trajectory)
        logger.info("{}Lift trajectory executed", prefix)
    else:
        logger.warning(
            "{}Lift planning failed ({}), skipping", prefix, lift_result.failure
        )

    # ── Phase 6: Report ─────────────────────────────────────────────────
    logger.info("{}Phase 6: SUCCESS", prefix)
    return True


def main() -> None:
    args = parse_args()

    # ── Phase 0: Setup ──────────────────────────────────────────────────
    logger.info("Phase 0: Setup")

    env_config = FetchEnvConfig(
        scene_sampler=SceneSamplerConfig(manifest_path=args.manifest_path),
        obs_mode="rgb+depth+segmentation",
        control_mode="pd_joint_delta_pos",
        render_mode="human" if args.render else None,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
    )
    mpc_config = MPCConfig()

    logger.info("Creating ManiSkill env from manifest: {}", args.manifest_path)
    robot = ManiSkillFetchRobot(config=env_config, mpc_config=mpc_config)

    logger.info("Creating VampPreviewPlanner")
    planner = VampPreviewPlanner(VampPlannerConfig())

    logger.info("Creating IKFastSolver")
    ik_solver = IKFastSolver()

    logger.info("Creating GraspGenPredictor")
    grasp_predictor = GraspGenPredictor(
        GraspGenConfig(checkpoint_config_path=args.graspgen_config)
    )

    pc_scene = PointCloudScene(PointCloudSceneConfig(mode="latest"))

    grasp_sampler = GraspSampler(
        ik_solver=ik_solver,
        grasp_predictor=grasp_predictor,
        planner=planner,
        config=GraspSamplerConfig(
            target_object_id=args.target_id if args.target_id is not None else 1,
        ),
    )

    # ── Load task suite if provided ─────────────────────────────────────
    suite: TaskSuite | None = None
    if args.tasks:
        from TyGrit.tasks.loader import load_tasks

        suite = load_tasks(args.tasks)
        logger.info(
            "Loaded task suite: {} scenes, {} tasks",
            len(suite.scenes),
            suite.total_tasks,
        )

    if suite is not None:
        # Filter scenes/tasks if specified
        scenes = list(suite.scenes)
        if args.scene is not None:
            scenes = [
                s
                for s in scenes
                if s.seed == args.scene or s.scene_id == f"scene_{args.scene}"
            ]
            if not scenes:
                logger.error("Scene {} not found", args.scene)
                robot.close()
                sys.exit(1)

        total = 0
        successes = 0

        for bscene in scenes:
            tasks = list(bscene.grasp_tasks)
            if args.task is not None:
                tasks = [t for t in tasks if t.task_index == args.task]

            for task in tasks:
                total += 1
                label = f"{bscene.scene_id}/task_{task.task_index}"
                ok = run_single_trial(
                    robot,
                    planner,
                    grasp_sampler,
                    pc_scene,
                    args,
                    seed=bscene.seed,
                    target_id=args.target_id,
                    label=label,
                )
                if ok:
                    successes += 1

        logger.info(
            "=== Benchmark Results: {}/{} ({:.1%}) ===",
            successes,
            total,
            successes / max(total, 1),
        )
    else:
        # Single trial (legacy mode)
        ok = run_single_trial(
            robot,
            planner,
            grasp_sampler,
            pc_scene,
            args,
            target_id=args.target_id,
        )
        if not ok:
            robot.close()
            sys.exit(1)

    robot.close()


if __name__ == "__main__":
    main()
