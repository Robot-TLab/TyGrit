"""End-to-end grasp example: approach → close gripper → lift.

Thin wrapper: load config → create shared components → build the
hierarchical subgoal generator → run the scheduler.

The ``GraspSubgoalGenerator`` owns all grasp-specific logic (sampling,
phase transitions, gripper target).  The scheduler handles all
continuous control (planning, MPC, stepping).  This script just
wires them together.

Usage::

    pixi run python examples/grasp_example.py --config config/grasp_example.toml
"""

from __future__ import annotations

import argparse

from loguru import logger

from TyGrit.checker import create_collision_check
from TyGrit.config import load_config
from TyGrit.controller.fetch import make_mpc_controller
from TyGrit.core.scheduler import Scheduler
from TyGrit.envs import create_env
from TyGrit.kinematics.fetch.camera import compute_camera_pose
from TyGrit.planning.planner import create_planner
from TyGrit.scene import PointCloudScene
from TyGrit.subgoal_generator import create_subgoal_generator


def main() -> None:
    parser = argparse.ArgumentParser(description="Grasp example")
    parser.add_argument(
        "--config",
        type=str,
        default="config/grasp_example.toml",
        help="Path to TOML config file",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    # Shared components
    robot = create_env(cfg.env, cfg.mpc)
    planner = create_planner("fetch", "vamp_preview", cfg.planner)
    scene = PointCloudScene(cfg.scene)

    # Hierarchical subgoal generator — owns sampling, phase transitions
    collision_check = create_collision_check(planner)
    generator = create_subgoal_generator(cfg, robot, collision_check)

    # Scheduler — owns planning, MPC, stepping
    robot.reset()
    result = Scheduler(
        robot=robot,
        scene=scene,
        planner=planner,
        generator=generator,
        controller_fn=make_mpc_controller(cfg.mpc),
        config=cfg.scheduler,
        camera_pose_fn=compute_camera_pose,
    ).run(max_iterations=500)

    logger.info("Result: {}", result.outcome.value)
    robot.close()


if __name__ == "__main__":
    main()
