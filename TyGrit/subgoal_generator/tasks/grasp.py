"""Grasp task — subgoal generator for grasping.

Implements the subgoal policy π_g from the receding-horizon framework.
Each call to :meth:`generate` samples a reachable arm configuration for
direct grasping.  The scheduler executes the returned subgoal via
closed-loop planning.  When it calls ``generate()`` again (trajectory
done or path/goal invalidated), the generator re-evaluates with the
updated scene.

Usage::

    from TyGrit.subgoal_generator import create_subgoal_generator

    generator = create_subgoal_generator(cfg, robot, collision_check)
    result = Scheduler(
        robot, scene, planner,
        generator=generator,
        controller_fn=make_mpc_controller(config.mpc),
        config=cfg.scheduler,
        camera_pose_fn=compute_camera_pose,
    ).run()
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from loguru import logger

from TyGrit.controller.fetch.gripper import GRIPPER_CLOSED
from TyGrit.kinematics.fetch.camera import compute_camera_pose
from TyGrit.kinematics.ik import create_ik_solver
from TyGrit.perception.grasping import create_grasp_predictor
from TyGrit.subgoal_generator.samplers.grasp_sampler import GraspSampler
from TyGrit.types.planning import PlanningMode, Subgoal

if TYPE_CHECKING:
    from TyGrit.checker.collision import CollisionCheckFn
    from TyGrit.config import SystemConfig
    from TyGrit.envs.base import RobotBase
    from TyGrit.scene.scene import Scene
    from TyGrit.subgoal_generator.config import GraspGeneratorConfig
    from TyGrit.types.planning import SchedulerFeedback
    from TyGrit.types.robot import RobotState


class _Phase(enum.Enum):
    """Task-level phases for the grasp pipeline."""

    APPROACH = "approach"  # Moving toward a grasp configuration
    LIFT = "lift"  # Raising the arm after grasping
    DONE = "done"  # Task complete


class GraspSubgoalGenerator:
    """Subgoal generator for grasping tasks.

    Acts as a state machine with phases APPROACH → LIFT → DONE.
    The approach-to-lift transition closes the gripper and computes a
    lift target.

    During APPROACH, each ``generate()`` call samples a grasp goal (arm
    config for direct grasping).  If the scheduler reports the current
    goal is still valid and not yet reached, it is reused so the
    scheduler only replans the path.

    Parameters
    ----------
    robot : RobotBase
        Robot environment (for snapshots and gripper control).
    config : SystemConfig
        Full system config — used to create IK solver, grasp predictor,
        and sampler internally.
    collision_check : CollisionCheckFn
        ``(WholeBodyConfig) → bool`` — collision validation against the
        shared environment maintained by the scheduler's planner.
        Typically ``planner.validate_config``.
    gen_config : GraspGeneratorConfig or None
        Grasp-specific thresholds.  Defaults to ``GraspGeneratorConfig()``.
    """

    def __init__(
        self,
        robot: RobotBase,
        config: SystemConfig,
        collision_check: CollisionCheckFn,
        gen_config: GraspGeneratorConfig | None = None,
    ) -> None:
        if gen_config is None:
            from TyGrit.subgoal_generator.config import GraspGeneratorConfig as _GGC

            gen_config = _GGC()
        self._robot = robot
        self._approach_threshold = gen_config.approach_threshold
        self._lift_height = gen_config.lift_height
        self._lift_threshold = gen_config.lift_threshold

        # Create sampling dependencies internally from config
        ik_solver = create_ik_solver("fetch", "ikfast_base")
        predictor = create_grasp_predictor(config.grasping)
        self._grasp_sampler = GraspSampler(ik_solver, predictor, collision_check)

        # State
        self._phase = _Phase.APPROACH
        self._grasp_target: npt.NDArray[np.float64] | None = None
        self._lift_target: npt.NDArray[np.float64] | None = None
        self._current_goal: npt.NDArray[np.float64] | None = None

    # ── Public interface (scheduler callbacks) ────────────────────────

    def generate(
        self,
        scene: Scene,
        robot_state: RobotState,
        feedback: SchedulerFeedback,
    ) -> Subgoal | None:
        """Return the next subgoal for the motion planner.

        Called by the scheduler as ``generator.generate(scene, state, feedback)``
        whenever a trajectory is exhausted, path/goal is invalidated, or
        a new plan is needed.
        """
        current = np.array(robot_state.planning_joints, dtype=np.float64)

        # ── Phase transitions ────────────────────────────────────────
        if self._phase == _Phase.APPROACH and self._grasp_target is not None:
            if (
                float(np.linalg.norm(current - self._grasp_target))
                < self._approach_threshold
            ):
                logger.info("Grasp approach complete — closing gripper")
                self._robot.control_gripper(GRIPPER_CLOSED)
                self._phase = _Phase.LIFT
                self._lift_target = self._compute_lift(robot_state)
                self._current_goal = self._lift_target
                return Subgoal(PlanningMode.INTERPOLATION, self._lift_target)

        if self._phase == _Phase.LIFT:
            if self._lift_target is not None:
                return Subgoal(PlanningMode.INTERPOLATION, self._lift_target)
            return None

        # ── APPROACH phase: hierarchical subgoal sampling ────────────

        # Reuse current goal if feedback says it's still valid and not yet reached
        if self._current_goal is not None:
            dist = float(np.linalg.norm(current - self._current_goal))
            if dist > self._approach_threshold:
                if feedback.is_goal_valid is not False:
                    return Subgoal(PlanningMode.ARM, self._current_goal)

        # 1. Grasp goal — most ambitious, direct manipulation
        goal = self._sample_grasp(robot_state)
        if goal is not None:
            self._grasp_target = goal
            self._current_goal = goal
            logger.info("Subgoal: grasp goal sampled")
            return Subgoal(PlanningMode.ARM, goal)

        # 2. Pre-grasp goal — base repositioning to enable grasp
        goal = self._sample_pregrasp(scene, robot_state)
        if goal is not None:
            self._current_goal = goal
            logger.info("Subgoal: pre-grasp goal sampled")
            return Subgoal(PlanningMode.WHOLE_BODY, goal)

        # 3. Observe goal — gather sensor data to improve the map
        goal = self._sample_observe(scene, robot_state)
        if goal is not None:
            self._current_goal = goal
            logger.info("Subgoal: observe goal sampled")
            return Subgoal(PlanningMode.ARM, goal)

        logger.warning("All subgoal strategies failed")
        self._current_goal = None
        return None

    def goal_predicate(self, state: RobotState) -> bool:
        """True when the entire grasp pipeline is complete.

        Checks both the phase flag and (for LIFT) the actual distance
        so the scheduler can exit as soon as the lift is reached.
        """
        if self._phase == _Phase.DONE:
            return True
        if self._phase == _Phase.LIFT and self._lift_target is not None:
            current = np.array(state.planning_joints, dtype=np.float64)
            return (
                float(np.linalg.norm(current - self._lift_target))
                < self._lift_threshold
            )
        return False

    # ── Sampling strategies ───────────────────────────────────────────

    def _sample_grasp(self, robot_state: RobotState) -> npt.NDArray[np.float64] | None:
        """Sample a direct grasp config without base motion."""
        snapshot = self._robot.get_sensor_snapshot("head")
        camera_pose = compute_camera_pose(snapshot.robot_state)
        return self._grasp_sampler.sample(snapshot, camera_pose, robot_state)

    def _sample_pregrasp(
        self,
        scene: Scene,  # noqa: ARG002
        robot_state: RobotState,  # noqa: ARG002
    ) -> npt.NDArray[np.float64] | None:
        """Sample a whole-body config with base repositioning."""
        return None

    def _sample_observe(
        self,
        scene: Scene,  # noqa: ARG002
        robot_state: RobotState,  # noqa: ARG002
    ) -> npt.NDArray[np.float64] | None:
        """Sample a standoff observation pose."""
        return None

    # ── Internal ──────────────────────────────────────────────────────

    def _compute_lift(self, robot_state: RobotState) -> npt.NDArray[np.float64]:
        """Compute lift target by raising the torso from current config."""
        joints = np.array(robot_state.planning_joints, dtype=np.float64)
        joints[0] += self._lift_height  # torso is the first planning joint
        return joints
