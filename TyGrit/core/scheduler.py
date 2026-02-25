"""Receding-horizon scheduler — the core control loop.

This replaces the monolithic ``move_to_config_with_replanning()`` from
grasp_anywhere.  There is no background thread, no ``time.sleep()``, no
locks.  Every iteration is sequential::

    observe → update map → {check/replan} → gaze → track → execute

The algorithm follows the receding-horizon structure:

- When the trajectory is **empty or finished**, check the goal predicate.
  If the goal is reached → SUCCESS.  Otherwise generate a new subgoal
  and plan.
- When the trajectory is **active**, validate the path and goal against
  the updated map.  If either is invalid → replan.
- Compute a gaze target (active perception) and a motion action (path
  tracking), then execute both.

All components are duck-typed / protocol-based so swapping VAMP for an RL
policy, or swapping the BT for a learned subgoal generator, requires zero
changes here.
"""

from __future__ import annotations

from typing import Callable  # used by camera_pose_fn annotation

import numpy as np
import numpy.typing as npt

from TyGrit.core.config import (
    CheckFn,
    ControllerFn,
    GazeFn,
    SchedulerConfig,
)
from TyGrit.envs.base import RobotBase
from TyGrit.logging import log
from TyGrit.planning.motion_planner import MotionPlanner
from TyGrit.scene.scene import Scene
from TyGrit.subgoal_generator.protocol import SubGoalGenerator
from TyGrit.types.planning import PlanningMode, SchedulerFeedback, Subgoal, Trajectory
from TyGrit.types.results import PlanResult, SchedulerOutcome, SchedulerResult
from TyGrit.types.robot import RobotState, WholeBodyConfig

# ── The scheduler ────────────────────────────────────────────────────────


class Scheduler:
    """Receding-horizon loop that coordinates perception, planning, and control.

    Parameters
    ----------
    robot : RobotBase
        Robot interface (calls ``step()`` synchronously).
    scene : Scene
        World model (belief state).
    planner : MotionPlanner
        Collision-free motion planner.
    generator : SubGoalGenerator
        High-level decision-maker: produces subgoals and checks goal
        completion via ``generate()`` and ``goal_predicate()``.
    controller_fn : ControllerFn
        Low-level: ``(robot_state, trajectory, waypoint_idx) → action``.
    config : SchedulerConfig
        Tuning parameters.
    check_fn : CheckFn, optional
        ``(trajectory, scene) → (is_path_valid, is_goal_valid)``.
        Called every iteration when a trajectory is active.  If either
        flag is ``False``, the scheduler replans.  If ``None``, the
        active trajectory is assumed valid until exhausted.
    gaze_fn : GazeFn, optional
        Active perception: ``(trajectory, waypoint_idx) → gaze target or None``.
    camera_pose_fn : callable, optional
        ``(robot_state) → (4, 4) camera pose``.  Needed for scene updates.
        If ``None``, scene updates are skipped.
    """

    def __init__(
        self,
        robot: RobotBase,
        scene: Scene,
        planner: MotionPlanner,
        generator: SubGoalGenerator,
        controller_fn: ControllerFn,
        config: SchedulerConfig | None = None,
        check_fn: CheckFn | None = None,
        gaze_fn: GazeFn | None = None,
        camera_pose_fn: Callable[[RobotState], npt.NDArray[np.float64]] | None = None,
    ) -> None:
        self.robot = robot
        self.scene = scene
        self.planner = planner
        self.generator = generator
        self.controller_fn = controller_fn
        self.config = config or SchedulerConfig()
        self.check_fn = check_fn
        self.gaze_fn = gaze_fn
        self.camera_pose_fn = camera_pose_fn

    # ── Main loop ────────────────────────────────────────────────────────

    def run(self, max_iterations: int = 1000) -> SchedulerResult:
        """Execute the receding-horizon loop until the goal is reached.

        Returns
        -------
        SchedulerResult
            Outcome, iteration count, and total steps taken.
        """
        trajectory: Trajectory | None = None
        waypoint_idx = 0
        total_steps = 0

        for iteration in range(max_iterations):
            # 1. Observe
            obs = self.robot.get_observation()
            state = obs.robot_state

            # 2. Update map → sync planner collision environment
            if self.camera_pose_fn is not None:
                camera_pose = self.camera_pose_fn(state)
                self.scene.update(obs, camera_pose)
            self.planner.update_environment(
                self.scene.get_pointcloud(), state.base_pose
            )

            # 3. Plan or validate
            if trajectory is None or waypoint_idx >= len(trajectory.arm_path):
                # ξ = ∅ or IsFinished(ξ) — check goal, then replan
                if self.generator.goal_predicate(state):
                    log.info("Goal reached after {} iterations", iteration)
                    return SchedulerResult(
                        outcome=SchedulerOutcome.SUCCESS,
                        iterations=iteration,
                        total_steps=total_steps,
                    )

                feedback = SchedulerFeedback(trajectory_exhausted=True)
                trajectory, waypoint_idx = self._replan(state, feedback)
                if trajectory is None:
                    log.warning("Planning failed at iteration {}", iteration)
                    return SchedulerResult(
                        outcome=SchedulerOutcome.PLAN_FAILURE,
                        iterations=iteration,
                        total_steps=total_steps,
                    )
            else:
                # ξ active — Check(ξ, M): validate path and goal
                if self.check_fn is not None:
                    is_path_valid, is_goal_valid = self.check_fn(trajectory, self.scene)
                    if not is_path_valid or not is_goal_valid:
                        log.info(
                            "Replanning: path_valid={}, goal_valid={}",
                            is_path_valid,
                            is_goal_valid,
                        )
                        feedback = SchedulerFeedback(
                            is_path_valid=is_path_valid,
                            is_goal_valid=is_goal_valid,
                        )
                        trajectory, waypoint_idx = self._replan(state, feedback)
                        if trajectory is None:
                            log.warning("Planning failed at iteration {}", iteration)
                            return SchedulerResult(
                                outcome=SchedulerOutcome.PLAN_FAILURE,
                                iterations=iteration,
                                total_steps=total_steps,
                            )

            # 4. Active perception (gaze)
            assert trajectory is not None
            if self.gaze_fn is not None:
                gaze_target = self.gaze_fn(trajectory, waypoint_idx)
                if gaze_target is not None:
                    self.robot.look_at(gaze_target, "head")

            # 5. Path tracking
            action = self.controller_fn(state, trajectory, waypoint_idx)

            # 6. Execute
            for _ in range(self.config.steps_per_iteration):
                obs = self.robot.step(action)
                total_steps += 1

            waypoint_idx += 1

        log.warning("Max iterations ({}) reached", max_iterations)
        return SchedulerResult(
            outcome=SchedulerOutcome.MAX_ITERATIONS,
            iterations=max_iterations,
            total_steps=total_steps,
        )

    # ── Internal ─────────────────────────────────────────────────────────

    def _replan(
        self, state: RobotState, feedback: SchedulerFeedback
    ) -> tuple[Trajectory | None, int]:
        """Generate a subgoal and plan a trajectory to it.

        Dispatches to the planner method indicated by ``subgoal.mode``:

        - **ARM** → ``plan_arm`` (collision-aware, arm only)
        - **WHOLE_BODY** → ``plan_whole_body`` (collision-aware, arm + base)
        - **INTERPOLATION** → ``plan_interpolation`` (linear, no collision)

        Returns ``(trajectory, 0)`` on success, ``(None, 0)`` on failure.
        """
        subgoal = self.generator.generate(self.scene, state, feedback)
        if subgoal is None:
            return None, 0

        start_arm = np.array(state.planning_joints, dtype=np.float64)
        result = self._dispatch_plan(subgoal, start_arm, state)
        if result.success and result.trajectory is not None:
            return result.trajectory, 0
        return None, 0

    def _dispatch_plan(
        self,
        subgoal: Subgoal,
        start_arm: npt.NDArray[np.float64],
        state: RobotState,
    ) -> PlanResult:
        """Call the correct planner method for *subgoal.mode*."""
        if subgoal.mode == PlanningMode.ARM:
            return self.planner.plan_arm(start_arm, subgoal.arm_joints)

        if subgoal.mode == PlanningMode.WHOLE_BODY:
            start_wb = WholeBodyConfig(arm_joints=start_arm, base_pose=state.base_pose)
            goal_wb = WholeBodyConfig(
                arm_joints=np.asarray(subgoal.arm_joints, dtype=np.float64),
                base_pose=subgoal.base_pose or state.base_pose,
            )
            return self.planner.plan_whole_body(start_wb, goal_wb)

        # PlanningMode.INTERPOLATION
        return self.planner.plan_interpolation(
            start_arm, subgoal.arm_joints, state.base_pose
        )
