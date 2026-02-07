"""Receding-horizon scheduler — the core control loop.

This replaces the monolithic ``move_to_config_with_replanning()`` from
grasp_anywhere.  There is no background thread, no ``time.sleep()``, no
locks.  Every iteration is sequential:

    observe → update map → check/replan → compute control → step × N

The scheduler is parameterised by pluggable components:

- **env** — ``RobotBase`` (calls ``step()`` synchronously)
- **scene** — ``SceneRepresentation`` (world model)
- **planner** — ``MotionPlanner`` (low-level motion planning)
- **subgoal_fn** — callable that produces the next subgoal
- **controller** — callable that maps (state, trajectory) → action
- **goal_predicate** — callable that decides when to stop

All of these are duck-typed / protocol-based so swapping VAMP for an RL
policy, or swapping the BT for a learned subgoal generator, requires zero
changes here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Protocol

import numpy as np
import numpy.typing as npt

from TyGrit.logging import log
from TyGrit.types.config import SchedulerConfig
from TyGrit.types.planning import PlanResult, Trajectory
from TyGrit.types.robot import RobotState
from TyGrit.types.sensor import SensorSnapshot

# ── Structural protocols (what the scheduler actually needs) ─────────────


class _EnvLike(Protocol):
    """Minimal env interface the scheduler needs."""

    def step(self, action: npt.NDArray[np.float32]) -> SensorSnapshot: ...

    def get_observation(self) -> SensorSnapshot: ...


class _SceneLike(Protocol):
    """Minimal scene interface the scheduler needs."""

    def update(
        self,
        snapshot: SensorSnapshot,
        camera_pose: npt.NDArray[np.float64],
    ) -> None: ...


class _PlannerLike(Protocol):
    """Minimal planner interface the scheduler needs."""

    def plan_arm(
        self,
        start: npt.NDArray[np.float64],
        goal: npt.NDArray[np.float64],
    ) -> PlanResult: ...


# Type aliases for the pluggable callables
SubGoalFn = Callable[[_SceneLike, RobotState], npt.NDArray[np.float64] | None]
ControllerFn = Callable[[RobotState, Trajectory, int], npt.NDArray[np.float32]]
GoalPredicate = Callable[[RobotState], bool]
GazeFn = Callable[[Trajectory, int], npt.NDArray[np.float64] | None]


# ── Result types ─────────────────────────────────────────────────────────


class SchedulerOutcome(Enum):
    """Possible outcomes of a scheduler run."""

    SUCCESS = "success"
    MAX_ITERATIONS = "max_iterations"
    PLAN_FAILURE = "plan_failure"


@dataclass
class SchedulerResult:
    """Result returned by :meth:`Scheduler.run`."""

    outcome: SchedulerOutcome
    iterations: int = 0
    total_steps: int = 0
    stats: dict[str, float] = field(default_factory=dict)


# ── The scheduler ────────────────────────────────────────────────────────


class Scheduler:
    """Receding-horizon loop that coordinates perception, planning, and control.

    Parameters
    ----------
    env
        Any object with ``step(action)`` and ``get_observation()``
        (e.g. a ``RobotBase`` subclass).
    scene
        Any object with ``update(snapshot, camera_pose)``
        (e.g. a ``SceneRepresentation``).
    planner
        Any object with ``plan_arm(start, goal)``
        (e.g. a ``MotionPlanner``).
    subgoal_fn : SubGoalFn
        High-level: ``(scene, robot_state) → goal config or None``.
    controller_fn : ControllerFn
        Low-level: ``(robot_state, trajectory, waypoint_idx) → action``.
    goal_predicate : GoalPredicate
        Returns ``True`` when the task is complete.
    config : SchedulerConfig
        Tuning parameters.
    gaze_fn : GazeFn, optional
        Active perception: ``(trajectory, waypoint_idx) → gaze target or None``.
    camera_pose_fn : callable, optional
        ``(robot_state) → (4, 4) camera pose``.  Needed for scene updates.
        If ``None``, scene updates are skipped.
    """

    def __init__(
        self,
        env: _EnvLike,
        scene: _SceneLike,
        planner: _PlannerLike,
        subgoal_fn: SubGoalFn,
        controller_fn: ControllerFn,
        goal_predicate: GoalPredicate,
        config: SchedulerConfig | None = None,
        gaze_fn: GazeFn | None = None,
        camera_pose_fn: Callable[[RobotState], npt.NDArray[np.float64]] | None = None,
    ) -> None:
        self.env = env
        self.scene = scene
        self.planner = planner
        self.subgoal_fn = subgoal_fn
        self.controller_fn = controller_fn
        self.goal_predicate = goal_predicate
        self.config = config or SchedulerConfig()
        self.gaze_fn = gaze_fn
        self.camera_pose_fn = camera_pose_fn

    # ── Main loop ────────────────────────────────────────────────────────

    def run(self, max_iterations: int = 1000) -> SchedulerResult:
        """Execute the receding-horizon loop until the goal is reached.

        Returns
        -------
        SchedulerResult
            Outcome, iteration count, and total env steps taken.
        """
        trajectory: Trajectory | None = None
        waypoint_idx = 0
        total_steps = 0

        for iteration in range(max_iterations):
            # 1. Observe
            obs = self.env.get_observation()
            state = obs.robot_state

            # 2. Update scene (if camera pose is available)
            if self.camera_pose_fn is not None:
                camera_pose = self.camera_pose_fn(state)
                self.scene.update(obs, camera_pose)

            # 3. Check goal
            if self.goal_predicate(state):
                log.info("Goal reached after {} iterations", iteration)
                return SchedulerResult(
                    outcome=SchedulerOutcome.SUCCESS,
                    iterations=iteration,
                    total_steps=total_steps,
                )

            # 4. Plan or reuse trajectory
            need_replan = trajectory is None or waypoint_idx >= len(trajectory.arm_path)
            if need_replan:
                trajectory, waypoint_idx = self._replan(state)
                if trajectory is None:
                    log.warning("Planning failed at iteration {}", iteration)
                    return SchedulerResult(
                        outcome=SchedulerOutcome.PLAN_FAILURE,
                        iterations=iteration,
                        total_steps=total_steps,
                    )

            # 5. Compute control action (trajectory guaranteed non-None here)
            assert trajectory is not None
            action = self.controller_fn(state, trajectory, waypoint_idx)

            # 6. Execute N steps
            for _ in range(self.config.steps_per_iteration):
                obs = self.env.step(action)
                total_steps += 1

            # Advance waypoint
            waypoint_idx += 1

        log.warning("Max iterations ({}) reached", max_iterations)
        return SchedulerResult(
            outcome=SchedulerOutcome.MAX_ITERATIONS,
            iterations=max_iterations,
            total_steps=total_steps,
        )

    # ── Internal ─────────────────────────────────────────────────────────

    def _replan(self, state: RobotState) -> tuple[Trajectory | None, int]:
        """Generate a subgoal and plan a trajectory to it.

        Returns ``(trajectory, 0)`` on success, ``(None, 0)`` on failure.
        """
        subgoal = self.subgoal_fn(self.scene, state)
        if subgoal is None:
            return None, 0

        start = np.array(state.planning_joints, dtype=np.float64)
        result = self.planner.plan_arm(start, subgoal)
        if result.success and result.trajectory is not None:
            return result.trajectory, 0
        return None, 0
