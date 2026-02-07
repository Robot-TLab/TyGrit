"""Tests for TyGrit.core.scheduler — the receding-horizon loop."""

from __future__ import annotations

import numpy as np

from TyGrit.core.scheduler import Scheduler, SchedulerConfig
from TyGrit.types.geometry import SE2Pose
from TyGrit.types.planning import Trajectory
from TyGrit.types.results import PlanResult, SchedulerOutcome
from TyGrit.types.robot import RobotState
from TyGrit.types.sensor import SensorSnapshot

# ── Helpers / Mocks ──────────────────────────────────────────────────────


def _make_robot_state(joints: tuple[float, ...] = (0.0,) * 8) -> RobotState:
    return RobotState(
        base_pose=SE2Pose(0.0, 0.0, 0.0),
        planning_joints=joints,
        head_joints=(0.0, 0.0),
    )


def _make_snapshot(state: RobotState | None = None) -> SensorSnapshot:
    if state is None:
        state = _make_robot_state()
    return SensorSnapshot(
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        depth=np.zeros((2, 2), dtype=np.float32),
        intrinsics=np.eye(3),
        robot_state=state,
    )


def _make_trajectory(n_waypoints: int = 5) -> Trajectory:
    arm_path = tuple(np.full(8, float(i)) for i in range(n_waypoints))
    base_configs = tuple(SE2Pose(0.0, 0.0, 0.0) for _ in range(n_waypoints))
    return Trajectory(arm_path=arm_path, base_configs=base_configs)


def _noop_subgoal(_scene, _state):
    return np.zeros(8)


def _noop_controller(_state, _traj, _idx):
    return np.zeros(10, dtype=np.float32)


class MockRobot:
    """Mock robot that counts steps."""

    def __init__(self) -> None:
        self.step_count = 0
        self._state = _make_robot_state()

    def step(self, action: np.ndarray) -> SensorSnapshot:  # noqa: ARG002
        self.step_count += 1
        return _make_snapshot(self._state)

    def get_observation(self) -> SensorSnapshot:
        return _make_snapshot(self._state)


class MockScene:
    """Mock scene that tracks update calls."""

    def __init__(self) -> None:
        self.update_count = 0

    def update(
        self, snapshot: SensorSnapshot, camera_pose: np.ndarray
    ) -> None:  # noqa: ARG002
        self.update_count += 1


class MockPlanner:
    """Mock planner that returns a straight-line trajectory."""

    def __init__(self, n_waypoints: int = 5) -> None:
        self.n_waypoints = n_waypoints
        self.plan_count = 0

    def plan_arm(
        self, start: np.ndarray, goal: np.ndarray
    ) -> PlanResult:  # noqa: ARG002
        self.plan_count += 1
        traj = _make_trajectory(self.n_waypoints)
        return PlanResult(success=True, trajectory=traj)


class FailPlanner:
    """Mock planner that always fails."""

    def plan_arm(
        self, start: np.ndarray, goal: np.ndarray
    ) -> PlanResult:  # noqa: ARG002
        return PlanResult(success=False)


# ── Tests ────────────────────────────────────────────────────────────────


class TestScheduler:
    def test_immediate_goal(self):
        """If goal is already reached, scheduler returns SUCCESS with 0 iterations."""
        scheduler = Scheduler(
            robot=MockRobot(),
            scene=MockScene(),
            planner=MockPlanner(),
            subgoal_fn=_noop_subgoal,
            controller_fn=_noop_controller,
            goal_predicate=lambda _state: True,
        )
        result = scheduler.run(max_iterations=100)
        assert result.outcome == SchedulerOutcome.SUCCESS
        assert result.iterations == 0
        assert result.total_steps == 0

    def test_runs_expected_steps(self):
        """Scheduler steps the robot steps_per_iteration times per iteration."""
        robot = MockRobot()
        n_waypoints = 3
        steps_per = 5
        config = SchedulerConfig(steps_per_iteration=steps_per)
        call_count = [0]

        def goal_after_n(_state):
            call_count[0] += 1
            return call_count[0] > n_waypoints + 1

        scheduler = Scheduler(
            robot=robot,
            scene=MockScene(),
            planner=MockPlanner(n_waypoints=n_waypoints),
            subgoal_fn=_noop_subgoal,
            controller_fn=_noop_controller,
            goal_predicate=goal_after_n,
            config=config,
        )
        result = scheduler.run(max_iterations=100)
        assert result.outcome == SchedulerOutcome.SUCCESS
        assert robot.step_count == result.iterations * steps_per

    def test_plan_failure(self):
        """If planning fails, scheduler returns PLAN_FAILURE."""
        scheduler = Scheduler(
            robot=MockRobot(),
            scene=MockScene(),
            planner=FailPlanner(),
            subgoal_fn=_noop_subgoal,
            controller_fn=_noop_controller,
            goal_predicate=lambda _state: False,
        )
        result = scheduler.run(max_iterations=100)
        assert result.outcome == SchedulerOutcome.PLAN_FAILURE

    def test_max_iterations(self):
        """Scheduler stops after max_iterations."""
        scheduler = Scheduler(
            robot=MockRobot(),
            scene=MockScene(),
            planner=MockPlanner(n_waypoints=1000),
            subgoal_fn=_noop_subgoal,
            controller_fn=_noop_controller,
            goal_predicate=lambda _state: False,
        )
        result = scheduler.run(max_iterations=5)
        assert result.outcome == SchedulerOutcome.MAX_ITERATIONS
        assert result.iterations == 5

    def test_replans_when_trajectory_exhausted(self):
        """Scheduler replans when all waypoints are consumed."""
        planner = MockPlanner(n_waypoints=2)
        call_count = [0]

        def goal_after_5(_state):
            call_count[0] += 1
            return call_count[0] > 5

        scheduler = Scheduler(
            robot=MockRobot(),
            scene=MockScene(),
            planner=planner,
            subgoal_fn=_noop_subgoal,
            controller_fn=_noop_controller,
            goal_predicate=goal_after_5,
        )
        result = scheduler.run(max_iterations=100)
        assert result.outcome == SchedulerOutcome.SUCCESS
        assert planner.plan_count >= 2

    def test_subgoal_none_causes_plan_failure(self):
        """If subgoal_fn returns None, planning fails."""
        scheduler = Scheduler(
            robot=MockRobot(),
            scene=MockScene(),
            planner=MockPlanner(),
            subgoal_fn=lambda _scene, _state: None,
            controller_fn=_noop_controller,
            goal_predicate=lambda _state: False,
        )
        result = scheduler.run(max_iterations=100)
        assert result.outcome == SchedulerOutcome.PLAN_FAILURE

    def test_scene_updated_when_camera_pose_fn_provided(self):
        """Scene.update is called when camera_pose_fn is provided."""
        scene = MockScene()
        call_count = [0]

        def goal_after_2(_state):
            call_count[0] += 1
            return call_count[0] > 2

        scheduler = Scheduler(
            robot=MockRobot(),
            scene=scene,
            planner=MockPlanner(),
            subgoal_fn=_noop_subgoal,
            controller_fn=_noop_controller,
            goal_predicate=goal_after_2,
            camera_pose_fn=lambda _state: np.eye(4),
        )
        result = scheduler.run()
        assert result.outcome == SchedulerOutcome.SUCCESS
        assert scene.update_count >= 2

    def test_scene_not_updated_without_camera_pose_fn(self):
        """Scene.update is NOT called when camera_pose_fn is None."""
        scene = MockScene()
        call_count = [0]

        def goal_after_1(_state):
            call_count[0] += 1
            return call_count[0] > 1

        scheduler = Scheduler(
            robot=MockRobot(),
            scene=scene,
            planner=MockPlanner(),
            subgoal_fn=_noop_subgoal,
            controller_fn=_noop_controller,
            goal_predicate=goal_after_1,
        )
        result = scheduler.run()
        assert result.outcome == SchedulerOutcome.SUCCESS
        assert scene.update_count == 0

    def test_controller_receives_trajectory(self):
        """Controller function receives the planned trajectory."""
        trajectories_seen = []
        call_count = [0]

        def record_controller(_state, traj, _idx):
            trajectories_seen.append(traj)
            return np.zeros(10, dtype=np.float32)

        def goal_after_2(_state):
            call_count[0] += 1
            return call_count[0] > 2

        scheduler = Scheduler(
            robot=MockRobot(),
            scene=MockScene(),
            planner=MockPlanner(n_waypoints=5),
            subgoal_fn=_noop_subgoal,
            controller_fn=record_controller,
            goal_predicate=goal_after_2,
        )
        result = scheduler.run()
        assert result.outcome == SchedulerOutcome.SUCCESS
        assert len(trajectories_seen) >= 2
        for traj in trajectories_seen:
            assert traj is not None
