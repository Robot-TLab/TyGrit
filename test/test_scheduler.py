"""Tests for TyGrit.core.scheduler — the receding-horizon loop."""

from __future__ import annotations

import numpy as np

from TyGrit.core.config import SchedulerConfig
from TyGrit.core.scheduler import Scheduler
from TyGrit.types.geometry import SE2Pose
from TyGrit.types.planning import PlanningMode, Subgoal, Trajectory
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


class MockGenerator:
    """Mock subgoal generator wrapping callables."""

    def __init__(self, subgoal_fn, goal_pred):
        self._fn = subgoal_fn
        self._pred = goal_pred

    def generate(self, scene, state, feedback):
        return self._fn(scene, state, feedback)

    def goal_predicate(self, state):
        return self._pred(state)


def _noop_subgoal(_scene, _state, _feedback):
    return Subgoal(PlanningMode.ARM, np.zeros(8))


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

    def get_pointcloud(self) -> np.ndarray:
        return np.empty((0, 3), dtype=np.float32)


class MockPlanner:
    """Mock planner that returns a straight-line trajectory."""

    def __init__(self, n_waypoints: int = 5) -> None:
        self.n_waypoints = n_waypoints
        self.plan_count = 0

    def update_environment(self, points: np.ndarray, base_pose) -> None:  # noqa: ARG002
        pass

    def plan_arm(self, start, goal) -> PlanResult:  # noqa: ARG002
        self.plan_count += 1
        return PlanResult(success=True, trajectory=_make_trajectory(self.n_waypoints))

    def plan_whole_body(self, start, goal) -> PlanResult:  # noqa: ARG002
        self.plan_count += 1
        return PlanResult(success=True, trajectory=_make_trajectory(self.n_waypoints))

    def plan_interpolation(self, start, goal, base_pose) -> PlanResult:  # noqa: ARG002
        self.plan_count += 1
        return PlanResult(success=True, trajectory=_make_trajectory(self.n_waypoints))


class FailPlanner:
    """Mock planner that always fails."""

    def update_environment(self, points: np.ndarray, base_pose) -> None:  # noqa: ARG002
        pass

    def plan_arm(self, start, goal) -> PlanResult:  # noqa: ARG002
        return PlanResult(success=False)

    def plan_whole_body(self, start, goal) -> PlanResult:  # noqa: ARG002
        return PlanResult(success=False)

    def plan_interpolation(self, start, goal, base_pose) -> PlanResult:  # noqa: ARG002
        return PlanResult(success=False)


# ── Tests ────────────────────────────────────────────────────────────────


class TestScheduler:
    def test_immediate_goal(self):
        """If goal is already reached, scheduler returns SUCCESS with 0 iterations."""
        scheduler = Scheduler(
            robot=MockRobot(),
            scene=MockScene(),
            planner=MockPlanner(),
            generator=MockGenerator(_noop_subgoal, lambda _state: True),
            controller_fn=_noop_controller,
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
            generator=MockGenerator(_noop_subgoal, goal_after_n),
            controller_fn=_noop_controller,
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
            generator=MockGenerator(_noop_subgoal, lambda _state: False),
            controller_fn=_noop_controller,
        )
        result = scheduler.run(max_iterations=100)
        assert result.outcome == SchedulerOutcome.PLAN_FAILURE

    def test_max_iterations(self):
        """Scheduler stops after max_iterations."""
        scheduler = Scheduler(
            robot=MockRobot(),
            scene=MockScene(),
            planner=MockPlanner(n_waypoints=1000),
            generator=MockGenerator(_noop_subgoal, lambda _state: False),
            controller_fn=_noop_controller,
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
            generator=MockGenerator(_noop_subgoal, goal_after_5),
            controller_fn=_noop_controller,
        )
        result = scheduler.run(max_iterations=100)
        assert result.outcome == SchedulerOutcome.SUCCESS
        assert planner.plan_count >= 2

    def test_subgoal_none_causes_plan_failure(self):
        """If generator returns None, planning fails."""
        scheduler = Scheduler(
            robot=MockRobot(),
            scene=MockScene(),
            planner=MockPlanner(),
            generator=MockGenerator(
                lambda _scene, _state, _feedback: None,
                lambda _state: False,
            ),
            controller_fn=_noop_controller,
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
            generator=MockGenerator(_noop_subgoal, goal_after_2),
            controller_fn=_noop_controller,
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
            generator=MockGenerator(_noop_subgoal, goal_after_1),
            controller_fn=_noop_controller,
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
            generator=MockGenerator(_noop_subgoal, goal_after_2),
            controller_fn=record_controller,
        )
        result = scheduler.run()
        assert result.outcome == SchedulerOutcome.SUCCESS
        assert len(trajectories_seen) >= 2
        for traj in trajectories_seen:
            assert traj is not None

    def test_feedback_trajectory_exhausted(self):
        """Subgoal receives trajectory_exhausted=True when trajectory is done."""
        feedbacks = []

        def record_subgoal(_scene, _state, feedback):
            feedbacks.append(feedback)
            return Subgoal(PlanningMode.ARM, np.zeros(8))

        call_count = [0]

        def goal_after_2(_state):
            call_count[0] += 1
            return call_count[0] > 2

        scheduler = Scheduler(
            robot=MockRobot(),
            scene=MockScene(),
            planner=MockPlanner(n_waypoints=1),
            generator=MockGenerator(record_subgoal, goal_after_2),
            controller_fn=_noop_controller,
        )
        scheduler.run(max_iterations=10)
        assert len(feedbacks) >= 1
        assert feedbacks[0].trajectory_exhausted is True

    def test_feedback_check_fn_path_invalid(self):
        """Subgoal receives is_path_valid=False when check_fn says path is bad."""
        feedbacks = []

        def record_subgoal(_scene, _state, feedback):
            feedbacks.append(feedback)
            return Subgoal(PlanningMode.ARM, np.zeros(8))

        call_count = [0]

        def goal_after_3(_state):
            call_count[0] += 1
            return call_count[0] > 3

        # check_fn returns path_invalid on second call (first iteration uses exhausted path)
        check_count = [0]

        def check_path_invalid(_traj, _scene):
            check_count[0] += 1
            return (False, True)  # path invalid, goal valid

        scheduler = Scheduler(
            robot=MockRobot(),
            scene=MockScene(),
            planner=MockPlanner(n_waypoints=10),
            generator=MockGenerator(record_subgoal, goal_after_3),
            controller_fn=_noop_controller,
            check_fn=check_path_invalid,
        )
        scheduler.run(max_iterations=10)
        # First feedback is trajectory_exhausted, subsequent are from check_fn
        check_feedbacks = [f for f in feedbacks if not f.trajectory_exhausted]
        assert len(check_feedbacks) >= 1
        assert check_feedbacks[0].is_path_valid is False
        assert check_feedbacks[0].is_goal_valid is True

    def test_feedback_check_fn_goal_invalid(self):
        """Subgoal receives is_goal_valid=False when check_fn says goal is bad."""
        feedbacks = []

        def record_subgoal(_scene, _state, feedback):
            feedbacks.append(feedback)
            return Subgoal(PlanningMode.ARM, np.zeros(8))

        call_count = [0]

        def goal_after_3(_state):
            call_count[0] += 1
            return call_count[0] > 3

        def check_goal_invalid(_traj, _scene):
            return (True, False)  # path valid, goal invalid

        scheduler = Scheduler(
            robot=MockRobot(),
            scene=MockScene(),
            planner=MockPlanner(n_waypoints=10),
            generator=MockGenerator(record_subgoal, goal_after_3),
            controller_fn=_noop_controller,
            check_fn=check_goal_invalid,
        )
        scheduler.run(max_iterations=10)
        check_feedbacks = [f for f in feedbacks if not f.trajectory_exhausted]
        assert len(check_feedbacks) >= 1
        assert check_feedbacks[0].is_path_valid is True
        assert check_feedbacks[0].is_goal_valid is False
