"""Tests for TyGrit.subgoal_generator.bt — behaviour-tree nodes and trees."""

from __future__ import annotations

import numpy as np
import py_trees

from TyGrit.subgoal_generator.bt.nodes import (
    GenerateSubGoal,
    IsGoalReached,
    IsTrajectoryValid,
    Observe,
    PlanMotion,
)
from TyGrit.subgoal_generator.tasks.grasp import build_grasp_tree
from TyGrit.types.geometry import SE2Pose
from TyGrit.types.planning import Trajectory
from TyGrit.types.results import PlanResult
from TyGrit.types.robot import RobotState
from TyGrit.types.sensor import SensorSnapshot

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_robot_state() -> RobotState:
    return RobotState(
        base_pose=SE2Pose(0.0, 0.0, 0.0),
        planning_joints=(0.0,) * 8,
        head_joints=(0.0, 0.0),
    )


def _make_snapshot() -> SensorSnapshot:
    return SensorSnapshot(
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        depth=np.zeros((2, 2), dtype=np.float32),
        intrinsics=np.eye(3),
        robot_state=_make_robot_state(),
    )


def _make_trajectory(n: int = 3) -> Trajectory:
    return Trajectory(
        arm_path=tuple(np.zeros(8) for _ in range(n)),
        base_configs=tuple(SE2Pose(0.0, 0.0, 0.0) for _ in range(n)),
    )


class MockEnv:
    def get_observation(self) -> SensorSnapshot:
        return _make_snapshot()


class MockPlanner:
    def __init__(self, succeed: bool = True):
        self._succeed = succeed

    def plan_arm(self, start, goal) -> PlanResult:
        if self._succeed:
            return PlanResult(success=True, trajectory=_make_trajectory())
        return PlanResult(success=False)


class MockCollisionChecker:
    def __init__(self, valid: bool = True):
        self._valid = valid

    def validate(self, config):
        return self._valid


def _setup_blackboard(**overrides):
    """Clear the blackboard and populate it with defaults + overrides."""
    py_trees.blackboard.Blackboard.enable_activity_stream()
    bb = py_trees.blackboard.Client(name="test")
    for key in [
        "env",
        "scene",
        "robot_state",
        "trajectory",
        "subgoal",
        "goal_predicate",
        "planner",
        "subgoal_fn",
        "collision_checker",
    ]:
        bb.register_key(key=key, access=py_trees.common.Access.WRITE)

    bb.env = MockEnv()
    bb.scene = None
    bb.robot_state = _make_robot_state()
    bb.trajectory = None
    bb.subgoal = None
    bb.goal_predicate = lambda state: False
    bb.planner = MockPlanner()
    bb.subgoal_fn = lambda scene, state: np.zeros(8)
    bb.collision_checker = None

    for key, val in overrides.items():
        setattr(bb, key, val)

    return bb


# ── IsGoalReached ────────────────────────────────────────────────────────


class TestIsGoalReached:
    def test_success_when_predicate_true(self):
        _setup_blackboard(goal_predicate=lambda s: True)
        node = IsGoalReached()
        node.setup()
        assert node.update() == py_trees.common.Status.SUCCESS

    def test_failure_when_predicate_false(self):
        _setup_blackboard(goal_predicate=lambda s: False)
        node = IsGoalReached()
        node.setup()
        assert node.update() == py_trees.common.Status.FAILURE


# ── IsTrajectoryValid ────────────────────────────────────────────────────


class TestIsTrajectoryValid:
    def test_failure_when_no_trajectory(self):
        _setup_blackboard(trajectory=None)
        node = IsTrajectoryValid()
        node.setup()
        assert node.update() == py_trees.common.Status.FAILURE

    def test_success_when_valid_trajectory(self):
        _setup_blackboard(
            trajectory=_make_trajectory(),
            collision_checker=MockCollisionChecker(valid=True),
        )
        node = IsTrajectoryValid()
        node.setup()
        assert node.update() == py_trees.common.Status.SUCCESS

    def test_failure_when_collision(self):
        _setup_blackboard(
            trajectory=_make_trajectory(),
            collision_checker=MockCollisionChecker(valid=False),
        )
        node = IsTrajectoryValid()
        node.setup()
        assert node.update() == py_trees.common.Status.FAILURE

    def test_success_without_checker(self):
        """If no collision checker, just check trajectory exists."""
        _setup_blackboard(trajectory=_make_trajectory(), collision_checker=None)
        node = IsTrajectoryValid()
        node.setup()
        assert node.update() == py_trees.common.Status.SUCCESS


# ── Observe ──────────────────────────────────────────────────────────────


class TestObserve:
    def test_updates_robot_state(self):
        bb = _setup_blackboard()
        node = Observe()
        node.setup()
        status = node.update()
        assert status == py_trees.common.Status.SUCCESS
        assert bb.robot_state is not None


# ── GenerateSubGoal ──────────────────────────────────────────────────────


class TestGenerateSubGoal:
    def test_success(self):
        bb = _setup_blackboard(subgoal_fn=lambda scene, state: np.ones(8))
        node = GenerateSubGoal()
        node.setup()
        assert node.update() == py_trees.common.Status.SUCCESS
        np.testing.assert_array_equal(bb.subgoal, np.ones(8))

    def test_failure_when_none(self):
        _setup_blackboard(subgoal_fn=lambda scene, state: None)
        node = GenerateSubGoal()
        node.setup()
        assert node.update() == py_trees.common.Status.FAILURE


# ── PlanMotion ───────────────────────────────────────────────────────────


class TestPlanMotion:
    def test_success(self):
        bb = _setup_blackboard(
            subgoal=np.zeros(8),
            planner=MockPlanner(succeed=True),
        )
        node = PlanMotion()
        node.setup()
        assert node.update() == py_trees.common.Status.SUCCESS
        assert bb.trajectory is not None

    def test_failure(self):
        _setup_blackboard(
            subgoal=np.zeros(8),
            planner=MockPlanner(succeed=False),
        )
        node = PlanMotion()
        node.setup()
        assert node.update() == py_trees.common.Status.FAILURE


# ── Tree structure ───────────────────────────────────────────────────────


class TestGraspTree:
    def test_builds_without_error(self):
        tree = build_grasp_tree()
        assert tree is not None
        assert tree.name == "grasp_root"

    def test_tick_goal_reached(self):
        """When goal is reached, tree returns SUCCESS."""
        _setup_blackboard(goal_predicate=lambda s: True)
        root = build_grasp_tree()
        root.setup_with_descendants()
        root.tick_once()
        # Observe succeeds, then IsGoalReached succeeds → root succeeds
        assert root.status == py_trees.common.Status.SUCCESS

    def test_tick_plan_and_execute(self):
        """When goal not reached, tree plans and returns SUCCESS."""
        _setup_blackboard(
            goal_predicate=lambda s: False,
            subgoal_fn=lambda scene, state: np.zeros(8),
            planner=MockPlanner(succeed=True),
            collision_checker=None,
        )
        root = build_grasp_tree()
        root.setup_with_descendants()
        root.tick_once()
        # Observe → IsGoalReached FAIL → IsTrajectoryValid FAIL →
        # GenerateSubGoal SUCCESS → PlanMotion SUCCESS → root SUCCESS
        assert root.status == py_trees.common.Status.SUCCESS
