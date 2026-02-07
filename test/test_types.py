"""Tests for TyGrit.types dataclasses."""

import numpy as np
import pytest

from TyGrit.types import (
    ExecutionFailure,
    GraspFailure,
    GraspPose,
    IKFailure,
    IKSolution,
    JointState,
    PerceptionFailure,
    PlannerFailure,
    PlanResult,
    RobotState,
    SceneConfig,
    SE2Pose,
    SensorSnapshot,
    StageResult,
    SystemConfig,
    Trajectory,
    WholeBodyConfig,
)

# ── geometry ─────────────────────────────────────────────────────────────────


class TestSE2Pose:
    def test_create(self):
        p = SE2Pose(1.0, 2.0, 0.5)
        assert p.x == 1.0
        assert p.y == 2.0
        assert p.theta == 0.5

    def test_frozen(self):
        p = SE2Pose(1.0, 2.0, 0.5)
        with pytest.raises(AttributeError):
            p.x = 3.0  # type: ignore[misc]


# ── robot ────────────────────────────────────────────────────────────────────


class TestJointState:
    def test_create(self):
        js = JointState(names=("a", "b"), positions=(1.0, 2.0))
        assert js.names == ("a", "b")

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            JointState(names=("a",), positions=(1.0, 2.0))


class TestIKSolution:
    def test_create(self):
        sol = IKSolution(
            joint_names=("j1", "j2", "j3"),
            positions=np.array([0.1, 0.2, 0.3]),
        )
        assert sol.joint_names == ("j1", "j2", "j3")
        assert sol.positions.shape == (3,)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            IKSolution(
                joint_names=("j1", "j2"),
                positions=np.array([0.1, 0.2, 0.3]),
            )


class TestRobotState:
    def test_create(self):
        rs = RobotState(
            base_pose=SE2Pose(0, 0, 0),
            planning_joints=(0.0,) * 8,
            head_joints=(0.0, 0.0),
        )
        assert len(rs.planning_joints) == 8
        assert len(rs.head_joints) == 2


class TestWholeBodyConfig:
    def test_create(self):
        wbc = WholeBodyConfig(
            arm_joints=np.zeros(8),
            base_pose=SE2Pose(0, 0, 0),
        )
        assert wbc.arm_joints.shape == (8,)


# ── sensor ───────────────────────────────────────────────────────────────────


class TestSensorSnapshot:
    def test_create(self):
        ss = SensorSnapshot(
            rgb=np.zeros((480, 640, 3), dtype=np.uint8),
            depth=np.zeros((480, 640), dtype=np.float32),
            intrinsics=np.eye(3),
            robot_state=RobotState(
                base_pose=SE2Pose(0, 0, 0),
                planning_joints=(0.0,) * 8,
                head_joints=(0.0, 0.0),
            ),
        )
        assert ss.segmentation is None


# ── planning ─────────────────────────────────────────────────────────────────


class TestSubsystemFailures:
    def test_planner_failure(self):
        assert PlannerFailure.NO_PATH_FOUND.value == "no_path_found"
        assert PlannerFailure.TIMEOUT.value == "timeout"

    def test_ik_failure(self):
        assert IKFailure.NO_SOLUTION.value == "no_solution"

    def test_grasp_failure(self):
        assert GraspFailure.NO_GRASPS_DETECTED.value == "no_grasps_detected"

    def test_perception_failure(self):
        assert PerceptionFailure.DEPTH_INVALID.value == "depth_invalid"

    def test_execution_failure(self):
        assert ExecutionFailure.COLLISION_DETECTED.value == "collision_detected"


class TestPlanResult:
    def test_success(self):
        traj = Trajectory(
            arm_path=(np.zeros(8),),
            base_configs=(SE2Pose(0, 0, 0),),
        )
        pr = PlanResult(success=True, trajectory=traj)
        assert pr.success
        assert pr.failure is None

    def test_failure(self):
        pr = PlanResult(success=False, failure=PlannerFailure.TIMEOUT)
        assert not pr.success
        assert pr.failure == PlannerFailure.TIMEOUT
        assert pr.trajectory is None


class TestStageResult:
    def test_success(self):
        sr = StageResult(success=True)
        assert sr.message == ""

    def test_failure_accepts_any_subsystem(self):
        """StageResult.failure accepts any subsystem failure enum."""
        sr = StageResult(
            success=False,
            failure=GraspFailure.ALL_UNREACHABLE,
            message="no IK",
        )
        assert sr.message == "no IK"

        sr2 = StageResult(success=False, failure=IKFailure.NO_SOLUTION)
        assert sr2.failure == IKFailure.NO_SOLUTION


class TestGraspPose:
    def test_create(self):
        gp = GraspPose(transform=np.eye(4), score=0.95)
        assert gp.score == 0.95


# ── config ───────────────────────────────────────────────────────────────────


class TestConfig:
    def test_system_config_defaults(self):
        cfg = SystemConfig()
        assert cfg.robot.name == "fetch"
        assert cfg.scene.ground_z_threshold == 0.3
        assert cfg.gaze.lookahead_window == 80
        assert cfg.planner.timeout == 5.0

    def test_scene_config_custom(self):
        sc = SceneConfig(ground_z_threshold=0.5, merge_radius=0.05)
        assert sc.ground_z_threshold == 0.5
        assert sc.merge_radius == 0.05
