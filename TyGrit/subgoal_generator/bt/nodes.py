"""py_trees behaviour-tree nodes for the receding-horizon scheduler.

Each node wraps a single call to one of the scheduler's pluggable components.
Nodes communicate via the py_trees blackboard.

Blackboard keys
---------------
- ``robot_state``  : RobotState (written by Observe)
- ``scene``        : SceneRepresentation (read/write by Observe, read by others)
- ``trajectory``   : Trajectory | None (written by PlanMotion)
- ``subgoal``      : NDArray | None (written by GenerateSubGoal)
- ``goal_predicate``: Callable[[RobotState], bool] (set at tree init)
- ``env``          : RobotBase
- ``planner``      : MotionPlanner
- ``subgoal_fn``   : Callable (high-level subgoal generator)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import py_trees

if TYPE_CHECKING:
    pass


# ── Condition nodes ──────────────────────────────────────────────────────


class IsGoalReached(py_trees.behaviour.Behaviour):
    """SUCCESS if the goal predicate holds for the current robot state."""

    def __init__(self, name: str = "IsGoalReached"):
        super().__init__(name=name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(
            key="robot_state", access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(
            key="goal_predicate", access=py_trees.common.Access.READ
        )

    def update(self) -> py_trees.common.Status:
        predicate = self.blackboard.goal_predicate
        state = self.blackboard.robot_state
        if predicate(state):
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class IsTrajectoryValid(py_trees.behaviour.Behaviour):
    """SUCCESS if the current trajectory exists and is collision-free.

    Checks that:
    1. A trajectory exists on the blackboard.
    2. The collision checker (if available) validates all waypoints.
    """

    def __init__(self, name: str = "IsTrajectoryValid"):
        super().__init__(name=name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(
            key="trajectory", access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(
            key="collision_checker", access=py_trees.common.Access.READ
        )

    def update(self) -> py_trees.common.Status:
        traj = self.blackboard.trajectory
        if traj is None:
            return py_trees.common.Status.FAILURE

        checker = self.blackboard.collision_checker
        if checker is not None:
            for config in traj.arm_path:
                if not checker.validate(config):
                    return py_trees.common.Status.FAILURE

        return py_trees.common.Status.SUCCESS


# ── Action nodes ─────────────────────────────────────────────────────────


class Observe(py_trees.behaviour.Behaviour):
    """Capture a sensor snapshot and update the scene."""

    def __init__(self, name: str = "Observe"):
        super().__init__(name=name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key="env", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="scene", access=py_trees.common.Access.READ)
        self.blackboard.register_key(
            key="robot_state", access=py_trees.common.Access.WRITE
        )

    def update(self) -> py_trees.common.Status:
        env = self.blackboard.env
        obs = env.get_observation()
        self.blackboard.robot_state = obs.robot_state
        return py_trees.common.Status.SUCCESS


class GenerateSubGoal(py_trees.behaviour.Behaviour):
    """Call the high-level subgoal generator and write the result."""

    def __init__(self, name: str = "GenerateSubGoal"):
        super().__init__(name=name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(
            key="subgoal_fn", access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(key="scene", access=py_trees.common.Access.READ)
        self.blackboard.register_key(
            key="robot_state", access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(key="subgoal", access=py_trees.common.Access.WRITE)

    def update(self) -> py_trees.common.Status:
        fn = self.blackboard.subgoal_fn
        scene = self.blackboard.scene
        state = self.blackboard.robot_state
        subgoal = fn(scene, state)
        if subgoal is None:
            return py_trees.common.Status.FAILURE
        self.blackboard.subgoal = subgoal
        return py_trees.common.Status.SUCCESS


class PlanMotion(py_trees.behaviour.Behaviour):
    """Plan a trajectory from the current config to the subgoal."""

    def __init__(self, name: str = "PlanMotion"):
        super().__init__(name=name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key="planner", access=py_trees.common.Access.READ)
        self.blackboard.register_key(
            key="robot_state", access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(key="subgoal", access=py_trees.common.Access.READ)
        self.blackboard.register_key(
            key="trajectory", access=py_trees.common.Access.WRITE
        )

    def update(self) -> py_trees.common.Status:
        import numpy as np

        planner = self.blackboard.planner
        state = self.blackboard.robot_state
        subgoal = self.blackboard.subgoal

        start = np.array(state.planning_joints, dtype=np.float64)
        result = planner.plan_arm(start, subgoal)

        if result.success and result.trajectory is not None:
            self.blackboard.trajectory = result.trajectory
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE
