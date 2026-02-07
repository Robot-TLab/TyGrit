"""Per-subsystem failure enums.

Each subsystem defines its own failure type.  These are leaf labels —
they describe *what* failed, not *why* the overall system failed.

Causal analysis (building the failure tree, identifying root cause) is
done separately by the analysis system, not here.

Add new enums as subsystems are built.  Keep variants minimal — only
failures we've actually observed or designed for.
"""

from enum import Enum


class PlannerFailure(Enum):
    """Motion planner failures."""

    COLLISION_AT_START = "collision_at_start"
    COLLISION_AT_GOAL = "collision_at_goal"
    NO_PATH_FOUND = "no_path_found"
    TIMEOUT = "timeout"


class IKFailure(Enum):
    """Inverse kinematics failures."""

    NO_SOLUTION = "no_solution"
    JOINT_LIMITS = "joint_limits"


class GraspFailure(Enum):
    """Grasp prediction failures."""

    NO_GRASPS_DETECTED = "no_grasps_detected"
    ALL_UNREACHABLE = "all_unreachable"


class PerceptionFailure(Enum):
    """Perception subsystem failures."""

    SEGMENTATION_FAILED = "segmentation_failed"
    DEPTH_INVALID = "depth_invalid"


class ExecutionFailure(Enum):
    """Trajectory execution failures."""

    COLLISION_DETECTED = "collision_detected"
    TIMEOUT = "timeout"
    JOINT_ERROR = "joint_error"
