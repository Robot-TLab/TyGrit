"""Event tracking system — the flight recorder.

Similar to the logging system, the tracker has configurable levels
that control how much data gets recorded:

- ``FAILURE_INFO``: failure label + step + small metadata.
- ``FAILURE_DATA``: + full context around failures.
- ``SUCCESS_INFO``: + success labels + step + small metadata.
- ``SUCCESS_DATA``: + full context for successes too.

Usage::

    from TyGrit.tracking import tracker, TrackingLevel, TrackingEvent
    from TyGrit.types.failures import PlannerFailure

    tracker.configure(TrackingLevel.FAILURE_DATA)

    tracker.record(TrackingEvent(
        subsystem="planner",
        stage="grasp",
        success=False,
        failure=PlannerFailure.NO_PATH_FOUND,
        step=tracker.tick(),
        info={"goal_config": [0.1, 0.2, ...]},
        data={"pointcloud": big_array, "collision_state": ...},
    ))
"""

from TyGrit.tracking.events import TrackingEvent
from TyGrit.tracking.levels import TrackingLevel
from TyGrit.tracking.tracker import Tracker, tracker

__all__ = [
    "TrackingEvent",
    "TrackingLevel",
    "Tracker",
    "tracker",
]
