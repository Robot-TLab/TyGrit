"""Tests for TyGrit.tracking module."""

import pytest

from TyGrit.tracking import Tracker, TrackingEvent, TrackingLevel, tracker
from TyGrit.types.failures import IKFailure, PlannerFailure


def _make_event(
    success: bool = False,
    failure=None,
    step: int = 1,
    subsystem: str = "planner",
    stage: str = "grasp",
    info: dict | None = None,
    data: dict | None = None,
) -> TrackingEvent:
    return TrackingEvent(
        subsystem=subsystem,
        stage=stage,
        success=success,
        failure=failure,
        step=step,
        info=info or {},
        data=data or {},
    )


# ── TrackingLevel ordering ──────────────────────────────────────────────────


class TestTrackingLevel:
    def test_ordering(self):
        assert TrackingLevel.OFF < TrackingLevel.FAILURE_INFO
        assert TrackingLevel.FAILURE_INFO < TrackingLevel.FAILURE_DATA
        assert TrackingLevel.FAILURE_DATA < TrackingLevel.SUCCESS_INFO
        assert TrackingLevel.SUCCESS_INFO < TrackingLevel.SUCCESS_DATA


# ── Tracker basics ──────────────────────────────────────────────────────────


class TestTrackerBasics:
    def test_default_level(self):
        t = Tracker()
        assert t.level == TrackingLevel.FAILURE_DATA

    def test_configure(self):
        t = Tracker()
        t.configure(TrackingLevel.SUCCESS_DATA)
        assert t.level == TrackingLevel.SUCCESS_DATA

    def test_tick_increments(self):
        t = Tracker()
        assert t.step == 0
        assert t.tick() == 1
        assert t.tick() == 2
        assert t.step == 2

    def test_clear_resets(self):
        t = Tracker(level=TrackingLevel.FAILURE_INFO)
        t.tick()
        t.record(_make_event(failure=PlannerFailure.TIMEOUT))
        t.clear()
        assert t.step == 0
        assert t.get_trace() == []


# ── Level filtering ─────────────────────────────────────────────────────────


class TestLevelFiltering:
    def test_off_records_nothing(self):
        t = Tracker(level=TrackingLevel.OFF)
        t.record(_make_event(success=False, failure=PlannerFailure.TIMEOUT))
        t.record(_make_event(success=True))
        assert t.get_trace() == []

    def test_failure_info_records_failures_only(self):
        t = Tracker(level=TrackingLevel.FAILURE_INFO)
        t.record(_make_event(success=False, failure=PlannerFailure.TIMEOUT))
        t.record(_make_event(success=True))
        trace = t.get_trace()
        assert len(trace) == 1
        assert trace[0].failure == PlannerFailure.TIMEOUT

    def test_failure_data_records_failures_only(self):
        t = Tracker(level=TrackingLevel.FAILURE_DATA)
        t.record(_make_event(success=False, failure=IKFailure.NO_SOLUTION))
        t.record(_make_event(success=True))
        assert len(t.get_trace()) == 1

    def test_success_info_records_both(self):
        t = Tracker(level=TrackingLevel.SUCCESS_INFO)
        t.record(_make_event(success=False, failure=PlannerFailure.TIMEOUT))
        t.record(_make_event(success=True))
        assert len(t.get_trace()) == 2

    def test_success_data_records_both(self):
        t = Tracker(level=TrackingLevel.SUCCESS_DATA)
        t.record(_make_event(success=False, failure=PlannerFailure.TIMEOUT))
        t.record(_make_event(success=True))
        assert len(t.get_trace()) == 2


# ── Data stripping ──────────────────────────────────────────────────────────


class TestDataStripping:
    def test_failure_info_strips_data(self):
        t = Tracker(level=TrackingLevel.FAILURE_INFO)
        t.record(
            _make_event(
                success=False,
                failure=PlannerFailure.TIMEOUT,
                info={"config": [1, 2]},
                data={"pointcloud": "big"},
            )
        )
        trace = t.get_trace()
        assert trace[0].info == {"config": [1, 2]}
        assert trace[0].data == {}

    def test_failure_data_keeps_data(self):
        t = Tracker(level=TrackingLevel.FAILURE_DATA)
        t.record(
            _make_event(
                success=False,
                failure=PlannerFailure.TIMEOUT,
                data={"pointcloud": "big"},
            )
        )
        assert t.get_trace()[0].data == {"pointcloud": "big"}

    def test_success_info_strips_data(self):
        t = Tracker(level=TrackingLevel.SUCCESS_INFO)
        t.record(
            _make_event(
                success=True,
                info={"step_time": 0.1},
                data={"observation": "large"},
            )
        )
        trace = t.get_trace()
        assert trace[0].info == {"step_time": 0.1}
        assert trace[0].data == {}

    def test_success_data_keeps_data(self):
        t = Tracker(level=TrackingLevel.SUCCESS_DATA)
        t.record(
            _make_event(
                success=True,
                data={"observation": "large"},
            )
        )
        assert t.get_trace()[0].data == {"observation": "large"}


# ── Event fields ────────────────────────────────────────────────────────────


class TestTrackingEvent:
    def test_frozen(self):
        ev = _make_event()
        with pytest.raises(AttributeError):
            ev.subsystem = "other"  # type: ignore[misc]

    def test_fields_preserved(self):
        ev = TrackingEvent(
            subsystem="ik",
            stage="prepose",
            success=False,
            failure=IKFailure.JOINT_LIMITS,
            step=42,
            info={"target": [0.1, 0.2]},
            data={"seed_configs": [[1, 2, 3]]},
        )
        assert ev.subsystem == "ik"
        assert ev.stage == "prepose"
        assert not ev.success
        assert ev.failure == IKFailure.JOINT_LIMITS
        assert ev.step == 42


# ── Module-level instance ───────────────────────────────────────────────────


class TestModuleTracker:
    def test_module_instance_exists(self):
        assert isinstance(tracker, Tracker)

    def test_module_instance_is_configurable(self):
        original = tracker.level
        tracker.configure(TrackingLevel.SUCCESS_DATA)
        assert tracker.level == TrackingLevel.SUCCESS_DATA
        tracker.configure(original)  # restore
