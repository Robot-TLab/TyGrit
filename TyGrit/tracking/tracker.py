"""Tracker — the flight recorder.

Records :class:`TrackingEvent` instances at a configurable verbosity
level.  Does **not** analyze or interpret failures — that is the
analysis system's job.

A module-level ``tracker`` instance is provided (like loguru's
``logger``).  Import it directly::

    from TyGrit.tracking import tracker

Or create your own instance for testing::

    t = Tracker(level=TrackingLevel.SUCCESS_DATA)
"""

from __future__ import annotations

import dataclasses

from TyGrit.tracking.events import TrackingEvent
from TyGrit.tracking.levels import TrackingLevel


class Tracker:
    """Configurable event recorder."""

    def __init__(self, level: TrackingLevel = TrackingLevel.FAILURE_DATA) -> None:
        self._level = level
        self._events: list[TrackingEvent] = []
        self._step = 0

    # ── configuration ────────────────────────────────────────────────────

    @property
    def level(self) -> TrackingLevel:
        return self._level

    def configure(self, level: TrackingLevel) -> None:
        """Change the recording level (does not affect already-recorded events)."""
        self._level = level

    # ── step counter ─────────────────────────────────────────────────────

    @property
    def step(self) -> int:
        return self._step

    def tick(self) -> int:
        """Advance and return the step counter."""
        self._step += 1
        return self._step

    # ── recording ────────────────────────────────────────────────────────

    def record(self, event: TrackingEvent) -> None:
        """Record an event if it passes the current level filter.

        At INFO levels the ``data`` dict is stripped (replaced with ``{}``)
        to save memory.  At DATA levels everything is kept.
        """
        if not self._should_record(event):
            return
        if self._should_strip_data(event):
            event = dataclasses.replace(event, data={})
        self._events.append(event)

    # ── retrieval ────────────────────────────────────────────────────────

    def get_trace(self) -> list[TrackingEvent]:
        """Return a copy of all recorded events."""
        return list(self._events)

    def clear(self) -> None:
        """Clear all recorded events and reset the step counter."""
        self._events.clear()
        self._step = 0

    # ── internals ────────────────────────────────────────────────────────

    def _should_record(self, event: TrackingEvent) -> bool:
        if self._level <= TrackingLevel.OFF:
            return False
        if event.success:
            return self._level >= TrackingLevel.SUCCESS_INFO
        return self._level >= TrackingLevel.FAILURE_INFO

    def _should_strip_data(self, event: TrackingEvent) -> bool:
        if event.success:
            return self._level < TrackingLevel.SUCCESS_DATA
        return self._level < TrackingLevel.FAILURE_DATA


# Module-level instance — import and use directly.
tracker = Tracker()
