"""Tracking verbosity levels.

Controls how much data the tracker records.  Higher levels include
everything from lower levels (cumulative threshold, like logging).

Level ordering (low → high verbosity):

- ``OFF``:          Nothing recorded.
- ``FAILURE_INFO``: Failure label + step + small metadata.
- ``FAILURE_DATA``: + full context around failures (observations, configs).
- ``SUCCESS_INFO``: + success labels + step + small metadata.
- ``SUCCESS_DATA``: + full context for successes too (everything).
"""

from enum import IntEnum


class TrackingLevel(IntEnum):
    OFF = 0
    FAILURE_INFO = 10
    FAILURE_DATA = 20
    SUCCESS_INFO = 30
    SUCCESS_DATA = 40
