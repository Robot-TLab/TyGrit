"""Collision checking function type and factory.

A collision check is a function that tests whether a robot configuration
is collision-free against the current environment.  The environment
itself is maintained elsewhere (by the scheduler / planner) — the check
function only queries it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from TyGrit.types.robot import WholeBodyConfig

if TYPE_CHECKING:
    from TyGrit.planning.motion_planner import MotionPlanner

#: A collision check function.
#:
#: Returns ``True`` if *config* is collision-free against the currently
#: loaded environment.
CollisionCheckFn = Callable[[WholeBodyConfig], bool]


def create_collision_check(planner: MotionPlanner) -> CollisionCheckFn:
    """Create a collision check function from a motion planner."""
    return planner.validate_config
