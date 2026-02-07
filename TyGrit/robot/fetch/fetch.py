"""Fetch robot — concrete implementation of RobotBase.

This is a placeholder for the Fetch robot class.  Concrete env backends
(ManiSkill, ROS 2, etc.) will fill in the abstract methods.
"""

from __future__ import annotations

from TyGrit.robot.base import RobotBase


class FetchRobot(RobotBase):
    """Fetch mobile manipulator.

    Subclass this for a specific backend (simulation or real hardware).
    Fetch-specific controllers live in ``robot.fetch.controller``.
    """

    # Concrete implementations will be added when env backends are built.
    # For now this serves as the type that downstream code references.
    ...
