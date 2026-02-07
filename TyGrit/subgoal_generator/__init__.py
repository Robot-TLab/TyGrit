"""Subgoal generation — high-level decision-making for the scheduler.

This module defines the ``SubGoalGenerator`` protocol and provides
concrete implementations.  The BT-based generator (``bt/``) is one option;
others (FSM, learned policy, hard-coded) can be added as siblings.
"""

from TyGrit.subgoal_generator.protocol import SubGoalGenerator

__all__ = ["SubGoalGenerator"]
