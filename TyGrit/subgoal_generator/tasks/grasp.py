"""Grasp task — BT-based subgoal generator for grasping.

Composes generic BT nodes with grasp-specific logic:
observe → check goal → validate/replan → control.

Future: will add grasp-specific nodes (SampleGraspPose, CheckReachability,
ApproachPrepose, ExecuteGrasp, Lift) that call into ``samplers/``.
"""

from __future__ import annotations

import py_trees

from TyGrit.subgoal_generator.bt.nodes import (
    GenerateSubGoal,
    IsGoalReached,
    IsTrajectoryValid,
    Observe,
    PlanMotion,
)


def build_grasp_tree() -> py_trees.behaviour.Behaviour:
    """Build the receding-horizon grasp tree.

    Structure::

        Sequence "grasp_root"
        ├── Observe
        ├── Selector "decide"
        │   ├── IsGoalReached          → SUCCESS exits the loop
        │   └── Sequence "plan_or_reuse"
        │       ├── Selector "reuse_or_replan"
        │       │   ├── IsTrajectoryValid
        │       │   └── Sequence "replan"
        │       │       ├── GenerateSubGoal
        │       │       └── PlanMotion
        │       └── (control / execute handled by scheduler)
    """
    # Replan branch
    replan = py_trees.composites.Sequence("replan", memory=False)
    replan.add_children([GenerateSubGoal(), PlanMotion()])

    # Reuse existing trajectory OR replan
    reuse_or_replan = py_trees.composites.Selector("reuse_or_replan", memory=False)
    reuse_or_replan.add_children([IsTrajectoryValid(), replan])

    plan_or_reuse = py_trees.composites.Sequence("plan_or_reuse", memory=False)
    plan_or_reuse.add_children([reuse_or_replan])

    # Top-level decide: goal reached (success) or keep working
    decide = py_trees.composites.Selector("decide", memory=False)
    decide.add_children([IsGoalReached(), plan_or_reuse])

    # Root sequence: observe, then decide
    root = py_trees.composites.Sequence("grasp_root", memory=False)
    root.add_children([Observe(), decide])

    return root
