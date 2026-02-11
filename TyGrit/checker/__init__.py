"""Collision and trajectory validation."""

from TyGrit.checker.collision import CollisionCheckFn, create_collision_check
from TyGrit.checker.occlusion import check_self_occlusion

__all__ = ["CollisionCheckFn", "check_self_occlusion", "create_collision_check"]
