"""Tests for TyGrit.robots.fetch — pure Python, no simulator dependencies.

The richer ``RobotCfg`` validators are tested in
``test/test_types_robot_cfg.py``; this file covers the Fetch-specific
``FETCH_CFG`` instance: that it imports from the package, has the
right joint cardinality, has its joint limits aligned with the
upstream constants, and is genuinely frozen.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from TyGrit.robots import FETCH_CFG
from TyGrit.robots.fetch.kinematics.constants import JOINT_LIMITS_LOWER


class TestFetchCfg:
    def test_import_from_package(self) -> None:
        assert FETCH_CFG.name == "fetch"

    def test_field_cardinality_matches_fetch_constants(self) -> None:
        assert len(FETCH_CFG.joint_limits_lower) == len(FETCH_CFG.planning_joint_names)
        assert len(FETCH_CFG.joint_limits_upper) == len(FETCH_CFG.planning_joint_names)
        assert len(FETCH_CFG.head_joint_names) == 2

    def test_is_frozen(self) -> None:
        with pytest.raises(FrozenInstanceError):
            setattr(FETCH_CFG, "name", "other")

    def test_joint_limits_round_trip(self) -> None:
        np.testing.assert_allclose(
            np.asarray(FETCH_CFG.joint_limits_lower),
            JOINT_LIMITS_LOWER,
        )

    def test_sim_uids_immutable(self) -> None:
        # ``RobotCfg.__post_init__`` wraps mapping fields in
        # ``MappingProxyType`` so external consumers can't silently
        # mutate them after construction.
        with pytest.raises(TypeError):
            FETCH_CFG.sim_uids["isaac"] = "fetch"  # type: ignore[index]
