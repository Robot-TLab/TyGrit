"""Tests for TyGrit.robots.fetch — pure Python, no simulator dependencies."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from TyGrit.kinematics.fetch.constants import JOINT_LIMITS_LOWER
from TyGrit.robots import FETCH_SPEC
from TyGrit.types.robots import RobotSpec


class TestFetchSpec:
    def test_import_from_package(self) -> None:
        assert FETCH_SPEC.name == "fetch"

    def test_field_cardinality_matches_fetch_constants(self) -> None:
        assert len(FETCH_SPEC.joint_limits_lower) == len(
            FETCH_SPEC.planning_joint_names
        )
        assert len(FETCH_SPEC.joint_limits_upper) == len(
            FETCH_SPEC.planning_joint_names
        )
        assert len(FETCH_SPEC.head_joint_names) == 2

    def test_is_frozen(self) -> None:
        with pytest.raises(FrozenInstanceError):
            setattr(FETCH_SPEC, "name", "other")

    def test_joint_limits_round_trip(self) -> None:
        np.testing.assert_allclose(
            np.asarray(FETCH_SPEC.joint_limits_lower),
            JOINT_LIMITS_LOWER,
        )


class TestRobotSpecValidation:
    def _make_spec(self, **overrides: object) -> RobotSpec:
        values: dict[str, object] = {
            "name": "testbot",
            "sim_uids": {"maniskill": "testbot"},
            "planning_joint_names": ("joint_a", "joint_b"),
            "head_joint_names": (),
            "base_joint_names": ("root_x", "root_y", "root_theta"),
            "is_mobile": True,
            "controller_order": ("arm", "base"),
            "camera_ids": ("head",),
            "camera_sensor_ids": {"head": "sensor_head"},
            "joint_limits_lower": (-1.0, -2.0),
            "joint_limits_upper": (1.0, 2.0),
            "default_spawn_pose": (0.0, 0.0, 0.0),
        }
        values.update(overrides)
        return RobotSpec(**values)

    def test_rejects_mismatched_joint_limit_lengths(self) -> None:
        with pytest.raises(ValueError, match="joint_limits_lower must match"):
            self._make_spec(joint_limits_lower=(-1.0,))
        with pytest.raises(ValueError, match="joint_limits_upper must match"):
            self._make_spec(joint_limits_upper=(1.0,))

    def test_rejects_fixed_base_robot_with_base_joint_names(self) -> None:
        with pytest.raises(ValueError, match="empty base_joint_names"):
            self._make_spec(is_mobile=False, base_joint_names=("root_x",))

    def test_rejects_duplicate_base_joint_names(self) -> None:
        with pytest.raises(ValueError, match="must not contain duplicates"):
            self._make_spec(base_joint_names=("root_x", "root_x", "root_theta"))

    def test_rejects_mobile_robot_without_base_triplet(self) -> None:
        with pytest.raises(ValueError, match="exactly 3 base_joint_names"):
            self._make_spec(base_joint_names=("root_x", "root_y"))

    def test_rejects_inverted_joint_limits(self) -> None:
        with pytest.raises(ValueError, match="must be <= upper"):
            self._make_spec(
                joint_limits_lower=(2.0, -2.0),
                joint_limits_upper=(1.0, 2.0),
            )

    def test_rejects_fixed_base_robot_with_spawn_pose(self) -> None:
        with pytest.raises(ValueError, match="must not define default_spawn_pose"):
            self._make_spec(
                is_mobile=False,
                base_joint_names=(),
                default_spawn_pose=(0.0, 0.0, 0.0),
            )

    def test_rejects_camera_sensor_ids_mismatch(self) -> None:
        with pytest.raises(ValueError, match="camera_sensor_ids"):
            self._make_spec(
                camera_ids=("head", "wrist"),
                camera_sensor_ids={"head": "sensor_head"},
            )


class TestRobotSpecImmutability:
    def test_sim_uids_mapping_is_read_only(self) -> None:
        # __post_init__ wraps the input dict in MappingProxyType so
        # downstream consumers can't silently mutate it.
        with pytest.raises(TypeError):
            FETCH_SPEC.sim_uids["isaac"] = "fetch"  # type: ignore[index]

    def test_camera_sensor_ids_mapping_is_read_only(self) -> None:
        with pytest.raises(TypeError):
            FETCH_SPEC.camera_sensor_ids["wrist"] = "wrist_cam"  # type: ignore[index]

    def test_mapping_input_is_copied_not_aliased(self) -> None:
        source = {"maniskill": "testbot"}
        spec = RobotSpec(
            name="testbot",
            sim_uids=source,
            planning_joint_names=("j0",),
            head_joint_names=(),
            base_joint_names=(),
            is_mobile=False,
            controller_order=("arm",),
            camera_ids=("head",),
            camera_sensor_ids={"head": "sensor_head"},
            joint_limits_lower=(-1.0,),
            joint_limits_upper=(1.0,),
            default_spawn_pose=None,
        )
        source["genesis"] = "leaked"
        assert "genesis" not in spec.sim_uids
