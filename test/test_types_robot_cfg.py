"""Tests for :class:`TyGrit.types.robots.RobotCfg` + :class:`ActuatorCfg`.

These are pure-type tests — no sim imports, run in the default pixi env.
"""

from __future__ import annotations

import pytest

from TyGrit.types.robots import ActuatorCfg, RobotCfg
from TyGrit.types.sensors import CameraSpec


def _make_valid_actuators() -> dict[str, ActuatorCfg]:
    return {
        "arm": ActuatorCfg(
            name="arm",
            joint_names=("j1", "j2"),
            control_mode="velocity",
            action_dim=2,
        ),
        "base": ActuatorCfg(
            name="base",
            joint_names=("bx", "by", "bth"),
            control_mode="velocity",
            action_dim=2,
            kind="base_twist",
        ),
    }


def _make_valid_cfg(**overrides) -> RobotCfg:
    defaults = dict(
        name="testbot",
        sim_uids={"maniskill": "testbot"},
        base_link_name="base_link",
        is_mobile=True,
        urdf_path="/tmp/testbot.urdf",
        base_joint_names=("bx", "by", "bth"),
        actuators=_make_valid_actuators(),
        controller_order=("arm", "base"),
        planning_joint_names=("j1", "j2"),
        head_joint_names=(),
        joint_limits_lower=(-1.0, -1.0),
        joint_limits_upper=(1.0, 1.0),
        cameras=(),
        default_spawn_pose=(0.0, 0.0, 0.0),
    )
    defaults.update(overrides)
    return RobotCfg(**defaults)


class TestActuatorCfg:
    def test_valid(self) -> None:
        a = ActuatorCfg(
            name="arm",
            joint_names=("j1", "j2", "j3"),
            control_mode="velocity",
            action_dim=3,
        )
        assert a.name == "arm"
        assert a.control_mode == "velocity"
        assert a.action_dim == 3

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name must be non-empty"):
            ActuatorCfg(
                name="", joint_names=("j",), control_mode="velocity", action_dim=1
            )

    def test_zero_action_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="action_dim must be positive"):
            ActuatorCfg(
                name="x", joint_names=("j",), control_mode="velocity", action_dim=0
            )

    def test_empty_joint_names_raises(self) -> None:
        with pytest.raises(ValueError, match="joint_names must be non-empty"):
            ActuatorCfg(name="x", joint_names=(), control_mode="velocity", action_dim=1)

    def test_duplicate_joint_names_raises(self) -> None:
        with pytest.raises(ValueError, match="joint_names must be unique"):
            ActuatorCfg(
                name="x",
                joint_names=("j", "j"),
                control_mode="velocity",
                action_dim=2,
            )

    def test_sim_params_frozen(self) -> None:
        a = ActuatorCfg(
            name="x",
            joint_names=("j",),
            control_mode="velocity",
            action_dim=1,
            sim_params={"maniskill": {"stiffness": 100.0}},
        )
        with pytest.raises(TypeError):
            a.sim_params["maniskill"]["stiffness"] = 0.0  # type: ignore[index]

    def test_command_mapping_required_when_dim_mismatches(self) -> None:
        with pytest.raises(
            ValueError, match="requires an explicit command_to_joint_mapping"
        ):
            ActuatorCfg(
                name="gripper",
                joint_names=("r", "l"),
                control_mode="position",
                action_dim=1,
            )

    def test_command_mapping_valid(self) -> None:
        a = ActuatorCfg(
            name="gripper",
            joint_names=("r", "l"),
            control_mode="position",
            action_dim=1,
            command_to_joint_mapping=(0, 0),
        )
        assert a.command_to_joint_mapping == (0, 0)

    def test_command_mapping_wrong_length(self) -> None:
        with pytest.raises(
            ValueError, match="length .* must equal len\\(joint_names\\)"
        ):
            ActuatorCfg(
                name="gripper",
                joint_names=("r", "l"),
                control_mode="position",
                action_dim=1,
                command_to_joint_mapping=(0,),
            )

    def test_command_mapping_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="must be in \\[0, action_dim"):
            ActuatorCfg(
                name="gripper",
                joint_names=("r", "l"),
                control_mode="position",
                action_dim=1,
                command_to_joint_mapping=(0, 5),
            )

    def test_base_twist_kind(self) -> None:
        a = ActuatorCfg(
            name="base",
            joint_names=("bx", "by", "bth"),
            control_mode="velocity",
            action_dim=2,
            kind="base_twist",
        )
        assert a.kind == "base_twist"
        assert a.command_to_joint_mapping is None

    def test_base_twist_rejects_command_mapping(self) -> None:
        with pytest.raises(ValueError, match="meaningless for kind='base_twist'"):
            ActuatorCfg(
                name="base",
                joint_names=("bx", "by", "bth"),
                control_mode="velocity",
                action_dim=2,
                kind="base_twist",
                command_to_joint_mapping=(0, 0, 0),
            )

    def test_joint_mismatched_with_no_mapping_rejected_even_if_equal(self) -> None:
        # Sanity: equal dims with a mapping is also rejected because
        # the mapping only makes sense when dims differ.
        with pytest.raises(
            ValueError, match="only legal when action_dim != len\\(joint_names\\)"
        ):
            ActuatorCfg(
                name="arm",
                joint_names=("j1", "j2"),
                control_mode="velocity",
                action_dim=2,
                command_to_joint_mapping=(0, 1),
            )


class TestRobotCfg:
    def test_valid(self) -> None:
        cfg = _make_valid_cfg()
        assert cfg.name == "testbot"
        assert cfg.total_action_dim() == 4  # 2 + 2
        slices = cfg.action_slices_from_order()
        assert slices["arm"] == slice(0, 2)
        assert slices["base"] == slice(2, 4)

    def test_no_asset_route_raises(self) -> None:
        with pytest.raises(ValueError, match="must set at least one of sim_uids"):
            _make_valid_cfg(
                sim_uids={},
                urdf_path=None,
                usd_path=None,
                mjcf_path=None,
            )

    def test_mobile_missing_base_joints_raises(self) -> None:
        with pytest.raises(ValueError, match="must define exactly 3 base_joint_names"):
            _make_valid_cfg(is_mobile=True, base_joint_names=("bx", "by"))

    def test_fixed_with_base_joints_raises(self) -> None:
        with pytest.raises(ValueError, match="must have empty base_joint_names"):
            _make_valid_cfg(
                is_mobile=False,
                base_joint_names=("bx", "by", "bth"),
                default_spawn_pose=None,
            )

    def test_fixed_with_default_spawn_raises(self) -> None:
        with pytest.raises(ValueError, match="must not set default_spawn_pose"):
            _make_valid_cfg(
                is_mobile=False,
                base_joint_names=(),
                default_spawn_pose=(0.0, 0.0, 0.0),
            )

    def test_controller_order_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="controller_order keys"):
            _make_valid_cfg(controller_order=("arm", "ghost"))

    def test_unknown_default_joint_positions_raises(self) -> None:
        with pytest.raises(
            ValueError, match="default_joint_positions references unknown joints"
        ):
            _make_valid_cfg(default_joint_positions={"not_a_joint": 0.0})

    def test_duplicate_camera_ids_raises(self) -> None:
        cam_a = CameraSpec(camera_id="head", parent_link="a")
        cam_b = CameraSpec(camera_id="head", parent_link="b")
        with pytest.raises(ValueError, match="camera_id values must be unique"):
            _make_valid_cfg(cameras=(cam_a, cam_b))

    def test_camera_by_id(self) -> None:
        cam = CameraSpec(camera_id="head", parent_link="head_link")
        cfg = _make_valid_cfg(cameras=(cam,))
        assert cfg.camera_by_id("head").parent_link == "head_link"
        with pytest.raises(KeyError, match="no camera with id"):
            cfg.camera_by_id("wrist")

    def test_frozen(self) -> None:
        cfg = _make_valid_cfg()
        with pytest.raises(Exception):
            cfg.name = "other"  # type: ignore[misc]

    def test_sim_uids_immutable(self) -> None:
        cfg = _make_valid_cfg()
        with pytest.raises(TypeError):
            cfg.sim_uids["genesis"] = "x"  # type: ignore[index]


class TestFetchCfg:
    def test_fetch_cfg_loads(self) -> None:
        from TyGrit.robots.fetch import FETCH_CFG

        assert FETCH_CFG.name == "fetch"
        assert FETCH_CFG.is_mobile is True
        assert FETCH_CFG.sim_uids["maniskill"] == "fetch"
        assert FETCH_CFG.urdf_path is not None and FETCH_CFG.urdf_path.endswith(
            "fetch.urdf"
        )
        assert set(FETCH_CFG.controller_order) == {"arm", "gripper", "body", "base"}

    def test_fetch_cfg_total_action_dim(self) -> None:
        from TyGrit.robots.fetch import FETCH_CFG

        # arm=7 + gripper=1 + body=3 + base=2 = 13
        assert FETCH_CFG.total_action_dim() == 13

    def test_fetch_cfg_cameras(self) -> None:
        from TyGrit.robots.fetch import FETCH_CFG

        assert len(FETCH_CFG.cameras) == 1
        assert FETCH_CFG.cameras[0].camera_id == "head"
