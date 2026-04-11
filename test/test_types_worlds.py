"""Tests for TyGrit.types.worlds — pure Python, no simulator dependencies."""

from __future__ import annotations

import dataclasses

import pytest

from TyGrit.types import (
    BuiltWorld,
    ObjectSpec,
    SceneSamplerConfig,
    SceneSpec,
)


class TestObjectSpec:
    def test_defaults_identity_pose(self) -> None:
        obj = ObjectSpec(name="x", builtin_id="ycb:001_chips_can")
        assert obj.position == (0.0, 0.0, 0.0)
        assert obj.orientation_xyzw == (0.0, 0.0, 0.0, 1.0)
        assert obj.scale == (1.0, 1.0, 1.0)
        assert obj.fix_base is False
        assert obj.is_articulated is False
        assert obj.joint_init == ()

    def test_requires_an_asset_source(self) -> None:
        with pytest.raises(ValueError, match="must set one of"):
            ObjectSpec(name="nothing")

    @pytest.mark.parametrize(
        "kwarg,value",
        [
            ("urdf_path", "/tmp/foo.urdf"),
            ("usd_path", "/tmp/foo.usd"),
            ("mjcf_path", "/tmp/foo.xml"),
            ("mesh_path", "/tmp/foo.glb"),
            ("builtin_id", "ycb:foo"),
        ],
    )
    def test_any_asset_field_satisfies_requirement(
        self, kwarg: str, value: str
    ) -> None:
        # Should not raise — any one of the asset fields is enough.
        ObjectSpec(name="x", **{kwarg: value})

    def test_asset_path_for_returns_matching_format(self) -> None:
        obj = ObjectSpec(
            name="x",
            urdf_path="/a.urdf",
            usd_path="/a.usd",
            mjcf_path="/a.xml",
            mesh_path="/a.glb",
        )
        assert obj.asset_path_for("urdf") == "/a.urdf"
        assert obj.asset_path_for("usd") == "/a.usd"
        assert obj.asset_path_for("mjcf") == "/a.xml"
        assert obj.asset_path_for("mesh") == "/a.glb"

    def test_asset_path_for_returns_none_for_unset_format(self) -> None:
        obj = ObjectSpec(name="x", urdf_path="/a.urdf")
        assert obj.asset_path_for("urdf") == "/a.urdf"
        assert obj.asset_path_for("usd") is None  # not set ≠ unknown format

    def test_asset_path_for_rejects_unknown_format(self) -> None:
        obj = ObjectSpec(name="x", urdf_path="/a.urdf")
        # "URDF" is a typo for "urdf"; must not silently return None.
        with pytest.raises(ValueError, match="unknown format"):
            obj.asset_path_for("URDF")

    def test_frozen_and_hashable(self) -> None:
        obj = ObjectSpec(name="x", builtin_id="ycb:foo")
        # frozen=True makes __setattr__ raise FrozenInstanceError on
        # any mutation attempt; we assert that contract here.
        with pytest.raises(dataclasses.FrozenInstanceError):
            obj.name = "y"  # type: ignore[misc]
        # Hashability is required for use as dict keys / set members.
        {obj}


class TestSceneSpec:
    def _mk_obj(self, name: str) -> ObjectSpec:
        return ObjectSpec(name=name, builtin_id=f"ycb:{name}")

    def test_minimal_scene(self) -> None:
        scene = SceneSpec(
            scene_id="replicacad/apt_0",
            source="replicacad",
            background_builtin_id="replicacad:apt_0",
        )
        assert scene.scene_id == "replicacad/apt_0"
        assert scene.source == "replicacad"
        assert scene.objects == ()
        assert scene.target_object_names == ()

    def test_background_less_scene_is_legal(self) -> None:
        # Pure-object scenes (e.g. a tabletop test) don't need a background.
        scene = SceneSpec(
            scene_id="custom/tabletop_1",
            source="custom",
            objects=(self._mk_obj("cup"),),
        )
        assert scene.background_builtin_id is None
        assert len(scene.objects) == 1

    def test_rejects_duplicate_object_names(self) -> None:
        with pytest.raises(ValueError, match="duplicate object names"):
            SceneSpec(
                scene_id="bad",
                source="custom",
                objects=(self._mk_obj("a"), self._mk_obj("a")),
            )

    def test_rejects_unknown_target_names(self) -> None:
        with pytest.raises(ValueError, match="unknown objects"):
            SceneSpec(
                scene_id="bad",
                source="custom",
                objects=(self._mk_obj("a"),),
                target_object_names=("a", "b"),
            )

    def test_object_by_name(self) -> None:
        a = self._mk_obj("a")
        b = self._mk_obj("b")
        scene = SceneSpec(
            scene_id="s",
            source="custom",
            objects=(a, b),
        )
        assert scene.object_by_name("a") is a
        assert scene.object_by_name("b") is b
        with pytest.raises(KeyError):
            scene.object_by_name("missing")


class TestBuiltWorld:
    def test_default_handles_empty(self) -> None:
        spec = SceneSpec(scene_id="s", source="custom")
        built = BuiltWorld(spec=spec)
        assert built.spec is spec
        assert built.navigable_positions is None
        assert dict(built.object_handles) == {}

    def test_holds_handles(self) -> None:
        spec = SceneSpec(scene_id="s", source="custom")
        handles = {"a": object(), "b": object()}
        built = BuiltWorld(spec=spec, object_handles=handles)
        assert built.object_handles is handles


class TestSceneSamplerConfig:
    def test_defaults(self) -> None:
        cfg = SceneSamplerConfig(manifest_path="manifests/replicacad.json")
        assert cfg.manifest_path == "manifests/replicacad.json"
        assert cfg.scene_ids is None
        assert cfg.base_seed == 0

    def test_filter_by_scene_ids(self) -> None:
        cfg = SceneSamplerConfig(
            manifest_path="m.json",
            scene_ids=("replicacad/apt_0", "replicacad/apt_3"),
            base_seed=42,
        )
        assert cfg.scene_ids == ("replicacad/apt_0", "replicacad/apt_3")
        assert cfg.base_seed == 42
