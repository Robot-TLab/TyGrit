"""Tests for TyGrit.worlds.manifest — pure Python, no simulator deps."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from TyGrit.types.worlds import ObjectSpec, SceneSpec
from TyGrit.worlds.manifest import (
    MANIFEST_VERSION,
    load_manifest,
    save_manifest,
)

BASELINE_MANIFEST = Path("resources/worlds/replicacad.json")


def _write(tmp_path: Path, payload: dict) -> Path:
    """Helper: write a dict as JSON to a temp path and return it."""
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(payload))
    return p


class TestLoadReplicaCADBaseline:
    """The bundled 6-apt manifest must load cleanly — it's our CI canary."""

    def test_loads_six_scenes(self) -> None:
        scenes = load_manifest(BASELINE_MANIFEST)
        assert len(scenes) == 6

    def test_all_entries_are_scenespec(self) -> None:
        scenes = load_manifest(BASELINE_MANIFEST)
        for scene in scenes:
            assert isinstance(scene, SceneSpec)

    def test_scene_ids_sequential(self) -> None:
        scenes = load_manifest(BASELINE_MANIFEST)
        ids = [s.scene_id for s in scenes]
        assert ids == [f"replicacad/apt_{i}" for i in range(6)]

    def test_source_is_replicacad(self) -> None:
        scenes = load_manifest(BASELINE_MANIFEST)
        assert all(s.source == "replicacad" for s in scenes)

    def test_builtin_ids_point_at_replicacad_loader(self) -> None:
        scenes = load_manifest(BASELINE_MANIFEST)
        assert all(
            s.background_builtin_id == f"replicacad:apt_{i}"
            for i, s in enumerate(scenes)
        )

    def test_no_objects_in_baseline(self) -> None:
        # The baseline delegates object spawning to ManiSkill's existing
        # ReplicaCADSceneBuilder, so the objects list is empty by design.
        scenes = load_manifest(BASELINE_MANIFEST)
        assert all(s.objects == () for s in scenes)

    def test_accepts_string_path(self) -> None:
        # Callers often pass plain strings; load_manifest must accept them.
        scenes = load_manifest(str(BASELINE_MANIFEST))
        assert len(scenes) == 6


class TestLoadErrors:
    """Every error path must raise with a clear message naming the file."""

    def test_missing_file(self, tmp_path: Path) -> None:
        # Path.read_text raises FileNotFoundError directly — we don't
        # catch it because the caller needs to know the file is missing.
        with pytest.raises(FileNotFoundError):
            load_manifest(tmp_path / "does_not_exist.json")

    def test_invalid_json(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text("{not valid json")
        with pytest.raises(ValueError, match="invalid JSON"):
            load_manifest(p)

    def test_rejects_non_object_toplevel(self, tmp_path: Path) -> None:
        p = tmp_path / "list.json"
        p.write_text("[1, 2, 3]")
        with pytest.raises(ValueError, match="top-level must be a JSON object"):
            load_manifest(p)

    def test_rejects_missing_version(self, tmp_path: Path) -> None:
        p = _write(tmp_path, {"scenes": []})
        with pytest.raises(ValueError, match="unsupported schema version"):
            load_manifest(p)

    def test_rejects_wrong_version(self, tmp_path: Path) -> None:
        p = _write(tmp_path, {"version": 999, "scenes": []})
        with pytest.raises(ValueError, match="unsupported schema version 999"):
            load_manifest(p)

    def test_rejects_missing_scenes_key(self, tmp_path: Path) -> None:
        p = _write(tmp_path, {"version": MANIFEST_VERSION})
        with pytest.raises(ValueError, match="'scenes' must be a list"):
            load_manifest(p)

    def test_rejects_scenes_not_list(self, tmp_path: Path) -> None:
        p = _write(tmp_path, {"version": MANIFEST_VERSION, "scenes": {}})
        with pytest.raises(ValueError, match="'scenes' must be a list"):
            load_manifest(p)

    def test_scene_entry_must_be_object(self, tmp_path: Path) -> None:
        p = _write(tmp_path, {"version": MANIFEST_VERSION, "scenes": ["nope"]})
        with pytest.raises(ValueError, match="scene entry must be a JSON object"):
            load_manifest(p)

    def test_scene_validator_error_is_wrapped_with_file_context(
        self, tmp_path: Path
    ) -> None:
        # SceneSpec.__post_init__ rejects duplicate object names. The
        # error must come through with both the manifest path and the
        # scene_id so debugging a bad manifest is straightforward.
        p = _write(
            tmp_path,
            {
                "version": MANIFEST_VERSION,
                "scenes": [
                    {
                        "scene_id": "bad/dup",
                        "source": "custom",
                        "background_builtin_id": "custom:x",
                        "objects": [
                            {"name": "a", "builtin_id": "ycb:a"},
                            {"name": "a", "builtin_id": "ycb:a"},
                        ],
                    }
                ],
            },
        )
        with pytest.raises(ValueError) as info:
            load_manifest(p)
        msg = str(info.value)
        assert str(p) in msg
        assert "'bad/dup'" in msg
        assert "duplicate object names" in msg

    def test_object_validator_error_is_wrapped(self, tmp_path: Path) -> None:
        # ObjectSpec with no asset source triggers its __post_init__.
        p = _write(
            tmp_path,
            {
                "version": MANIFEST_VERSION,
                "scenes": [
                    {
                        "scene_id": "bad/noasset",
                        "source": "custom",
                        "objects": [{"name": "orphan"}],
                    }
                ],
            },
        )
        with pytest.raises(ValueError) as info:
            load_manifest(p)
        msg = str(info.value)
        assert "'orphan'" in msg
        assert "must set one of" in msg


class TestSaveAndRoundtrip:
    def _complex_scene(self) -> SceneSpec:
        return SceneSpec(
            scene_id="replicacad/apt_0",
            source="replicacad",
            background_builtin_id="replicacad:apt_0",
            navmesh_path="resources/navmesh/apt_0.obj",
            objects=(
                ObjectSpec(
                    name="target_cup",
                    builtin_id="ycb:065-a_cups",
                    position=(1.5, 0.2, 0.75),
                    orientation_xyzw=(0.0, 0.0, 0.707, 0.707),
                ),
                ObjectSpec(
                    name="drawer",
                    urdf_path="resources/partnet/drawer.urdf",
                    fix_base=True,
                    is_articulated=True,
                    joint_init=(("drawer_top", 0.0), ("drawer_bot", 0.1)),
                ),
            ),
            target_object_names=("target_cup",),
            lighting="default",
        )

    def test_save_writes_version(self, tmp_path: Path) -> None:
        out = tmp_path / "m.json"
        save_manifest(out, [self._complex_scene()])
        payload = json.loads(out.read_text())
        assert payload["version"] == MANIFEST_VERSION

    def test_save_creates_parent_directories(self, tmp_path: Path) -> None:
        out = tmp_path / "nested" / "subdir" / "m.json"
        save_manifest(out, [self._complex_scene()])
        assert out.exists()

    def test_save_writes_optional_metadata(self, tmp_path: Path) -> None:
        out = tmp_path / "m.json"
        save_manifest(
            out,
            [self._complex_scene()],
            source="replicacad",
            generator="tests",
        )
        payload = json.loads(out.read_text())
        assert payload["source"] == "replicacad"
        assert payload["generator"] == "tests"

    def test_save_omits_unset_metadata(self, tmp_path: Path) -> None:
        out = tmp_path / "m.json"
        save_manifest(out, [self._complex_scene()])
        payload = json.loads(out.read_text())
        assert "source" not in payload
        assert "generator" not in payload

    def test_output_ends_with_newline(self, tmp_path: Path) -> None:
        out = tmp_path / "m.json"
        save_manifest(out, [self._complex_scene()])
        assert out.read_text().endswith("\n")

    def test_simple_roundtrip(self, tmp_path: Path) -> None:
        scenes = (self._complex_scene(),)
        out = tmp_path / "m.json"
        save_manifest(out, scenes)
        restored = load_manifest(out)
        assert restored == scenes

    def test_roundtrip_preserves_tuple_types(self, tmp_path: Path) -> None:
        # Frozen dataclasses store tuples, not lists. JSON only has
        # arrays, so the loader must restore tuples for equality and
        # hashability to work.
        scenes = (self._complex_scene(),)
        out = tmp_path / "m.json"
        save_manifest(out, scenes)
        restored = load_manifest(out)[0]

        assert isinstance(restored.objects, tuple)
        assert isinstance(restored.objects[0].position, tuple)
        assert isinstance(restored.objects[0].orientation_xyzw, tuple)
        assert isinstance(restored.target_object_names, tuple)

        drawer = restored.objects[1]
        assert isinstance(drawer.joint_init, tuple)
        assert isinstance(drawer.joint_init[0], tuple)
        assert drawer.joint_init[0] == ("drawer_top", 0.0)

    def test_roundtrip_multiple_scenes(self, tmp_path: Path) -> None:
        a = SceneSpec(
            scene_id="a",
            source="custom",
            background_builtin_id="custom:a",
        )
        b = SceneSpec(
            scene_id="b",
            source="custom",
            background_builtin_id="custom:b",
        )
        out = tmp_path / "m.json"
        save_manifest(out, [a, b])
        assert load_manifest(out) == (a, b)

    def test_roundtrip_accepts_list_or_tuple_input(self, tmp_path: Path) -> None:
        scene = self._complex_scene()
        out = tmp_path / "m.json"
        # save_manifest advertises Iterable[SceneSpec]; verify list works
        # (the test above uses tuple).
        save_manifest(out, [scene])
        assert load_manifest(out) == (scene,)


class TestForwardCompat:
    def test_unknown_toplevel_keys_ignored(self, tmp_path: Path) -> None:
        p = _write(
            tmp_path,
            {
                "version": MANIFEST_VERSION,
                "future_metadata": {"anything": "goes"},
                "scenes": [],
            },
        )
        assert load_manifest(p) == ()

    def test_unknown_scene_keys_ignored(self, tmp_path: Path) -> None:
        p = _write(
            tmp_path,
            {
                "version": MANIFEST_VERSION,
                "scenes": [
                    {
                        "scene_id": "s",
                        "source": "custom",
                        "background_builtin_id": "custom:s",
                        "future_scene_field": "ignored",
                    }
                ],
            },
        )
        (scene,) = load_manifest(p)
        assert scene.scene_id == "s"

    def test_unknown_object_keys_ignored(self, tmp_path: Path) -> None:
        p = _write(
            tmp_path,
            {
                "version": MANIFEST_VERSION,
                "scenes": [
                    {
                        "scene_id": "s",
                        "source": "custom",
                        "objects": [
                            {
                                "name": "x",
                                "builtin_id": "ycb:x",
                                "future_object_field": 42,
                            }
                        ],
                    }
                ],
            },
        )
        (scene,) = load_manifest(p)
        assert scene.objects[0].name == "x"
