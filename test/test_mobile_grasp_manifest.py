"""Round-trip + cross-backend validation for the mobile-grasp manifest.

Pure-Python unit tests — no simulator import, runnable in the default
pixi env: ``pixi run test test/test_mobile_grasp_manifest.py -v``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from TyGrit.types.mobile_grasp import MobileGraspDatapoint, MobileGraspDataset
from TyGrit.types.worlds import ObjectSpec, SceneSpec
from TyGrit.worlds.mobile_grasp_manifest import (
    CROSS_BACKEND_SCENE_SOURCES,
    load_mobile_grasp_manifest,
    save_mobile_grasp_manifest,
    validate_cross_backend,
)


def _holodeck_scene(
    scene_id: str = "holodeck/holodeck-objaverse-train/train_0",
) -> SceneSpec:
    return SceneSpec(
        scene_id=scene_id,
        source="holodeck",
        background_mjcf="assets/molmospaces/mjcf/scenes/holodeck-objaverse-train/train_0.xml",
    )


def _objaverse_object(name: str = "obj0", pos=(1.0, 0.5, 0.7)) -> ObjectSpec:
    return ObjectSpec(
        name=name,
        mesh_path="assets/objaverse/meshes/005e3725b8d9484e94a71aeb9495aea6.glb",
        position=pos,
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        scale=(0.06, 0.06, 0.06),
    )


def _datapoint(name: str = "obj0", pos=(1.0, 0.5, 0.7)) -> MobileGraspDatapoint:
    return MobileGraspDatapoint(
        scene=_holodeck_scene(),
        object=_objaverse_object(name=name, pos=pos),
        base_pose=(0.3, 0.5, 1.2),
        init_qpos={
            "torso_lift_joint": 0.2,
            "head_pan_joint": 0.0,
            "head_tilt_joint": 0.5,
            "shoulder_pan_joint": 0.0,
        },
        grasp_hint=(1.0, 0.5, 0.85, 0.0, 0.7071, 0.0, 0.7071),
    )


def test_roundtrip_preserves_fields(tmp_path: Path) -> None:
    original = MobileGraspDataset(
        entries=(_datapoint("a"), _datapoint("b", pos=(2.0, -0.5, 0.9))),
        metadata={"generator": "test", "seed": "42"},
    )
    path = tmp_path / "mobile_grasp.json"

    save_mobile_grasp_manifest(path, original, generator="test-gen")
    loaded = load_mobile_grasp_manifest(path)

    assert len(loaded) == 2
    assert loaded.metadata == {"generator": "test", "seed": "42"}
    for original_dp, loaded_dp in zip(original.entries, loaded.entries):
        assert loaded_dp == original_dp


def test_roundtrip_handles_gzip(tmp_path: Path) -> None:
    dataset = MobileGraspDataset(entries=(_datapoint(),))
    path = tmp_path / "mobile_grasp.json.gz"

    save_mobile_grasp_manifest(path, dataset)
    loaded = load_mobile_grasp_manifest(path)

    assert loaded.entries == dataset.entries


def test_roundtrip_grasp_hint_none(tmp_path: Path) -> None:
    dp = MobileGraspDatapoint(
        scene=_holodeck_scene(),
        object=_objaverse_object(),
        base_pose=(0.0, 0.0, 0.0),
    )
    path = tmp_path / "mobile_grasp.json"
    save_mobile_grasp_manifest(path, MobileGraspDataset(entries=(dp,)))
    loaded = load_mobile_grasp_manifest(path)
    assert loaded.entries[0].grasp_hint is None


def test_accepts_iterable_input(tmp_path: Path) -> None:
    path = tmp_path / "mobile_grasp.json"
    save_mobile_grasp_manifest(path, [_datapoint()])
    loaded = load_mobile_grasp_manifest(path)
    assert len(loaded) == 1
    assert loaded.metadata == {}


def test_validate_cross_backend_accepts_holodeck_objaverse() -> None:
    dataset = MobileGraspDataset(entries=(_datapoint(),))
    validate_cross_backend(dataset)


def test_validate_cross_backend_rejects_replicacad() -> None:
    scene = SceneSpec(
        scene_id="replicacad/apt_0",
        source="replicacad",
        background_builtin_id="replicacad:apt_0",
    )
    dp = MobileGraspDatapoint(
        scene=scene,
        object=_objaverse_object(),
        base_pose=(0.0, 0.0, 0.0),
    )
    with pytest.raises(ValueError, match="not cross-backend"):
        validate_cross_backend((dp,))


def test_validate_cross_backend_rejects_ycb_builtin() -> None:
    ycb = ObjectSpec(name="soup", builtin_id="ycb:005_tomato_soup_can")
    dp = MobileGraspDatapoint(
        scene=_holodeck_scene(),
        object=ycb,
        base_pose=(0.0, 0.0, 0.0),
    )
    with pytest.raises(ValueError, match="mesh_path"):
        validate_cross_backend((dp,))


def test_cross_backend_constant_is_holodeck_only() -> None:
    # Pinning the set so broadening it is a deliberate edit (with matching
    # generator + adapter work) rather than a silent drift.
    assert CROSS_BACKEND_SCENE_SOURCES == frozenset({"holodeck"})


def test_load_bad_version_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"version": 99, "entries": []}')
    with pytest.raises(ValueError, match="unsupported schema version"):
        load_mobile_grasp_manifest(path)


def test_load_missing_scene_key_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        '{"version": 1, "entries": [{"object": {}, "base_pose": [0, 0, 0]}]}'
    )
    with pytest.raises(ValueError, match="missing required key"):
        load_mobile_grasp_manifest(path)
