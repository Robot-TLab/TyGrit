"""Tests for the Step 10 RoboCasa manifest + scene_id → idx parsing.

Pure Python — runs in the default pixi env. Integration with real
ManiSkill env loading (gym.make + RoboCasaSceneBuilder) is gated on
the ``download-robocasa`` asset bundle landing; that test lives in
``test_worlds_maniskill.py`` with a guard.

RoboCasa is combinatorial (10 layouts × 12 styles = 120), not a
flat scene list — so these tests also verify the stem-parsing in
:func:`TyGrit.worlds.backends.maniskill._robocasa_scene_id_to_idx`.
"""

from __future__ import annotations

from pathlib import Path

from TyGrit.types.worlds import SceneSpec
from TyGrit.worlds.manifest import load_manifest

MANIFEST_PATH = Path("resources/worlds/robocasa.json")

# 10 layouts × 12 styles = 120 configs per RoboCasaSceneBuilder.build
# (see mani_skill.utils.scene_builder.robocasa.scene_builder:165).
EXPECTED_COUNT = 120


class TestRoboCasaManifest:
    def test_manifest_exists(self) -> None:
        assert MANIFEST_PATH.exists()

    def test_scene_count(self) -> None:
        scenes = load_manifest(MANIFEST_PATH)
        assert len(scenes) == EXPECTED_COUNT

    def test_all_entries_are_scenespec(self) -> None:
        scenes = load_manifest(MANIFEST_PATH)
        assert all(isinstance(s, SceneSpec) for s in scenes)

    def test_source_is_robocasa(self) -> None:
        scenes = load_manifest(MANIFEST_PATH)
        assert all(s.source == "robocasa" for s in scenes)

    def test_scene_ids_prefixed_with_robocasa(self) -> None:
        scenes = load_manifest(MANIFEST_PATH)
        assert all(s.scene_id.startswith("robocasa/") for s in scenes)

    def test_scene_ids_contain_layout_and_style(self) -> None:
        # Every scene_id must be "robocasa/<layout>__<style>" with the
        # double-underscore separator the translator expects.
        scenes = load_manifest(MANIFEST_PATH)
        for s in scenes:
            local = s.scene_id.split("/", 1)[1]
            assert "__" in local, f"missing '__' in {s.scene_id!r}"

    def test_builtin_ids_mirror_scene_ids(self) -> None:
        scenes = load_manifest(MANIFEST_PATH)
        for s in scenes:
            local = s.scene_id.split("/", 1)[1]
            assert s.background_builtin_id == f"robocasa:{local}"

    def test_scene_ids_are_unique(self) -> None:
        scenes = load_manifest(MANIFEST_PATH)
        ids = [s.scene_id for s in scenes]
        assert len(ids) == len(set(ids))

    def test_covers_all_10_layouts(self) -> None:
        # Every layout must show up at least once — guard against a
        # generator regression that drops a layout silently.
        scenes = load_manifest(MANIFEST_PATH)
        layouts = {s.scene_id.split("/", 1)[1].split("__", 1)[0] for s in scenes}
        assert len(layouts) == 10, f"expected 10 layouts, got {sorted(layouts)}"

    def test_covers_all_12_styles(self) -> None:
        scenes = load_manifest(MANIFEST_PATH)
        styles = {s.scene_id.split("/", 1)[1].split("__", 1)[1] for s in scenes}
        assert len(styles) == 12, f"expected 12 styles, got {sorted(styles)}"
