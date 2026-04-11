"""Tests for the Step 8 AI2THOR manifests and stem-map dispatch.

Pure Python — runs in the default pixi env. The manifest files
themselves are produced by TyGrit.worlds.generators.ai2thor (which
runs in the world env because it imports mani_skill for the metadata
file path) but the loader + _build_stem_map logic is sim-agnostic
and verifiable here.

Integration with real ManiSkill env loading is gated on the AI2THOR
asset download succeeding — that test lives in
test_worlds_maniskill.py with a pytest.importorskip("mani_skill")
guard. See the `pixi run -e world download-ai2thor` task.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from TyGrit.types.worlds import SceneSpec
from TyGrit.worlds.manifest import load_manifest

WORLDS_DIR = Path("resources/worlds")

# Scene counts locked in by the generator — match the shipped ManiSkill
# metadata JSONs at the pinned mani-skill version in pixi.lock. If these
# numbers change upstream we want tests to fail loudly so we can decide
# whether to regenerate the manifest.
EXPECTED_COUNTS = {
    "procthor": 12000,
    "ithor": 150,
    "robothor": 75,
    "architecthor": 10,
}

# Per-variant file extension. ProcTHOR uses .json.gz because the
# uncompressed 12 000-entry manifest (~4.4 MB) exceeds pre-commit's
# check-added-large-files 500 KB default. load_manifest sniffs the
# .gz suffix transparently so callers don't care.
MANIFEST_FILENAMES = {
    "procthor": "procthor.json.gz",
    "ithor": "ithor.json",
    "robothor": "robothor.json",
    "architecthor": "architecthor.json",
}


def _manifest_path(variant: str) -> Path:
    return WORLDS_DIR / MANIFEST_FILENAMES[variant]


@pytest.mark.parametrize("variant,expected", list(EXPECTED_COUNTS.items()))
class TestAI2THORManifests:
    def test_manifest_exists(self, variant: str, expected: int) -> None:
        assert (_manifest_path(variant)).exists()

    def test_scene_count_matches_upstream_metadata(
        self, variant: str, expected: int
    ) -> None:
        scenes = load_manifest(_manifest_path(variant))
        assert len(scenes) == expected

    def test_all_entries_are_scenespec(self, variant: str, expected: int) -> None:
        scenes = load_manifest(_manifest_path(variant))
        assert all(isinstance(s, SceneSpec) for s in scenes)

    def test_source_matches_variant(self, variant: str, expected: int) -> None:
        scenes = load_manifest(_manifest_path(variant))
        assert all(s.source == variant for s in scenes)

    def test_scene_ids_prefixed_with_variant(self, variant: str, expected: int) -> None:
        scenes = load_manifest(_manifest_path(variant))
        assert all(s.scene_id.startswith(f"{variant}/") for s in scenes)

    def test_background_builtin_ids_mirror_scene_ids(
        self, variant: str, expected: int
    ) -> None:
        scenes = load_manifest(_manifest_path(variant))
        for s in scenes:
            stem = s.scene_id.split("/", 1)[1]
            assert s.background_builtin_id == f"{variant}:{stem}"

    def test_no_objects_in_manifest(self, variant: str, expected: int) -> None:
        # Like the ReplicaCAD manifest, object spawning is delegated to
        # the ManiSkill AI2THORBaseSceneBuilder (static-only for AI2THOR
        # since the original Unity articulations don't port to URDF).
        scenes = load_manifest(_manifest_path(variant))
        assert all(s.objects == () for s in scenes)

    def test_no_duplicate_scene_ids(self, variant: str, expected: int) -> None:
        # Stem uniqueness is verified in the generator, but regression-
        # test the loaded manifest too so any future corruption of the
        # on-disk JSON (e.g. a bad merge) surfaces immediately.
        scenes = load_manifest(_manifest_path(variant))
        ids = [s.scene_id for s in scenes]
        assert len(ids) == len(set(ids))
