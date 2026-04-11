"""Tests for the Step 9 YCB object manifest.

Pure Python — the YCB manifest is a committed artifact; this file
only verifies it loads cleanly and has the expected shape. The
generator that produced it requires the ycb asset bundle to be
downloaded (needs the ``world`` pixi env); regenerating via::

    pixi run -e world python -m TyGrit.worlds.generators.ycb

Integration with :class:`SpecBackedSceneBuilder` object spawning is
tested in ``test_worlds_maniskill.py::TestSpecObjectSpawning``.
"""

from __future__ import annotations

from pathlib import Path

from TyGrit.types.worlds import ObjectSpec
from TyGrit.worlds.manifest import load_object_manifest

YCB_MANIFEST = Path("resources/worlds/objects/ycb.json")

#: Expected entry count — locks in ManiSkill's info_pick_v0.json
#: content at the pinned mani-skill version in pixi.lock. If the
#: upstream set changes (objects added/removed) we want this test
#: to fail loudly so we can decide whether to regenerate.
EXPECTED_YCB_COUNT = 78


class TestYCBManifest:
    def test_manifest_exists(self) -> None:
        assert YCB_MANIFEST.exists()

    def test_object_count_matches_upstream(self) -> None:
        objects = load_object_manifest(YCB_MANIFEST)
        assert len(objects) == EXPECTED_YCB_COUNT

    def test_all_entries_are_objectspec(self) -> None:
        objects = load_object_manifest(YCB_MANIFEST)
        assert all(isinstance(o, ObjectSpec) for o in objects)

    def test_all_have_ycb_builtin_id(self) -> None:
        objects = load_object_manifest(YCB_MANIFEST)
        for o in objects:
            assert o.builtin_id is not None
            assert o.builtin_id.startswith("ycb:")

    def test_object_name_matches_builtin_model_id(self) -> None:
        objects = load_object_manifest(YCB_MANIFEST)
        for o in objects:
            model_id = o.builtin_id.split(":", 1)[1]  # type: ignore[union-attr]
            assert o.name == model_id, (
                f"object name {o.name!r} should match the builtin "
                f"model id {model_id!r} — otherwise the ObjectSampler "
                f"filter uses one value and the loader another, and "
                f"debugging which object failed to spawn becomes hard"
            )

    def test_names_are_unique(self) -> None:
        objects = load_object_manifest(YCB_MANIFEST)
        names = [o.name for o in objects]
        assert len(names) == len(set(names))

    def test_no_explicit_file_paths(self) -> None:
        # YCB objects should route through the builtin_id dispatch,
        # not via urdf_path/mesh_path, so the SpecBackedSceneBuilder
        # picks the fast ManiSkill get_ycb_builder path.
        objects = load_object_manifest(YCB_MANIFEST)
        for o in objects:
            assert o.urdf_path is None
            assert o.usd_path is None
            assert o.mjcf_path is None
            assert o.mesh_path is None

    def test_known_model_ids_present(self) -> None:
        # Spot-check a few well-known YCB objects are included so we
        # catch cases where the generator accidentally dropped whole
        # categories (e.g. regex filter gone wrong).
        objects = load_object_manifest(YCB_MANIFEST)
        names = {o.name for o in objects}
        known = {
            "002_master_chef_can",
            "003_cracker_box",
            "004_sugar_box",
            "065-a_cups",
        }
        missing = known - names
        assert not missing, f"YCB manifest missing known model IDs: {sorted(missing)}"
