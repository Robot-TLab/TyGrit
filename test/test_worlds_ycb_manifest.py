"""Tests for the Step 9 YCB object manifest.

Pure Python — the YCB manifest is a committed artifact; this file
only verifies it loads cleanly and has the expected shape.

The committed manifest holds the **fetch_graspable** subset (50
objects curated in grasp_anywhere v1 for Fetch's parallel-jaw
gripper). To regenerate::

    pixi run -e world python -m TyGrit.worlds.generators.ycb
    # or with the full 78-object upstream set:
    pixi run -e world python -m TyGrit.worlds.generators.ycb --subset all

Integration with :class:`SpecBackedSceneBuilder` object spawning is
tested in ``test_worlds_maniskill.py::TestSpecObjectSpawning``.
"""

from __future__ import annotations

from pathlib import Path

from TyGrit.types.worlds import ObjectSpec
from TyGrit.worlds.generators.ycb import FETCH_GRASPABLE_YCB
from TyGrit.worlds.manifest import load_object_manifest

YCB_MANIFEST = Path("resources/worlds/objects/ycb.json")

#: Expected count locked to the fetch_graspable subset length via
#: import — if the curated list changes in the generator, this
#: automatically tracks it without requiring a manual test update.
#: v1's curated subset is 50 objects.
EXPECTED_YCB_COUNT = len(FETCH_GRASPABLE_YCB)


class TestYCBManifest:
    def test_manifest_exists(self) -> None:
        assert YCB_MANIFEST.exists()

    def test_object_count_matches_fetch_graspable_subset(self) -> None:
        objects = load_object_manifest(YCB_MANIFEST)
        assert len(objects) == EXPECTED_YCB_COUNT

    def test_manifest_matches_fetch_graspable_subset_exactly(self) -> None:
        # Not just the COUNT — the full set of names should match
        # FETCH_GRASPABLE_YCB. Catches a regression where the generator
        # silently picks a different 50-object subset.
        objects = load_object_manifest(YCB_MANIFEST)
        names = sorted(o.name for o in objects)
        assert names == sorted(FETCH_GRASPABLE_YCB)

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
        # Spot-check a handful of well-known YCB objects that are in
        # the fetch_graspable subset. 003_cracker_box is intentionally
        # NOT here — v1 dropped it because Fetch can't pinch a box
        # that wide.
        objects = load_object_manifest(YCB_MANIFEST)
        names = {o.name for o in objects}
        known = {
            "002_master_chef_can",
            "004_sugar_box",
            "011_banana",
            "013_apple",
            "024_bowl",
            "025_mug",
            "065-a_cups",
            "077_rubiks_cube",
        }
        missing = known - names
        assert not missing, f"YCB manifest missing known model IDs: {sorted(missing)}"

    def test_cracker_box_not_in_fetch_graspable_subset(self) -> None:
        # Regression guard: 003_cracker_box is too wide for Fetch's
        # parallel-jaw gripper and was dropped by v1. If it ends up
        # back in the manifest, the subset filter was bypassed.
        objects = load_object_manifest(YCB_MANIFEST)
        assert "003_cracker_box" not in {o.name for o in objects}
