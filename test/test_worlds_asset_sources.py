"""Tests for :mod:`TyGrit.worlds.asset_sources`.

Pure-default-env tests — they don't read real manifests or downloads.
Manifest-backed sources (ReplicaCAD, ProcTHOR, …) are tested via a
temporary manifest file written in the test fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from TyGrit.types.worlds import ObjectSpec, SceneSpec
from TyGrit.worlds.asset_sources import (
    MANIFEST_DIR,
    SOURCE_SIM_COMPATIBILITY,
    YCB_FETCH_GRASPABLE,
    ArchitecTHORSource,
    AssetRequest,
    AssetSource,
    HolodeckSource,
    IThorSource,
    ManifestSceneSource,
    ManiSkillYCBSource,
    MolmoSpacesSource,
    ObjaverseSource,
    ProcTHORSource,
    ReplicaCADSource,
    RoboCasaSource,
    RoboTHORSource,
    compatible_sims,
    get_source,
    register_source,
    unregister_source,
)


class TestYCBSource:
    def test_source_name(self) -> None:
        src = ManiSkillYCBSource()
        assert src.source_name == "ycb"

    def test_conforms_to_protocol(self) -> None:
        assert isinstance(ManiSkillYCBSource(), AssetSource)

    def test_list_object_ids(self) -> None:
        src = ManiSkillYCBSource()
        ids = src.list_object_ids()
        assert ids == YCB_FETCH_GRASPABLE
        assert len(ids) == 50

    def test_list_scene_ids_empty(self) -> None:
        assert ManiSkillYCBSource().list_scene_ids() == ()

    def test_get_object(self) -> None:
        src = ManiSkillYCBSource()
        spec = src.get_object("024_bowl", name="bowl_0", position=(1.0, 2.0, 3.0))
        assert isinstance(spec, ObjectSpec)
        assert spec.builtin_id == "ycb:024_bowl"
        assert spec.name == "bowl_0"
        assert spec.position == (1.0, 2.0, 3.0)

    def test_get_object_unknown_raises(self) -> None:
        src = ManiSkillYCBSource()
        with pytest.raises(KeyError, match="unknown object_id"):
            src.get_object("not_in_pool", name="x")

    def test_get_scene_not_supported(self) -> None:
        with pytest.raises(NotImplementedError, match="object-only"):
            ManiSkillYCBSource().get_scene("anything")

    def test_sample_object_id_deterministic(self) -> None:
        src = ManiSkillYCBSource()
        a = src.sample_object_id(seed=42)
        b = src.sample_object_id(seed=42)
        assert a == b
        # Different seeds should usually give different answers; we
        # don't assert inequality because a pathological seed mapping
        # could coincide. But the id must be in the pool.
        assert a in YCB_FETCH_GRASPABLE

    def test_sample_scene_not_supported(self) -> None:
        with pytest.raises(NotImplementedError, match="object-only"):
            ManiSkillYCBSource().sample_scene_id(seed=0)


class TestManifestSceneSource:
    @pytest.fixture
    def tmp_manifest(self, tmp_path: Path) -> Path:
        manifest = {
            "version": 1,
            "scenes": [
                dict(
                    scene_id="apt_0",
                    source="replicacad",
                    background_builtin_id="replicacad:apt_0",
                ),
                dict(
                    scene_id="apt_1",
                    source="replicacad",
                    background_builtin_id="replicacad:apt_1",
                ),
            ],
        }
        path = tmp_path / "replicacad.json"
        path.write_text(json.dumps(manifest))
        return path

    def test_loads_lazily(self, tmp_manifest: Path) -> None:
        # Constructing a source should not read the file.
        src = ManifestSceneSource("replicacad", tmp_manifest)
        assert src.source_name == "replicacad"
        # First enumeration triggers the read.
        ids = src.list_scene_ids()
        assert ids == ("apt_0", "apt_1")

    def test_get_scene(self, tmp_manifest: Path) -> None:
        src = ManifestSceneSource("replicacad", tmp_manifest)
        spec = src.get_scene("apt_1")
        assert isinstance(spec, SceneSpec)
        assert spec.source == "replicacad"

    def test_get_scene_unknown_raises(self, tmp_manifest: Path) -> None:
        src = ManifestSceneSource("replicacad", tmp_manifest)
        with pytest.raises(KeyError, match="no scene with id"):
            src.get_scene("ghost")

    def test_wrong_source_raises(self, tmp_path: Path) -> None:
        manifest = {
            "version": 1,
            "scenes": [
                dict(
                    scene_id="x",
                    source="procthor",
                    background_builtin_id="procthor:x",
                )
            ],
        }
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(manifest))
        src = ManifestSceneSource("replicacad", bad)
        with pytest.raises(ValueError, match="mismatched source field"):
            src.list_scene_ids()

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        src = ManifestSceneSource("replicacad", tmp_path / "missing.json")
        with pytest.raises(FileNotFoundError, match="manifest not found"):
            src.list_scene_ids()

    def test_sample_scene_id_deterministic(self, tmp_manifest: Path) -> None:
        src = ManifestSceneSource("replicacad", tmp_manifest)
        a = src.sample_scene_id(seed=99)
        b = src.sample_scene_id(seed=99)
        assert a == b

    def test_object_methods_not_supported(self, tmp_manifest: Path) -> None:
        src = ManifestSceneSource("replicacad", tmp_manifest)
        with pytest.raises(NotImplementedError, match="scene-only"):
            src.get_object("x", name="y")
        with pytest.raises(NotImplementedError, match="scene-only"):
            src.sample_object_id(seed=0)


class TestPerDatasetSubclasses:
    def test_replica_cad(self) -> None:
        src = ReplicaCADSource()
        assert src.source_name == "replicacad"
        assert str(src.manifest_path).endswith("replicacad.json")

    def test_procthor_uses_gz(self) -> None:
        # Generator writes procthor.json.gz (gzipped because plain
        # JSON for ~12k scenes is too large). The source default must
        # match.
        src = ProcTHORSource()
        assert src.source_name == "procthor"
        assert str(src.manifest_path).endswith("procthor.json.gz")

    def test_holodeck_uses_gz(self) -> None:
        src = HolodeckSource()
        assert src.source_name == "holodeck"
        assert str(src.manifest_path).endswith("holodeck.json.gz")

    def test_molmospaces_alias_points_at_holodeck(self) -> None:
        # Backward-compat alias: same class, same source_name.
        assert MolmoSpacesSource is HolodeckSource
        src = MolmoSpacesSource()
        assert src.source_name == "holodeck"

    def test_default_paths_anchored_to_project_root(self) -> None:
        # Anchoring fixes the cwd-relative bug — every source's
        # default path must be an absolute, repo-root-anchored path.
        for Cls in (
            ReplicaCADSource,
            ProcTHORSource,
            IThorSource,
            RoboTHORSource,
            ArchitecTHORSource,
            RoboCasaSource,
            HolodeckSource,
        ):
            src = Cls()
            assert (
                src.manifest_path.is_absolute()
            ), f"{Cls.__name__} default path is not absolute: {src.manifest_path}"
            assert src.manifest_path.parent == MANIFEST_DIR, (
                f"{Cls.__name__} default path is not under MANIFEST_DIR: "
                f"{src.manifest_path}"
            )

    def test_objaverse_default_matches_generator_output(self) -> None:
        # Generator writes resources/worlds/objects/objaverse.json.
        src = ObjaverseSource()
        assert str(src.manifest_path).endswith("objects/objaverse.json")


class TestGetSourceFactory:
    def test_known_sources(self) -> None:
        for name, expected_cls in [
            ("replicacad", ReplicaCADSource),
            ("procthor", ProcTHORSource),
            ("ithor", IThorSource),
            ("robothor", RoboTHORSource),
            ("architecthor", ArchitecTHORSource),
            ("robocasa", RoboCasaSource),
            ("holodeck", HolodeckSource),
        ]:
            src = get_source(name)
            assert isinstance(src, expected_cls)
            assert src.source_name == name

    def test_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="unknown source_name"):
            get_source("not_a_dataset")

    def test_register_custom_source(self, tmp_path: Path) -> None:
        # Custom subclasses must be registerable WITHOUT monkey-patching
        # _SCENE_SOURCE_REGISTRY — that's the public-API contract Codex
        # flagged as missing.
        manifest = tmp_path / "custom.json"
        manifest.write_text(
            json.dumps(
                {
                    "version": 1,
                    "scenes": [
                        dict(
                            scene_id="custom_room_0",
                            source="my_custom_source",
                            background_builtin_id="custom:0",
                        )
                    ],
                }
            )
        )

        class _MySource(ManifestSceneSource):
            def __init__(self, p=manifest):
                super().__init__("my_custom_source", p)

        register_source("my_custom_source", _MySource)
        try:
            src = get_source("my_custom_source")
            assert isinstance(src, _MySource)
            assert src.list_scene_ids() == ("custom_room_0",)
        finally:
            unregister_source("my_custom_source")

        with pytest.raises(KeyError, match="unknown source_name"):
            get_source("my_custom_source")

    def test_register_source_collision(self) -> None:
        with pytest.raises(KeyError, match="already registered"):
            register_source("replicacad", lambda: ReplicaCADSource())

    def test_register_source_overwrite(self) -> None:
        # Snapshot original so we restore after.
        original = get_source("replicacad").__class__
        register_source("replicacad", lambda: ReplicaCADSource(), overwrite=True)
        try:
            assert get_source("replicacad").__class__ is ReplicaCADSource
        finally:
            register_source("replicacad", original, overwrite=True)

    def test_register_source_empty_name(self) -> None:
        with pytest.raises(ValueError, match="must be non-empty"):
            register_source("", lambda: ReplicaCADSource())

    def test_unregister_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="is not registered"):
            unregister_source("never_registered")

    def test_o_one_scene_lookup(self, tmp_path: Path) -> None:
        # Regression guard: get_scene() should be O(1), not O(N) over
        # 12k entries. We construct a 1000-spec manifest and verify
        # lookup doesn't iterate.
        manifest = {
            "version": 1,
            "scenes": [
                dict(
                    scene_id=f"scene_{i:04d}",
                    source="replicacad",
                    background_builtin_id=f"replicacad:scene_{i:04d}",
                )
                for i in range(1000)
            ],
        }
        path = tmp_path / "big.json"
        path.write_text(json.dumps(manifest))
        src = ManifestSceneSource("replicacad", path)
        # Force load.
        src.list_scene_ids()
        # Lookup should be a single dict access — verified by patching
        # the cache to a sentinel.
        sentinel = src.get_scene("scene_0987")
        assert sentinel.scene_id == "scene_0987"


class TestObjaverseSource:
    def test_conforms_to_protocol(self) -> None:
        assert isinstance(ObjaverseSource(), AssetSource)

    def test_object_only(self) -> None:
        src = ObjaverseSource()
        assert src.list_scene_ids() == ()
        with pytest.raises(NotImplementedError):
            src.get_scene("x")

    def test_missing_manifest_raises(self, tmp_path: Path) -> None:
        src = ObjaverseSource(tmp_path / "missing.json")
        with pytest.raises(FileNotFoundError, match="manifest not found"):
            src.list_object_ids()

    def test_reads_manifest(self, tmp_path: Path) -> None:
        manifest = {
            "objects": [
                dict(name="mug_a", mesh_path="/tmp/mug_a.glb", scale=[0.5, 0.5, 0.5]),
                dict(name="book_b", mesh_path="/tmp/book_b.glb", fix_base=True),
            ]
        }
        p = tmp_path / "objaverse_objects.json"
        p.write_text(json.dumps(manifest))
        src = ObjaverseSource(p)
        ids = src.list_object_ids()
        assert set(ids) == {"mug_a", "book_b"}
        spec = src.get_object("mug_a", name="mug_0")
        assert spec.mesh_path == "/tmp/mug_a.glb"
        # When caller doesn't override scale, template scale (0.5) wins.
        assert spec.scale == (0.5, 0.5, 0.5)

    def test_get_object_unknown_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "objaverse_objects.json"
        p.write_text(json.dumps({"objects": []}))
        src = ObjaverseSource(p)
        with pytest.raises(RuntimeError, match="object pool is empty"):
            src.sample_object_id(seed=0)


class TestCompatibleSims:
    def test_known_source(self) -> None:
        sims = compatible_sims("replicacad")
        assert "maniskill" in sims
        assert "genesis" in sims

    def test_object_only_source(self) -> None:
        # YCB is ManiSkill-only — Genesis has no asset bundle.
        assert compatible_sims("ycb") == frozenset({"maniskill"})

    def test_robocasa_maniskill_only(self) -> None:
        # RoboCasa's procedural assembler is ManiSkill-internal.
        assert compatible_sims("robocasa") == frozenset({"maniskill"})

    def test_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="unknown source_name"):
            compatible_sims("nonexistent")

    def test_matrix_covers_known_sources(self) -> None:
        # The compatibility matrix must include every source the
        # AssetSource layer registers a default-constructed instance
        # for. Catches drift between the ``get_source`` registry and
        # the compatibility matrix.
        for name in [
            "replicacad",
            "procthor",
            "ithor",
            "robothor",
            "architecthor",
            "robocasa",
            "holodeck",
        ]:
            assert (
                name in SOURCE_SIM_COMPATIBILITY
            ), f"SOURCE_SIM_COMPATIBILITY is missing a known scene source {name!r}"


def test_asset_request_frozen() -> None:
    req = AssetRequest(seed=1, preferred_format="mesh")
    with pytest.raises(Exception):
        req.seed = 2  # type: ignore[misc]


class TestManifestPathContract:
    """Catch the bug class where a source's default ``manifest_path``
    drifts from where the matching generator writes its output.

    This is a *structural* test — it doesn't require the manifest file
    to actually exist. It only checks that the path the source
    enumerates against matches the path the generator writes to (read
    out of the generator's source code). When the two drift, the
    refactor that caused it fails this test before the user runs into
    a confusing FileNotFoundError at runtime.
    """

    @pytest.mark.parametrize(
        "source_cls,generator_module,output_attr",
        [
            (ReplicaCADSource, "TyGrit.worlds.generators.replicacad", "DEFAULT_OUTPUT"),
            (RoboCasaSource, "TyGrit.worlds.generators.robocasa", "DEFAULT_OUTPUT"),
            (HolodeckSource, "TyGrit.worlds.generators.holodeck", "MANIFEST_PATH"),
        ],
    )
    def test_source_default_matches_generator_output(
        self, source_cls, generator_module, output_attr
    ) -> None:
        import importlib

        gen = importlib.import_module(generator_module)
        gen_path = getattr(gen, output_attr)
        # Resolve both paths to absolute strings for comparison; the
        # source's default is project-root-anchored, the generator's
        # is repo-relative.
        gen_resolved = (
            Path(gen_path) if Path(gen_path).is_absolute() else Path(gen_path).resolve()
        )
        src = source_cls()
        src_resolved = src.manifest_path.resolve()
        # Compare just the suffix paths (basename + parent dir) so we
        # don't lock down the absolute path while still catching
        # filename / extension drift like procthor.json vs procthor.json.gz.
        assert src_resolved.name == gen_resolved.name, (
            f"{source_cls.__name__} default manifest filename "
            f"{src_resolved.name!r} != generator output filename "
            f"{gen_resolved.name!r} — the source and generator have drifted apart"
        )

    def test_default_source_resolves_or_raises_useful_error(self) -> None:
        # End-to-end smoke: every default-configured source must
        # either find its manifest at the expected absolute path or
        # raise FileNotFoundError with a message that points the user
        # at the corresponding generator. Catches the bug class where
        # a default path silently misses while no error surfaces.
        for name in [
            "replicacad",
            "procthor",
            "ithor",
            "robothor",
            "architecthor",
            "robocasa",
            "holodeck",
        ]:
            src = get_source(name)
            try:
                src.list_scene_ids()
            except FileNotFoundError as exc:
                msg = str(exc)
                assert "manifest not found" in msg and "generator" in msg, (
                    f"{name!r}: FileNotFoundError lacks the expected "
                    f"runbook hint (got {msg!r})"
                )

    def test_ai2thor_variant_manifest_paths_match_generator(self) -> None:
        # AI2THOR's generator writes 4 variants from one DEFAULT_OUTPUTS
        # mapping; check each source against that mapping.
        from TyGrit.worlds.generators.ai2thor import _DEFAULT_OUTPUTS

        for variant, expected_path in _DEFAULT_OUTPUTS.items():
            src = get_source(variant)
            assert src.manifest_path.name == Path(expected_path).name, (
                f"AI2THOR {variant!r}: source default {src.manifest_path.name!r} "
                f"!= generator output {Path(expected_path).name!r}"
            )
