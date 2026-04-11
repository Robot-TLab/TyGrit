"""Generate a world manifest for AllenAI's Holodeck scene set (MolmoSpaces).

Holodeck is an LLM-driven procedural scene generator built on top of THOR's
room/object placement system. AllenAI ships it as one MJCF per scene inside
the broader `MolmoSpaces`_ dataset on HuggingFace. Unlike ReplicaCAD /
AI2THOR / RoboCasa (all loaded via ManiSkill's shipped ``SceneBuilder``
subclasses), Holodeck has **no native ManiSkill loader**; MolmoSpaces ships
its own ``molmo_spaces_maniskill.assets.loader.MjcfSceneLoader`` which takes
a ``sapien.Scene`` or ``ManiSkillScene`` and calls ``create_actor_builder``
+ ``create_articulation_builder`` directly. TyGrit's own
``SpecBackedSceneBuilder`` will learn to dispatch to that loader in
Step 11b (``worlds/backends/molmospaces_maniskill.py``); this module's
job is strictly the manifest side of things — enumerate the scenes on
disk, write one ``SceneSpec`` per ``.xml``, and pull the bulk data down so
future runtime code has something to point at.

.. _MolmoSpaces: https://huggingface.co/datasets/allenai/molmospaces

Pipeline (one-shot, run manually)::

    pixi run -e world generate-holodeck-scenes

Steps:

1. Construct an ``allenai/molmospaces`` :class:`HFRemoteStorage` pointing
   at the ``mujoco`` repo prefix (we use the MuJoCo MJCF export, not the
   Isaac USD export — MJCF is what ``MjcfSceneLoader`` consumes).
2. Call :func:`setup_resource_manager` with pinned versions for
   ``scenes/holodeck-objaverse-train``, ``objects/thor``, and
   ``objects/objaverse``. ``setup()`` pulls only the small per-subset
   index files (<20 MB total).
3. For each of the three sources, call
   :meth:`ResourceManager.install_all_for_source` to pull the full bulk
   data. This is the ~153 GB step:

   * ``scenes/holodeck-objaverse-train`` — 99 997 scenes, ~13.5 GB
   * ``objects/thor`` — 17 THOR furniture packages, ~1.5 GB
   * ``objects/objaverse`` — 129 725 curated Objaverse meshes, ~138.6 GB

   The Objaverse pool is the dominant cost because every Holodeck scene
   references a mix of THOR and Objaverse objects via relative mesh
   paths inside its MJCF. Skipping the Objaverse pool would produce
   scenes that spawn with missing geometry.
4. Walk ``manager.symlink_path("scenes", "holodeck-objaverse-train")``
   for ``*.xml`` files, sort for deterministic ordering, and emit one
   :class:`~TyGrit.types.worlds.SceneSpec` per file with
   ``background_mjcf`` pointing at the absolute (but repo-local) path
   and ``source="holodeck"``. ``scene_id`` is
   ``"holodeck/<source_subset>/<stem>"``.
5. Write the manifest to ``resources/worlds/holodeck.json.gz``
   (gzipped because 100 k SceneSpecs → ~15 MB uncompressed).

Opt-in flags
------------

* ``--include-val`` — also download and enumerate
  ``holodeck-objaverse-val`` (~1.1 GB additional, +10 001 scenes).
  Off by default because val is for eval and most RL runs don't need it.
* ``--no-download`` — skip the bulk download and only rebuild the
  manifest from whatever is already on disk under
  ``assets/molmospaces/mjcf/``. Use this to regenerate after tweaking
  naming conventions without re-paying the 153 GB cost.

Disk layout
-----------

All writes land under the gitignored project-local
``assets/molmospaces/`` tree to match the pattern we use for
ManiSkill / Objaverse caches::

    assets/molmospaces/
    ├── cache/                   # versioned tarball cache
    │   ├── scenes/holodeck-objaverse-train/20251217/...
    │   ├── objects/thor/20251117/...
    │   └── objects/objaverse/20260131/...
    └── mjcf/                    # extracted + symlinked working tree
        ├── scenes/holodeck-objaverse-train/*.xml
        ├── objects/thor/...
        └── objects/objaverse/...

Version pins
------------

Dates in :data:`VERSIONS` are the latest ``mujoco/`` MolmoSpaces snapshots
observed on HF as of 2026-04-11. Bumping them is a deliberate act — a
fresh snapshot may reshuffle scene counts or rename files, so any bump
should come paired with a manifest regeneration and (ideally) a spot
check that random sampled scenes still load via
:class:`MjcfSceneLoader`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from TyGrit.types.worlds import SceneSpec
from TyGrit.worlds.manifest import save_manifest

#: HF dataset hosting the MolmoSpaces bulk data.
HF_REPO_ID = "allenai/molmospaces"

#: Sub-directory inside the HF repo holding the MuJoCo-format MJCF
#: assets. The same repo also ships ``isaac/`` (USD) — we use MuJoCo
#: because AllenAI's ``MjcfSceneLoader`` (the only available ManiSkill-
#: adjacent loader) consumes MJCF XML, not USD.
HF_REPO_PREFIX = "mujoco"

#: Pinned version dates per (data_type, source). The ResourceManager
#: uses these to locate the right snapshot under ``mujoco/<data_type>/
#: <source>/<version>/`` on HF. All four are the latest mujoco snapshots
#: observed on 2026-04-11; bumping any date is a deliberate act.
VERSIONS: dict[str, dict[str, str]] = {
    "scenes": {
        "holodeck-objaverse-train": "20251217",
        # val is opt-in via --include-val; we keep the version pin
        # here so enabling the flag doesn't need a second constant.
        "holodeck-objaverse-val": "20251217",
    },
    "objects": {
        "thor": "20251117",
        "objaverse": "20260131",
    },
}

#: Default set of scene sources to download + enumerate. ``val`` is
#: added by the ``--include-val`` CLI flag.
DEFAULT_SCENE_SOURCES: tuple[str, ...] = ("holodeck-objaverse-train",)

#: Project-local paths. Everything under ``assets/molmospaces/`` is
#: gitignored (see .gitignore, added alongside this generator).
MOLMOSPACES_ROOT = Path("assets/molmospaces")
MJCF_DIR = MOLMOSPACES_ROOT / "mjcf"
CACHE_DIR = MOLMOSPACES_ROOT / "cache"
MANIFEST_PATH = Path("resources/worlds/holodeck.json.gz")


def _filter_versions(scene_sources: tuple[str, ...]) -> dict[str, dict[str, str]]:
    """Return the :data:`VERSIONS` dict reduced to the requested scene sources.

    ``objects/thor`` and ``objects/objaverse`` are always included
    because every Holodeck scene references meshes from both pools.
    Filtering out a scene source the caller didn't ask for avoids an
    unnecessary download if a future caller only wants val.
    """
    if not scene_sources:
        raise ValueError("scene_sources must be non-empty")
    unknown = set(scene_sources) - set(VERSIONS["scenes"])
    if unknown:
        raise ValueError(
            f"Unknown scene source(s) {sorted(unknown)}; "
            f"expected a subset of {sorted(VERSIONS['scenes'])}"
        )
    return {
        "scenes": {src: VERSIONS["scenes"][src] for src in scene_sources},
        "objects": dict(VERSIONS["objects"]),
    }


def download_molmospaces(scene_sources: tuple[str, ...]) -> None:
    """Download + extract the MolmoSpaces MJCF bulk data for Holodeck.

    Blocks until every scene package and every referenced object package
    is cached on disk. Idempotent: reruns skip sources that are already
    fully installed (the ResourceManager tracks install state via
    per-source ``*_complete`` flag files in the cache dir, so a
    partially-completed previous run resumes from where it stopped).

    Parameters
    ----------
    scene_sources
        A subset of :data:`VERSIONS`'s ``"scenes"`` keys to pull. Always
        includes the shared ``objects/thor`` + ``objects/objaverse``
        pools because Holodeck scene MJCFs reference meshes from both.

    Raises
    ------
    ImportError
        If ``molmospaces_resources`` isn't installed. This dep lives in
        the ``world`` pixi feature — run via
        ``pixi run -e world generate-holodeck-scenes`` so the right
        env is active.
    """
    # Deferred import: molmospaces_resources only lives in the `world`
    # pixi feature. Importing it at module load time would break the
    # default env's ability to even import TyGrit.worlds.generators —
    # which breaks test discovery. The same deferred-import pattern is
    # used by the AI2THOR / Objaverse generators for exactly this reason.
    from molmospaces_resources import HFRemoteStorage, setup_resource_manager

    MJCF_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    storage = HFRemoteStorage(repo_id=HF_REPO_ID, repo_prefix=HF_REPO_PREFIX)
    versions = _filter_versions(scene_sources)

    # setup_resource_manager fetches the per-source index files
    # (~20 MB total) and returns a ready-to-use manager. It does NOT
    # download the bulk shards; that requires an explicit
    # install_all_for_source call per source below.
    manager = setup_resource_manager(
        remote_storage=storage,
        symlink_dir=MJCF_DIR,
        cache_dir=CACHE_DIR,
        versions=versions,
        force_install=False,
        cache_lock=True,
    )

    # Now pull the bulk data. We call install_all_for_source per
    # (data_type, source) so the tqdm progress bars have a clear label
    # and failures are scoped to one source at a time.
    print("  downloading MolmoSpaces bulk data — this may take a long time...")
    for data_type, sources in versions.items():
        for source in sources:
            print(
                f"  install_all_for_source(data_type={data_type!r}, source={source!r})"
            )
            manager.install_all_for_source(data_type, source)


def list_scene_mjcfs(scene_source: str) -> list[Path]:
    """Return every Holodeck MJCF path on disk for a scene source.

    Walks ``MJCF_DIR / "scenes" / scene_source`` recursively for
    ``*.xml`` files and sorts the result so manifest output is stable
    across platforms. Raises :class:`FileNotFoundError` with a helpful
    message if the expected directory doesn't exist — almost always
    means :func:`download_molmospaces` hasn't been run yet.
    """
    scene_dir = MJCF_DIR / "scenes" / scene_source
    if not scene_dir.exists():
        raise FileNotFoundError(
            f"Expected Holodeck scene dir at {scene_dir} but it does not "
            f"exist. Run `pixi run -e world generate-holodeck-scenes` "
            f"(without --no-download) to fetch the MolmoSpaces bulk data."
        )
    paths = sorted(scene_dir.rglob("*.xml"))
    if not paths:
        raise FileNotFoundError(
            f"Holodeck scene dir {scene_dir} exists but contains no "
            f".xml files. The bulk download may have been interrupted; "
            f"rerun the generator without --no-download to resume."
        )
    return paths


def build_holodeck_manifest(
    scene_sources: tuple[str, ...] = DEFAULT_SCENE_SOURCES,
) -> tuple[SceneSpec, ...]:
    """Enumerate downloaded Holodeck scenes as a :class:`SceneSpec` tuple.

    Each spec is::

        SceneSpec(
            scene_id            = f"holodeck/{source}/{stem}",
            source              = "holodeck",
            background_mjcf     = str(mjcf_path),  # relative to repo root
        )

    ``objects`` is left empty — a Holodeck MJCF is self-contained: all
    furniture and Objaverse props are inlined in the XML via ``<include>``
    / ``<body>`` references to the shared ``objects/thor`` and
    ``objects/objaverse`` pools (downloaded alongside the scenes), so the
    SpecBackedSceneBuilder doesn't need a per-object spawn list — the
    MjcfSceneLoader parses the whole tree in one go during ``build()``.
    """
    specs: list[SceneSpec] = []
    for source in scene_sources:
        for mjcf in list_scene_mjcfs(source):
            stem = mjcf.stem
            specs.append(
                SceneSpec(
                    scene_id=f"holodeck/{source}/{stem}",
                    source="holodeck",
                    background_mjcf=str(mjcf),
                )
            )
    return tuple(specs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download AllenAI's Holodeck scene set from MolmoSpaces and "
            "write a TyGrit world manifest. Default pulls the train split "
            "(~100k scenes) plus the shared THOR + Objaverse object pools; "
            "full bulk download is ~153 GB (Objaverse dominates). "
            "Step 11b will add a runtime ManiSkill backend adapter that "
            "dispatches scene_id='holodeck/...' entries to AllenAI's "
            "MjcfSceneLoader; until then this manifest is data-only."
        )
    )
    parser.add_argument(
        "--include-val",
        action="store_true",
        help=(
            "Also download and enumerate the holodeck-objaverse-val split "
            "(~1.1 GB additional, +10001 scenes). Off by default because "
            "RL training only consumes the train split."
        ),
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help=(
            "Skip the bulk download and rebuild the manifest from whatever "
            "is already present under assets/molmospaces/mjcf/. Use this to "
            "regenerate after tweaking the manifest format without paying "
            "the 153 GB download cost again."
        ),
    )
    args = parser.parse_args()

    scene_sources: list[str] = list(DEFAULT_SCENE_SOURCES)
    if args.include_val:
        scene_sources.append("holodeck-objaverse-val")
    sources_tuple = tuple(scene_sources)

    if args.no_download:
        print("[1/2] Skipping download (--no-download)")
    else:
        print(
            f"[1/2] Downloading MolmoSpaces bulk data for {sources_tuple} + "
            f"shared objects/thor + objects/objaverse (~153 GB one-time)..."
        )
        download_molmospaces(sources_tuple)

    print(f"[2/2] Enumerating MJCFs + writing manifest to {MANIFEST_PATH}...")
    specs = build_holodeck_manifest(sources_tuple)
    save_manifest(
        MANIFEST_PATH,
        specs,
        source="holodeck",
        generator=(
            "TyGrit.worlds.generators.holodeck " f"--sources={','.join(sources_tuple)}"
        ),
    )
    print(f"       wrote {len(specs)} SceneSpecs to {MANIFEST_PATH}")
    print()
    print(
        f"Done. Commit {MANIFEST_PATH} (gzipped, small). The (gitignored) "
        f"bulk cache stays in {MOLMOSPACES_ROOT}/ for local use — Step 11b "
        f"will wire scene_id='holodeck/...' entries to MjcfSceneLoader so "
        f"they spawn in ManiSkill."
    )


if __name__ == "__main__":
    main()
