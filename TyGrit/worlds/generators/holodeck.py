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
Step 11b (``worlds/backends/_maniskill_holodeck.py``); this module's
job is strictly the manifest side of things — enumerate the scenes on
disk, write one ``SceneSpec`` per ``.xml``, and pull down *only the
objects those scenes actually reference* so future runtime code has
something to point at without costing half a terabyte of mesh storage.

.. _MolmoSpaces: https://huggingface.co/datasets/allenai/molmospaces

Why the pipeline filters the Objaverse pool
-------------------------------------------

A naive ``install_all_for_source("objects", "objaverse")`` call would pull
every one of NVIDIA's 129 725 curated Objaverse UUIDs — ~138 GB compressed
and far more on disk after extraction. Empirically, a single Holodeck
scene references only **~50 unique Objaverse UUIDs**, and the 100 k
scenes share heavy overlap between them (many scenes reuse the same
chairs / tables / decor). The actual working set is almost certainly
well under 30 k UUIDs — roughly 1/5 of the full pool — which is the
difference between a ~150 GB run and a usable ~75 GB run (plus a more
modest ~15 GB smoke-test if ``--count`` is used to subsample scenes).

We filter by parsing each extracted scene's MJCF for
``../../objects/objaverse/<uuid>/`` references, unioning the set across
scenes, and then calling :meth:`ResourceManager.install_packages` with
that exact list. The ``find_archives`` helper on the resource manager
translates UUID-qualified paths into Objaverse package names.

Pipeline (one-shot, run manually)::

    pixi run -e world generate-holodeck-scenes                    # full train
    pixi run -e world generate-holodeck-scenes --count 1000       # smoke
    pixi run -e world generate-holodeck-scenes --include-val      # + val

Stages:

1. **setup** — ``setup_resource_manager`` downloads the per-source
   index files (~30 MB total) and, because ``objects/thor`` is declared
   ``EAGER``, auto-extracts the full THOR pool (~3.2 GB on disk,
   ~1.5 GB compressed).
2. **scene subsample** — if ``--count N`` is given, deterministically
   sample N scene package names from the scene source's
   ``mjthor_resource_file_to_size_mb.json`` under the per-seed RNG;
   otherwise use all 99 997 train scenes.
3. **scene install** — ``install_packages("scenes", {source: [pkgs]})``
   downloads the HF shards containing those scenes and extracts each
   into ``cache/scenes/<source>/<version>/train_<N>.{xml,json,...}``.
4. **scene linking** — manually symlink every scene file from its
   cache location into ``mjcf/scenes/<source>/train_<N>.*``. The
   ``ResourceManager``'s own ``PER_FILE`` linker for scenes is broken
   because the scene source ships an empty trie dict; we recreate the
   equivalent layout ourselves so that the MJCF's ``../../objects/...``
   relative references resolve through ``mjcf/objects/{thor,objaverse}/``
   (both of which *are* symlinks the ResourceManager creates correctly).
5. **objaverse filter + install** — parse every linked MJCF for
   Objaverse UUIDs, union them, call ``find_archives`` to resolve the
   working set to package names, then
   ``install_packages("objects", {"objaverse": pkgs})``. This is the
   only stage whose cost scales with the number of unique UUIDs rather
   than scene count.
6. **manifest** — emit one :class:`~TyGrit.types.worlds.SceneSpec` per
   linked MJCF with ``source="holodeck"``,
   ``scene_id="holodeck/<source_subset>/<stem>"``, and
   ``background_mjcf`` pointing at the (project-local) symlink path.
   Write to ``resources/worlds/holodeck.json.gz`` (gzipped because
   100 k SceneSpecs uncompressed exceeds the pre-commit large-file
   threshold).

Opt-in flags
------------

* ``--count N`` — subsample to N scenes (deterministic via
  ``--seed``). Off by default ⇒ use every scene in the split.
  Start with something like ``--count 1000`` for a fast smoke test,
  then rerun without the flag once you've confirmed the pipeline
  works end-to-end.
* ``--seed S`` — RNG seed for the subsample step. Default ``0``.
* ``--include-val`` — also pull ``holodeck-objaverse-val`` (+10 001
  scenes, adds ~2-4 GB in scenes + a delta of Objaverse packages that
  aren't already in the train working set).
* ``--no-download`` — skip every network / extraction step and only
  rebuild the manifest from whatever is already linked under
  ``assets/molmospaces/mjcf/scenes/<source>/``. Useful for format
  tweaks without re-paying the download cost.

Disk layout
-----------

Everything lands under the gitignored project-local
``assets/molmospaces/`` tree::

    assets/molmospaces/
    ├── cache/                   # extracted .tar.zst payloads
    │   ├── scenes/holodeck-objaverse-train/20251217/
    │   │   ├── train_<N>.xml
    │   │   ├── train_<N>_assets/
    │   │   └── ...
    │   ├── objects/thor/20251117/...        (auto-installed, EAGER)
    │   └── objects/objaverse/20260131/      (filtered subset only)
    │       └── <uuid>/<uuid>_visual.obj ...
    └── mjcf/                    # symlink tree the MJCFs resolve against
        ├── scenes/holodeck-objaverse-train/
        │   ├── train_<N>.xml -> ../../cache/.../20251217/train_<N>.xml
        │   ├── train_<N>_assets -> ../../cache/.../20251217/train_<N>_assets
        │   └── ...
        ├── objects/thor -> ../../cache/objects/thor/20251117
        └── objects/objaverse -> ../../cache/objects/objaverse/20260131

Version pins
------------

:data:`VERSIONS` tracks the latest ``mujoco/`` MolmoSpaces snapshots as
of 2026-04-11. Bumping any date is deliberate — a fresh snapshot may
reshuffle scenes or rename object UUIDs, which would invalidate the
filtered working set the manifest was generated against.
"""

from __future__ import annotations

import argparse
import json
import random
import re
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

#: Pinned version dates per (data_type, source). Bumping any date is a
#: deliberate act — a fresh snapshot may reshuffle scenes or rename
#: object UUIDs, which would invalidate a generated manifest's filtered
#: working set.
VERSIONS: dict[str, dict[str, str]] = {
    "scenes": {
        "holodeck-objaverse-train": "20251217",
        # val is opt-in via --include-val; we keep the version pin here
        # so enabling the flag doesn't need a second constant.
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

DEFAULT_SEED: int = 0

#: Project-local paths. Everything under ``assets/molmospaces/`` is
#: gitignored.
MOLMOSPACES_ROOT = Path("assets/molmospaces")
MJCF_DIR = MOLMOSPACES_ROOT / "mjcf"
CACHE_DIR = MOLMOSPACES_ROOT / "cache"
MANIFEST_PATH = Path("resources/worlds/holodeck.json.gz")

#: Files the ResourceManager already symlinks at the top of
#: ``mjcf/scenes/<source>/``. Our manual linker must skip these so it
#: doesn't try to re-create them.
_RESOURCE_MANAGER_INDEX_FILES = frozenset(
    {
        "mjthor_resource_file_to_size_mb.json",
        "mjthor_resources_combined_meta.json.gz",
    }
)

#: Regex matching ``file="../../objects/objaverse/<uuid>/..."`` references
#: inside Holodeck MJCFs. The captured group is just the UUID.
_OBJAVERSE_REF_RE = re.compile(r'"\.\./\.\./objects/objaverse/([^/]+)/')

#: Regex that pulls the ``train_<N>`` stem out of an extracted scene
#: file's ``name`` (e.g. ``train_91962.xml`` → ``train_91962``,
#: ``train_91962_assets`` → ``train_91962``). Flag files start with
#: ``.`` and index files start with ``mjthor_`` so they fail to match
#: and are naturally excluded.
_SCENE_STEM_RE = re.compile(r"^(train_\d+)(?:[._]|$)")


def _scene_stem(pkg: str) -> str | None:
    """Return ``train_<N>`` for scene packages, or ``None`` for auxiliaries.

    The MolmoSpaces scene manifest (``mjthor_resource_file_to_size_mb.json``)
    lists the ~100k ``<source>_train_<N>.tar.zst`` scene packages plus a
    handful of auxiliary config packages (observed: ``housegen_build_
    settings.json.tar.zst``). The auxiliaries get downloaded + extracted
    by ``install_packages`` but aren't scenes, so the manifest/UUID-
    collection stages have to skip them.
    """
    stem = pkg.rsplit(".tar.zst", 1)[0].split("_", 1)[-1]
    if stem.startswith("train_") and stem[len("train_") :].isdigit():
        return stem
    return None


def _filter_versions(scene_sources: tuple[str, ...]) -> dict[str, dict[str, str]]:
    """Reduce :data:`VERSIONS` to only the requested scene sources.

    Object sources are always included because Holodeck scenes reference
    meshes from both THOR and Objaverse pools. Filtering out a scene
    source the caller didn't ask for avoids paying the download cost
    for a source they won't enumerate.
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


def _build_manager(scene_sources: tuple[str, ...]):
    """Stand up a configured :class:`ResourceManager` for the requested sources.

    Deferred ``molmospaces_resources`` import so the generator module
    stays importable in the default pixi env (which doesn't have the
    dep installed). Same pattern the Objaverse + AI2THOR generators
    use for their heavy imports.
    """
    from molmospaces_resources import HFRemoteStorage, setup_resource_manager

    MJCF_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    storage = HFRemoteStorage(repo_id=HF_REPO_ID, repo_prefix=HF_REPO_PREFIX)
    return setup_resource_manager(
        remote_storage=storage,
        symlink_dir=MJCF_DIR,
        cache_dir=CACHE_DIR,
        versions=_filter_versions(scene_sources),
        force_install=False,
        cache_lock=True,
    )


def _scene_cache_version_dir(manager, source: str) -> Path:
    """Return the extracted-cache version directory for a scene source.

    ``ResourceManager.cache_path`` points at the versioned subdir where
    scene packages land after ``install_packages`` extracts them. This
    is the *authoritative* on-disk location for scene MJCFs; the
    ``mjcf/scenes/<source>/`` tree is a symlink layer on top (which
    our manual linker below populates).
    """
    return manager.cache_path("scenes", source)


def fixup_object_symlinks(manager) -> None:
    """Rewrite ``mjcf/objects/{thor,objaverse}`` with absolute targets.

    ``ResourceManager`` creates those two top-level object symlinks
    with a *repo-relative* target like
    ``"assets/molmospaces/cache/objects/thor/20251117"``. On most
    filesystems, relative symlink targets are resolved against the
    symlink's parent directory, NOT the process CWD — so the OS ends
    up looking for ``mjcf/objects/assets/molmospaces/cache/objects/...``
    which doesn't exist. The consequence is that every scene MJCF's
    ``../../objects/thor/...`` reference silently dangles.

    We delete those bad symlinks and recreate them pointing at the
    *absolute* cache version directory. Absolute targets make the
    links survive ``cd`` in any consumer while still living under
    the gitignored ``assets/`` tree. Running this is idempotent — if
    the link already has an absolute target we leave it alone.
    """
    for obj_source in VERSIONS["objects"]:
        link = MJCF_DIR / "objects" / obj_source
        cache_target = manager.cache_path("objects", obj_source).resolve()
        if link.is_symlink():
            current = Path(link.readlink())
            if current.is_absolute() and current == cache_target:
                continue
            link.unlink()
        elif link.exists():
            # A plain directory exists where we need the symlink —
            # refuse to touch it so we don't accidentally nuke user data.
            raise FileExistsError(
                f"{link} exists as a regular directory; expected either "
                f"nothing or a ResourceManager-created symlink. Move or "
                f"delete it before rerunning the generator."
            )
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(cache_target)


def load_scene_package_names(manager, source: str) -> list[str]:
    """Return every scene package name (``*.tar.zst``) the source ships.

    Reads ``mjthor_resource_file_to_size_mb.json`` under the cache
    version dir — that file is downloaded by ``setup()`` before any
    per-package install, so this works from the moment
    :func:`_build_manager` returns.
    """
    version_dir = _scene_cache_version_dir(manager, source)
    manifest_path = version_dir / "mjthor_resource_file_to_size_mb.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Expected scene package manifest at {manifest_path} but it "
            f"is missing. setup_resource_manager should have fetched it; "
            f"if it didn't, the manager construction likely failed."
        )
    with manifest_path.open() as f:
        return sorted(json.load(f).keys())


def subsample_scene_packages(
    packages: list[str], count: int | None, seed: int
) -> list[str]:
    """Deterministically sample ``count`` scene package names.

    ``count = None`` returns the full sorted list. Otherwise uses
    :class:`random.Random` seeded on ``seed`` for repeatability
    independent of numpy / hashseed state, then re-sorts so the
    manifest diff is stable across reruns.
    """
    if count is None or count >= len(packages):
        return list(packages)
    rng = random.Random(seed)
    return sorted(rng.sample(packages, count))


def install_scene_packages(manager, source: str, packages: list[str]) -> None:
    """Download + extract the requested scene packages via the resource manager.

    Idempotent: packages that the ResourceManager has already marked as
    extracted (via ``_complete_extract_flag`` files in the cache dir)
    are skipped, so reruns cost only the incremental download.
    """
    if not packages:
        return
    print(
        f"  install_packages(scenes, {source}) — "
        f"{len(packages)} package(s) to install/verify"
    )
    manager.install_packages("scenes", {source: packages})


def _scene_file_entries(
    cache_version_dir: Path, package_names: list[str]
) -> list[Path]:
    """Resolve each scene package name to the on-disk files it extracted to.

    Each package ``holodeck-objaverse-train_train_<N>.tar.zst`` extracts
    to a set of sibling files sharing the ``train_<N>`` stem:
    ``train_<N>.xml``, ``train_<N>.json``, ``train_<N>_metadata.json``,
    ``train_<N>_ceiling.xml``, and a ``train_<N>_assets/`` directory.
    Our linker needs to glob all of them without assuming a fixed set
    of suffixes (fresh snapshots have added new sidecars in the past).

    Returns a flat list of paths inside ``cache_version_dir`` — both
    top-level files and the ``_assets`` directory (as a single Path,
    not its recursive contents).

    Implementation: single ``iterdir()`` pass, bucketing by ``train_<N>``
    stem. The previous O(N²) version re-iterated the whole cache dir
    for each of the ~100k packages, which froze the pipeline for hours
    at a ~500k-entry cache. Now we scan once (O(N)) and do O(1) dict
    lookups per package.
    """
    by_stem: dict[str, list[Path]] = {}
    for candidate in cache_version_dir.iterdir():
        match = _SCENE_STEM_RE.match(candidate.name)
        if match is None:
            # Index files (mjthor_*), flag files (.<pkg>_complete_*) and
            # any other unrelated entries fall through here and are
            # correctly ignored.
            continue
        by_stem.setdefault(match.group(1), []).append(candidate)

    entries: list[Path] = []
    for pkg in package_names:
        stem = _scene_stem(pkg)
        if stem is None:
            continue
        entries.extend(by_stem.get(stem, ()))
    return entries


def link_scenes_into_mjcf_dir(manager, source: str, package_names: list[str]) -> Path:
    """Symlink extracted scene files from cache into ``mjcf/scenes/<source>/``.

    The ResourceManager's PER_FILE linking for the scenes source is a
    no-op because its scene trie dict is empty (a quirk we don't need
    to debug — the scene package trie *is* populated in the LMDB
    archive index used by ``find_archives``, just not the higher-level
    ``tries()`` helper). We create the equivalent symlink layout
    ourselves: for each scene package, link every extracted sibling
    file / dir at ``mjcf/scenes/<source>/<filename>`` pointing at the
    cache copy. This makes the MJCF's ``../../objects/...`` references
    resolve through ``mjcf/objects/{thor,objaverse}/`` symlinks that
    the resource manager *does* create correctly.

    Returns the target directory, which is then walked by
    :func:`build_holodeck_manifest` to produce the manifest entries.
    """
    cache_version_dir = _scene_cache_version_dir(manager, source)
    target_dir = MJCF_DIR / "scenes" / source
    target_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot existing link names once so the per-entry existence check
    # is an O(1) set lookup instead of a stat syscall. At 100k packages
    # * 5 files each = 500k iterations, the stat cost adds up to minutes.
    existing = {p.name for p in target_dir.iterdir()}

    linked = 0
    for entry in _scene_file_entries(cache_version_dir, package_names):
        if entry.name in existing:
            continue
        # Use absolute paths so the link survives `cd` in a future
        # consumer, and so a walk from MJCF_DIR finds the actual files
        # instead of broken relative targets.
        (target_dir / entry.name).symlink_to(entry.resolve())
        existing.add(entry.name)
        linked += 1

    print(f"  linked {linked} scene entries into {target_dir}")
    return target_dir


def collect_objaverse_uuids(
    mjcf_source_dir: Path, package_names: list[str]
) -> set[str]:
    """Scan the linked MJCFs for referenced Objaverse UUIDs.

    Iterates only the ``train_<N>.xml`` files that correspond to the
    installed ``package_names`` — we do NOT recurse into every link the
    directory has accumulated, because a prior generator run may have
    left orphan links from a different package set. Each MJCF is read
    as text and regex-scanned for ``../../objects/objaverse/<uuid>/``
    references; the union across all scenes is returned.
    """
    uuids: set[str] = set()
    for pkg in package_names:
        stem = _scene_stem(pkg)
        if stem is None:
            continue
        mjcf = mjcf_source_dir / f"{stem}.xml"
        if not mjcf.exists():
            # Shouldn't happen if link_scenes_into_mjcf_dir ran, but surface
            # rather than silently missing the UUIDs.
            raise FileNotFoundError(
                f"Expected linked MJCF at {mjcf}; was link_scenes_into_mjcf_dir "
                f"called with the same package_names?"
            )
        uuids.update(_OBJAVERSE_REF_RE.findall(mjcf.read_text()))
    return uuids


def install_filtered_objaverse(manager, uuids: set[str]) -> None:
    """Download only the Objaverse packages referenced by our scenes.

    Turns each UUID into a sample path (``<uuid>/<uuid>_visual.obj``),
    calls :meth:`ResourceManager.find_archives` to translate those to
    package names, de-duplicates, and installs the resulting subset.
    With an average of ~50 Objaverse UUIDs per scene and heavy scene-
    to-scene overlap, this is typically 1/5 the size of the full
    Objaverse pool — and far less on disk after extraction.
    """
    if not uuids:
        print("  no Objaverse UUIDs referenced; skipping Objaverse install")
        return
    sample_paths = [f"{u}/{u}_visual.obj" for u in sorted(uuids)]
    print(
        f"  find_archives(objects, objaverse, <{len(sample_paths)} paths>) "
        f"to resolve the working set..."
    )
    pkgs = sorted(set(manager.find_archives("objects", "objaverse", sample_paths)))
    print(
        f"  install_packages(objects, objaverse) — {len(pkgs)} package(s) "
        f"(filtered from the full 129725-UUID pool)"
    )
    manager.install_packages("objects", {"objaverse": pkgs})


def build_holodeck_manifest(
    mjcf_source_dir: Path, source: str, package_names: list[str]
) -> tuple[SceneSpec, ...]:
    """Emit one :class:`SceneSpec` per linked Holodeck MJCF.

    ``scene_id`` is ``"holodeck/<source>/<stem>"`` where ``stem`` is
    the MJCF filename without extension (globally unique inside a
    source). ``background_mjcf`` is the project-local symlink path
    under ``mjcf/scenes/<source>/`` — guaranteed to be the right
    location for the MJCF's ``../../objects/...`` references to
    resolve. ``objects`` is empty because Holodeck MJCFs inline their
    object placements via ``<include>`` / ``<body>`` references; the
    per-scene spawn happens inside ``MjcfSceneLoader.load()`` at
    Step 11b runtime rather than as a list of
    :class:`~TyGrit.types.worlds.ObjectSpec` entries here.
    """
    specs: list[SceneSpec] = []
    for pkg in package_names:
        stem = _scene_stem(pkg)
        if stem is None:
            continue
        mjcf = mjcf_source_dir / f"{stem}.xml"
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
            "write a TyGrit world manifest. Only downloads the Objaverse "
            "meshes actually referenced by the selected scenes; the shared "
            "THOR pool (~3 GB) is pulled automatically by the resource "
            "manager as an eager source. Step 11b will add a ManiSkill "
            "backend adapter that dispatches scene_id='holodeck/...' "
            "entries to AllenAI's MjcfSceneLoader; until then this manifest "
            "is data-only."
        )
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help=(
            "Subsample this many scenes deterministically (default: all). "
            "Recommended smoke-test value is 1000; full train set is 99997."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="RNG seed for --count subsampling. Same seed -> same subset.",
    )
    parser.add_argument(
        "--include-val",
        action="store_true",
        help=(
            "Also download and enumerate the holodeck-objaverse-val split "
            "(+10001 scenes). Off by default because RL training only "
            "consumes the train split."
        ),
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help=(
            "Skip every download / extract step and rebuild the manifest "
            "from whatever is already linked under "
            "assets/molmospaces/mjcf/scenes/<source>/. Useful for format "
            "tweaks without re-paying the download cost."
        ),
    )
    args = parser.parse_args()

    scene_sources: list[str] = list(DEFAULT_SCENE_SOURCES)
    if args.include_val:
        scene_sources.append("holodeck-objaverse-val")
    sources_tuple = tuple(scene_sources)

    print(f"[1/6] Standing up ResourceManager for sources={sources_tuple}...")
    manager = _build_manager(sources_tuple)
    fixup_object_symlinks(manager)

    all_specs: list[SceneSpec] = []
    # Per-source install + link + UUID collect. Each source is
    # independent so failures in one don't corrupt another.
    all_referenced_uuids: set[str] = set()
    per_source_selected: dict[str, list[str]] = {}

    for source in sources_tuple:
        all_pkgs = load_scene_package_names(manager, source)
        selected = subsample_scene_packages(all_pkgs, args.count, args.seed)
        per_source_selected[source] = selected
        print(
            f"[2/6] Scene source {source!r}: "
            f"{len(all_pkgs)} total, {len(selected)} selected "
            f"(count={args.count}, seed={args.seed})"
        )

        if args.no_download:
            print(f"[3/6] (skipped install for {source}, --no-download)")
        else:
            print(f"[3/6] Installing scene packages for {source}...")
            install_scene_packages(manager, source, selected)

        print(f"[4/6] Linking scenes into mjcf dir for {source}...")
        mjcf_dir = link_scenes_into_mjcf_dir(manager, source, selected)

        print(f"[5a/6] Parsing MJCFs for Objaverse refs in {source}...")
        uuids = collect_objaverse_uuids(mjcf_dir, selected)
        print(f"       {source}: {len(uuids)} unique Objaverse UUIDs referenced")
        all_referenced_uuids |= uuids

        all_specs.extend(build_holodeck_manifest(mjcf_dir, source, selected))

    if args.no_download:
        print("[5b/6] (skipped Objaverse install, --no-download)")
    else:
        print(
            f"[5b/6] Installing filtered Objaverse subset "
            f"({len(all_referenced_uuids)} unique UUIDs across all sources)..."
        )
        install_filtered_objaverse(manager, all_referenced_uuids)

    print(f"[6/6] Writing manifest to {MANIFEST_PATH}...")
    save_manifest(
        MANIFEST_PATH,
        tuple(all_specs),
        source="holodeck",
        generator=(
            "TyGrit.worlds.generators.holodeck "
            f"--sources={','.join(sources_tuple)} "
            f"--count={args.count} --seed={args.seed}"
        ),
    )
    print(f"       wrote {len(all_specs)} SceneSpecs to {MANIFEST_PATH}")
    print()
    print(
        f"Done. Commit {MANIFEST_PATH}. The (gitignored) bulk cache "
        f"stays in {MOLMOSPACES_ROOT}/ for local use — Step 11b "
        f"will wire scene_id='holodeck/...' entries to MjcfSceneLoader so "
        f"they spawn in ManiSkill."
    )


if __name__ == "__main__":
    main()
