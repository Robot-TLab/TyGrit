"""Generate an object manifest from the NVIDIA-curated Objaverse-LVIS subset.

The meshes we use here are **Objaverse** — specifically the 8,515
objects NVIDIA pre-filtered from Objaverse-XL LVIS during their
GraspGen project (arXiv 2507.13097, ICRA'26). We inherit NVIDIA's
curation as a *graspability prior* (a UUID making it into their split
means it passed their geometric plausibility checks) and we also
inherit their **per-object mesh scale** — see the "Mesh scale" section
below. We still do not use NVIDIA's *grasp label* poses: TyGrit runs
its own grasp oracle at inference time, so only scale + mesh matter.

The "source" field on each emitted :class:`ObjectSpec` is therefore
``"objaverse"`` — honest about what the data IS. The specific filter
(GraspGen/robotiq_2f_140/etc.) is recorded in the manifest's top-level
``generator`` metadata field so future callers can see exactly which
subset was sampled and with what seed.

Pipeline (one-shot, run manually)::

    pixi run -e world generate-objaverse-objects --count 200

Steps:

1. Fetch the UUID split text file from
   ``huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GraspGen/
   splits/<gripper>/train.txt`` via ``huggingface_hub`` (~100 KB).
2. Subsample N UUIDs deterministically via :mod:`random` seeded by
   ``--seed`` — same seed + same input list always produces the same
   subset.
3. Load the GraspGen per-object scales for each sampled UUID by
   reading the grasp JSONs out of NVIDIA's webdataset shards — see
   the "Mesh scale" section below for the full rationale and the
   bandwidth footprint.
4. Download per-UUID ``.glb`` meshes via the ``objaverse`` pip
   package. Each mesh gets cached in ``~/.objaverse/`` then copied
   to ``assets/objaverse/meshes/<uuid>.glb`` so the project-local
   path is the authoritative one referenced by the manifest.
5. Write an ObjectSpec manifest to
   ``resources/worlds/objects/objaverse.json`` with each entry's
   ``mesh_path`` pointing at the project-local file and
   ``scale=(s, s, s)`` carrying the GraspGen-recorded uniform scale.
   The :class:`SpecBackedSceneBuilder` file-path spawn branch wires
   this into ManiSkill as ``scene.create_actor_builder()`` +
   ``add_visual_from_file(..., scale=list(obj.scale))`` +
   ``add_convex_collision_from_file(..., scale=list(obj.scale))``.

Defaults: ``--gripper robotiq_2f_140`` (closest to Fetch's
parallel-jaw geometry), ``--count 200``, ``--seed 0``. Adjust to
taste — the full ``robotiq_2f_140`` split has ~8500 training entries.

Mesh scale
----------

Raw Objaverse meshes have **wildly inconsistent native scales** —
some meshes are ~1 mm across, some ~10 km, and there is no reliable
closed-form rule for normalizing them. NVIDIA's GraspGen data-gen
pipeline solves this per-object and records the result as
``grasps_dict["object"]["scale"]`` inside each grasp JSON: a single
uniform float such that, when applied to the raw mesh, the scaled
object is physically sensible for the ``robotiq_2f_140`` / franka
/ suction gripper that GraspGen simulated against. Since
robotiq_2f_140 is the closest match to Fetch's parallel-jaw
geometry, NVIDIA's scale is exactly what we want.

NVIDIA's grasp data is distributed on HuggingFace as 8 webdataset
tar shards per gripper, totalling ~6.9 GB for robotiq_2f_140. The
scales we need are a single float per UUID inside those tars, so
this step has an unavoidable **one-time ~6.9 GB download** the
first time the generator runs. Shards are cached under
``assets/objaverse/grasp_data/<gripper>/`` (gitignored); subsequent
runs read scales directly from the local cache with no network
traffic. The 200 × ~float64 resulting scales are compact enough to
live inside the committed manifest; users who want to free disk
after a successful generation can safely delete the
``grasp_data/`` subtree.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import tarfile
from pathlib import Path

from TyGrit.types.worlds import ObjectSpec
from TyGrit.worlds.manifest import save_object_manifest

#: HuggingFace dataset repo hosting NVIDIA's GraspGen UUID splits and
#: grasp data shards.
HF_REPO_ID = "nvidia/PhysicalAI-Robotics-GraspGen"

#: Number of grasp-data shards NVIDIA publishes per gripper. Fixed at
#: 8 in NVIDIA's webdataset layout — see
#: ``grasp_gen/dataset/webdataset_utils.py::convert_to_webdataset``
#: in the ``thirdparty/GraspGen`` submodule. We encode the number
#: explicitly rather than globbing because the layout is stable and
#: we don't want a partial download to silently report "all shards
#: present".
NUM_GRASPGEN_SHARDS = 8

#: Every gripper GraspGen shipped splits for. Robotiq-2f-140 is the
#: default because its parallel-jaw geometry (~140mm opening) is
#: closest to Fetch's gripper — objects NVIDIA judged graspable with
#: robotiq_2f_140 are likely graspable with Fetch too, and the scale
#: NVIDIA recorded was computed for robotiq_2f_140's gripper opening.
VALID_GRIPPERS = ("franka_panda", "robotiq_2f_140", "suction")
DEFAULT_GRIPPER = "robotiq_2f_140"
DEFAULT_COUNT = 200
DEFAULT_SEED = 0

#: Project-local paths. The assets directory is gitignored (see
#: .gitignore); only the small manifest JSON lives in git.
ASSETS_ROOT = Path("assets/objaverse")
MESHES_DIR = ASSETS_ROOT / "meshes"
MANIFEST_PATH = Path("resources/worlds/objects/objaverse.json")


def fetch_uuid_splits(gripper: str) -> Path:
    """Download NVIDIA's ``train.txt`` UUID splits for a gripper.

    Parameters
    ----------
    gripper
        One of :data:`VALID_GRIPPERS`.

    Returns
    -------
    Path
        Local path to the fetched ``train.txt``, under
        ``assets/objaverse/splits/<gripper>/``.

    Raises
    ------
    ValueError
        If ``gripper`` is not one of the three GraspGen-shipped splits.
    """
    if gripper not in VALID_GRIPPERS:
        raise ValueError(
            f"Unknown gripper {gripper!r}; expected one of {VALID_GRIPPERS}"
        )

    from huggingface_hub import hf_hub_download

    ASSETS_ROOT.mkdir(parents=True, exist_ok=True)
    train_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        filename=f"splits/{gripper}/train.txt",
        local_dir=str(ASSETS_ROOT),
    )
    return Path(train_path)


def load_uuids(split_file: Path) -> list[str]:
    """Read a GraspGen split text file. One UUID per line; blanks skipped."""
    with split_file.open() as f:
        return [line.strip() for line in f if line.strip()]


def subsample_uuids(uuids: list[str], count: int, seed: int) -> list[str]:
    """Deterministically sample ``count`` UUIDs from the full list.

    Uses :class:`random.Random` rather than numpy so the same seed +
    input always produces the same subset across machines regardless
    of numpy version. Results are sorted so the manifest diff is
    stable across runs.
    """
    if count >= len(uuids):
        return sorted(uuids)
    rng = random.Random(seed)
    return sorted(rng.sample(uuids, count))


def load_grasp_scales(gripper: str, uuids: list[str]) -> dict[str, float]:
    """Read the GraspGen-recorded per-UUID mesh scales for ``uuids``.

    Pulls NVIDIA's ``uuid_index.json`` (UUID → shard index mapping) to
    figure out which of the 8 grasp-data shards contain the requested
    UUIDs, then downloads exactly those shards via
    :func:`huggingface_hub.hf_hub_download` (cached under
    ``assets/objaverse/grasp_data/<gripper>/``) and extracts the
    single ``object.scale`` float from each UUID's ``grasps.json``
    webdataset member. Returns a UUID → float dict.

    Most 200-UUID subsamples will span all 8 shards (~6.9 GB total
    one-time download); downstream calls are hit from the local cache.

    Parameters
    ----------
    gripper
        One of :data:`VALID_GRIPPERS`. Selects which gripper's
        grasp-data shards to read — scales are gripper-specific
        because NVIDIA's generation pipeline picks a scale that makes
        the object graspable for THAT gripper's opening width.
    uuids
        The UUIDs to load scales for. Must all appear in NVIDIA's
        ``uuid_index.json`` for the chosen gripper, else
        :class:`ValueError` is raised naming the first missing UUID.

    Returns
    -------
    dict[str, float]
        UUID → scale mapping. Always contains exactly ``len(uuids)``
        entries on success.

    Raises
    ------
    ValueError
        If any UUID is missing from the grasp-data index (indicates
        the caller sampled from a different split than
        ``grasp_data/<gripper>/`` covers, which shouldn't happen when
        ``uuids`` come from :func:`fetch_uuid_splits` / :func:`load_uuids`
        of the same gripper).
    FileNotFoundError
        If a UUID is in the index but missing from its shard's .tar
        content — this would indicate a corrupt shard and should
        surface loudly.
    """
    from huggingface_hub import hf_hub_download

    ASSETS_ROOT.mkdir(parents=True, exist_ok=True)

    # Step 1: fetch the tiny uuid_index.json (a dict of UUID → shard int).
    index_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        filename=f"grasp_data/{gripper}/uuid_index.json",
        local_dir=str(ASSETS_ROOT),
    )
    with open(index_path) as f:
        uuid_to_shard: dict[str, int] = json.load(f)

    # Step 2: bucket requested UUIDs by shard. Missing UUIDs fail loudly
    # rather than silently dropping entries — a missing UUID usually means
    # the caller subsampled from a different split file than the grasp
    # data was built from, which would produce a broken manifest.
    shards_needed: dict[int, set[str]] = {}
    missing: list[str] = []
    for uuid in uuids:
        shard_idx = uuid_to_shard.get(uuid)
        if shard_idx is None:
            missing.append(uuid)
            continue
        shards_needed.setdefault(shard_idx, set()).add(uuid)
    if missing:
        raise ValueError(
            f"{len(missing)} of {len(uuids)} UUID(s) not found in "
            f"grasp_data/{gripper}/uuid_index.json "
            f"(first missing: {missing[0]!r}). Did you pass UUIDs from "
            f"a different gripper's split file?"
        )

    # Step 3: for each shard that has UUIDs we care about, download it
    # once (hf_hub_download caches subsequent calls as a no-op) and
    # stream-extract the grasps.json members. Webdataset tars pair each
    # sample as two members: "<uuid>.grasps.json" + "<uuid>.integer_id".
    # We need only the former and only the "object.scale" float from it.
    scales: dict[str, float] = {}
    for shard_idx in sorted(shards_needed):
        wanted: set[str] = shards_needed[shard_idx]
        print(
            f"  shard {shard_idx}: extracting {len(wanted)} scale(s) "
            f"(downloads ~862 MB on first run, cached thereafter)"
        )
        shard_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            filename=f"grasp_data/{gripper}/shard_{shard_idx:03d}.tar",
            local_dir=str(ASSETS_ROOT),
        )
        remaining: set[str] = set(wanted)
        with tarfile.open(shard_path, "r") as tar:
            for member in tar:
                if not member.name.endswith(".grasps.json"):
                    continue
                # Member name format: "<uuid>.grasps.json"
                uuid = member.name[: -len(".grasps.json")]
                if uuid not in remaining:
                    continue
                fobj = tar.extractfile(member)
                if fobj is None:
                    # extractfile returns None for non-regular-file
                    # members (symlinks, directories); webdataset shards
                    # don't contain those, so this would indicate a
                    # malformed shard.
                    raise FileNotFoundError(
                        f"shard {shard_idx}: member {member.name!r} is "
                        f"not a regular file (corrupt shard?)"
                    )
                grasps_dict = json.load(fobj)
                scales[uuid] = float(grasps_dict["object"]["scale"])
                remaining.discard(uuid)
                if not remaining:
                    # Every UUID we need from this shard is loaded —
                    # stop iterating instead of reading every remaining
                    # member's header (a shard has ~2000 members).
                    break
        if remaining:
            # UUIDs were in uuid_index.json but absent from their shard
            # .tar: either the index is stale or the shard is truncated.
            raise FileNotFoundError(
                f"shard {shard_idx}: {len(remaining)} UUID(s) indexed "
                f"but not found in the tar (first: {next(iter(remaining))!r})"
            )

    return scales


def download_meshes(uuids: list[str]) -> dict[str, Path]:
    """Download each UUID's mesh via the ``objaverse`` package and copy locally.

    ``objaverse.load_objects`` caches downloads in ``~/.objaverse/``;
    we then copy each file to :data:`MESHES_DIR` so the authoritative
    path is project-local and users can delete ``~/.objaverse/``
    afterward without breaking anything.

    Returns
    -------
    dict[str, Path]
        Mapping of ``uuid → project_local_mesh_path`` for every UUID
        the package successfully returned. Missing UUIDs are logged
        and omitted (no partial-state failure).
    """
    import objaverse  # type: ignore[import-untyped]

    MESHES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  objaverse.load_objects on {len(uuids)} UUIDs...")
    cache_paths = objaverse.load_objects(uids=uuids)

    out: dict[str, Path] = {}
    missing: list[str] = []
    for uuid in uuids:
        src = cache_paths.get(uuid)
        if src is None:
            missing.append(uuid)
            continue
        src_path = Path(src)
        # Preserve whatever extension objaverse returned (usually .glb).
        dest = MESHES_DIR / f"{uuid}{src_path.suffix}"
        shutil.copy2(src_path, dest)
        out[uuid] = dest

    if missing:
        print(
            f"  WARNING objaverse returned no path for {len(missing)} "
            f"UUID(s); example: {missing[0]!r}"
        )
    return out


def build_manifest(
    mesh_paths: dict[str, Path],
    scales: dict[str, float],
) -> tuple[ObjectSpec, ...]:
    """Create one :class:`ObjectSpec` per downloaded mesh, sorted by UUID.

    The name is the UUID (unique, hashable, debug-friendly). The
    ``mesh_path`` routes through
    :meth:`~TyGrit.worlds.backends.maniskill.SpecBackedSceneBuilder._spawn_one_object`'s
    file-path branch, which calls ManiSkill's
    ``add_visual_from_file`` + ``add_convex_collision_from_file`` on
    the .glb directly with ``scale=list(obj.scale)``. The per-UUID
    scale comes from :func:`load_grasp_scales` — the uniform float
    NVIDIA's GraspGen pipeline recorded for each object. We store it
    as a three-tuple ``(s, s, s)`` because :class:`ObjectSpec.scale`
    is per-axis; uniform scaling is just the diagonal case.

    Any UUID in ``mesh_paths`` that's missing from ``scales`` is
    skipped with a warning rather than emitted with a default scale
    — a missing scale almost always means ``load_grasp_scales``
    silently dropped something and should be investigated, not
    papered over with ``1.0``.
    """
    specs: list[ObjectSpec] = []
    skipped: list[str] = []
    for uuid, path in sorted(mesh_paths.items()):
        scale = scales.get(uuid)
        if scale is None:
            skipped.append(uuid)
            continue
        specs.append(
            ObjectSpec(
                name=uuid,
                mesh_path=str(path),
                scale=(scale, scale, scale),
            )
        )
    if skipped:
        print(
            f"  WARNING {len(skipped)} UUID(s) had mesh but no GraspGen "
            f"scale and were dropped; example: {skipped[0]!r}"
        )
    return tuple(specs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sample N meshes from NVIDIA's GraspGen-curated Objaverse-LVIS "
            "subset and write a TyGrit object manifest. Uses NVIDIA's "
            "UUID splits as a graspability prior AND their per-object "
            "mesh scales; we ignore their grasp-pose labels."
        )
    )
    parser.add_argument(
        "--gripper",
        default=DEFAULT_GRIPPER,
        choices=VALID_GRIPPERS,
        help=(
            "GraspGen gripper split to sample from. robotiq_2f_140 is "
            "closest to Fetch's parallel-jaw geometry — and the scales "
            "NVIDIA recorded were computed for this gripper's opening."
        ),
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help=(
            "Number of objects to sample. Full robotiq_2f_140 train "
            "set has ~8500 entries; 200 is enough for RL variety "
            "while keeping the mesh download bounded."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for subsampling. Same seed -> same subset.",
    )
    args = parser.parse_args()

    print(f"[1/5] Fetching UUID splits from {HF_REPO_ID} (gripper={args.gripper})...")
    split_file = fetch_uuid_splits(args.gripper)
    all_uuids = load_uuids(split_file)
    print(f"       full {args.gripper}/train set: {len(all_uuids)} UUIDs")

    print(f"[2/5] Subsampling {args.count} UUIDs (seed={args.seed})...")
    selected = subsample_uuids(all_uuids, args.count, args.seed)
    print(f"       selected: {len(selected)} UUIDs")

    print(
        f"[3/5] Loading GraspGen per-UUID scales "
        f"(first run downloads up to {NUM_GRASPGEN_SHARDS} × ~862 MB shards)..."
    )
    scales = load_grasp_scales(args.gripper, selected)
    print(f"       loaded scales for {len(scales)} UUIDs")

    print("[4/5] Downloading meshes (this may take a while)...")
    mesh_paths = download_meshes(selected)
    print(
        f"       downloaded: {len(mesh_paths)}/{len(selected)} meshes "
        f"into {MESHES_DIR}/"
    )

    print(f"[5/5] Writing manifest to {MANIFEST_PATH}...")
    specs = build_manifest(mesh_paths, scales)
    save_object_manifest(
        MANIFEST_PATH,
        specs,
        source="objaverse",
        generator=(
            f"TyGrit.worlds.generators.objaverse "
            f"--gripper={args.gripper} --count={args.count} --seed={args.seed}"
        ),
    )
    print(f"       wrote {len(specs)} ObjectSpecs to {MANIFEST_PATH}")
    print()
    print(
        f"Done. Commit {MANIFEST_PATH} and the (gitignored) mesh + "
        f"grasp-data cache stays in {ASSETS_ROOT}/ for local use."
    )


if __name__ == "__main__":
    main()
