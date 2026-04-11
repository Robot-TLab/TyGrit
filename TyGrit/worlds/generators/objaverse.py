"""Generate an object manifest from the NVIDIA-curated Objaverse-LVIS subset.

The meshes we use here are **Objaverse** — specifically the 8,515
objects NVIDIA pre-filtered from Objaverse-XL LVIS during their
GraspGen project (arXiv 2507.13097, ICRA'26). We inherit NVIDIA's
curation as a *graspability prior* (a UUID making it into their split
means it passed their geometric plausibility checks) but **do not use
any of their grasp labels** — TyGrit runs its own grasp oracle at
inference time, so we only need the meshes themselves.

The "source" field on each emitted :class:`ObjectSpec` is therefore
``"objaverse"`` — honest about what the data IS. The specific filter
(GraspGen/robotiq_2f_140/etc.) is recorded in the manifest's top-level
``generator`` metadata field so future callers can see exactly which
subset was sampled and with what seed.

Pipeline (one-shot, run manually)::

    pixi run -e world generate-objaverse-objects --count 200

Steps:

1. Fetch UUID split files from
   ``huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GraspGen/
   splits/<gripper>/train.txt`` via ``huggingface_hub`` (~100 KB).
2. Subsample N UUIDs deterministically via :mod:`random` seeded by
   ``--seed`` — same seed + same input list always produces the same
   subset.
3. Download per-UUID ``.glb`` meshes via the ``objaverse`` pip
   package. Each mesh gets cached in ``~/.objaverse/`` then copied
   to ``assets/objaverse/meshes/<uuid>.glb`` so the project-local
   path is the authoritative one referenced by the manifest.
4. Write an ObjectSpec manifest to
   ``resources/worlds/objects/objaverse.json`` with each entry's
   ``mesh_path`` pointing at the project-local file. The
   :class:`SpecBackedSceneBuilder` file-path spawn branch wires this
   into ManiSkill as ``scene.create_actor_builder()`` +
   ``add_visual_from_file()`` + ``add_convex_collision_from_file()``.

Defaults: ``--gripper robotiq_2f_140`` (closest to Fetch's parallel-jaw
geometry), ``--count 200``, ``--seed 0``. Adjust to taste — the
full ``robotiq_2f_140`` split has ~6000 training entries.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from TyGrit.types.worlds import ObjectSpec
from TyGrit.worlds.manifest import save_object_manifest

#: HuggingFace dataset repo hosting NVIDIA's GraspGen UUID splits.
HF_REPO_ID = "nvidia/PhysicalAI-Robotics-GraspGen"

#: Every gripper GraspGen shipped splits for. Robotiq-2f-140 is the
#: default because its parallel-jaw geometry (~140mm opening) is
#: closest to Fetch's gripper — objects NVIDIA judged graspable with
#: robotiq_2f_140 are likely graspable with Fetch too.
VALID_GRIPPERS = ("franka_panda", "robotiq_2f_140", "suction")
DEFAULT_GRIPPER = "robotiq_2f_140"
DEFAULT_COUNT = 200
DEFAULT_SEED = 0

#: Project-local paths. The assets directory is gitignored (see
#: .gitignore); only the small manifest JSON lives in git.
ASSETS_ROOT = Path("assets/objaverse")
SPLITS_DIR = ASSETS_ROOT / "splits"
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

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        filename=f"splits/{gripper}/train.txt",
        local_dir=str(SPLITS_DIR),
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


def build_manifest(mesh_paths: dict[str, Path]) -> tuple[ObjectSpec, ...]:
    """Create one :class:`ObjectSpec` per downloaded mesh, sorted by UUID.

    The name is the UUID (unique, hashable, debug-friendly). The
    ``mesh_path`` routes through
    :meth:`~TyGrit.worlds.backends.maniskill.SpecBackedSceneBuilder._spawn_one_object`'s
    file-path branch, which calls ManiSkill's
    ``add_visual_from_file`` + ``add_convex_collision_from_file`` on
    the .glb directly.
    """
    return tuple(
        ObjectSpec(name=uuid, mesh_path=str(path))
        for uuid, path in sorted(mesh_paths.items())
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sample N meshes from NVIDIA's GraspGen-curated Objaverse-LVIS "
            "subset and write a TyGrit object manifest. Uses NVIDIA's "
            "UUID splits as a graspability prior but ignores their "
            "grasp labels."
        )
    )
    parser.add_argument(
        "--gripper",
        default=DEFAULT_GRIPPER,
        choices=VALID_GRIPPERS,
        help=(
            "GraspGen gripper split to sample from. robotiq_2f_140 is "
            "closest to Fetch's parallel-jaw geometry."
        ),
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help=(
            "Number of objects to sample. Full robotiq_2f_140 train "
            "set has ~6000 entries; 200 is enough for RL variety "
            "while keeping the download bounded."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for subsampling. Same seed -> same subset.",
    )
    args = parser.parse_args()

    print(
        f"[1/4] Fetching UUID splits from {HF_REPO_ID} " f"(gripper={args.gripper})..."
    )
    split_file = fetch_uuid_splits(args.gripper)
    all_uuids = load_uuids(split_file)
    print(f"       full {args.gripper}/train set: {len(all_uuids)} UUIDs")

    print(f"[2/4] Subsampling {args.count} UUIDs (seed={args.seed})...")
    selected = subsample_uuids(all_uuids, args.count, args.seed)
    print(f"       selected: {len(selected)} UUIDs")

    print("[3/4] Downloading meshes (this may take a while)...")
    mesh_paths = download_meshes(selected)
    print(
        f"       downloaded: {len(mesh_paths)}/{len(selected)} meshes "
        f"into {MESHES_DIR}/"
    )

    print(f"[4/4] Writing manifest to {MANIFEST_PATH}...")
    specs = build_manifest(mesh_paths)
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
        f"Done. Commit {MANIFEST_PATH} and the (gitignored) mesh cache "
        f"stays in {ASSETS_ROOT}/ for local use."
    )


if __name__ == "__main__":
    main()
