"""Generate world manifests for ManiSkill's AI2THOR family.

ManiSkill ships four AI2THOR variants under a single base class
(:class:`AI2THORBaseSceneBuilder`), all hosted in the HuggingFace
``hssd/ai2thor-hab`` conversion and pinned to one of the four
``scene_dataset`` values:

* **ProcTHOR** — 12 000 procedurally generated houses
* **iTHOR**    — 150 hand-authored interactive rooms
* **RoboTHOR** — 75 apartment-scale RoboTHOR scenes
* **ArchitecTHOR** — 10 architect-authored scenes

The scene lists are bundled with ManiSkill as JSON metadata files
(``mani_skill/utils/scene_builder/ai2thor/metadata/<variant>.json``),
so the generator doesn't depend on the bulk assets being downloaded —
it only needs the ManiSkill package installed. The bulk assets are
downloaded separately via::

    pixi run -e world download-ai2thor

Usage::

    pixi run -e world python -m TyGrit.worlds.generators.ai2thor

Writes four manifests under ``resources/worlds/``:

* ``procthor.json``
* ``ithor.json``
* ``robothor.json``
* ``architecthor.json``

Note on interactivity
---------------------
The ManiSkill port of AI2THOR is static-only — the original AI2THOR
simulator is Unity-based and uses "magic articulation scripts" that
couldn't be ported to URDF. Scenes load with furniture positioned but
without interactable articulations (no opening drawers/cabinets). That
limits manipulation diversity compared to ReplicaCAD (which ships
articulated URDFs), but the scene count more than compensates: 12 235
AI2THOR-family scenes vs 90 ReplicaCAD.
"""

from __future__ import annotations

from pathlib import Path

from TyGrit.types.worlds import SceneSpec
from TyGrit.worlds.manifest import save_manifest

#: Per-variant output path. ProcTHOR uses ``.json.gz`` because the
#: uncompressed 12 000-entry manifest is ~4.4 MB and exceeds
#: pre-commit's ``check-added-large-files`` 500 KB threshold.
#: :func:`TyGrit.worlds.manifest.save_manifest` sniffs the ``.gz``
#: suffix and gzip-compresses transparently; load_manifest does the
#: same on read. Other variants are small enough for plain JSON.
_DEFAULT_OUTPUTS: dict[str, Path] = {
    "procthor": Path("resources/worlds/procthor.json.gz"),
    "ithor": Path("resources/worlds/ithor.json"),
    "robothor": Path("resources/worlds/robothor.json"),
    "architecthor": Path("resources/worlds/architecthor.json"),
}


#: Variant name → metadata JSON filename under the ManiSkill package.
#: ManiSkill's ``AI2THORBaseSceneBuilder`` reads these same files at
#: construction time, but reading them directly lets the generator run
#: without the full multi-GB AI2THOR asset download (the metadata JSONs
#: are shipped with the Python package; only the bulk GLB assets need
#: to be downloaded via ``pixi run -e world download-ai2thor``).
_VARIANT_TO_METADATA_FILE: dict[str, str] = {
    "procthor": "ProcTHOR.json",
    "ithor": "iTHOR.json",
    "robothor": "RoboTHOR.json",
    "architecthor": "ArchitecTHOR.json",
}


def _ai2thor_metadata_dir() -> Path:
    """Return the package-shipped AI2THOR metadata directory.

    Deferred import so the generator module stays importable even when
    ``mani_skill`` isn't installed (e.g. inspection in the default env).
    """
    from mani_skill.utils.scene_builder.ai2thor import scene_builder as _ai2_module

    return Path(_ai2_module.__file__).parent / "metadata"


def list_scene_stems(variant: str) -> list[str]:
    """Return every scene stem for an AI2THOR variant.

    Reads ManiSkill's package-shipped ``metadata/<variant>.json`` file
    directly rather than instantiating the corresponding SceneBuilder
    subclass, because the subclass constructor's
    :func:`load_ai2thor_metadata` call touches
    ``ai2thor-hab/configs/object_semantic_id_mapping.json`` under the
    asset dir — which isn't needed just to enumerate scene stems, and
    which would force a 15 GB asset download just to regenerate a
    manifest.

    Parameters
    ----------
    variant
        One of ``"procthor"``, ``"ithor"``, ``"robothor"``,
        ``"architecthor"``.

    Returns
    -------
    list[str]
        Stripped scene stems in file order. Stem is
        ``Path(path).stem.split(".")[0]`` — strips the
        ``.scene_instance.json`` suffix and any ``./<subdir>/`` prefix.
    """
    import json

    if variant not in _VARIANT_TO_METADATA_FILE:
        raise ValueError(
            f"list_scene_stems: unknown variant {variant!r}; "
            f"expected one of {sorted(_VARIANT_TO_METADATA_FILE)}"
        )

    metadata_path = _ai2thor_metadata_dir() / _VARIANT_TO_METADATA_FILE[variant]
    with metadata_path.open() as f:
        scenes: list[str] = json.load(f)["scenes"]

    # Each entry is a string like "./2/ProcTHOR-Train-293.scene_instance.json".
    # Strip directory prefix via Path.stem (drops .json), then split at "."
    # to drop ".scene_instance". Stems are already unique per variant
    # (verified: 12000/12000, 150/150, 75/75, 10/10 unique).
    return [Path(scene).stem.split(".")[0] for scene in scenes]


def build_variant_manifest(variant: str) -> tuple[SceneSpec, ...]:
    """Construct SceneSpecs for one AI2THOR variant.

    Each spec has:

    * ``scene_id`` = ``"<variant>/<stem>"``
    * ``source`` = the lowercase variant name
    * ``background_builtin_id`` = ``"<variant>:<stem>"`` — the
      :class:`SpecBackedSceneBuilder` dispatches on the variant prefix
      to pick the right ManiSkill AI2THOR subclass.
    * ``objects`` = empty — ManiSkill's builder spawns the scene's
      furniture automatically (static only, no articulations).
    """
    stems = list_scene_stems(variant)
    return tuple(
        SceneSpec(
            scene_id=f"{variant}/{stem}",
            source=variant,
            background_builtin_id=f"{variant}:{stem}",
        )
        for stem in stems
    )


def main() -> None:
    """Write all four AI2THOR variant manifests and print a summary."""
    for variant, output in _DEFAULT_OUTPUTS.items():
        specs = build_variant_manifest(variant)
        save_manifest(
            output,
            specs,
            source=variant,
            generator="TyGrit.worlds.generators.ai2thor",
        )
        print(f"wrote {len(specs):>5} scenes to {output}")


if __name__ == "__main__":
    main()
