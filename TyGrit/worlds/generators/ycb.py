"""Generate an object manifest for ManiSkill's shipped YCB models.

ManiSkill's ``ycb`` asset bundle unpacks to
``assets/mani_skill2_ycb/`` with a top-level
``info_pick_v0.json`` whose keys are model IDs (e.g. ``002_master_chef_can``)
and whose values carry bbox / scale / density metadata. This generator
reads that JSON and emits one :class:`~TyGrit.types.worlds.ObjectSpec`
per model ID, with ``builtin_id = "ycb:<model_id>"`` so
:class:`~TyGrit.worlds.backends.maniskill.SpecBackedSceneBuilder` can
dispatch to ManiSkill's :func:`get_ycb_builder` loader at spawn time.

Usage::

    pixi run -e world python -m TyGrit.worlds.generators.ycb

Requires the ``world`` pixi env (or any env with ``mani-skill``
installed + ``MS_ASSET_DIR`` pointing at a location that has the
``ycb`` asset bundle downloaded). If the YCB download hasn't been
run yet, first run::

    pixi run -e world download-ycb

The generator reads the bundle's metadata JSON via ManiSkill's own
``ASSET_DIR`` so it automatically picks up whichever path
``MS_ASSET_DIR`` configured — no hard-coded paths.
"""

from __future__ import annotations

from pathlib import Path

from TyGrit.types.worlds import ObjectSpec
from TyGrit.worlds.manifest import save_object_manifest

#: Default output path. Committed small; no gzip needed for ~78 entries.
DEFAULT_OUTPUT: Path = Path("resources/worlds/objects/ycb.json")


def list_ycb_model_ids() -> list[str]:
    """Return every YCB model ID in ManiSkill's shipped pick-v0 set.

    Reads ``info_pick_v0.json`` at ``ASSET_DIR/assets/mani_skill2_ycb/``,
    which is the same file :func:`mani_skill.utils.building.actors.ycb.get_ycb_builder`
    consumes at spawn time — so every ID returned here is guaranteed
    to be spawnable via that loader.

    Raises
    ------
    FileNotFoundError
        If the YCB bundle hasn't been downloaded yet. The error message
        points at the download command.
    """
    from mani_skill import ASSET_DIR
    from mani_skill.utils.io_utils import load_json

    info_path = Path(ASSET_DIR) / "assets/mani_skill2_ycb/info_pick_v0.json"
    if not info_path.exists():
        raise FileNotFoundError(
            f"YCB metadata not found at {info_path}. Run "
            f"`pixi run -e world download-ycb` first to fetch the bundle."
        )
    data = load_json(str(info_path))
    return sorted(data.keys())


def build_ycb_manifest() -> tuple[ObjectSpec, ...]:
    """Construct one ObjectSpec per YCB model ID.

    Each spec is keyed by the model ID (e.g. ``"002_master_chef_can"``)
    and uses ``builtin_id="ycb:<model_id>"`` so the ManiSkill spawn
    path can dispatch to :func:`get_ycb_builder`. Poses/scales are
    left at defaults — callers that want a specific placement should
    construct their own :class:`ObjectSpec` copies with overridden
    ``position`` / ``orientation_xyzw`` / ``scale`` fields.
    """
    model_ids = list_ycb_model_ids()
    return tuple(
        ObjectSpec(
            name=model_id,
            builtin_id=f"ycb:{model_id}",
        )
        for model_id in model_ids
    )


def main(output: Path = DEFAULT_OUTPUT) -> None:
    """Write the manifest to ``output`` and print a summary."""
    objects = build_ycb_manifest()
    save_object_manifest(
        output,
        objects,
        source="ycb",
        generator="TyGrit.worlds.generators.ycb",
    )
    print(f"wrote {len(objects)} YCB objects to {output}")


if __name__ == "__main__":
    main()
