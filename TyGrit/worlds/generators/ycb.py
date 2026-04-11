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

import argparse
from pathlib import Path

from TyGrit.types.worlds import ObjectSpec
from TyGrit.worlds.manifest import save_object_manifest

#: Default output path. Committed small; no gzip needed for ~50 entries.
DEFAULT_OUTPUT: Path = Path("resources/worlds/objects/ycb.json")

#: Hand-curated subset of YCB that Fetch's parallel-jaw gripper can
#: actually grasp. Copied verbatim from grasp_anywhere v1's
#: ``tools/generate_grasp_benchmark.py`` (``YCB_OBJECTS``). The list
#: drops 28 objects from the full 78-entry ``info_pick_v0.json``:
#: large boxes (``003_cracker_box``, ``008_pudding_box``,
#: ``009_gelatin_box``, ``036_wood_block``), flat/thin objects
#: (``026_sponge``, ``029_plate``), elongated tools
#: (``030_fork``..``033_spatula``, ``037_scissors``,
#: ``042_adjustable_wrench``, ``043_phillips_screwdriver``), wide
#: bottles (``019_pitcher_base``, ``022_windex_bottle``), and
#: miscellaneous shapes Fetch can't reliably pinch.
#:
#: Source: /media/run/Work/paper/mobile_grasping_uncertainty/version_1/
#: grasp_anywhere_v1/tools/generate_grasp_benchmark.py:28-79 (50 objects).
FETCH_GRASPABLE_YCB: tuple[str, ...] = (
    "002_master_chef_can",
    "004_sugar_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "007_tuna_fish_can",
    "010_potted_meat_can",
    "011_banana",
    "012_strawberry",
    "013_apple",
    "014_lemon",
    "015_peach",
    "016_pear",
    "017_orange",
    "018_plum",
    "021_bleach_cleanser",
    "024_bowl",
    "025_mug",
    "035_power_drill",
    "044_flat_screwdriver",
    "048_hammer",
    "050_medium_clamp",
    "052_extra_large_clamp",
    "053_mini_soccer_ball",
    "054_softball",
    "055_baseball",
    "056_tennis_ball",
    "057_racquetball",
    "058_golf_ball",
    "061_foam_brick",
    "063-a_marbles",
    "063-b_marbles",
    "065-a_cups",
    "065-b_cups",
    "065-c_cups",
    "065-d_cups",
    "065-e_cups",
    "065-f_cups",
    "065-g_cups",
    "065-h_cups",
    "065-i_cups",
    "065-j_cups",
    "070-a_colored_wood_blocks",
    "072-c_toy_airplane",
    "073-a_lego_duplo",
    "073-b_lego_duplo",
    "073-c_lego_duplo",
    "073-d_lego_duplo",
    "073-e_lego_duplo",
    "073-f_lego_duplo",
    "077_rubiks_cube",
)

#: Valid ``--subset`` values for :func:`main`.
VALID_SUBSETS = ("fetch_graspable", "all")
DEFAULT_SUBSET = "fetch_graspable"


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


def _select_model_ids(subset: str) -> list[str]:
    """Filter the full ManiSkill YCB set by ``subset``.

    Parameters
    ----------
    subset
        One of :data:`VALID_SUBSETS`:

        * ``"fetch_graspable"`` — the 50-object hand-curated list from
          grasp_anywhere v1 that drops boxes, thin objects, elongated
          tools, and other items Fetch's parallel-jaw gripper can't
          reliably pick up. See :data:`FETCH_GRASPABLE_YCB`.
        * ``"all"`` — every model ID in ManiSkill's
          ``info_pick_v0.json`` (78 entries at the pinned version).

    Raises
    ------
    ValueError
        If ``subset`` is unknown, or if the ``fetch_graspable`` list
        has entries that aren't in ManiSkill's shipped metadata (which
        would indicate upstream drift we should surface, not silently
        ignore).
    """
    if subset not in VALID_SUBSETS:
        raise ValueError(f"Unknown subset {subset!r}; expected one of {VALID_SUBSETS}")
    all_ids = set(list_ycb_model_ids())
    if subset == "all":
        return sorted(all_ids)

    missing = [m for m in FETCH_GRASPABLE_YCB if m not in all_ids]
    if missing:
        raise ValueError(
            f"FETCH_GRASPABLE_YCB references {len(missing)} model IDs "
            f"not in ManiSkill's info_pick_v0.json at the pinned "
            f"mani-skill version; example: {missing[0]!r}. Either the "
            f"upstream set shrank or the curated list drifted. Check "
            f"grasp_anywhere v1's YCB_OBJECTS and the FETCH_GRASPABLE_YCB "
            f"constant in TyGrit/worlds/generators/ycb.py."
        )
    return list(FETCH_GRASPABLE_YCB)


def build_ycb_manifest(subset: str = DEFAULT_SUBSET) -> tuple[ObjectSpec, ...]:
    """Construct one ObjectSpec per selected YCB model ID.

    Each spec is keyed by the model ID (e.g. ``"002_master_chef_can"``)
    and uses ``builtin_id="ycb:<model_id>"`` so the ManiSkill spawn
    path dispatches to :func:`get_ycb_builder`. Poses/scales are left
    at defaults — callers that want a specific placement construct
    their own :class:`ObjectSpec` copies with overridden fields.
    """
    model_ids = _select_model_ids(subset)
    return tuple(
        ObjectSpec(
            name=model_id,
            builtin_id=f"ycb:{model_id}",
        )
        for model_id in model_ids
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a YCB ObjectSpec manifest for TyGrit's worlds layer. "
            "Default subset is 'fetch_graspable' (50 objects curated in "
            "grasp_anywhere v1 for Fetch's parallel-jaw gripper)."
        )
    )
    parser.add_argument(
        "--subset",
        default=DEFAULT_SUBSET,
        choices=VALID_SUBSETS,
        help=(
            "Which filter to apply. fetch_graspable (50) drops boxes, "
            "thin objects, and elongated tools Fetch can't pinch "
            "reliably. all (78) uses every model ManiSkill ships."
        ),
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Manifest output path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    objects = build_ycb_manifest(args.subset)
    save_object_manifest(
        Path(args.output),
        objects,
        source="ycb",
        generator=f"TyGrit.worlds.generators.ycb --subset={args.subset}",
    )
    print(f"wrote {len(objects)} YCB objects ({args.subset}) to {args.output}")


if __name__ == "__main__":
    main()
