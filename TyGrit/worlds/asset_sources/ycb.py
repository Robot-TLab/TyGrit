"""ManiSkill YCB — the curated 50-object subset shipped with ManiSkill3.

YCB is the simplest :class:`AssetSource`:

* object-only (no scenes),
* ids are ManiSkill builtin ids of the form ``"<ycb_model_id>"``
  (prefixed with ``"ycb:"`` when written into
  :attr:`ObjectSpec.builtin_id`),
* resolved via ManiSkill's asset registry at runtime, so this source
  only has to declare the catalogue — no file paths, no physics
  proxies, no quirks.

Catalogue
---------

The v1 ``grasp_anywhere`` project picked 50 YCB objects graspable by
Fetch; that list is the canonical TyGrit subset. We enumerate them
here as plain strings; ManiSkill validates each id at spawn time and
errors early for typos or missing assets.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from TyGrit.types.worlds import ObjectSpec, SceneSpec
from TyGrit.worlds.asset_sources.base import AssetRequest

if TYPE_CHECKING:
    pass


#: 50-object Fetch-graspable YCB subset carried over from v1
#: (see ``grasp_anywhere/benchmark/ycb_pool.py``). Edit this tuple to
#: expand / contract the pool; keep it sorted so diffs stay readable.
YCB_FETCH_GRASPABLE: tuple[str, ...] = (
    "002_master_chef_can",
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "007_tuna_fish_can",
    "008_pudding_box",
    "009_gelatin_box",
    "010_potted_meat_can",
    "011_banana",
    "012_strawberry",
    "013_apple",
    "014_lemon",
    "015_peach",
    "016_pear",
    "017_orange",
    "018_plum",
    "019_pitcher_base",
    "021_bleach_cleanser",
    "024_bowl",
    "025_mug",
    "026_sponge",
    "029_plate",
    "030_fork",
    "031_spoon",
    "032_knife",
    "033_spatula",
    "035_power_drill",
    "036_wood_block",
    "037_scissors",
    "038_padlock",
    "040_large_marker",
    "042_adjustable_wrench",
    "043_phillips_screwdriver",
    "044_flat_screwdriver",
    "048_hammer",
    "050_medium_clamp",
    "051_large_clamp",
    "052_extra_large_clamp",
    "053_mini_soccer_ball",
    "054_softball",
    "055_baseball",
    "056_tennis_ball",
    "057_racquetball",
    "058_golf_ball",
    "061_foam_brick",
    "062_dice",
    "063-a_marbles",
    "065-a_cups",
    "077_rubiks_cube",
)


class ManiSkillYCBSource:
    """:class:`AssetSource` for ManiSkill-registered YCB objects."""

    source_name: str = "ycb"

    def __init__(self, object_ids: tuple[str, ...] = YCB_FETCH_GRASPABLE) -> None:
        self._object_ids = tuple(object_ids)

    # ── enumeration ────────────────────────────────────────────────────

    def list_scene_ids(self, *, split: str | None = None) -> tuple[str, ...]:
        return ()

    def list_object_ids(self, *, split: str | None = None) -> tuple[str, ...]:
        if split is not None:
            raise ValueError(
                f"ManiSkillYCBSource: split={split!r} not supported; "
                f"pass split=None"
            )
        return self._object_ids

    # ── lookup ─────────────────────────────────────────────────────────

    def get_scene(
        self, scene_id: str, *, request: AssetRequest | None = None
    ) -> SceneSpec:
        raise NotImplementedError("ManiSkillYCBSource is object-only")

    def get_object(
        self,
        object_id: str,
        *,
        name: str,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
        fix_base: bool = False,
        request: AssetRequest | None = None,
    ) -> ObjectSpec:
        if object_id not in self._object_ids:
            raise KeyError(
                f"ManiSkillYCBSource: unknown object_id {object_id!r}. "
                f"Known ids: {self._object_ids!r}"
            )
        return ObjectSpec(
            name=name,
            builtin_id=f"ycb:{object_id}",
            position=position,
            orientation_xyzw=orientation_xyzw,
            scale=scale,
            fix_base=fix_base,
            is_articulated=False,
        )

    # ── sampling ───────────────────────────────────────────────────────

    def sample_scene_id(self, *, seed: int, split: str | None = None) -> str:
        raise NotImplementedError("ManiSkillYCBSource is object-only")

    def sample_object_id(self, *, seed: int, split: str | None = None) -> str:
        if split is not None:
            raise ValueError(
                f"ManiSkillYCBSource.sample_object_id: split={split!r} not supported"
            )
        if not self._object_ids:
            raise RuntimeError(
                "ManiSkillYCBSource.sample_object_id: object pool is empty"
            )
        # random.Random is deterministic per-seed and re-hydratable;
        # using it (instead of numpy) keeps the sampler self-contained
        # without pulling heavy deps into the default pixi env.
        rng = random.Random(int(seed))
        return rng.choice(self._object_ids)


__all__ = ["ManiSkillYCBSource", "YCB_FETCH_GRASPABLE"]
