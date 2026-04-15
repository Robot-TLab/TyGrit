"""Per-source ManiSkill scene-builder delegate factories.

Extracted from :mod:`TyGrit.worlds.backends.maniskill` on 2026-04-15
(rule 5 split — the parent file was 824 lines bundling the
:class:`SpecBackedSceneBuilder` orchestration class with these
six-source delegate selectors).

The orchestration class consumes :func:`make_delegate` to build the
right ManiSkill SceneBuilder subclass for a SceneSpec source, and
:func:`translate_specs_to_delegate_idxs` to map TyGrit scene_ids to
the delegate's internal build-config indices.

Each per-source helper imports its sim-specific dependency lazily so
this module stays importable even when only ReplicaCAD is installed.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mani_skill.utils.scene_builder import SceneBuilder
from mani_skill.utils.scene_builder.replicacad import ReplicaCADSceneBuilder

from TyGrit.types.worlds import SceneSpec


def make_delegate(
    source: str,
    env: Any,
    robot_init_qpos_noise: float,
    specs: Sequence[SceneSpec],
) -> SceneBuilder:
    """Instantiate the ManiSkill scene builder for a SceneSpec ``source``.

    ManiSkill ships different builder classes per dataset:

    * **replicacad** → :class:`ReplicaCADSceneBuilder` with
      ``include_staging_scenes=True`` to unlock all 90 apartments.
    * **procthor / ithor / robothor / architecthor** → the four AI2THOR
      variants from :mod:`mani_skill.utils.scene_builder.ai2thor.variants`.
      They all share the :class:`AI2THORBaseSceneBuilder` base class and
      differ only by ``scene_dataset`` class attribute, so we dispatch
      via a lookup dict.
    * **holodeck** → :class:`~TyGrit.worlds.backends._maniskill_holodeck.HolodeckSceneBuilder`,
      which wraps AllenAI's :class:`MjcfSceneLoader` to load Holodeck
      MJCFs into a Sapien scene. Construction needs the per-spec
      ``background_mjcf`` paths up front because the delegate does not
      enumerate scenes from any shipped manifest — the caller's spec
      pool *is* the build-config list.

    AI2THOR / RoboCasa / Holodeck imports are deferred to this
    function so the module stays importable when only ReplicaCAD is
    installed.
    """
    if source == "replicacad":
        return ReplicaCADSceneBuilder(
            env,
            robot_init_qpos_noise=robot_init_qpos_noise,
            include_staging_scenes=True,
        )

    if source in {"procthor", "ithor", "robothor", "architecthor"}:
        from mani_skill.utils.scene_builder.ai2thor.variants import (
            ArchitecTHORSceneBuilder,
            ProcTHORSceneBuilder,
            RoboTHORSceneBuilder,
            iTHORSceneBuilder,
        )

        variant_cls: dict[str, type[SceneBuilder]] = {
            "procthor": ProcTHORSceneBuilder,
            "ithor": iTHORSceneBuilder,
            "robothor": RoboTHORSceneBuilder,
            "architecthor": ArchitecTHORSceneBuilder,
        }
        return variant_cls[source](env, robot_init_qpos_noise=robot_init_qpos_noise)

    if source == "robocasa":
        from mani_skill.utils.scene_builder.robocasa.scene_builder import (
            RoboCasaSceneBuilder,
        )

        return RoboCasaSceneBuilder(env, robot_init_qpos_noise=robot_init_qpos_noise)

    if source == "holodeck":
        from TyGrit.worlds.backends._maniskill_holodeck import (
            HolodeckSceneBuilder,
        )

        # Holodeck SceneSpecs always populate background_mjcf — that's
        # what TyGrit.worlds.generators.holodeck emits. Surface a
        # specific error rather than letting None reach Path() and
        # raising deep inside MjcfSceneLoader.
        mjcf_paths: list[str] = []
        for spec in specs:
            if spec.background_mjcf is None:
                raise ValueError(
                    f"make_delegate(holodeck): spec {spec.scene_id!r} has "
                    f"background_mjcf=None. Holodeck specs must point at the "
                    f"on-disk MJCF symlink under "
                    f"assets/molmospaces/mjcf/scenes/<source>/."
                )
            mjcf_paths.append(spec.background_mjcf)
        return HolodeckSceneBuilder(
            env,
            mjcf_paths=mjcf_paths,
            robot_init_qpos_noise=robot_init_qpos_noise,
        )

    # _SUPPORTED_SOURCES is checked upstream in set_specs, so hitting
    # this branch means the frozenset and this dispatch got out of sync
    # — a programming error, not a data error.
    raise ValueError(
        f"make_delegate: internal dispatch missing for source {source!r}; "
        f"update make_delegate to match _SUPPORTED_SOURCES"
    )


def translate_specs_to_delegate_idxs(
    source: str,
    specs: Sequence[SceneSpec],
    delegate: SceneBuilder,
) -> list[int]:
    """Map each spec's scene_id to an index in ``delegate.build_configs``.

    Different sources expose their scenes differently:

    * **ReplicaCAD and AI2THOR variants** publish ``build_configs``
      as a flat list where each entry's stem encodes a unique scene
      id (``apt_0``, ``ProcTHOR-Train-293`` …). A stem-to-index map
      built via :func:`_build_stem_map` handles both.
    * **RoboCasa** does NOT publish a ``build_configs`` list at all —
      its scenes are generated combinatorially from 10 layouts × 12
      styles (index = ``layout * 12 + style``, total 120). Spec
      scene_ids encode the layout/style names explicitly, e.g.
      ``"robocasa/one_wall_small__industrial"``, and the translator
      parses them via :func:`_robocasa_scene_id_to_idx` without
      touching the delegate.
    * **Holodeck** uses an identity mapping. The delegate
      (:class:`~TyGrit.worlds.backends._maniskill_holodeck.HolodeckSceneBuilder`)
      stores the per-spec MJCF paths on its ``build_configs`` in the
      same order as the spec pool, so ``spec_idx == delegate_idx``.

    Raises
    ------
    ValueError
        If any scene_id doesn't resolve to a valid delegate index.
    """
    if source == "robocasa":
        return [_robocasa_scene_id_to_idx(s.scene_id) for s in specs]

    if source == "holodeck":
        return list(range(len(specs)))

    # All other supported sources (replicacad, procthor, ithor,
    # robothor, architecthor) use the stem-map approach because their
    # delegates populate build_configs at __init__ time.
    stem_to_idx = _build_stem_map(delegate)
    out: list[int] = []
    for spec in specs:
        local_id = spec.scene_id.split("/", 1)[-1]
        if local_id not in stem_to_idx:
            raise ValueError(
                f"SpecBackedSceneBuilder: scene {spec.scene_id!r} maps "
                f"to local id {local_id!r} which is not in the {source} "
                f"build_configs ({len(stem_to_idx)} scenes available)"
            )
        out.append(stem_to_idx[local_id])
    return out


def _robocasa_scene_id_to_idx(scene_id: str) -> int:
    """Parse a RoboCasa scene_id into a ``build_config_idx``.

    RoboCasa scenes are addressed as ``<layout_name>__<style_name>``
    with names drawn from
    :class:`mani_skill.utils.scene_builder.robocasa.utils.scene_registry.LayoutType`
    and ``StyleType`` (both ``IntEnum``). The expected format is::

        "robocasa/<layout_name>__<style_name>"

    where ``layout_name`` is a lowercase enum member (e.g.
    ``one_wall_small``, ``l_shaped_large``) and ``style_name`` is
    likewise (``industrial``, ``modern_1``, …). The index is
    computed as ``layout_value * 12 + style_value`` to match
    :meth:`RoboCasaSceneBuilder.build`'s internal decoding::

        layout_idx = build_config_idx // 12
        style_idx  = build_config_idx %  12

    Separating with ``__`` (two underscores) rather than a single
    one avoids collisions with enum names that themselves contain
    underscores (``l_shaped_small``, ``modern_1`` …).

    Raises
    ------
    ValueError
        If the format is unrecognised or either name isn't a valid
        ``LayoutType``/``StyleType`` member.
    """
    from mani_skill.utils.scene_builder.robocasa.utils.scene_registry import (
        LayoutType,
        StyleType,
    )

    local = scene_id.split("/", 1)[-1]
    if "__" not in local:
        raise ValueError(
            f"SpecBackedSceneBuilder: RoboCasa scene_id {scene_id!r} "
            f"must be formatted '<layout>__<style>' with a double-"
            f"underscore separator, e.g. 'one_wall_small__industrial'"
        )
    layout_name, _, style_name = local.partition("__")

    try:
        layout_val = LayoutType[layout_name.upper()].value
    except KeyError as exc:
        valid = sorted(m.name.lower() for m in LayoutType if m.value >= 0)
        raise ValueError(
            f"SpecBackedSceneBuilder: RoboCasa layout {layout_name!r} "
            f"in {scene_id!r} is not a valid LayoutType. Valid layouts: "
            f"{valid}"
        ) from exc

    try:
        style_val = StyleType[style_name.upper()].value
    except KeyError as exc:
        valid = sorted(m.name.lower() for m in StyleType if m.value >= 0)
        raise ValueError(
            f"SpecBackedSceneBuilder: RoboCasa style {style_name!r} "
            f"in {scene_id!r} is not a valid StyleType. Valid styles: "
            f"{valid}"
        ) from exc

    return layout_val * 12 + style_val


def _build_stem_map(delegate: SceneBuilder) -> dict[str, int]:
    """Key the delegate's ``build_configs`` by stripped scene stem.

    Handles two shapes of ``build_configs`` entries:

    * **ReplicaCAD**: plain strings like ``"apt_0.scene_instance.json"``
      — stem is ``"apt_0"``.
    * **AI2THOR variants**: :class:`AI2BuildConfig` dataclasses with a
      ``config_file`` string field like
      ``"./2/ProcTHOR-Train-293.scene_instance.json"`` — stem is
      ``"ProcTHOR-Train-293"``.

    Both strip everything after the first ``.`` on the path stem, which
    drops ``scene_instance`` and any further suffixes so the resulting
    key is stable regardless of ManiSkill's internal file naming.
    """
    out: dict[str, int] = {}
    for idx, entry in enumerate(delegate.build_configs):
        if isinstance(entry, str):
            config_file = entry
        else:
            # Duck-type on AI2BuildConfig.config_file — avoids an import
            # of ai2thor.constants from a module that may be called
            # before AI2THOR support is wired.
            config_file = getattr(entry, "config_file", None)
            if not isinstance(config_file, str):
                raise TypeError(
                    f"SpecBackedSceneBuilder: unexpected build_config "
                    f"entry type {type(entry).__name__} at index {idx}; "
                    f"expected str or AI2BuildConfig-like with a "
                    f"'config_file: str' field"
                )
        stem = Path(config_file).stem.split(".")[0]
        out[stem] = idx
    return out


__all__ = ["make_delegate", "translate_specs_to_delegate_idxs"]
