"""Generate a world manifest for ManiSkill's RoboCasa scene set.

Unlike ReplicaCAD and the AI2THOR variants, RoboCasa does NOT publish
a flat list of scene files. Its scenes are synthesised combinatorially
at build time from 10 kitchen **layouts** and 12 kitchen **styles**
(see ``LayoutType`` / ``StyleType`` in
``mani_skill.utils.scene_builder.robocasa.utils.scene_registry``).
The total scene count is fixed at **10 × 12 = 120** regardless of the
shipped asset version; the (layout, style) pair fully determines the
kitchen YAML files ManiSkill loads during
:meth:`RoboCasaSceneBuilder.build`.

This generator enumerates all 120 combinations and emits one
:class:`~TyGrit.types.worlds.SceneSpec` per pair with::

    scene_id             = "robocasa/<layout_name>__<style_name>"
    source               = "robocasa"
    background_builtin_id = "robocasa:<layout_name>__<style_name>"

The double-underscore separator avoids collision with the underscores
inside enum names (``l_shaped_small``, ``modern_1``, …). The
:class:`~TyGrit.worlds.backends.maniskill.SpecBackedSceneBuilder`
dispatch parses this format via
:func:`~TyGrit.worlds.backends.maniskill._robocasa_scene_id_to_idx`.

Usage::

    pixi run -e world python -m TyGrit.worlds.generators.robocasa

Requires the ``world`` pixi env (imports
``mani_skill.utils.scene_builder.robocasa.utils.scene_registry`` to
enumerate the enums — the enum definitions are in the Python package,
no asset download needed to *generate* the manifest). Actually
*loading* a RoboCasa scene at runtime needs the bulk assets, which
is a separate download::

    pixi run -e world download-robocasa
"""

from __future__ import annotations

from pathlib import Path

from TyGrit.types.worlds import SceneSpec
from TyGrit.worlds.manifest import save_manifest

#: Default output path. 120 entries, small plain JSON, no gzip needed.
DEFAULT_OUTPUT: Path = Path("resources/worlds/robocasa.json")


def list_robocasa_pairs() -> list[tuple[str, str]]:
    """Return every ``(layout_name, style_name)`` pair RoboCasa knows about.

    Reads ``LayoutType`` and ``StyleType`` enums from ManiSkill's
    ``scene_registry`` module. Skips enum members with negative values
    because those are logical groups (``ALL``, ``NO_ISLAND`` …), not
    actual scene configurations.
    """
    from mani_skill.utils.scene_builder.robocasa.utils.scene_registry import (
        LayoutType,
        StyleType,
    )

    layouts = [m for m in LayoutType if m.value >= 0]
    styles = [m for m in StyleType if m.value >= 0]
    return [
        (layout.name.lower(), style.name.lower())
        for layout in layouts
        for style in styles
    ]


def build_robocasa_manifest() -> tuple[SceneSpec, ...]:
    """Construct one SceneSpec per (layout, style) pair.

    Order matches the 120-config layout ManiSkill's
    :meth:`RoboCasaSceneBuilder.build` expects: outer loop on layout
    (``layout_idx * 12 + style_idx``), so iterating ``build_config_idx``
    from 0 upward gives the same sequence as iterating the manifest
    in file order.
    """
    pairs = list_robocasa_pairs()
    return tuple(
        SceneSpec(
            scene_id=f"robocasa/{layout}__{style}",
            source="robocasa",
            background_builtin_id=f"robocasa:{layout}__{style}",
        )
        for layout, style in pairs
    )


def main(output: Path = DEFAULT_OUTPUT) -> None:
    """Write the manifest to ``output`` and print a summary."""
    specs = build_robocasa_manifest()
    save_manifest(
        output,
        specs,
        source="robocasa",
        generator="TyGrit.worlds.generators.robocasa",
    )
    print(f"wrote {len(specs)} scenes to {output}")


if __name__ == "__main__":
    main()
