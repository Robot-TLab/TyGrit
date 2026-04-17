"""Generate a world manifest for all 90 ReplicaCAD apartments.

Reads the canonical list of ReplicaCAD scene configs from ManiSkill's
shipped :class:`ReplicaCADSceneBuilder` (with ``include_staging_scenes=
True`` so both the 6 main apts and the 84 staging scenes are included)
and emits one :class:`~TyGrit.types.worlds.SceneSpec` per stem, wrapped
into a JSON manifest at ``resources/worlds/replicacad.json``.

The previous hand-crafted baseline listed only the 6 main apts; this
generator replaces it with the full 90, unlocking the staging pool
that was referenced in the v1 grasp_anywhere bug analysis (v1 never
enabled staging, so the sampling pool was only 6 instead of 90).

Usage::

    pixi run -e world python -m TyGrit.worlds.generators.replicacad

Requires the ``world`` pixi env because the generator imports
``mani_skill.utils.scene_builder.replicacad.ReplicaCADSceneBuilder``
to enumerate scene configs via ManiSkill's public API rather than
reaching into shipped metadata files directly.
"""

from __future__ import annotations

from pathlib import Path

from TyGrit.types.worlds import SceneSpec
from TyGrit.worlds.manifest import save_manifest

#: Default output path, relative to the repo root.
DEFAULT_OUTPUT: Path = Path("resources/worlds/replicacad.json")


class _FakeEnv:
    """Minimal stand-in so ``ReplicaCADSceneBuilder.__init__`` accepts us.

    The builder constructor only stores ``env`` and reads metadata
    files; it does not touch ``env.num_envs`` or ``env.scene`` until
    ``build()`` is called, which the generator never does.
    """

    num_envs: int = 1


def list_replicacad_stems(include_staging: bool = True) -> list[str]:
    """Return every ReplicaCAD scene stem ManiSkill knows about.

    The stems are extracted from :attr:`ReplicaCADSceneBuilder.build_configs`,
    which ManiSkill populates from its shipped
    ``metadata/scene_configs.json``. With ``include_staging=True`` the
    list is 6 main apts (``apt_0`` … ``apt_5``) plus 84 staging scenes
    (``v3_sc0_staging_00`` … ``v3_sc3_staging_20``) for 90 total.
    """
    from mani_skill.utils.scene_builder.replicacad import ReplicaCADSceneBuilder

    builder = ReplicaCADSceneBuilder(
        _FakeEnv(),
        include_staging_scenes=include_staging,
    )
    # build_configs is a list of filenames like "apt_0.scene_instance.json".
    # Strip the extension to match our scene_id convention.
    return [name.split(".")[0] for name in builder.build_configs]


def build_replicacad_manifest(
    include_staging: bool = True,
) -> tuple[SceneSpec, ...]:
    """Construct SceneSpecs for every ReplicaCAD scene.

    Each spec has:

    * ``scene_id`` = ``"replicacad/<stem>"``
    * ``source`` = ``"replicacad"``
    * ``background_builtin_id`` = ``"replicacad:<stem>"`` — the
      :class:`SpecBackedSceneBuilder` dispatches on this prefix to
      delegate to ManiSkill's shipped ``ReplicaCADSceneBuilder``.
    * ``objects`` = empty — ManiSkill's builder spawns ReplicaCAD's
      shipped furniture automatically.
    """
    stems = list_replicacad_stems(include_staging=include_staging)
    return tuple(
        SceneSpec(
            scene_id=f"replicacad/{stem}",
            source="replicacad",
            background_builtin_id=f"replicacad:{stem}",
        )
        for stem in stems
    )


def main(output: Path = DEFAULT_OUTPUT) -> None:
    """Write the manifest to ``output`` and print a summary."""
    specs = build_replicacad_manifest(include_staging=True)
    save_manifest(
        output,
        specs,
        source="replicacad",
        generator="TyGrit.worlds.generators.replicacad",
    )
    print(f"wrote {len(specs)} scenes to {output}")


if __name__ == "__main__":
    main()
