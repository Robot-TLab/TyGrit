"""ManiSkill SceneBuilder wrapper around AllenAI's MjcfSceneLoader.

For Holodeck (MolmoSpaces) scenes, neither sapien nor mani_skill ship
a native MJCF loader. We delegate to ``MjcfSceneLoader`` from
``molmo_spaces_maniskill.assets.loader`` (vendored as a git submodule
under ``thirdparty/molmospaces``; install via
``pixi run -e world install-molmo-spaces-maniskill``).

This module's :class:`HolodeckSceneBuilder` adapts that loader to the
mani_skill ``SceneBuilder`` interface so the central
:class:`~TyGrit.worlds.backends.maniskill.SpecBackedSceneBuilder` can
dispatch ``source="holodeck"`` SceneSpecs through it.

Limitations
-----------
* **Per-env MJCF variation is not supported yet.** ManiSkill GPU sim
  exposes ``ManiSkillScene.create_actor_builder()`` which by default
  spawns into every parallel sub-scene. ``MjcfSceneLoader`` calls
  ``builder.build()`` without a ``set_scene_idxs([env_idx])`` restriction,
  so loading one MJCF spawns its actors into every parallel env. We
  raise :class:`NotImplementedError` if ``build_config_idxs`` contains
  more than one unique value, instead of silently producing a wrong
  scene. Workaround: have the :class:`~TyGrit.worlds.sampler.SceneSampler`
  return a single index replicated across all envs (or use
  ``num_envs=1``).
* **No navmesh.** Holodeck MJCFs don't ship navmeshes;
  :attr:`navigable_positions` returns ``None``. The Fetch env wrapper
  has to spawn the robot via a different mechanism (e.g. a fixed
  start pose, or by reading the per-scene metadata JSON that
  Holodeck does ship at ``train_<N>_metadata.json``).
* ``builds_lighting = True`` because ``MjcfSceneLoader`` reads
  ``<light type="directional">`` entries from the MJCF; ManiSkill
  must skip its default lights so we don't double-light.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mani_skill.utils.scene_builder import SceneBuilder


class HolodeckSceneBuilder(SceneBuilder):
    """``SceneBuilder`` that loads Holodeck MJCFs via ``MjcfSceneLoader``.

    Constructed with the env plus a fixed sequence of MJCF paths
    (one per :class:`~TyGrit.types.worlds.SceneSpec` in the pool).
    :meth:`build` looks up the path at the requested index and hands
    it to the loader.
    """

    #: ``MjcfSceneLoader`` adds directional lights from the MJCF; tell
    #: ManiSkill to skip its own default lights.
    builds_lighting: bool = True

    def __init__(
        self,
        env: Any,
        mjcf_paths: Sequence[str],
        robot_init_qpos_noise: float = 0.02,
    ) -> None:
        super().__init__(env, robot_init_qpos_noise=robot_init_qpos_noise)
        self.build_configs = tuple(mjcf_paths)
        self.scene_objects: dict[str, Any] = {}
        self.movable_objects: dict[str, Any] = {}
        self.articulations: dict[str, Any] = {}

    # ─────────────────────── SceneBuilder API ───────────────────────

    def build(self, build_config_idxs: list[int]) -> None:
        """Load the MJCF at the requested index into the env's scene.

        ``build_config_idxs`` is a length-``num_envs`` list of indices
        into :attr:`build_configs`. See module-level *Limitations*
        for why every entry must currently be the same value.
        """
        unique_idxs = set(build_config_idxs)
        if len(unique_idxs) > 1:
            raise NotImplementedError(
                f"HolodeckSceneBuilder.build: per-env MJCF variation is "
                f"not yet supported (got idxs {build_config_idxs}). "
                f"MjcfSceneLoader has no scene_idxs hook, so loading "
                f"different MJCFs across parallel envs would spawn "
                f"every scene's actors into every subscene. Workaround: "
                f"have SceneSampler return a single index replicated "
                f"across all envs, or use num_envs=1."
            )

        # Deferred import keeps this module importable in envs that
        # don't have molmo_spaces_maniskill installed (e.g. the default
        # pixi env). The world env's setup.sh installs it via
        # `pixi run -e world install-molmo-spaces-maniskill`.
        from molmo_spaces_maniskill.assets.loader import MjcfSceneLoader

        mjcf_path = Path(self.build_configs[build_config_idxs[0]])
        loader = MjcfSceneLoader(self.scene)
        actors, articulations = loader.load(mjcf_path)

        # All Holodeck actors are static — the loader assigns
        # body_type="dynamic" only when a body has a free joint, and
        # Holodeck MJCFs only emit __STRUCTURAL_*__ and
        # __ARTICULABLE_DYNAMIC_MJT__ classes (no free joints at the
        # scene root). So movable_objects stays empty.
        self.scene_objects = dict(actors)
        self.articulations = dict(articulations)
        self.movable_objects = {}

    def initialize(self, env_idx: Any) -> None:
        """No-op: Holodeck scenes have no per-episode randomisation.

        The MJCF places every actor at its default pose during
        :meth:`build`. There's nothing to reset between episodes (no
        free-jointed bodies, no per-episode init configs).
        """
        del env_idx  # unused — see docstring

    @property
    def navigable_positions(self) -> Any:
        """Holodeck MJCFs don't ship navmeshes — returns ``None``.

        See module-level *Limitations*. The Fetch env wrapper has to
        provide its own spawn position when the active scene is a
        Holodeck one.
        """
        return None
