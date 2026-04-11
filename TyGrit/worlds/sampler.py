"""Seeded per-reset sampling of :class:`SceneSpec` from a manifest.

A :class:`SceneSampler` owns a deterministic random draw from a filtered
list of SceneSpecs. It derives an independent seed for every
``(env_idx, reset_count)`` pair via :class:`numpy.random.SeedSequence`,
so:

* Two calls with the same ``(base_seed, env_idx, reset_count)`` always
  pick the same scene (reproducibility).
* Successive ``reset()`` calls on the same env get DIFFERENT scenes —
  even when the RL loop passes the same base seed — because
  ``reset_count`` participates in seed derivation. This is the specific
  fix for the grasp_anywhere v1 "repeating scene" bug, where the same
  integer seed was reused across every task loop iteration and
  ``torch.randint`` inside ManiSkill's ``sample_build_config_idxs``
  always returned the same apartment index.
* Parallel env workers with distinct ``env_idx`` draw from independent
  streams, so scenes are diverse across the batch without any global
  RNG coordination.

Typical use::

    cfg = SceneSamplerConfig(manifest_path="resources/worlds/replicacad.json")
    sampler = create_sampler(cfg)
    scene = sampler.sample(env_idx=3, reset_count=0)

Or, if you already have the SceneSpecs loaded::

    sampler = SceneSampler(cfg, scenes)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from TyGrit.types.worlds import SceneSamplerConfig, SceneSpec
from TyGrit.worlds.manifest import load_manifest


class SceneSampler:
    """Deterministic, per-reset SceneSpec sampler.

    The sampler holds an immutable filtered list of scenes and derives a
    fresh seed for every sample via :class:`numpy.random.SeedSequence`.
    It has no internal mutable state — calling :meth:`sample` twice with
    the same arguments always returns the same scene — which makes it
    safe to share across parallel workers.
    """

    def __init__(
        self,
        config: SceneSamplerConfig,
        scenes: tuple[SceneSpec, ...],
    ) -> None:
        """Construct a sampler from a config and a pre-loaded scene pool.

        Parameters
        ----------
        config
            Sampler config. ``base_seed`` controls determinism;
            ``scene_ids`` optionally filters the pool to a subset.
        scenes
            The scenes to sample from — typically the output of
            :func:`TyGrit.worlds.load_manifest`. Must be non-empty after
            any ``scene_ids`` filter is applied.

        Raises
        ------
        ValueError
            If ``scenes`` is empty, if ``config.scene_ids`` references
            any id not present in ``scenes``, or if the filter leaves no
            scenes to sample from.
        """
        if not scenes:
            raise ValueError("SceneSampler: scene pool is empty — cannot sample")

        if config.scene_ids is None:
            filtered: tuple[SceneSpec, ...] = scenes
        else:
            available = {s.scene_id: s for s in scenes}
            missing = [sid for sid in config.scene_ids if sid not in available]
            if missing:
                raise ValueError(
                    f"SceneSampler: config.scene_ids references unknown "
                    f"scene_ids {missing!r}; available: {sorted(available)}"
                )
            # Preserve the order given in config.scene_ids so the caller
            # controls the sampling order when it matters (debugging).
            filtered = tuple(available[sid] for sid in config.scene_ids)
            if not filtered:
                raise ValueError("SceneSampler: scene_ids filter left zero scenes")

        self._scenes: tuple[SceneSpec, ...] = filtered
        self._base_seed: int = config.base_seed

    @property
    def scene_count(self) -> int:
        """Number of scenes in the (filtered) sampling pool."""
        return len(self._scenes)

    @property
    def base_seed(self) -> int:
        """The root seed used for per-reset derivation."""
        return self._base_seed

    @property
    def scenes(self) -> tuple[SceneSpec, ...]:
        """The filtered scene pool this sampler draws from.

        Callers that need to share the SAME pool with a downstream
        component (e.g. a ManiSkill scene builder created via
        :func:`TyGrit.worlds.backends.maniskill.bind_specs`) should
        pass this tuple rather than reloading the manifest — otherwise
        indices returned by :meth:`sample_idx` won't line up with the
        consumer's internal index space.
        """
        return self._scenes

    def sample_idx(self, env_idx: int, reset_count: int) -> int:
        """Pick the **index** of a scene for a given (env_idx, reset_count).

        Prefer this over :meth:`sample` when you already hold the
        scene pool and want the integer index directly — callers that
        forward indices to ManiSkill's ``build_config_idxs`` path use
        this one. See :meth:`sample` for the deterministic mapping
        contract.
        """
        # numpy.random.SeedSequence accepts an int or a sequence of ints
        # and deterministically mixes them into a 128-bit entropy pool.
        # It's explicitly designed for "derive independent seeds from a
        # root" workflows like this one, and is stable across numpy
        # versions per numpy's API compat policy.
        seq = np.random.SeedSequence([self._base_seed, int(env_idx), int(reset_count)])
        rng = np.random.default_rng(seq)
        return int(rng.integers(0, len(self._scenes)))

    def sample(self, env_idx: int, reset_count: int) -> SceneSpec:
        """Pick a scene for a given (env_idx, reset_count) pair.

        The mapping is deterministic and independent across workers.
        Two calls with the same arguments always return the same scene;
        two calls with any argument different (even just ``reset_count``
        incrementing) get independent draws.

        Parameters
        ----------
        env_idx
            Parallel env worker index. Different workers sample from
            independent streams so their scenes are unrelated.
        reset_count
            Monotonic reset counter for this env — 0 at first reset,
            incrementing by one per ``env.reset()``. This is the piece
            that prevents the v1 "repeating scene" bug: reusing the
            same integer across resets is no longer possible because
            the counter participates in seed derivation.

        Returns
        -------
        SceneSpec
            One of the scenes in the filtered pool.
        """
        return self._scenes[self.sample_idx(env_idx, reset_count)]


def create_sampler(config: SceneSamplerConfig) -> SceneSampler:
    """Load a manifest and return a :class:`SceneSampler` ready to draw.

    Convenience wrapper that combines
    :func:`TyGrit.worlds.manifest.load_manifest` with the
    :class:`SceneSampler` constructor. Use this when your caller has a
    path, not a pre-loaded scene tuple.

    Raises
    ------
    FileNotFoundError
        If ``config.manifest_path`` does not exist. Surfaces from
        :func:`load_manifest`.
    ValueError
        If the manifest is malformed, or the sampler cannot construct
        (empty pool, unknown scene_ids filter).
    """
    scenes = load_manifest(Path(config.manifest_path))
    return SceneSampler(config, scenes)
