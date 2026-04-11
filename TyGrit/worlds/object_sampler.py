"""Seeded per-reset sampling of :class:`ObjectSpec` from a manifest.

Parallel to :class:`~TyGrit.worlds.sampler.SceneSampler` but for object
pools. An :class:`ObjectSampler` owns a deterministic draw from a
filtered list of ObjectSpecs and derives an independent seed for every
``(env_idx, reset_count)`` pair via
:class:`numpy.random.SeedSequence`, so it inherits the same contract:

* Same ``(base_seed, env_idx, reset_count)`` → same object (and same
  k-subset when sampling multiple objects at once).
* Successive ``reset()`` calls on the same env get DIFFERENT objects
  even when the caller passes the same base seed.
* Parallel env workers with distinct ``env_idx`` draw from independent
  streams.

Typical use — a training loop that needs one target + two distractors
per reset::

    cfg = ObjectSamplerConfig(manifest_path="resources/worlds/objects/ycb.json")
    sampler = create_object_sampler(cfg)
    target, *distractors = sampler.sample_k(
        env_idx=0, reset_count=step, k=3,
    )
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from TyGrit.types.worlds import ObjectSamplerConfig, ObjectSpec
from TyGrit.worlds.manifest import load_object_manifest


class ObjectSampler:
    """Deterministic, per-reset ObjectSpec sampler.

    Holds an immutable filtered pool of objects and derives a fresh
    seed per draw via :class:`numpy.random.SeedSequence`. No mutable
    state — safe to share across parallel workers. API mirrors
    :class:`SceneSampler` for symmetry.
    """

    def __init__(
        self,
        config: ObjectSamplerConfig,
        objects: tuple[ObjectSpec, ...],
    ) -> None:
        """Construct a sampler from a config and a pre-loaded object pool.

        Parameters
        ----------
        config
            Sampler config. ``base_seed`` controls determinism;
            ``object_names`` optionally filters the pool to a subset.
        objects
            The objects to sample from — typically the output of
            :func:`TyGrit.worlds.manifest.load_object_manifest`. Must
            be non-empty after any ``object_names`` filter is applied.

        Raises
        ------
        ValueError
            If ``objects`` is empty, if ``config.object_names``
            references any name not present in ``objects``, or if the
            filter leaves no objects to sample from.
        """
        if not objects:
            raise ValueError("ObjectSampler: object pool is empty — cannot sample")

        if config.object_names is None:
            filtered: tuple[ObjectSpec, ...] = objects
        else:
            available = {o.name: o for o in objects}
            # Detecting duplicate names in the INPUT pool is important —
            # a dict collapses them silently, and the sampler would then
            # draw from a pool smaller than the caller expected.
            if len(available) != len(objects):
                seen: set[str] = set()
                dups: list[str] = []
                for o in objects:
                    if o.name in seen:
                        dups.append(o.name)
                    else:
                        seen.add(o.name)
                raise ValueError(
                    f"ObjectSampler: object pool has duplicate names "
                    f"{sorted(set(dups))}"
                )
            missing = [n for n in config.object_names if n not in available]
            if missing:
                raise ValueError(
                    f"ObjectSampler: config.object_names references unknown "
                    f"objects {missing!r}; available: {sorted(available)}"
                )
            # Preserve the order the caller gave so filter semantics are
            # reproducible across runs regardless of pool ordering.
            filtered = tuple(available[n] for n in config.object_names)
            if not filtered:
                raise ValueError("ObjectSampler: object_names filter left zero objects")

        self._objects: tuple[ObjectSpec, ...] = filtered
        self._base_seed: int = config.base_seed

    # ─────────────────────── basic properties ───────────────────────

    @property
    def object_count(self) -> int:
        """Number of objects in the (filtered) sampling pool."""
        return len(self._objects)

    @property
    def base_seed(self) -> int:
        """The root seed used for per-reset derivation."""
        return self._base_seed

    @property
    def objects(self) -> tuple[ObjectSpec, ...]:
        """The filtered object pool this sampler draws from.

        Callers that need to share the SAME pool with a downstream
        component — e.g. a task layer that spawns actors from the
        same list this sampler drew from — should pass this tuple
        rather than reloading the manifest, so indices returned by
        :meth:`sample_idx` line up with the consumer's view.
        """
        return self._objects

    # ─────────────────────── single-object sampling ───────────────────────

    def sample_idx(self, env_idx: int, reset_count: int) -> int:
        """Return the index of a single sampled object in the pool."""
        rng = self._rng_for(env_idx, reset_count)
        return int(rng.integers(0, len(self._objects)))

    def sample(self, env_idx: int, reset_count: int) -> ObjectSpec:
        """Pick a single object for a given (env_idx, reset_count)."""
        return self._objects[self.sample_idx(env_idx, reset_count)]

    # ─────────────────────── k-object sampling ───────────────────────

    def sample_k_idxs(
        self,
        env_idx: int,
        reset_count: int,
        k: int,
        *,
        replace: bool = False,
    ) -> tuple[int, ...]:
        """Return k object indices for a given (env_idx, reset_count).

        Parameters
        ----------
        env_idx, reset_count
            Routed through :class:`numpy.random.SeedSequence` the same
            way as :meth:`sample_idx`, so the same arguments always
            produce the same k-subset.
        k
            Number of indices to draw.
        replace
            If False (default), draw without replacement — ``k`` must
            be <= :attr:`object_count`. If True, allow duplicates,
            which lets the caller request more than ``object_count``
            samples.

        Returns
        -------
        tuple[int, ...]
            Length-``k`` tuple of indices into :attr:`objects`.

        Raises
        ------
        ValueError
            If ``k`` is negative, or if ``replace=False`` and
            ``k > object_count`` (no valid way to satisfy the request).
        """
        if k < 0:
            raise ValueError(f"ObjectSampler.sample_k_idxs: k must be >= 0, got {k}")
        if k == 0:
            return ()
        if not replace and k > len(self._objects):
            raise ValueError(
                f"ObjectSampler.sample_k_idxs: k={k} exceeds pool size "
                f"{len(self._objects)} and replace=False"
            )
        rng = self._rng_for(env_idx, reset_count)
        picks = rng.choice(len(self._objects), size=k, replace=replace)
        return tuple(int(i) for i in picks)

    def sample_k(
        self,
        env_idx: int,
        reset_count: int,
        k: int,
        *,
        replace: bool = False,
    ) -> tuple[ObjectSpec, ...]:
        """Return k ObjectSpecs — see :meth:`sample_k_idxs` for the contract."""
        idxs = self.sample_k_idxs(env_idx, reset_count, k, replace=replace)
        return tuple(self._objects[i] for i in idxs)

    # ─────────────────────── internals ───────────────────────

    def _rng_for(self, env_idx: int, reset_count: int) -> np.random.Generator:
        """Derive an independent numpy Generator for (env_idx, reset_count).

        Uses the same :class:`SeedSequence` pattern as
        :class:`SceneSampler` so the two samplers produce statistically
        independent but reproducible streams for the same (base_seed,
        env_idx, reset_count) inputs — the 128-bit entropy pool makes
        collisions between "scene stream" and "object stream" for the
        same base_seed effectively impossible.
        """
        seq = np.random.SeedSequence([self._base_seed, int(env_idx), int(reset_count)])
        return np.random.default_rng(seq)


def create_object_sampler(config: ObjectSamplerConfig) -> ObjectSampler:
    """Load an object manifest and return an :class:`ObjectSampler`.

    Convenience wrapper combining
    :func:`TyGrit.worlds.manifest.load_object_manifest` with the
    :class:`ObjectSampler` constructor. Use this when your caller has
    a path, not a pre-loaded object tuple.

    Raises
    ------
    FileNotFoundError
        If ``config.manifest_path`` does not exist. Propagated from
        :func:`load_object_manifest`.
    ValueError
        If the manifest is malformed, or the sampler cannot construct
        (empty pool, unknown names filter).
    """
    objects = load_object_manifest(Path(config.manifest_path))
    return ObjectSampler(config, objects)
