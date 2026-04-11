"""ManiSkill adapter for TyGrit :class:`SceneSpec` entries.

Implements :class:`SpecBackedSceneBuilder`, a subclass of ManiSkill's
``SceneBuilder`` that consumes a list of TyGrit ``SceneSpec``\\ s and
delegates actual geometry loading to the shipped builder appropriate to
each spec's ``source`` field. For Step 4 the only supported source is
``"replicacad"``; HSSD, RoboCasa, and custom file-based sources follow
in later steps.

Deterministic scene selection
-----------------------------
The v1 grasp_anywhere benchmark hit a "repeating scene" bug because the
caller reused one fixed ``seed`` across every ``env.reset()``, and
ManiSkill's :meth:`SceneBuilder.sample_build_config_idxs` then drew the
same ``torch.randint`` result every time — so the same apartment
loaded repeatedly. :class:`SpecBackedSceneBuilder` eliminates that trap
in two ways:

1. It **raises** from :meth:`sample_build_config_idxs`. Callers must
   pass explicit per-env indices through
   ``env.reset(options=dict(reconfigure=True, build_config_idxs=[...]))``.
2. :class:`~TyGrit.worlds.sampler.SceneSampler` (Step 3) is the
   canonical producer of those indices, and its deterministic derivation
   from ``(base_seed, env_idx, reset_count)`` guarantees successive
   resets never reuse a single integer seed.

Usage
-----
Typical setup goes through :func:`bind_specs` because ManiSkill's
``gym.make(scene_builder_cls=...)`` wants a class, not an instance::

    from TyGrit.worlds import load_manifest
    from TyGrit.worlds.backends.maniskill import bind_specs, build_world

    specs = load_manifest("resources/worlds/replicacad.json")
    env = gym.make(
        "SceneManipulation-v1",
        scene_builder_cls=bind_specs(specs),
        build_config_idxs=[0],
        num_envs=1,
    )
    built = build_world(env, specs, per_env_scene_idxs=[0])
    # built.navigable_positions is a list of trimesh objects, one per
    # parallel env, sourced from the underlying ReplicaCAD navmesh.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mani_skill.utils.scene_builder import SceneBuilder
from mani_skill.utils.scene_builder.replicacad import ReplicaCADSceneBuilder

from TyGrit.types.worlds import BuiltWorld, SceneSpec

#: Source values this adapter knows how to dispatch. Each entry maps to
#: a ManiSkill shipped scene builder class that loads its own metadata
#: and asset data. Add new entries here when a new ManiSkill-native
#: scene source is wired up (HSSD, RoboCasa in Step 10, Genesis in
#: Step 12 has its own backend at TyGrit/worlds/backends/genesis.py).
_SUPPORTED_SOURCES = frozenset(
    {"replicacad", "procthor", "ithor", "robothor", "architecthor"}
)


class SpecBackedSceneBuilder(SceneBuilder):
    """A :class:`SceneBuilder` driven by TyGrit :class:`SceneSpec` entries.

    The builder holds a tuple of ``SceneSpec``\\ s (its ``build_configs``)
    and translates per-env indices into them to indices in an internal
    delegate's build_configs. For each spec whose ``source`` is
    ``"replicacad"`` the delegate is a
    :class:`~mani_skill.utils.scene_builder.replicacad.ReplicaCADSceneBuilder`
    instantiated with ``include_staging_scenes=True`` so all 90 apts
    (6 main + 84 staging) are reachable.

    Instances should be constructed via :func:`bind_specs` (which bakes
    the spec list into a closure subclass) so that
    ``gym.make(scene_builder_cls=bound_cls)`` "just works". Direct
    construction is supported for tests and interactive use via
    :meth:`set_specs`.
    """

    #: Class-level spec list set by :func:`bind_specs`. Subclasses that
    #: override this are picked up in :meth:`__init__`.
    _cls_specs: tuple[SceneSpec, ...] = ()

    def __init__(self, env: Any, robot_init_qpos_noise: float = 0.02) -> None:
        super().__init__(env, robot_init_qpos_noise=robot_init_qpos_noise)
        self._delegate: SceneBuilder | None = None
        # Per-spec index → delegate build_config index. Populated by
        # set_specs; empty until then.
        self._spec_to_delegate_idxs: tuple[int, ...] = ()
        # Copy the class-level default so instance-level mutations via
        # set_specs don't shadow the class attribute in confusing ways.
        self.build_configs = ()
        if self._cls_specs:
            self.set_specs(self._cls_specs)

    # ─────────────────────── configuration ───────────────────────

    def set_specs(self, specs: Sequence[SceneSpec]) -> None:
        """Configure the scene pool and build an internal delegate.

        ``specs`` is converted to a tuple and stored as
        :attr:`build_configs` — ManiSkill's :meth:`SceneBuilder.build_configs`
        hook. An internal delegate (currently always
        :class:`ReplicaCADSceneBuilder`) is created so that
        :meth:`build` can forward calls after translating indices.

        Raises
        ------
        ValueError
            If ``specs`` contains a mix of sources (not supported yet),
            if the source isn't one of :data:`_SUPPORTED_SOURCES`, or
            if any spec's ``scene_id`` does not resolve to an entry in
            the delegate's own ``build_configs`` list.
        """
        specs_tuple: tuple[SceneSpec, ...] = tuple(specs)
        if not specs_tuple:
            # Empty pool is legal only during the transient gap between
            # __init__ and the first real set_specs() call; we reset
            # state rather than raising so callers can reconfigure.
            self._delegate = None
            self._spec_to_delegate_idxs = ()
            self.build_configs = ()
            return

        sources = {spec.source for spec in specs_tuple}
        if len(sources) > 1:
            raise ValueError(
                f"SpecBackedSceneBuilder: mixed scene sources not yet "
                f"supported; got {sorted(sources)}"
            )
        (source,) = sources
        if source not in _SUPPORTED_SOURCES:
            raise ValueError(
                f"SpecBackedSceneBuilder: source {source!r} not yet "
                f"supported (have {sorted(_SUPPORTED_SOURCES)})"
            )

        delegate = _make_delegate(source, self.env, self.robot_init_qpos_noise)
        stem_to_delegate_idx = _build_stem_map(delegate)

        spec_to_delegate: list[int] = []
        for spec in specs_tuple:
            # scene_id is conventionally "<source>/<local_id>". Strip
            # the source prefix; fall back to the whole id if no "/"
            # separator is present (defensive — manifests should always
            # be qualified).
            local_id = spec.scene_id.split("/", 1)[-1]
            if local_id not in stem_to_delegate_idx:
                raise ValueError(
                    f"SpecBackedSceneBuilder: scene {spec.scene_id!r} "
                    f"maps to local id {local_id!r} which is not in the "
                    f"ReplicaCAD build_configs "
                    f"({len(stem_to_delegate_idx)} scenes including staging)"
                )
            spec_to_delegate.append(stem_to_delegate_idx[local_id])

        self._delegate = delegate
        self._spec_to_delegate_idxs = tuple(spec_to_delegate)
        self.build_configs = specs_tuple

    # ─────────────────────── SceneBuilder API ───────────────────────

    def build(self, build_config_idxs: int | list[int] | None = None) -> None:
        """Translate per-env spec indices and delegate to the inner builder.

        ``build_config_idxs`` is interpreted as indices into
        :attr:`build_configs` (i.e. into the spec list), *not* indices
        into the delegate's config list. The translation happens here
        via :attr:`_spec_to_delegate_idxs`.

        Raises
        ------
        RuntimeError
            If no specs have been configured yet (delegate is ``None``).
            This catches the footgun of instantiating without
            :func:`bind_specs` and forgetting to call :meth:`set_specs`.
        ValueError
            If the argument length does not match ``env.num_envs``.
        """
        if self._delegate is None:
            raise RuntimeError(
                "SpecBackedSceneBuilder.build: no specs configured. "
                "Construct the builder via bind_specs(specs) or call "
                "set_specs(specs) before the first reset."
            )

        # Normalise to an explicit list[int] up-front so the rest of
        # the method has a single clean type to work with. None is the
        # "implicit sampling" case we refuse; int is ManiSkill's
        # broadcast-to-all-envs convenience; list is the normal path.
        if build_config_idxs is None:
            raise ValueError(
                "SpecBackedSceneBuilder.build: build_config_idxs is None. "
                "Pass explicit per-env indices via env.reset(options=dict("
                "reconfigure=True, build_config_idxs=[...])) — implicit "
                "sampling is disabled to avoid the v1 repeating-scene bug."
            )
        if isinstance(build_config_idxs, int):
            idxs: list[int] = [build_config_idxs] * self.env.num_envs
        else:
            idxs = list(build_config_idxs)

        if len(idxs) != self.env.num_envs:
            raise ValueError(
                f"SpecBackedSceneBuilder.build: expected "
                f"{self.env.num_envs} indices (one per parallel env), "
                f"got {len(idxs)}"
            )

        # Translate spec indices → delegate indices through the map we
        # precomputed in set_specs. A spec index is an int in
        # [0, len(build_configs)); the map maps it to the corresponding
        # entry in delegate.build_configs.
        delegate_idxs = [self._spec_to_delegate_idxs[spec_idx] for spec_idx in idxs]
        self._delegate.build(delegate_idxs)

        # Mirror the delegate's scene inventories so callers using the
        # standard SceneBuilder interface (env wrappers, debug tools)
        # still see spawned actors.
        self.scene_objects = self._delegate.scene_objects
        self.movable_objects = self._delegate.movable_objects
        self.articulations = self._delegate.articulations

    def initialize(
        self,
        env_idx: Any,
        init_config_idxs: list[int] | None = None,
    ) -> None:
        """Delegate :meth:`SceneBuilder.initialize` to the inner builder.

        ReplicaCAD's ``initialize`` resets object poses to their defaults
        and teleports the Fetch robot to a safe spawn. We simply forward.
        ``init_config_idxs`` is accepted for API compatibility with the
        base :class:`SceneBuilder` signature (ManiSkill's env calls
        ``initialize(env_idx, init_config_idxs)`` when ``init_configs``
        is set), but we don't use it because ReplicaCAD's builder
        doesn't take it either.
        """
        del init_config_idxs  # unused — see docstring
        if self._delegate is None:
            raise RuntimeError("SpecBackedSceneBuilder.initialize: no specs configured")
        self._delegate.initialize(env_idx)

    def sample_build_config_idxs(self) -> list[int]:
        """Disabled — implicit sampling is the v1 trap we're avoiding.

        ManiSkill's :class:`SceneManipulationEnv` calls this method from
        :meth:`_load_scene` only when ``self.build_config_idxs is None``
        at reconfigure time. There are two situations in which that
        happens, and both have clean fixes:

        1. *Constructor-time auto-reset.* ``gym.make`` → env ``__init__``
           → ``BaseEnv.__init__`` calls ``env.reset(reconfigure=True)``
           before the user's code runs. Fix: pass
           ``build_config_idxs=[i]`` as a kwarg to ``gym.make`` so
           ``SceneManipulationEnv.__init__`` stashes it up front.
        2. *User-driven reset without options.* A ``reset()`` that
           reconfigures but forgets to pass ``build_config_idxs``. Fix:
           pass ``options=dict(reconfigure=True, build_config_idxs=[...])``
           — that's what :func:`build_world` does, and what Step 5's
           Fetch env wrapper routes from
           :class:`~TyGrit.worlds.sampler.SceneSampler`.

        Raises
        ------
        RuntimeError
            Always. See the two fixes above.
        """
        raise RuntimeError(
            "SpecBackedSceneBuilder.sample_build_config_idxs: implicit "
            "sampling is disabled to avoid the grasp_anywhere v1 "
            "'repeating scene' bug. Fix: pass build_config_idxs=[...] "
            "as a kwarg to gym.make() (for the constructor-time reset), "
            "AND pass options=dict(reconfigure=True, build_config_idxs=[...]) "
            "to every subsequent env.reset() — or use TyGrit.worlds.maniskill."
            "build_world(), which routes indices from a SceneSampler."
        )

    @property
    def navigable_positions(self) -> Any:
        """Per-env navmesh data, delegated to the inner builder.

        For ReplicaCAD this is a list of :class:`trimesh.Trimesh` objects
        (one per parallel env), loaded from each scene's
        ``*.fetch.navigable_positions.obj`` file. Returns ``None`` if
        the delegate hasn't been built yet.
        """
        if self._delegate is None:
            return None
        return self._delegate.navigable_positions

    @property
    def builds_lighting(self) -> bool:
        """Mirror the delegate's ``builds_lighting`` flag.

        ReplicaCAD provides its own lighting via
        :attr:`ReplicaCADSceneBuilder.builds_lighting`, so we must
        return ``True`` for ManiSkill to skip its default lights.
        """
        if self._delegate is None:
            return False
        return self._delegate.builds_lighting


# ─────────────────────── source → delegate dispatch ─────────────────────


def _make_delegate(
    source: str,
    env: Any,
    robot_init_qpos_noise: float,
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

    AI2THOR imports are deferred to this function so the module stays
    importable when only ReplicaCAD is installed — ManiSkill loads the
    AI2THOR metadata JSONs from its own package directory at import
    time of the variants module, but those JSONs ship with the package
    so the import itself is safe. The asset files themselves are
    downloaded separately.
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

    # _SUPPORTED_SOURCES is checked upstream in set_specs, so hitting
    # this branch means the frozenset and this dispatch got out of sync
    # — a programming error, not a data error.
    raise ValueError(
        f"_make_delegate: internal dispatch missing for source {source!r}; "
        f"update _make_delegate to match _SUPPORTED_SOURCES"
    )


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


def bind_specs(
    specs: Sequence[SceneSpec],
) -> type[SpecBackedSceneBuilder]:
    """Return a :class:`SpecBackedSceneBuilder` subclass with ``specs`` baked in.

    ``gym.make(scene_builder_cls=...)`` expects a class and instantiates
    it as ``cls(env)``. This factory produces a closure subclass whose
    ``_cls_specs`` class attribute is the fixed spec list, so the
    standard ManiSkill flow wires everything up automatically — no
    need to reach into the env after construction to set specs.

    Parameters
    ----------
    specs
        The scene pool. Must be non-empty and share a single ``source``.

    Returns
    -------
    type[SpecBackedSceneBuilder]
        A new subclass, not registered anywhere. Safe to create many of
        these (one per unique spec list) without polluting ManiSkill's
        scene-builder registry.
    """
    bound_specs: tuple[SceneSpec, ...] = tuple(specs)

    class _Bound(SpecBackedSceneBuilder):
        _cls_specs = bound_specs

    _Bound.__name__ = "SpecBackedSceneBuilder_bound"
    _Bound.__qualname__ = _Bound.__name__
    return _Bound


def build_world(
    env: Any,
    specs: Sequence[SceneSpec],
    per_env_scene_idxs: Sequence[int],
    *,
    seed: int | list[int] | None = None,
) -> BuiltWorld:
    """Reconfigure ``env`` for the given per-env scene selection and return a :class:`BuiltWorld`.

    This is the integration entry point used by the Fetch env wrapper:
    given an existing ManiSkill env whose scene builder is a
    :class:`SpecBackedSceneBuilder`, and a per-env list of scene
    indices chosen by the :class:`~TyGrit.worlds.sampler.SceneSampler`,
    trigger a reconfiguration that loads the corresponding geometry.

    Parameters
    ----------
    env
        A gym env whose ``env.unwrapped.scene_builder`` is a
        :class:`SpecBackedSceneBuilder` — typically created via
        ``gym.make("SceneManipulation-v1", scene_builder_cls=bind_specs(specs))``.
    specs
        The full scene pool (same list that was bound via
        :func:`bind_specs`). Stored on the returned :class:`BuiltWorld`
        for caller inspection.
    per_env_scene_idxs
        One integer per parallel env, indexing into ``specs``.
    seed
        Optional ``seed`` argument forwarded to ``env.reset``. Use this
        to keep the ManiSkill ``_episode_rng`` state aligned with the
        outer (RL loop) seed schedule while still passing explicit
        scene indices. ``None`` lets ManiSkill pick its own episode
        seed.

    Returns
    -------
    BuiltWorld
        Wrapper carrying the representative :class:`SceneSpec`
        (``specs[per_env_scene_idxs[0]]``), the delegate's
        ``navigable_positions``, and an (initially empty) ``object_handles``
        map that Step 5 will populate.

    Raises
    ------
    TypeError
        If ``env.unwrapped.scene_builder`` is not a
        :class:`SpecBackedSceneBuilder`. Surfaces early if the caller
        forgot to pass ``scene_builder_cls=bind_specs(...)`` to
        ``gym.make``.
    """
    scene_builder = env.unwrapped.scene_builder
    if not isinstance(scene_builder, SpecBackedSceneBuilder):
        raise TypeError(
            f"build_world: env.scene_builder must be a "
            f"SpecBackedSceneBuilder, got {type(scene_builder).__name__}"
        )

    idxs = list(per_env_scene_idxs)
    # env.reset(options=dict(reconfigure=True, build_config_idxs=...))
    # is the documented entry point from SceneManipulationEnv.reset —
    # it stashes the indices on self.build_config_idxs and triggers
    # _reconfigure → _load_scene → scene_builder.build(idxs). See
    # mani_skill/envs/scenes/base_env.py (reset + _load_scene).
    env.reset(
        seed=seed,
        options={"reconfigure": True, "build_config_idxs": idxs},
    )

    return BuiltWorld(
        spec=tuple(specs)[idxs[0]],
        navigable_positions=scene_builder.navigable_positions,
        object_handles={},
    )
