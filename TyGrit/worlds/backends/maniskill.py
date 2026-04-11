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

from TyGrit.types.worlds import BuiltWorld, ObjectSpec, SceneSpec

#: Source values this adapter knows how to dispatch. Each entry maps to
#: a ManiSkill shipped scene builder class that loads its own metadata
#: and asset data. Add new entries here when a new ManiSkill-native
#: scene source is wired up. Genesis (Step 12) has its own backend at
#: TyGrit/worlds/backends/genesis.py.
_SUPPORTED_SOURCES = frozenset(
    {"replicacad", "procthor", "ithor", "robothor", "architecthor", "robocasa"}
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
        spec_to_delegate = _translate_specs_to_delegate_idxs(
            source, specs_tuple, delegate
        )

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
        # still see spawned actors. Copy rather than assign so our
        # per-spec object spawning below can add entries without
        # mutating the delegate's own dicts.
        #
        # Coalesce to empty dict: ManiSkill's SceneBuilder base class
        # declares these attrs as Optional[dict[...]] (default None at
        # the class level). ReplicaCAD and AI2THOR overwrite them with
        # {} during build(), but RoboCasaSceneBuilder.build() never
        # assigns them at all — so reading them still returns the base
        # class None. Both are valid per the interface; we only care
        # that our mirror is a dict callers can iterate safely.
        self.scene_objects = dict(self._delegate.scene_objects or {})
        self.movable_objects = dict(self._delegate.movable_objects or {})
        self.articulations = dict(self._delegate.articulations or {})

        # Spawn per-spec custom objects on top of the delegate's scene.
        # Heterogeneous parallel envs each draw their own SceneSpec from
        # the sampler (Step 5), and each spec may list its own objects
        # to spawn — see SceneSpec.objects. Objects are keyed by
        # (env_idx, obj.name) because two parallel envs drawing the
        # same spec spawn two independent actors, one per env_idx.
        self._spawn_per_spec_objects(idxs)

    def _spawn_per_spec_objects(self, per_env_spec_idxs: list[int]) -> None:
        """Spawn :attr:`SceneSpec.objects` for each parallel env.

        Groups envs by which spec they drew, so multiple envs sharing
        the same spec spawn each object once via ``set_scene_idxs`` —
        matching the pattern ReplicaCADSceneBuilder uses internally for
        background meshes. Dispatches each :class:`ObjectSpec` to the
        appropriate ManiSkill actor loader based on its ``builtin_id``
        prefix (currently only ``ycb:`` is wired; ``gso:`` will be
        added when we wire a GSO loader; explicit file paths from
        ``urdf_path``/``mesh_path`` raise NotImplementedError until
        their code path is written).
        """
        from collections import defaultdict

        envs_by_spec: dict[int, list[int]] = defaultdict(list)
        for env_idx, spec_idx in enumerate(per_env_spec_idxs):
            envs_by_spec[spec_idx].append(env_idx)

        for spec_idx, env_ids in envs_by_spec.items():
            spec = self.build_configs[spec_idx]
            for obj_spec in spec.objects:
                self._spawn_one_object(obj_spec, env_ids)

    def _spawn_one_object(self, obj: ObjectSpec, env_ids: list[int]) -> None:
        """Spawn a single :class:`ObjectSpec` into ``env_ids``.

        Dispatch on ``builtin_id`` prefix:

        * ``ycb:<model_id>`` → ManiSkill's
          :func:`mani_skill.utils.building.actors.ycb.get_ycb_builder`,
          which reads ``info_pick_v0.json`` and creates an actor
          builder with the right collision + visual files and scale.
        * anything else → :class:`NotImplementedError` (explicit so
          future backend additions fail loudly instead of silently
          no-op'ing).

        Secondary branch: if ``builtin_id`` is ``None`` but
        ``mesh_path`` is set, fall through to a file-based spawn via
        :meth:`scene.create_actor_builder` +
        :meth:`add_visual_from_file` + :meth:`add_convex_collision_from_file`.
        This is the path used by the Objaverse generator (
        :mod:`TyGrit.worlds.generators.objaverse`), where each entry
        points at a downloaded ``.glb`` file under
        ``assets/objaverse/meshes/``.

        Pose: ``ObjectSpec.position`` is world-frame ``(x, y, z)``
        and ``orientation_xyzw`` is a unit quaternion in our project
        convention. Sapien takes ``wxyz``, so we swap at this boundary.

        Fix-base: :meth:`ObjectSpec.fix_base` controls static vs
        dynamic spawning. Dynamic actors also land in
        :attr:`movable_objects` so code that iterates movable objects
        (e.g. the task layer) sees them alongside delegate movables.
        """
        import sapien

        if obj.builtin_id is not None:
            builder = self._make_builtin_actor_builder(obj)
        elif obj.mesh_path is not None:
            builder = self._make_mesh_path_actor_builder(obj)
        else:
            raise NotImplementedError(
                f"SpecBackedSceneBuilder: object {obj.name!r} has "
                f"neither builtin_id nor mesh_path set. URDF/USD/MJCF "
                f"file paths are not yet supported — use mesh_path "
                f"(.glb/.obj/.stl) or builtin_id instead."
            )

        # TyGrit convention: xyzw; Sapien convention: wxyz. Swap at
        # the boundary so downstream stays consistent.
        x, y, z, w = obj.orientation_xyzw
        builder.initial_pose = sapien.Pose(p=list(obj.position), q=[w, x, y, z])
        builder.set_scene_idxs(env_ids)

        # Suffix with env_ids so parallel envs spawning the same object
        # still produce distinct actor names — ManiSkill requires
        # unique names across all parallel actors.
        env_tag = ",".join(str(e) for e in env_ids)
        actor_name = f"tygrit_{obj.name}_envs[{env_tag}]"

        if obj.fix_base:
            actor = builder.build_static(name=actor_name)
        else:
            actor = builder.build(name=actor_name)

        for env_id in env_ids:
            self.scene_objects[f"env-{env_id}_{obj.name}"] = actor
            if not obj.fix_base:
                self.movable_objects[f"env-{env_id}_{obj.name}"] = actor

    def _make_builtin_actor_builder(self, obj: ObjectSpec):
        """Dispatch a ``builtin_id``-qualified ObjectSpec to a ManiSkill loader.

        Currently wired: ``ycb:<model_id>`` via
        :func:`mani_skill.utils.building.actors.ycb.get_ycb_builder`,
        which reads ``info_pick_v0.json`` for the object's bbox, scale,
        and density and constructs an actor builder with the shipped
        collision + visual files.

        Raises
        ------
        NotImplementedError
            If the ``builtin_id`` prefix is not yet supported. Add new
            branches here when wiring new per-source actor loaders.
        """
        from mani_skill.utils.building.actors.ycb import get_ycb_builder

        # obj.builtin_id is guaranteed non-None by _spawn_one_object.
        assert obj.builtin_id is not None
        prefix, _, model_id = obj.builtin_id.partition(":")
        if prefix != "ycb":
            raise NotImplementedError(
                f"SpecBackedSceneBuilder: builtin_id prefix {prefix!r} "
                f"in {obj.builtin_id!r} is not yet supported. Currently "
                f"wired: 'ycb:<model_id>'. GSO and other sources land "
                f"in follow-up steps."
            )
        return get_ycb_builder(self.scene, id=model_id)

    def _make_mesh_path_actor_builder(self, obj: ObjectSpec):
        """Build an actor from a mesh file on disk.

        Used by :class:`ObjectSpec` entries that don't have a
        ``builtin_id`` (e.g. Objaverse-curated meshes from
        :mod:`TyGrit.worlds.generators.objaverse`). We use the mesh
        itself as a single convex collision hull via
        :meth:`add_convex_collision_from_file` — simplest, works for
        most Objaverse-LVIS objects which tend toward simple shapes.
        Generators that need concave collision can pre-decompose via
        CoACD/V-HACD and store the decomposed collision alongside
        the visual mesh; we would then switch to
        ``add_multiple_convex_collisions_from_file``.
        """
        # obj.mesh_path is guaranteed non-None by _spawn_one_object.
        assert obj.mesh_path is not None
        builder = self.scene.create_actor_builder()
        # ManiSkill's per-axis scale API takes a length-3 list.
        scale_list = list(obj.scale)
        builder.add_visual_from_file(filename=obj.mesh_path, scale=scale_list)
        builder.add_convex_collision_from_file(filename=obj.mesh_path, scale=scale_list)
        return builder

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

    if source == "robocasa":
        from mani_skill.utils.scene_builder.robocasa.scene_builder import (
            RoboCasaSceneBuilder,
        )

        return RoboCasaSceneBuilder(env, robot_init_qpos_noise=robot_init_qpos_noise)

    # _SUPPORTED_SOURCES is checked upstream in set_specs, so hitting
    # this branch means the frozenset and this dispatch got out of sync
    # — a programming error, not a data error.
    raise ValueError(
        f"_make_delegate: internal dispatch missing for source {source!r}; "
        f"update _make_delegate to match _SUPPORTED_SOURCES"
    )


def _translate_specs_to_delegate_idxs(
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

    Raises
    ------
    ValueError
        If any scene_id doesn't resolve to a valid delegate index.
    """
    if source == "robocasa":
        return [_robocasa_scene_id_to_idx(s.scene_id) for s in specs]

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
