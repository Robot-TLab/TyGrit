"""Genesis physics-engine adapter for TyGrit :class:`SceneSpec` entries.

Where :mod:`TyGrit.worlds.backends.maniskill` plugs SceneSpecs into
ManiSkill3 via its shipped ``SceneBuilder`` subclasses, this module
builds a :class:`genesis.Scene` directly because Genesis has no
``SceneBuilder`` abstraction — you call ``scene.add_entity(morph)``
once per object and then ``scene.build()`` to finalise.

Scope (what this module supports)
---------------------------------

* **source="holodeck"** — loads the MJCF pointed at by
  :attr:`SceneSpec.background_mjcf` via :class:`gs.morphs.MJCF`.
  Genesis natively parses MuJoCo XML, so Holodeck scenes drop in
  without any extra translation layer (unlike the ManiSkill side,
  which needed AllenAI's standalone ``MjcfSceneLoader``).
* **Per-spec :class:`ObjectSpec` entries** — ``mesh_path`` goes
  through :class:`gs.morphs.Mesh`; ``urdf_path`` goes through
  :class:`gs.morphs.URDF`. ``builtin_id`` prefixes (``ycb:…``)
  raise :class:`NotImplementedError` because Genesis doesn't ship
  the YCB asset bundle — a caller wanting YCB under Genesis has
  to download the assets and hand the file paths to a pre-existing
  spec via ``mesh_path``.

Deliberately NOT supported
--------------------------

ReplicaCAD, AI2THOR (iTHOR / ProcTHOR / RoboTHOR / ArchitecTHOR),
and RoboCasa are ManiSkill-shipped datasets loaded by their own
``SceneBuilder`` subclasses inside the ManiSkill repo. Porting
them to Genesis is out of scope — each has its own asset format
and scene-instance JSON schema. Specs with those sources raise
:class:`NotImplementedError`.

Single-build constraint
-----------------------

Genesis calls :meth:`Scene.build` exactly once; entities cannot be
added or removed afterward. That means scene-switch-on-reset (the
deterministic per-reset scene sampling in
:class:`~TyGrit.worlds.sampler.SceneSampler`) requires constructing
a fresh :class:`genesis.Scene`, which the env-wrapper layer
(:class:`~TyGrit.envs.fetch.genesis.GenesisFetchSimBackend`) does
by destroying and recreating on each :meth:`reset_to_idx`.

Performance note
----------------

Genesis's default path runs CoACD convex decomposition on every
mesh referenced by an MJCF/URDF so it can generate accurate
collision shapes. Holodeck scenes reference ~50-100 Objaverse
meshes, each taking seconds-to-minutes to decompose on CPU → full
loads can take hours. :func:`add_spec_to_scene` passes
``convexify=False`` on the Holodeck MJCF so the visual geometry is
used as-is and the decomposition pass is skipped. This is adequate
for our use case (free-space navigation / coarse grasp planning
driven by a separate graspable-object pool) but produces less
accurate collision for the background mesh. If tighter collisions
are needed later, pre-decompose offline and plumb the output through
:class:`gs.morphs.CoacdOptions`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from TyGrit.types.worlds import ObjectSpec, SceneSpec

if TYPE_CHECKING:
    import genesis as gs

#: Sources this backend can load. Other sources (replicacad, ithor,
#: procthor, robothor, architecthor, robocasa) are ManiSkill-only and
#: raise :class:`NotImplementedError` when encountered.
_SUPPORTED_SOURCES = frozenset({"holodeck"})


def add_spec_to_scene(scene: "gs.Scene", spec: SceneSpec) -> None:
    """Populate ``scene`` from a :class:`SceneSpec`.

    Adds the background scene geometry (if any) and every
    :class:`ObjectSpec` in ``spec.objects``. Must be called *before*
    :meth:`gs.Scene.build`.

    Raises
    ------
    NotImplementedError
        If ``spec.source`` is not in :data:`_SUPPORTED_SOURCES`, or
        if an object uses a ``builtin_id`` (no YCB bridge yet), or
        if the spec uses neither a recognised background nor lists
        any objects.
    ValueError
        If a Holodeck spec has ``background_mjcf=None``.
    """
    import genesis as gs

    if spec.source not in _SUPPORTED_SOURCES:
        raise NotImplementedError(
            f"add_spec_to_scene: source {spec.source!r} not supported by the "
            f"Genesis backend. Supported sources: {sorted(_SUPPORTED_SOURCES)}. "
            f"ReplicaCAD / AI2THOR variants / RoboCasa are ManiSkill-only — "
            f"port them if Genesis parity is needed."
        )

    if spec.source == "holodeck":
        if spec.background_mjcf is None:
            raise ValueError(
                f"add_spec_to_scene(holodeck): spec {spec.scene_id!r} has "
                f"background_mjcf=None. Holodeck specs must point at the MJCF "
                f"file (typically under assets/molmospaces/mjcf/scenes/...)."
            )
        scene.add_entity(
            gs.morphs.MJCF(
                file=spec.background_mjcf,
                # Skip CoACD decomposition of the Holodeck mesh pool —
                # see module docstring's *Performance note*. Using the
                # visual mesh as-is for collision is adequate for our
                # use case; pre-decompose offline if you need better.
                convexify=False,
            ),
            name=f"scene__{spec.scene_id}",
        )

    for obj in spec.objects:
        _add_object_to_scene(scene, obj)


def _add_object_to_scene(scene: "gs.Scene", obj: ObjectSpec) -> None:
    """Add a single :class:`ObjectSpec` to ``scene``.

    Dispatch:

    * ``mesh_path`` → :class:`gs.morphs.Mesh`
    * ``urdf_path`` → :class:`gs.morphs.URDF`
    * ``builtin_id`` → :class:`NotImplementedError` — Genesis has no
      YCB asset bundle; callers wanting YCB under Genesis must
      resolve builtin ids to file paths themselves.

    Pose and scale are passed through. The quaternion is swapped
    from TyGrit's ``[x, y, z, w]`` convention to Genesis's
    ``[w, x, y, z]`` convention (which follows Sapien / MuJoCo).
    """
    import genesis as gs

    # TyGrit convention: xyzw. Genesis / Sapien / MuJoCo: wxyz.
    x, y, z, w = obj.orientation_xyzw
    quat = (w, x, y, z)
    pos = tuple(obj.position)

    if obj.mesh_path is not None:
        morph = gs.morphs.Mesh(
            file=obj.mesh_path,
            pos=pos,
            quat=quat,
            scale=tuple(obj.scale),
            fixed=obj.fix_base,
        )
    elif obj.urdf_path is not None:
        morph = gs.morphs.URDF(
            file=obj.urdf_path,
            pos=pos,
            quat=quat,
            scale=obj.scale[0],  # URDF takes scalar scale, not per-axis
            fixed=obj.fix_base,
        )
    elif obj.builtin_id is not None:
        raise NotImplementedError(
            f"_add_object_to_scene: builtin_id {obj.builtin_id!r} (object "
            f"{obj.name!r}) is not wired for Genesis. YCB and other built-in "
            f"asset bundles are ManiSkill-only; resolve to an explicit "
            f"mesh_path / urdf_path if Genesis parity is needed."
        )
    else:
        raise NotImplementedError(
            f"_add_object_to_scene: object {obj.name!r} has none of "
            f"mesh_path / urdf_path / builtin_id set. Nothing to load."
        )

    scene.add_entity(morph, name=f"obj__{obj.name}")


def build_scene_for_spec(
    spec: SceneSpec,
    *,
    show_viewer: bool = False,
) -> "gs.Scene":
    """Convenience: construct a fresh :class:`genesis.Scene`, populate it
    from ``spec``, and :meth:`build` it.

    Returns the built scene ready for :meth:`step`. Most callers (e.g.
    :class:`~TyGrit.envs.fetch.genesis.GenesisFetchSimBackend`) will
    instead hand-roll the :class:`genesis.Scene` with a Fetch robot
    added before calling :func:`add_spec_to_scene` + :meth:`build`,
    but this wrapper is useful for smoke-testing scene loading in
    isolation.
    """
    import genesis as gs

    scene = gs.Scene(show_viewer=show_viewer)
    add_spec_to_scene(scene, spec)
    scene.build()
    return scene
