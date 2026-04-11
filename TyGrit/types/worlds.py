"""Pure data types for the world / scene layer.

A *world* is everything in the simulated environment EXCEPT the robot:
a background asset, a set of rigid or articulated objects at known
poses, an optional navmesh for mobile bases, and lighting.

These dataclasses are the canonical sim-agnostic representation. They
have **no simulator imports** and hold no runtime state. Per-sim
adapters live in :mod:`TyGrit.worlds` and translate a :class:`SceneSpec`
into the native scene representation of ManiSkill, Genesis, Isaac Lab,
etc.

Design notes
------------
* All quaternions use the project-wide ``[x, y, z, w]`` convention
  (SciPy / ROS). Sim adapters convert at the backend boundary
  (Sapien uses ``wxyz``).
* Dataclasses are frozen; collection fields use tuples — never lists or
  dicts — so instances are hashable and safe to share across parallel
  workers.
* Asset paths follow MetaSim's ``_FileBasedMixin`` pattern: one
  spec carries all of ``urdf_path`` / ``usd_path`` / ``mjcf_path`` /
  ``mesh_path``, and each sim adapter picks the variant native to its
  backend. ``builtin_id`` is a fast path for simulator-registered
  loaders such as ``"ycb:063-a_marbles"``.
* Task goals (grasp this, navigate there) live one level up in
  :mod:`TyGrit.types.tasks` and reference worlds by ``scene_id``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

# Quaternion identity in [x, y, z, w] convention.
_IDENTITY_QUAT_XYZW: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class ObjectSpec:
    """A single object placement in a world.

    Exactly one of the asset path fields must be populated: ``urdf_path``,
    ``usd_path``, ``mjcf_path``, ``mesh_path``, or ``builtin_id``. Sim
    adapters pick the variant native to their backend.

    Parameters
    ----------
    name
        Unique identifier within a :class:`SceneSpec`. Used by the task
        layer to look up target objects and by :class:`BuiltWorld` to
        key spawned actor handles.
    urdf_path, usd_path, mjcf_path, mesh_path
        Per-format asset paths. Absolute or relative to the repo root.
    builtin_id
        Alternative to a file path — references a loader registered with
        the simulator. Format: ``"<source>:<model_id>"``, e.g.
        ``"ycb:063-a_marbles"`` or ``"gso:<model_id>"``.
    position
        World-frame position ``(x, y, z)`` in metres.
    orientation_xyzw
        World-frame rotation as a unit quaternion ``(x, y, z, w)``.
    scale
        Per-axis mesh scale. ``(1.0, 1.0, 1.0)`` for no scaling.
    fix_base
        If True, the object is anchored to the world (static furniture,
        walls, floors). If False, the object has free-body dynamics.
    is_articulated
        Set True for URDFs with multiple links connected by joints
        (drawers, cabinets, faucets, articulated gripper props).
    joint_init
        Initial joint positions for articulated objects, as a tuple of
        ``(joint_name, qpos)`` pairs. Empty tuple means defaults.
    """

    name: str
    urdf_path: str | None = None
    usd_path: str | None = None
    mjcf_path: str | None = None
    mesh_path: str | None = None
    builtin_id: str | None = None
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation_xyzw: tuple[float, float, float, float] = _IDENTITY_QUAT_XYZW
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    fix_base: bool = False
    is_articulated: bool = False
    joint_init: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        asset_fields = (
            self.urdf_path,
            self.usd_path,
            self.mjcf_path,
            self.mesh_path,
            self.builtin_id,
        )
        if all(a is None for a in asset_fields):
            raise ValueError(
                f"ObjectSpec {self.name!r}: must set one of "
                "urdf_path / usd_path / mjcf_path / mesh_path / builtin_id"
            )

    def asset_path_for(self, preferred: str) -> str | None:
        """Return the asset path matching ``preferred`` format, if present.

        ``preferred`` must be one of ``"urdf"``, ``"usd"``, ``"mjcf"``, or
        ``"mesh"``. Returns ``None`` if that format is simply not populated
        on this spec (callers should then fall back to ``builtin_id`` or
        raise). Raises :class:`ValueError` if ``preferred`` is not a
        recognised format — this distinguishes a missing asset from a
        caller typo so bugs surface immediately.
        """
        table = {
            "urdf": self.urdf_path,
            "usd": self.usd_path,
            "mjcf": self.mjcf_path,
            "mesh": self.mesh_path,
        }
        if preferred not in table:
            raise ValueError(
                f"ObjectSpec.asset_path_for: unknown format {preferred!r}; "
                f"expected one of {sorted(table)}"
            )
        return table[preferred]


@dataclass(frozen=True)
class SceneSpec:
    """Full specification of a simulated world.

    A scene consists of a background asset (static environment geometry),
    a set of :class:`ObjectSpec` placements, an optional navmesh for
    mobile robots, and lighting. Scene specs are self-contained —
    manifests serialize/deserialize one ``SceneSpec`` per entry with no
    external references.

    Parameters
    ----------
    scene_id
        Unique identifier, conventionally ``"<source>/<local_id>"`` —
        for example ``"replicacad/apt_0"`` or ``"hssd/108736824"``.
    source
        Upstream dataset identifier: ``"replicacad"``, ``"hssd"``,
        ``"robocasa"``, ``"procthor"``, ``"molmospaces"``, or
        ``"custom"`` for ad-hoc scenes.
    background_urdf, background_usd, background_mjcf, background_mesh
        Background asset paths (multi-format, same pattern as
        :class:`ObjectSpec`). May all be ``None`` if
        ``background_builtin_id`` is used or if the scene is composed
        entirely of ``objects`` (legal — e.g. a tabletop-only test scene).
    background_builtin_id
        Alternative: a simulator-registered scene loader id, e.g.
        ``"replicacad:apt_0"``.
    navmesh_path
        Path to a navmesh OBJ (or ``None`` if the scene has no
        navigable regions, such as a tabletop-only scene).
    objects
        Rigid / articulated objects to spawn in the scene.
    target_object_names
        Subset of ``objects`` that can legally be targeted by a task
        goal. Used by task samplers; empty tuple means all objects
        are valid targets.
    lighting
        Optional lighting preset name or explicit path. ``None`` uses
        the simulator's default.
    """

    scene_id: str
    source: str
    background_urdf: str | None = None
    background_usd: str | None = None
    background_mjcf: str | None = None
    background_mesh: str | None = None
    background_builtin_id: str | None = None
    navmesh_path: str | None = None
    objects: tuple[ObjectSpec, ...] = ()
    target_object_names: tuple[str, ...] = ()
    lighting: str | None = None

    def __post_init__(self) -> None:
        object_names = {o.name for o in self.objects}
        if len(object_names) != len(self.objects):
            raise ValueError(f"SceneSpec {self.scene_id!r}: duplicate object names")
        unknown_targets = set(self.target_object_names) - object_names
        if unknown_targets:
            raise ValueError(
                f"SceneSpec {self.scene_id!r}: target_object_names "
                f"references unknown objects {sorted(unknown_targets)}"
            )

    def object_by_name(self, name: str) -> ObjectSpec:
        """Look up an ObjectSpec by name. Raises KeyError if missing."""
        for obj in self.objects:
            if obj.name == name:
                return obj
        raise KeyError(f"SceneSpec {self.scene_id!r}: no object {name!r}")


@dataclass(frozen=True)
class BuiltWorld:
    """Return type from a sim-specific ``build_world`` call.

    Holds references to spawned sim objects so the robot env layer can
    query them (navmesh sampling, target actor lookup) without reaching
    into simulator internals. Handles are typed as ``Any`` so this
    module can stay sim-free — concrete types are Sapien actors,
    Genesis entities, Isaac articulations, etc.

    Parameters
    ----------
    spec
        The source :class:`SceneSpec` that was built.
    navigable_positions
        A walkable-region dataset for mobile-base sampling, or ``None``.
        Concrete shape depends on the adapter — typically an ``(N, 3)``
        numpy array of world-frame points.
    object_handles
        Mapping from ``ObjectSpec.name`` to the simulator's opaque
        handle for the spawned actor.
    """

    spec: SceneSpec
    navigable_positions: Any = None
    object_handles: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SceneSamplerConfig:
    """Configuration for a scene sampler.

    Parameters
    ----------
    manifest_path
        Path to a JSON manifest emitted by a ``tools/make_manifest/``
        generator. The manifest is loaded lazily by the sampler.
    scene_ids
        Optional filter: only sample scenes whose ``scene_id`` appears
        in this tuple. ``None`` uses every scene in the manifest.
    base_seed
        Root seed for deterministic per-reset seed derivation. Actual
        per-reset seeds are derived as
        ``hash((base_seed, env_idx, reset_count))`` so successive
        ``reset()`` calls never reuse a single integer (the grasp_anywhere
        v1 "repeating scene" bug).
    """

    manifest_path: str
    scene_ids: tuple[str, ...] | None = None
    base_seed: int = 0
