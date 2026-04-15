"""The :class:`AssetSource` Protocol.

Each dataset gets one implementation. Sources produce canonical
:class:`~TyGrit.types.worlds.SceneSpec` / :class:`~TyGrit.types.worlds.ObjectSpec`
values for sim-agnostic consumption by :mod:`TyGrit.sim` handlers.

A source is responsible for:

* **Enumeration.** :meth:`list_scene_ids` / :meth:`list_object_ids`
  return the full catalogue (optionally filtered by ``split``).
* **Lookup.** :meth:`get_scene` / :meth:`get_object` turn an id into a
  frozen ``SceneSpec`` / ``ObjectSpec``. This is where dataset-specific
  quirks live — scale normalisation, orientation fixes, convex-hull
  proxy paths, per-asset bbox metadata.
* **Sampling.** :meth:`sample_scene_id` is a deterministic seeded
  selector over the catalogue. Callers use this from a scene sampler
  or task generator; sources can add dataset-specific filters (e.g.
  "only kitchen scenes" for RoboCasa) without changing the generic
  sampler logic.

Sources must **not** import simulator SDKs — they produce pure data.
Simulator-specific resolution (VHACD mesh-proxy generation for
Genesis / Isaac, asset-registry lookups for ManiSkill) lives in
:mod:`TyGrit.sim.<sim>` and reads :class:`ObjectSpec.builtin_id` /
``asset_path_for(...)`` to pick the native format.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from TyGrit.types.worlds import ObjectSpec, SceneSpec

#: Preferred asset format a caller can request. Sources try to produce
#: the requested format; they fall back (documented per-source) when
#: the dataset doesn't ship that format.
AssetFormat = Literal["builtin", "urdf", "usd", "mjcf", "mesh"]


@dataclass(frozen=True)
class AssetRequest:
    """Caller preferences for lookup / sampling.

    Parameters
    ----------
    split
        Optional dataset split filter (``"train"`` / ``"val"`` / …).
        Meaning is per-dataset; sources that don't support splits
        either ignore this or raise :class:`ValueError` on unknown
        splits.
    seed
        Seed for deterministic random choices inside :meth:`get_scene`
        / :meth:`get_object` (e.g. RoboCasa's procedural kitchen
        assembly). ``None`` lets the source pick its own default seed.
    preferred_format
        Preferred asset format; see :data:`AssetFormat`. Sources fall
        back to the closest available format.
    ensure_local
        If True, the source may block to download / cache dataset
        files before returning. If False, the source returns an
        ``AssetSpec`` whose paths may point at non-existent files and
        the caller is responsible for ensuring they exist.
    """

    split: str | None = None
    seed: int | None = None
    preferred_format: AssetFormat | None = None
    ensure_local: bool = True


@runtime_checkable
class AssetSource(Protocol):
    """One implementation per asset dataset."""

    @property
    def source_name(self) -> str:
        """Stable TyGrit identifier (``"ycb"``, ``"replicacad"`` …).

        Matches :attr:`ObjectSpec.builtin_id` / :attr:`SceneSpec.source`
        prefixes so callers can round-trip: find the source that
        produced a spec by looking it up in
        :func:`TyGrit.worlds.asset_sources.get_source`.
        """
        ...

    # ── enumeration ────────────────────────────────────────────────────

    def list_scene_ids(self, *, split: str | None = None) -> tuple[str, ...]:
        """Return every scene id the source can provide.

        Object-only sources (YCB, Objaverse) return an empty tuple.
        """
        ...

    def list_object_ids(self, *, split: str | None = None) -> tuple[str, ...]:
        """Return every object id the source can provide.

        Scene-only sources return an empty tuple.
        """
        ...

    # ── lookup ─────────────────────────────────────────────────────────

    def get_scene(
        self, scene_id: str, *, request: AssetRequest | None = None
    ) -> SceneSpec:
        """Resolve ``scene_id`` to a frozen :class:`SceneSpec`.

        Raises :class:`KeyError` if ``scene_id`` is not in
        :meth:`list_scene_ids`. Raises :class:`NotImplementedError` on
        object-only sources.
        """
        ...

    def get_object(
        self,
        object_id: str,
        *,
        name: str,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
        fix_base: bool = False,
        request: AssetRequest | None = None,
    ) -> ObjectSpec:
        """Resolve ``object_id`` to a frozen :class:`ObjectSpec`.

        ``name`` is the :attr:`ObjectSpec.name` of the resulting spec
        (needed because the same object can be placed multiple times
        in one scene with different names). ``position`` / ``scale``
        / ``fix_base`` / ``orientation_xyzw`` are placement parameters.

        Raises :class:`KeyError` if ``object_id`` is not in
        :meth:`list_object_ids`. Raises :class:`NotImplementedError`
        on scene-only sources.
        """
        ...

    # ── sampling ───────────────────────────────────────────────────────

    def sample_scene_id(self, *, seed: int, split: str | None = None) -> str:
        """Deterministically pick one scene id for ``seed``.

        Object-only sources raise :class:`NotImplementedError`.
        """
        ...

    def sample_object_id(self, *, seed: int, split: str | None = None) -> str:
        """Deterministically pick one object id for ``seed``.

        Scene-only sources raise :class:`NotImplementedError`.
        """
        ...


#: Per-source × per-sim compatibility matrix.
#:
#: Caller-facing answer to "which simulators can load this dataset?".
#: The set values are the canonical ``sim_name`` identifiers used in
#: :attr:`RobotCfg.sim_uids` and :mod:`TyGrit.sim` module names.
#:
#: Keep in lockstep with the ``_SUPPORTED_SOURCES`` frozensets in
#: :mod:`TyGrit.worlds.backends.maniskill` and
#: :mod:`TyGrit.worlds.backends.genesis`. The matrix here is the
#: source of truth; the per-backend frozensets become a runtime
#: assertion that the matrix wasn't lied to.
SOURCE_SIM_COMPATIBILITY: dict[str, frozenset[str]] = {
    # YCB: object-only, ManiSkill registry-only (Genesis / Isaac Sim
    # would need the asset bundle materialised first).
    "ycb": frozenset({"maniskill"}),
    # ReplicaCAD + AI2THOR variants: Habitat-schema, work in
    # ManiSkill via shipped builders and in Genesis via the
    # _genesis_habitat parser. Isaac Sim parity needs pre-converted
    # USDs of the per-scene Objaverse mesh pool — out of scope today.
    "replicacad": frozenset({"maniskill", "genesis"}),
    "procthor": frozenset({"maniskill", "genesis"}),
    "ithor": frozenset({"maniskill", "genesis"}),
    "robothor": frozenset({"maniskill", "genesis"}),
    "architecthor": frozenset({"maniskill", "genesis"}),
    # RoboCasa: procedural assembler is ManiSkill-internal; Genesis /
    # Isaac Sim would each need the assembler ported (large
    # workstream).
    "robocasa": frozenset({"maniskill"}),
    # Holodeck: MJCF — ManiSkill + Genesis load it natively. Isaac
    # Sim loads it via MjcfConverterCfg → USD round-trip (see
    # TyGrit.worlds.backends.isaac_sim).
    "holodeck": frozenset({"maniskill", "genesis", "isaac_sim"}),
    # Objaverse: raw mesh files. ManiSkill + Genesis spawn via
    # mesh_path; Isaac Sim spawns via MeshConverterCfg.
    "objaverse": frozenset({"maniskill", "genesis", "isaac_sim"}),
}


def compatible_sims(source_name: str) -> frozenset[str]:
    """Return the set of sim identifiers that can load ``source_name``.

    Useful when wiring an env from a SceneSpec without prior knowledge
    of which simulator the spec was generated for. Raises
    :class:`KeyError` for unregistered sources — call sites should
    register custom sources via :func:`register_source` and update
    :data:`SOURCE_SIM_COMPATIBILITY` together.
    """
    if source_name not in SOURCE_SIM_COMPATIBILITY:
        raise KeyError(
            f"compatible_sims: unknown source_name {source_name!r}. Known: "
            f"{sorted(SOURCE_SIM_COMPATIBILITY)!r}"
        )
    return SOURCE_SIM_COMPATIBILITY[source_name]


__all__ = [
    "AssetFormat",
    "AssetRequest",
    "AssetSource",
    "SOURCE_SIM_COMPATIBILITY",
    "compatible_sims",
]
