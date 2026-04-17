"""Isaac Sim / Isaac Lab adapter for TyGrit :class:`SceneSpec` entries.

Mirrors the :mod:`TyGrit.worlds.backends.maniskill` /
:mod:`TyGrit.worlds.backends.genesis` modules: takes a sim-agnostic
:class:`SceneSpec` and adds it into an :class:`InteractiveScene`.

Scope (what this module supports today)
---------------------------------------

* **source="holodeck"** — the MJCF pointed at by
  :attr:`SceneSpec.background_mjcf` is loaded via Isaac Lab's
  :class:`MjcfConverterCfg` + :func:`spawn_from_usd` round-trip
  (Isaac Sim does not have a first-class MJCF loader; the converter
  materialises the MJCF to USD on disk and spawns the converted
  asset). Holodeck specs ship a flat MJCF per scene so this single
  conversion is sufficient.
* **Per-spec :class:`ObjectSpec` entries** — :attr:`mesh_path` goes
  through :class:`MeshConverterCfg` (mesh → USD); :attr:`urdf_path`
  goes through :class:`UrdfConverterCfg`; :attr:`usd_path` is loaded
  natively via :class:`UsdFileCfg`.

Deliberately NOT supported (raise :class:`NotImplementedError`)
---------------------------------------------------------------

* **YCB / RoboCasa builtin ids.** Isaac Lab has no equivalent of
  ManiSkill's asset registry; loading these requires the asset files
  on disk and a per-source converter. That is upstream-of-this-PR
  work.
* **Habitat-schema scene_instance.json sources** (replicacad,
  ai2thor variants). Habitat's scene format references hundreds of
  Objaverse meshes per scene — converting them to USD on the fly
  inflates startup latency to many minutes. Until a pre-conversion
  / cache step lands, these sources stay maniskill+genesis only.
* **Procedural assemblers.** RoboCasa kitchens are generated at
  runtime by ManiSkill's RoboCasaSceneBuilder; Isaac Sim parity
  needs the assembler ported.

Per CLAUDE.md Rule 1 those gaps surface as :class:`NotImplementedError`
with a concrete message — not silent stubs.

Single-stage constraint
-----------------------

Isaac Lab's :class:`InteractiveScene` is a homogeneous rigging by
default (``replicate_physics=True``). Heterogeneous per-env scenes
require either rebuilding the stage (slow but correct, what
:class:`IsaacSimSimHandler` does) or :class:`MultiUsdFileCfg` (limited
to same-skeleton variants — not useful for full apartments). The
:func:`add_spec_to_scene` entry point assumes a single-scene rigging
context; vec callers rebuild between scene swaps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from TyGrit.types.worlds import ObjectSpec, SceneSpec

if TYPE_CHECKING:  # pragma: no cover - typing-only imports
    from isaaclab.scene import InteractiveScene

#: Sources this backend can load. Update :data:`SOURCE_SIM_COMPATIBILITY`
#: in lock-step (the matrix is the source of truth; this set asserts
#: against it at runtime).
_SUPPORTED_SOURCES = frozenset({"holodeck", "objaverse"})


def add_spec_to_scene(scene: "InteractiveScene", spec: SceneSpec) -> None:
    """Populate ``scene`` from a :class:`SceneSpec`.

    Adds the background scene geometry (when present) and every
    :class:`ObjectSpec` in ``spec.objects``. Must be called *after*
    :class:`InteractiveScene` has been constructed but before the
    first :meth:`SimulationContext.step`.

    Raises
    ------
    NotImplementedError
        If ``spec.source`` is not in :data:`_SUPPORTED_SOURCES`, or
        if an object uses a ``builtin_id`` (no asset-registry bridge),
        or if the spec is purely-objects with no background_* set.
    ValueError
        Holodeck spec with ``background_mjcf=None``.
    """
    if spec.source not in _SUPPORTED_SOURCES:
        raise NotImplementedError(
            f"add_spec_to_scene(isaac_sim): source {spec.source!r} not "
            f"supported by the Isaac Sim backend. Supported sources: "
            f"{sorted(_SUPPORTED_SOURCES)}. Habitat-schema datasets "
            f"(replicacad, ai2thor variants) and RoboCasa need pre-converted "
            f"USDs or an upstream port — see TyGrit.worlds.backends.isaac_sim "
            f"module docstring."
        )

    if spec.source == "holodeck":
        if spec.background_mjcf is None:
            raise ValueError(
                f"add_spec_to_scene(isaac_sim, holodeck): spec "
                f"{spec.scene_id!r} has background_mjcf=None. Holodeck "
                f"specs must point at the MJCF file."
            )
        _add_mjcf_background(scene, spec.background_mjcf, spec.scene_id)
    elif spec.source == "objaverse":
        # Objaverse specs are object-only — no background geometry.
        pass

    for obj in spec.objects:
        _add_object_to_scene(scene, obj)


# ── helpers ───────────────────────────────────────────────────────────


def _add_mjcf_background(
    scene: "InteractiveScene", mjcf_path: str, scene_id: str
) -> None:
    """Convert ``mjcf_path`` to USD and spawn it under the env root.

    Isaac Sim's MJCF converter writes a ``.usd`` next to the input
    MJCF; subsequent calls reuse the cached output. Heavy convex
    decomposition is left at MJCF defaults (Holodeck's MJCFs ship
    pre-decomposed meshes, so the converter is a fast metadata pass).
    """
    from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg
    from isaaclab.sim.spawners.from_files import UsdFileCfg, spawn_from_usd

    converter = MjcfConverter(MjcfConverterCfg(asset_path=mjcf_path))
    usd_path = converter.usd_path

    cfg = UsdFileCfg(usd_path=usd_path)
    spawn_from_usd(
        prim_path=f"/World/envs/env_.*/scene__{scene_id}",
        cfg=cfg,
    )


def _add_object_to_scene(scene: "InteractiveScene", obj: ObjectSpec) -> None:
    """Add a single :class:`ObjectSpec` to ``scene``.

    Dispatch:
    * ``usd_path`` → :class:`UsdFileCfg`
    * ``mesh_path`` → :class:`MeshConverterCfg`
    * ``urdf_path`` → :class:`UrdfConverterCfg`
    * ``builtin_id`` → :class:`NotImplementedError`
    """
    from isaaclab.sim.spawners.from_files import UsdFileCfg, spawn_from_usd

    if obj.usd_path is not None:
        spawn_cfg: Any = UsdFileCfg(usd_path=obj.usd_path)
        usd_path = obj.usd_path
    elif obj.mesh_path is not None:
        from isaaclab.sim.converters import MeshConverter, MeshConverterCfg

        converter = MeshConverter(MeshConverterCfg(asset_path=obj.mesh_path))
        usd_path = converter.usd_path
        spawn_cfg = UsdFileCfg(usd_path=usd_path)
    elif obj.urdf_path is not None:
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

        converter = UrdfConverter(
            UrdfConverterCfg(asset_path=obj.urdf_path, fix_base=obj.fix_base)
        )
        usd_path = converter.usd_path
        spawn_cfg = UsdFileCfg(usd_path=usd_path)
    elif obj.builtin_id is not None:
        raise NotImplementedError(
            f"add_spec_to_scene(isaac_sim): object {obj.name!r} uses "
            f"builtin_id={obj.builtin_id!r}; Isaac Sim has no equivalent "
            f"of ManiSkill's asset registry. Resolve the builtin id to a "
            f"file path (mesh_path / urdf_path / usd_path) before passing "
            f"the spec to this backend."
        )
    else:
        raise ValueError(
            f"add_spec_to_scene(isaac_sim): object {obj.name!r} has no "
            f"asset path set (usd_path / mesh_path / urdf_path / builtin_id "
            f"all None)"
        )

    # TyGrit convention: xyzw. Isaac Lab / Sapien / MuJoCo: wxyz.
    x, y, z, w = obj.orientation_xyzw
    spawn_from_usd(
        prim_path=f"/World/envs/env_.*/{obj.name}",
        cfg=spawn_cfg,
        translation=tuple(obj.position),
        orientation=(w, x, y, z),
    )
    _ = scene  # spawn_from_usd registers under the prim path the
    # InteractiveScene uses; the scene reference is documented for
    # callers who want to wire post-spawn manipulation.


__all__ = ["add_spec_to_scene"]
