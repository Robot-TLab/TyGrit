"""Habitat scene-instance loader for the Genesis backend.

``ReplicaCAD`` and the four AI2THOR variants (iTHOR / ProcTHOR /
RoboTHOR / ArchitecTHOR) all ship their scene data in Habitat's
``scene_instance.json`` schema — a flat JSON listing a background
``stage_instance``, a list of ``object_instances`` (rigid bodies),
and a list of ``articulated_object_instances`` (URDFs). ManiSkill
consumes those files through its shipped
:class:`ReplicaCADSceneBuilder` / :class:`AI2THORBaseSceneBuilder`
subclasses; on the Genesis side, there is no shipped loader, so we
parse the schema ourselves and translate each entry into a
``gs.morphs.{Mesh, URDF}`` entity.

This module is intentionally decoupled from :mod:`genesis` at import
time (imports are deferred inside functions) so a caller in a non-
genesis env can still import :mod:`TyGrit.worlds.backends.genesis`
— the Habitat helper only materialises Genesis objects when actually
called.

Dataset layout (learned from ManiSkill's reference loader)
---------------------------------------------------------

ManiSkill downloads the datasets under
``assets/maniskill/data/scene_datasets/`` (the project-local
``MS_ASSET_DIR``). The on-disk layout differs between datasets and we
mirror ManiSkill's path resolution:

* **ReplicaCAD** — ``replica_cad_dataset/``. Templates resolve as
  ``<dataset_root>/<template>.glb`` (e.g.
  ``stages/frl_apartment_stage.glb``). Articulated URDFs live under
  ``urdf/<template>/<template>.urdf`` with an optional
  ``<template>_dynamic.urdf`` sibling (ManiSkill's loader prefers the
  dynamic variant and so do we).
* **AI2THOR (iTHOR / ProcTHOR / RoboTHOR / ArchitecTHOR)** — split
  across two sibling roots:

  - stages live at ``ai2thor/ai2thor-hab/assets/<template>.glb``
  - objects live at ``ai2thor/ai2thorhab-uncompressed/assets/<template>.glb``

  ManiSkill's :class:`AI2THORBaseSceneBuilder` reaches into both roots
  in exactly this way. The "uncompressed" object pool is a superset of
  the compressed one — ProcTHOR scenes reference objects that only
  exist in the uncompressed pool.

ProcTHOR scene files are **bucketed** into hex-style subdirectories
(``configs/scenes/ProcTHOR/1/…``, ``…/2/…`` through ``…/9/…``, ``…/a/…``
through ``…/c/…``). We scan the buckets on first use and cache the
stem→path map; a flat listing would miss every ProcTHOR scene.

Coordinate convention (Habitat Y-up → Genesis Z-up)
---------------------------------------------------

Habitat GLBs are stored Y-up while Genesis (like Sapien / MuJoCo) is
Z-up. ManiSkill applies a 90° rotation about +x to every Habitat
asset it loads, plus an additional 90° rotation about -y for
ProcTHOR specifically (the ProcTHOR GLBs from hssd were packaged
with a different yaw). We apply the same transforms here — to the
stage entity as a pose rotation, and to every object/articulation
by composing with each instance's own rotation + rotating its
translation.

Habitat + Genesis both use ``[w, x, y, z]`` quaternion ordering, so
no wxyz↔xyzw swap is needed — only the coordinate rotation.

template_name resolution
------------------------

- Stage: ``stage_instance.template_name`` is a dataset-relative path
  prefix; append ``.glb``. Examples:
  ``stages/frl_apartment_stage`` (ReplicaCAD),
  ``stages/iTHOR/FloorPlan10_physics`` (iTHOR),
  ``stages/ProcTHOR/1/ProcTHOR-Test-1`` (ProcTHOR).
- Object: ``template_name`` is likewise a dataset-relative path
  prefix (ReplicaCAD: ``objects/frl_apartment_sofa``; AI2THOR:
  ``objects/Chair_007_1``).
- Articulated (ReplicaCAD only): ``template_name`` is a short name
  like ``fridge``; the URDF is at ``urdf/<name>/<name>.urdf`` with
  a preferred ``_dynamic.urdf`` variant when present.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    import genesis as gs

#: Project-local root that ManiSkill downloads scene datasets into.
#: Matches ``MS_ASSET_DIR`` from ``pixi.toml``'s feature.maniskill
#: activation env; hard-coded because a deviation there is a setup
#: bug, not a knob.
_MS_ASSET_ROOT = Path("assets/maniskill/data/scene_datasets")

#: Per-source root for *scene_instance.json configs* and *stage GLBs*.
#: For AI2THOR variants this is the ai2thor-hab root; stage templates
#: resolve under ``<root>/assets/<template>.glb``.
_SCENE_ROOTS: dict[str, Path] = {
    "replicacad": _MS_ASSET_ROOT / "replica_cad_dataset",
    "ithor": _MS_ASSET_ROOT / "ai2thor" / "ai2thor-hab",
    "procthor": _MS_ASSET_ROOT / "ai2thor" / "ai2thor-hab",
    "robothor": _MS_ASSET_ROOT / "ai2thor" / "ai2thor-hab",
    "architecthor": _MS_ASSET_ROOT / "ai2thor" / "ai2thor-hab",
}

#: Per-source root for *object GLBs*. ReplicaCAD keeps everything in
#: one dataset root, but AI2THOR splits object GLBs into the
#: ``ai2thorhab-uncompressed`` sibling — matching
#: ManiSkill's :class:`AI2THORBaseSceneBuilder`, which reads objects
#: from this root specifically.
_OBJECT_ROOTS: dict[str, Path] = {
    "replicacad": _MS_ASSET_ROOT / "replica_cad_dataset",
    "ithor": _MS_ASSET_ROOT / "ai2thor" / "ai2thorhab-uncompressed",
    "procthor": _MS_ASSET_ROOT / "ai2thor" / "ai2thorhab-uncompressed",
    "robothor": _MS_ASSET_ROOT / "ai2thor" / "ai2thorhab-uncompressed",
    "architecthor": _MS_ASSET_ROOT / "ai2thor" / "ai2thorhab-uncompressed",
}

#: Path prefix (relative to a source's scene root) where stage GLBs
#: live. ReplicaCAD puts stages at the dataset root; AI2THOR wraps
#: them under ``assets/``.
_STAGE_ASSET_PREFIX: dict[str, str] = {
    "replicacad": "",
    "ithor": "assets",
    "procthor": "assets",
    "robothor": "assets",
    "architecthor": "assets",
}

#: Same idea for object GLBs.
_OBJECT_ASSET_PREFIX: dict[str, str] = {
    "replicacad": "",
    "ithor": "assets",
    "procthor": "assets",
    "robothor": "assets",
    "architecthor": "assets",
}

#: AI2THOR variants keep their scene files in a per-variant subfolder
#: of the shared ai2thor-hab root. ReplicaCAD puts everything under
#: ``configs/scenes/`` directly (no subfolder).
_SCENE_SUBDIR: dict[str, str] = {
    "replicacad": "configs/scenes",
    "ithor": "configs/scenes/iTHOR",
    "procthor": "configs/scenes/ProcTHOR",
    "robothor": "configs/scenes/RoboTHOR",
    "architecthor": "configs/scenes/ArchitecTHOR",
}

#: Habitat GLBs are Y-up; Genesis is Z-up. ManiSkill applies this 90°
#: +x rotation to every Habitat asset and we mirror that. ProcTHOR
#: GLBs additionally need a -90° +y rotation to face the same way.
_BASE_FIX = Rotation.from_euler("x", 90, degrees=True)
_PROCTHOR_EXTRA = Rotation.from_euler("y", -90, degrees=True)


def _source_fix(source: str) -> Rotation:
    """Return the Habitat→Genesis coord fix rotation for a source.

    ReplicaCAD + iTHOR/RoboTHOR/ArchitecTHOR need only the 90° +x
    rotation. ProcTHOR additionally composes a -90° +y because the
    hssd ProcTHOR GLBs were exported at a different yaw — same quirk
    ManiSkill's AI2THOR builder compensates for.
    """
    if source == "procthor":
        return _BASE_FIX * _PROCTHOR_EXTRA
    return _BASE_FIX


def _quat_wxyz_to_rot(q_wxyz: tuple[float, float, float, float]) -> Rotation:
    """wxyz (Habitat / Genesis) → scipy Rotation (stores xyzw internally)."""
    w, x, y, z = q_wxyz
    return Rotation.from_quat([x, y, z, w])


def _rot_to_quat_wxyz(r: Rotation) -> tuple[float, float, float, float]:
    """scipy Rotation → wxyz (Habitat / Genesis convention)."""
    x, y, z, w = r.as_quat()
    return (float(w), float(x), float(y), float(z))


def _apply_fix(
    fix: Rotation,
    translation: tuple[float, float, float],
    rotation_wxyz: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Apply the Habitat→Genesis coord fix to a pose.

    Both translation and rotation are left-multiplied by ``fix``:
    ``p' = R_fix · p`` and ``q' = q_fix · q_instance``. Matches
    ManiSkill's AI2THOR loader where it does
    ``(x, -z, y)`` on positions (= 90° +x applied to the translation
    vector) and ``qmult(q_fix, q_instance)`` on rotations.
    """
    p_fixed = fix.apply(np.asarray(translation, dtype=float))
    q_fixed = fix * _quat_wxyz_to_rot(rotation_wxyz)
    return (float(p_fixed[0]), float(p_fixed[1]), float(p_fixed[2])), _rot_to_quat_wxyz(
        q_fixed
    )


@lru_cache(maxsize=None)
def _procthor_stem_index() -> dict[str, Path]:
    """Map every ProcTHOR scene stem → its on-disk path.

    ProcTHOR scenes are sharded across ``configs/scenes/ProcTHOR/<b>/``
    subdirectories (buckets ``1`` through ``9`` and ``a`` through
    ``c``). A flat ``scene_instance.json`` lookup can't find them;
    we glob every bucket once and cache the stem→path map so
    repeated resolver calls stay cheap.
    """
    root = _SCENE_ROOTS["procthor"] / _SCENE_SUBDIR["procthor"]
    out: dict[str, Path] = {}
    for path in root.glob("*/*.scene_instance.json"):
        stem = path.name.removesuffix(".scene_instance.json")
        out[stem] = path
    return out


def resolve_scene_instance_path(source: str, scene_id: str) -> Path:
    """Map a TyGrit ``SceneSpec.scene_id`` to its scene_instance.json.

    ``scene_id`` is ``"<source>/<stem>"`` (e.g.
    ``"replicacad/apt_0"`` or
    ``"procthor/ProcTHOR-Train-293"``). The stem is the basename the
    dataset uses; we append ``.scene_instance.json`` to reach the
    actual file.

    ProcTHOR scene files are bucketed across numbered subdirectories,
    so we dispatch through :func:`_procthor_stem_index` rather than
    treating the scene subdir as flat.

    Raises
    ------
    ValueError
        If ``source`` isn't a known Habitat-schema dataset.
    FileNotFoundError
        If the resolved path doesn't exist on disk — the dataset
        hasn't been downloaded, or the stem doesn't match a shipped
        scene.
    """
    if source not in _SCENE_ROOTS:
        raise ValueError(
            f"resolve_scene_instance_path: source {source!r} is not a "
            f"Habitat-schema dataset. Known: {sorted(_SCENE_ROOTS)}."
        )
    stem = scene_id.split("/", 1)[-1]

    if source == "procthor":
        index = _procthor_stem_index()
        if stem not in index:
            raise FileNotFoundError(
                f"resolve_scene_instance_path: ProcTHOR scene stem "
                f"{stem!r} not found under "
                f"{_SCENE_ROOTS['procthor'] / _SCENE_SUBDIR['procthor']} "
                f"(searched {len(index)} bucketed scenes). Either the "
                f"dataset isn't downloaded or the stem is unknown."
            )
        return index[stem]

    path = _SCENE_ROOTS[source] / _SCENE_SUBDIR[source] / f"{stem}.scene_instance.json"
    if not path.exists():
        raise FileNotFoundError(
            f"resolve_scene_instance_path: {path} does not exist. "
            f"Either the {source} dataset isn't downloaded "
            f"(`pixi run -e world download-{source}` where applicable), "
            f"or the scene stem {stem!r} isn't in this dataset version."
        )
    return path


def add_habitat_scene_to(
    genesis_scene: "gs.Scene",
    source: str,
    scene_id: str,
) -> None:
    """Parse a Habitat scene_instance.json and add its entities to ``genesis_scene``.

    Must be called *before* ``genesis_scene.build()``. Emits one
    ``gs.morphs.Mesh`` per ``stage_instance`` + ``object_instances``
    entry, and one ``gs.morphs.URDF`` per
    ``articulated_object_instances`` entry.

    Raises the same errors as :func:`resolve_scene_instance_path`,
    plus :class:`FileNotFoundError` if any referenced mesh/URDF file
    is missing (e.g. partial dataset download).
    """
    import genesis as gs

    scene_path = resolve_scene_instance_path(source, scene_id)
    with open(scene_path) as f:
        data = json.load(f)

    fix = _source_fix(source)
    _add_stage(genesis_scene, gs, source, fix, data.get("stage_instance"))

    for i, obj in enumerate(data.get("object_instances", [])):
        _add_object(genesis_scene, gs, source, fix, obj, i)

    for i, art in enumerate(data.get("articulated_object_instances", [])):
        _add_articulated(genesis_scene, gs, source, fix, art, i)


# ─────────────────────────── internals ───────────────────────────


def _stage_glb_path(source: str, template: str) -> Path:
    prefix = _STAGE_ASSET_PREFIX[source]
    root = _SCENE_ROOTS[source]
    return (root / prefix / f"{template}.glb") if prefix else (root / f"{template}.glb")


def _object_glb_path(source: str, template: str) -> Path:
    prefix = _OBJECT_ASSET_PREFIX[source]
    root = _OBJECT_ROOTS[source]
    return (root / prefix / f"{template}.glb") if prefix else (root / f"{template}.glb")


def _add_stage(
    genesis_scene: "gs.Scene",
    gs,
    source: str,
    fix: Rotation,
    stage_instance: dict | None,
) -> None:
    """Add the scene's background stage mesh.

    ``stage_instance.template_name`` is a dataset-relative path
    prefix; the GLB lives at ``<asset_root>/<template>.glb`` where
    ``asset_root`` is the dataset root for ReplicaCAD and
    ``<dataset_root>/assets`` for AI2THOR variants (see
    :data:`_STAGE_ASSET_PREFIX`).

    The Habitat→Genesis coord fix rotation is applied as the entity's
    pose so the mesh stands up the right way in Z-up Genesis.
    """
    if stage_instance is None:
        return
    template = stage_instance["template_name"]
    glb_path = _stage_glb_path(source, template)
    if not glb_path.exists():
        raise FileNotFoundError(
            f"_add_stage: stage GLB {glb_path} does not exist. "
            f"source={source!r} template_name={template!r}."
        )
    quat = _rot_to_quat_wxyz(fix)
    genesis_scene.add_entity(
        gs.morphs.Mesh(
            file=str(glb_path),
            quat=quat,
            fixed=True,
            # Skip CoACD convex decomposition — matching the Holodeck
            # branch's rationale in :mod:`TyGrit.worlds.backends.genesis`:
            # ReplicaCAD + AI2THOR stages reference hundreds of sub-
            # meshes and decomposing each takes seconds-to-minutes,
            # blowing up scene construction to hours. Visual geometry
            # as-is is adequate for our free-space navigation / coarse
            # grasp planning use case.
            convexify=False,
        ),
        name=f"stage__{Path(template).name}",
    )


def _add_object(
    genesis_scene: "gs.Scene",
    gs,
    source: str,
    fix: Rotation,
    obj: dict,
    idx: int,
) -> None:
    """Add one object_instance entry as a Mesh morph.

    ``template_name`` resolves to ``<asset_root>/<template>.glb``
    (different asset root per source; see :data:`_OBJECT_ROOTS`).
    ``motion_type`` determines whether the body is fixed; we respect
    ``STATIC`` and ``KEEP_FIXED`` (both treated as fixed) and
    ``DYNAMIC`` (free). ``non_uniform_scale`` carries per-axis scale
    when present (AI2THOR objects set it; ReplicaCAD objects don't).
    """
    template = obj["template_name"]
    glb_path = _object_glb_path(source, template)
    if not glb_path.exists():
        raise FileNotFoundError(
            f"_add_object: source={source!r} template_name={template!r} "
            f"→ {glb_path} does not exist."
        )

    motion = obj.get("motion_type", "STATIC").upper()
    fixed = motion in ("STATIC", "KEEP_FIXED")
    pos, quat = _apply_fix(
        fix,
        tuple(obj.get("translation", (0.0, 0.0, 0.0))),
        tuple(obj.get("rotation", (1.0, 0.0, 0.0, 0.0))),  # [w, x, y, z]
    )
    scale = tuple(obj.get("non_uniform_scale", (1.0, 1.0, 1.0)))

    genesis_scene.add_entity(
        gs.morphs.Mesh(
            file=str(glb_path),
            pos=pos,
            quat=quat,
            scale=scale,
            fixed=fixed,
            # See _add_stage above for the CoACD-skip rationale.
            convexify=False,
        ),
        name=f"obj_{idx}__{Path(template).name}",
    )


def _add_articulated(
    genesis_scene: "gs.Scene",
    gs,
    source: str,
    fix: Rotation,
    art: dict,
    idx: int,
) -> None:
    """Add one articulated_object_instance entry as a URDF morph.

    ReplicaCAD ships both ``<name>.urdf`` and ``<name>_dynamic.urdf``
    for each articulated object. The ``_dynamic`` variant has the
    interactive joints un-fixed (drawers that slide, doors that
    swing); ManiSkill's loader uses it by default and so do we.
    ``fixed_base`` on the instance controls whether the whole root
    is pinned in world space (``True`` for most furniture).
    ``uniform_scale`` is optional; URDF takes a scalar scale.
    """
    template = art["template_name"]
    urdf_dir = _OBJECT_ROOTS[source] / "urdf" / template
    dynamic = urdf_dir / f"{template}_dynamic.urdf"
    default = urdf_dir / f"{template}.urdf"
    if dynamic.exists():
        urdf_path = dynamic
    elif default.exists():
        urdf_path = default
    else:
        raise FileNotFoundError(
            f"_add_articulated: no URDF found for source={source!r} "
            f"template_name={template!r} (looked at {dynamic} and {default})."
        )

    pos, quat = _apply_fix(
        fix,
        tuple(art.get("translation", (0.0, 0.0, 0.0))),
        tuple(art.get("rotation", (1.0, 0.0, 0.0, 0.0))),  # [w, x, y, z]
    )
    fixed_base = bool(art.get("fixed_base", True))
    scale = float(art.get("uniform_scale", 1.0))

    genesis_scene.add_entity(
        gs.morphs.URDF(
            file=str(urdf_path), pos=pos, quat=quat, scale=scale, fixed=fixed_base
        ),
        name=f"art_{idx}__{template}",
    )
