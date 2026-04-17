"""JSON manifest I/O for the mobile-grasping dataset.

Complement to :mod:`TyGrit.worlds.manifest` — same schema conventions
(``version: 1``, pretty-printed with ``indent=2``, trailing newline,
parent directories created on save, transparent ``.gz`` handling) but
wraps :class:`~TyGrit.types.mobile_grasp.MobileGraspDatapoint` tuples
instead of raw :class:`~TyGrit.types.worlds.SceneSpec` lists.

Schema (version 1)::

    {
      "version": 1,
      "source": "mobile_grasp_v1",
      "generator": "TyGrit.worlds.generators.mobile_grasp ...",
      "metadata": {"key": "value", ...},
      "entries": [
        {
          "scene": { ... SceneSpec fields ... },
          "object": { ... ObjectSpec fields ... },
          "base_pose": [x, y, theta],
          "init_qpos": {"joint_name": qpos, ...},
          "grasp_hint": [x, y, z, qx, qy, qz, qw] | null
        },
        ...
      ]
    }

Cross-backend compatibility
---------------------------

Every datapoint must be loadable under every sim backend TyGrit
targets (ManiSkill + Genesis + Isaac Sim). :func:`validate_cross_backend`
enforces the static asset-level requirements:

* ``scene.source`` must appear in :data:`CROSS_BACKEND_SCENE_SOURCES`
  — currently only ``"holodeck"`` is accepted by all three adapters.
* ``object.mesh_path`` must be populated; ``builtin_id`` alone is
  rejected because registry-based loaders (``"ycb:..."``) are
  ManiSkill-only.

Per-backend dynamics checks (does the scene parse? does the object
settle?) require a live sim and are the generator's responsibility,
not this module's.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from TyGrit.types.mobile_grasp import MobileGraspDatapoint, MobileGraspDataset
from TyGrit.worlds.manifest import (
    MANIFEST_VERSION,
    _object_from_dict,
    _read_manifest_text,
    _scene_from_dict,
    _to_dict,
    _write_manifest_text,
)

#: Scene sources that every sim backend TyGrit targets can load.
#: Keep in sync with
#: :data:`TyGrit.worlds.asset_sources.base.SOURCE_SIM_COMPATIBILITY`.
#: Holodeck (MolmoSpaces MJCF) is the one source currently accepted by
#: all three of ManiSkill (via ``_maniskill_holodeck.HolodeckSceneBuilder``),
#: Genesis (via ``_genesis_habitat`` / direct MJCF load), and Isaac Sim.
CROSS_BACKEND_SCENE_SOURCES: frozenset[str] = frozenset({"holodeck"})


def load_mobile_grasp_manifest(path: str | Path) -> MobileGraspDataset:
    """Load a mobile-grasp dataset from a JSON file.

    Transparently handles ``.json`` and ``.json.gz`` via the same
    extension-sniffing helpers as :func:`TyGrit.worlds.manifest.load_manifest`.

    Parameters
    ----------
    path
        Filesystem path to the manifest.

    Returns
    -------
    MobileGraspDataset
        All datapoints in file order plus any top-level metadata.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file is not valid JSON, the schema version is unsupported,
        or any entry is malformed. The error message includes the
        manifest path for quick diagnosis.
    """
    path = Path(path)
    text = _read_manifest_text(path)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        # Same wrapping pattern as TyGrit.worlds.manifest.load_manifest:
        # re-raise with the file path so the caller sees which manifest
        # failed instead of a bare "line N column M" error.
        raise ValueError(f"mobile-grasp manifest {path}: invalid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"mobile-grasp manifest {path}: top-level must be a JSON object, "
            f"got {type(data).__name__}"
        )

    version = data.get("version")
    if version != MANIFEST_VERSION:
        raise ValueError(
            f"mobile-grasp manifest {path}: unsupported schema version "
            f"{version!r} (this build handles version {MANIFEST_VERSION})"
        )

    entries_raw = data.get("entries")
    if not isinstance(entries_raw, list):
        raise ValueError(
            f"mobile-grasp manifest {path}: 'entries' must be a list, got "
            f"{type(entries_raw).__name__}"
        )

    entries = tuple(_datapoint_from_dict(entry, path) for entry in entries_raw)

    metadata_raw = data.get("metadata", {})
    if not isinstance(metadata_raw, Mapping):
        raise ValueError(
            f"mobile-grasp manifest {path}: 'metadata' must be a JSON object, got "
            f"{type(metadata_raw).__name__}"
        )
    metadata = {str(k): str(v) for k, v in metadata_raw.items()}

    return MobileGraspDataset(entries=entries, metadata=metadata)


def save_mobile_grasp_manifest(
    path: str | Path,
    dataset: MobileGraspDataset | Iterable[MobileGraspDatapoint],
    *,
    source: str | None = "mobile_grasp_v1",
    generator: str | None = None,
) -> None:
    """Write a mobile-grasp dataset to a JSON file.

    Parent directories are created as needed. Output is pretty-printed
    with ``indent=2``. Accepts either a :class:`MobileGraspDataset`
    (metadata preserved) or a bare iterable of datapoints (metadata
    becomes empty).

    Parameters
    ----------
    path
        Destination filesystem path. Use ``.json.gz`` for gzip-compressed
        output.
    dataset
        Either a :class:`MobileGraspDataset` or an iterable of
        :class:`MobileGraspDatapoint`.
    source
        Optional upstream dataset label recorded as top-level metadata.
    generator
        Optional identifier of the tool that produced the manifest.
    """
    if isinstance(dataset, MobileGraspDataset):
        entries = dataset.entries
        metadata = dict(dataset.metadata)
    else:
        entries = tuple(dataset)
        metadata = {}

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {"version": MANIFEST_VERSION}
    if source is not None:
        data["source"] = source
    if generator is not None:
        data["generator"] = generator
    if metadata:
        data["metadata"] = metadata
    data["entries"] = [_datapoint_to_dict(dp) for dp in entries]

    _write_manifest_text(path, json.dumps(data, indent=2) + "\n")


def validate_cross_backend(
    dataset: MobileGraspDataset | Iterable[MobileGraspDatapoint],
) -> None:
    """Raise if any datapoint's assets are not loadable by every backend.

    Checks the static asset invariants spelled out in this module's
    docstring: scene source is in :data:`CROSS_BACKEND_SCENE_SOURCES`
    and object has a populated ``mesh_path``. Dynamic checks (does the
    MJCF parse? does the mesh exist on disk?) are the generator's
    responsibility — this function only validates the manifest itself.

    Raises
    ------
    ValueError
        On the first offending entry. The message names the entry
        index, the scene_id / object.name, and which invariant failed
        so the generator's bug is obvious.
    """
    entries = (
        dataset.entries if isinstance(dataset, MobileGraspDataset) else tuple(dataset)
    )
    for idx, dp in enumerate(entries):
        if dp.scene.source not in CROSS_BACKEND_SCENE_SOURCES:
            raise ValueError(
                f"entry {idx} ({dp.scene.scene_id!r}): scene source "
                f"{dp.scene.source!r} is not cross-backend; expected one of "
                f"{sorted(CROSS_BACKEND_SCENE_SOURCES)}"
            )
        if not dp.object.mesh_path:
            raise ValueError(
                f"entry {idx} ({dp.scene.scene_id!r} / {dp.object.name!r}): "
                f"object must carry mesh_path to be portable across backends; "
                f"builtin_id-only specs (e.g. ycb:...) are ManiSkill-only"
            )


# ─────────────────────────── internals ───────────────────────────


def _datapoint_to_dict(dp: MobileGraspDatapoint) -> dict[str, Any]:
    """Serialize one :class:`MobileGraspDatapoint` to a JSON-friendly dict.

    Reuses :func:`TyGrit.worlds.manifest._to_dict` for the nested scene
    and object so we stay in lockstep with the existing manifest format
    — any future field added to ``SceneSpec``/``ObjectSpec`` is picked
    up for free without a parallel serializer.
    """
    return {
        "scene": _to_dict(dp.scene),
        "object": _to_dict(dp.object),
        "base_pose": list(dp.base_pose),
        "init_qpos": {k: float(v) for k, v in dp.init_qpos.items()},
        "grasp_hint": list(dp.grasp_hint) if dp.grasp_hint is not None else None,
    }


def _datapoint_from_dict(data: Any, path: Path) -> MobileGraspDatapoint:
    """Reconstruct one :class:`MobileGraspDatapoint` from a JSON dict."""
    if not isinstance(data, Mapping):
        raise ValueError(
            f"mobile-grasp manifest {path}: entry must be a JSON object, got "
            f"{type(data).__name__}"
        )

    try:
        scene_raw = data["scene"]
        object_raw = data["object"]
        base_raw = data["base_pose"]
    except KeyError as exc:
        # KeyError: manifest entry is missing a required top-level key.
        # We re-wrap with the manifest path so the generator's bug
        # surfaces as "entry N of file X is missing scene" rather than
        # a bare KeyError.
        raise ValueError(
            f"mobile-grasp manifest {path}: entry missing required key {exc}"
        ) from exc

    scene = _scene_from_dict(scene_raw, path)
    obj = _object_from_dict(object_raw, path)

    if not isinstance(base_raw, (list, tuple)) or len(base_raw) != 3:
        raise ValueError(
            f"mobile-grasp manifest {path}: base_pose must be a 3-element list "
            f"[x, y, theta]; got {base_raw!r}"
        )
    base_pose = (float(base_raw[0]), float(base_raw[1]), float(base_raw[2]))

    init_qpos_raw = data.get("init_qpos", {})
    if not isinstance(init_qpos_raw, Mapping):
        raise ValueError(
            f"mobile-grasp manifest {path}: init_qpos must be a JSON object, got "
            f"{type(init_qpos_raw).__name__}"
        )
    init_qpos = {str(k): float(v) for k, v in init_qpos_raw.items()}

    grasp_hint_raw = data.get("grasp_hint")
    if grasp_hint_raw is None:
        grasp_hint = None
    elif isinstance(grasp_hint_raw, (list, tuple)) and len(grasp_hint_raw) == 7:
        grasp_hint = tuple(float(v) for v in grasp_hint_raw)
    else:
        raise ValueError(
            f"mobile-grasp manifest {path}: grasp_hint must be null or a "
            f"7-element list [x, y, z, qx, qy, qz, qw]; got {grasp_hint_raw!r}"
        )

    return MobileGraspDatapoint(
        scene=scene,
        object=obj,
        base_pose=base_pose,
        init_qpos=init_qpos,
        grasp_hint=grasp_hint,
    )
