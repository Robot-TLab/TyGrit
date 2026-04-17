"""JSON manifests of :class:`~TyGrit.types.worlds.SceneSpec` entries.

A manifest is a self-contained JSON file listing one or more SceneSpecs.
The format is deliberately simple so manifests are human-readable, diff
cleanly in PRs, and can be hand-edited when needed.

Schema (version 1)::

    {
      "version": 1,
      "source": "replicacad",           # optional: upstream dataset label
      "generator": "tools/make_manifest/replicacad.py",  # optional
      "scenes": [
        {
          "scene_id": "replicacad/apt_0",
          "source": "replicacad",
          "background_builtin_id": "replicacad:apt_0",
          "objects": [],
          "target_object_names": []
        },
        ...
      ]
    }

Each entry in ``scenes`` mirrors the fields of
:class:`~TyGrit.types.worlds.SceneSpec`. Tuples serialize as JSON arrays;
nested :class:`~TyGrit.types.worlds.ObjectSpec` instances become nested
dicts inside the ``objects`` array.

Use :func:`load_manifest` to read and :func:`save_manifest` to write.
Unknown top-level or per-scene fields are silently ignored so newer
manifests remain forward-compatible with older readers.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from TyGrit.types.worlds import ObjectSpec, SceneSpec

#: Current manifest schema version. Incremented only on breaking changes.
MANIFEST_VERSION = 1


def _read_manifest_text(path: Path) -> str:
    """Read a manifest file as text, transparently unzipping ``.gz``.

    The ``.json.gz`` variant is used for large manifests (e.g. the
    12,000-entry ProcTHOR manifest) so the committed file fits under
    pre-commit's ``check-added-large-files`` 500 KB threshold. Load
    behavior is otherwise identical — callers never need to care
    whether the file is compressed.
    """
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return f.read()
    return path.read_text()


def _write_manifest_text(path: Path, text: str) -> None:
    """Write a manifest file as text, transparently gzipping ``.gz``.

    Mirror of :func:`_read_manifest_text` — both functions have to
    agree on the extension sniffing or the round-trip breaks.
    """
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(text)
    else:
        path.write_text(text)


def load_manifest(path: str | Path) -> tuple[SceneSpec, ...]:
    """Load a world manifest from a JSON file.

    Parameters
    ----------
    path
        Filesystem path to the manifest.

    Returns
    -------
    tuple[SceneSpec, ...]
        All scenes in file order.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist. Surfaces from
        :meth:`pathlib.Path.read_text`.
    ValueError
        If the file is not valid JSON, the top-level structure is wrong,
        the schema version is unsupported, or any scene entry is
        malformed. The error message includes the manifest path and
        (where possible) the ``scene_id`` of the offending entry.
    """
    path = Path(path)
    text = _read_manifest_text(path)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        # json.loads raises JSONDecodeError on syntactically invalid
        # JSON. Re-wrap with the file path so callers know which manifest
        # is broken instead of getting a bare "line 3 column 5" error.
        raise ValueError(f"manifest {path}: invalid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"manifest {path}: top-level must be a JSON object, "
            f"got {type(data).__name__}"
        )

    version = data.get("version")
    if version != MANIFEST_VERSION:
        raise ValueError(
            f"manifest {path}: unsupported schema version {version!r} "
            f"(this build handles version {MANIFEST_VERSION})"
        )

    scenes_raw = data.get("scenes")
    if not isinstance(scenes_raw, list):
        raise ValueError(
            f"manifest {path}: 'scenes' must be a list, "
            f"got {type(scenes_raw).__name__}"
        )

    return tuple(_scene_from_dict(entry, path) for entry in scenes_raw)


def save_manifest(
    path: str | Path,
    scenes: Iterable[SceneSpec],
    *,
    source: str | None = None,
    generator: str | None = None,
) -> None:
    """Write a world manifest to a JSON file.

    Parent directories are created as needed. Output is pretty-printed
    with ``indent=2`` so manifest diffs are reviewable in PRs.

    Parameters
    ----------
    path
        Destination filesystem path.
    scenes
        The SceneSpecs to write. May be any iterable; order is preserved.
    source
        Optional upstream dataset label, recorded as top-level metadata
        (``"replicacad"``, ``"hssd"``, …).
    generator
        Optional identifier of the tool that produced the manifest, e.g.
        ``"tools/make_manifest/replicacad.py@abc123"``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Build the dict in reading order: version, metadata, then scenes.
    # Python dicts preserve insertion order, so this makes the on-disk
    # JSON human-friendly.
    data: dict[str, Any] = {"version": MANIFEST_VERSION}
    if source is not None:
        data["source"] = source
    if generator is not None:
        data["generator"] = generator
    data["scenes"] = [_to_dict(scene) for scene in scenes]

    _write_manifest_text(path, json.dumps(data, indent=2) + "\n")


# ─────────────────────────── object manifests ───────────────────────────
#
# Parallel to the scene manifest API but for ObjectSpec pools. The
# schema wraps a list of self-contained ObjectSpec dicts under an
# "objects" key:
#
#     {
#       "version": 1,
#       "source": "ycb",
#       "generator": "...",
#       "objects": [
#         { "name": "...", "builtin_id": "ycb:..." },
#         ...
#       ]
#     }
#
# Kept separate from scene manifests so the same object pool can be
# reused across multiple scene sources (e.g. YCB objects in both
# ReplicaCAD and HSSD scenes) without duplication.


def load_object_manifest(path: str | Path) -> tuple[ObjectSpec, ...]:
    """Load an object manifest from a JSON file.

    Parameters
    ----------
    path
        Filesystem path to the manifest.

    Returns
    -------
    tuple[ObjectSpec, ...]
        All objects in file order.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist. Surfaces from
        :meth:`pathlib.Path.read_text`.
    ValueError
        If the file is not valid JSON, the top-level structure is wrong,
        the schema version is unsupported, or any object entry is
        malformed. The error message includes the manifest path and
        (where possible) the ``name`` of the offending entry.
    """
    path = Path(path)
    text = _read_manifest_text(path)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        # Same wrapping pattern as load_manifest: re-raise with the file
        # path so the caller sees which manifest failed.
        raise ValueError(f"object manifest {path}: invalid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"object manifest {path}: top-level must be a JSON object, "
            f"got {type(data).__name__}"
        )

    version = data.get("version")
    if version != MANIFEST_VERSION:
        raise ValueError(
            f"object manifest {path}: unsupported schema version {version!r} "
            f"(this build handles version {MANIFEST_VERSION})"
        )

    objects_raw = data.get("objects")
    if not isinstance(objects_raw, list):
        raise ValueError(
            f"object manifest {path}: 'objects' must be a list, "
            f"got {type(objects_raw).__name__}"
        )

    return tuple(_object_from_dict(entry, path) for entry in objects_raw)


def save_object_manifest(
    path: str | Path,
    objects: Iterable[ObjectSpec],
    *,
    source: str | None = None,
    generator: str | None = None,
) -> None:
    """Write an object manifest to a JSON file.

    Same formatting conventions as :func:`save_manifest` (parent dirs
    created, pretty-printed with ``indent=2``, trailing newline).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {"version": MANIFEST_VERSION}
    if source is not None:
        data["source"] = source
    if generator is not None:
        data["generator"] = generator
    data["objects"] = [_to_dict(obj) for obj in objects]

    _write_manifest_text(path, json.dumps(data, indent=2) + "\n")


# ─────────────────────────── internals ───────────────────────────


def _to_dict(obj: Any) -> Any:
    """Recursively convert a frozen-dataclass tree to JSON-friendly types.

    Tuples become lists (JSON has no tuple type), nested dataclasses
    become dicts, and mappings are preserved. Primitives pass through.
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (tuple, list)):
        return [_to_dict(x) for x in obj]
    if isinstance(obj, Mapping):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj


def _scene_from_dict(data: Any, path: Path) -> SceneSpec:
    """Reconstruct a :class:`SceneSpec` from a JSON-decoded dict.

    Raises
    ------
    ValueError
        If ``data`` is not a dict, or the SceneSpec constructor rejects
        the coerced kwargs (unknown field, missing required field, or a
        ``__post_init__`` contract violation such as duplicate object
        names). The original exception is chained via ``from`` and its
        message is appended for context.
    """
    if not isinstance(data, Mapping):
        raise ValueError(
            f"manifest {path}: scene entry must be a JSON object, "
            f"got {type(data).__name__}"
        )

    # `objects` is optional in the manifest schema; absent and empty-list
    # both legally mean "scene has no spawned objects". A non-list value
    # under that key is a malformed manifest and we raise rather than
    # silently treating it as empty (the previous `or ()` swallowed
    # truthy bugs like a stray dict or string).
    raw_objects = data.get("objects", ())
    if not isinstance(raw_objects, (list, tuple)):
        raise ValueError(
            f"manifest {path}: scene entry 'objects' must be a list, got "
            f"{type(raw_objects).__name__}"
        )
    objects = tuple(_object_from_dict(o, path) for o in raw_objects)

    kwargs: dict[str, Any] = {"objects": objects}
    for f in fields(SceneSpec):
        if f.name == "objects" or f.name not in data:
            continue
        kwargs[f.name] = _coerce_json_value(f.name, data[f.name])

    scene_id_hint = data.get("scene_id", "<unknown>")
    try:
        return SceneSpec(**kwargs)
    except TypeError as exc:
        # TypeError: unknown kwarg or missing required kwarg — i.e. the
        # manifest dict doesn't match the SceneSpec signature. Surfaces
        # as e.g. "SceneSpec.__init__() missing 1 required positional
        # argument: 'source'".
        raise ValueError(
            f"manifest {path}: invalid SceneSpec {scene_id_hint!r}: {exc}"
        ) from exc
    except ValueError as exc:
        # ValueError: raised by SceneSpec.__post_init__ validators
        # (duplicate object names, unknown target_object_names).
        raise ValueError(
            f"manifest {path}: invalid SceneSpec {scene_id_hint!r}: {exc}"
        ) from exc


def _object_from_dict(data: Any, path: Path) -> ObjectSpec:
    """Reconstruct an :class:`ObjectSpec` from a JSON-decoded dict."""
    if not isinstance(data, Mapping):
        raise ValueError(
            f"manifest {path}: object entry must be a JSON object, "
            f"got {type(data).__name__}"
        )

    kwargs: dict[str, Any] = {}
    for f in fields(ObjectSpec):
        if f.name not in data:
            continue
        kwargs[f.name] = _coerce_json_value(f.name, data[f.name])

    name_hint = data.get("name", "<unknown>")
    try:
        return ObjectSpec(**kwargs)
    except TypeError as exc:
        # TypeError: unknown kwarg in the manifest dict. Unknown fields
        # were already filtered out by the fields() loop, so this can
        # only trigger if someone shadows a method or passes a non-str
        # for a str-typed field in a way Python catches at __init__.
        raise ValueError(
            f"manifest {path}: invalid ObjectSpec {name_hint!r}: {exc}"
        ) from exc
    except ValueError as exc:
        # ValueError: ObjectSpec.__post_init__ rejects specs with no
        # asset source (all five path/id fields are None).
        raise ValueError(
            f"manifest {path}: invalid ObjectSpec {name_hint!r}: {exc}"
        ) from exc


def _coerce_json_value(name: str, raw: Any) -> Any:
    """Coerce one JSON-decoded value into the type a frozen field expects.

    JSON has no tuple type, so lists need to become tuples for
    frozen-dataclass compatibility. ``joint_init`` is special-cased
    because it's a tuple of ``(str, float)`` tuples, not a flat tuple.
    """
    if raw is None:
        return None
    if name == "joint_init" and isinstance(raw, list):
        return tuple((str(pair[0]), float(pair[1])) for pair in raw)
    if isinstance(raw, list):
        return tuple(raw)
    return raw
