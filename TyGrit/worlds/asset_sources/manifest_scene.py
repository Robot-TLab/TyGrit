"""Manifest-backed :class:`AssetSource` for scene datasets.

Six of TyGrit's seven scene-source variants (MolmoSpaces / Holodeck,
4 AI2-THOR variants, ReplicaCAD, RoboCasa) share the same
integration pattern:

1. A generator under :mod:`TyGrit.worlds.generators` scans the
   dataset once and writes a JSON / gzipped-JSON manifest of
   :class:`SceneSpec` entries under ``resources/worlds/<dataset>.*``.
2. :func:`TyGrit.worlds.manifest.load_manifest` reads that manifest
   into a tuple of frozen specs.
3. A sim handler builds whichever spec the caller selects.

Since the load step is uniform, we expose one generic
:class:`ManifestSceneSource` and one four-line subclass per dataset.
The subclass's only job is pinning the canonical manifest path so
callers don't have to memorise file names. Adding a new scene
dataset = adding a generator + a four-line subclass here. **No
backend edits required.**

The default manifest paths are anchored to the repository root via
:data:`MANIFEST_DIR` so a source constructed from any working
directory resolves correctly.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path

from TyGrit.types.worlds import ObjectSpec, SceneSpec
from TyGrit.worlds.asset_sources.base import AssetRequest
from TyGrit.worlds.manifest import load_manifest

#: Repo root (``/media/run/Work/TyGrit/`` on the dev box) — derived
#: by walking up from this file's location. Anchoring manifest paths
#: here means a source constructed from any cwd resolves correctly.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: Canonical directory generators write to. Not configurable per
#: source: every generator under ``TyGrit/worlds/generators/`` writes
#: here.
MANIFEST_DIR: Path = _PROJECT_ROOT / "resources" / "worlds"


class ManifestSceneSource:
    """Scene source backed by a JSON manifest of :class:`SceneSpec`\\ s.

    Parameters
    ----------
    source_name
        TyGrit source identifier (``"replicacad"``, ``"procthor"``, …).
        Must match the ``source`` field of every :class:`SceneSpec` in
        the manifest; mismatches raise :class:`ValueError` at load
        time so a wrong-manifest-path bug fails fast instead of
        producing confusing backend dispatch errors later.
    manifest_path
        Path to the JSON manifest. Absolute, or relative to the cwd
        at first enumeration. Lazy-loaded — constructing a source is
        cheap even when the file is large (ProcTHOR is ~12k entries).
    """

    def __init__(self, source_name: str, manifest_path: str | Path) -> None:
        self._source_name = source_name
        self._manifest_path = Path(manifest_path)
        # Cache as a tuple (preserves manifest order) plus a dict for
        # O(1) lookup by scene_id. Both populated in _ensure_loaded.
        self._scenes_cache: tuple[SceneSpec, ...] | None = None
        self._by_id: dict[str, SceneSpec] | None = None

    # ── identity ───────────────────────────────────────────────────────

    @property
    def source_name(self) -> str:
        return self._source_name

    @property
    def manifest_path(self) -> Path:
        return self._manifest_path

    # ── internals ──────────────────────────────────────────────────────

    def _ensure_loaded(self) -> tuple[SceneSpec, ...]:
        if self._scenes_cache is not None:
            return self._scenes_cache
        if not self._manifest_path.exists():
            raise FileNotFoundError(
                f"ManifestSceneSource[{self._source_name!r}]: manifest not found "
                f"at {self._manifest_path}. Run the corresponding generator "
                f"(`pixi run -e world generate-{self._source_name}-scenes` or the "
                f"module entry point at TyGrit.worlds.generators.{self._source_name}) "
                f"to produce it."
            )
        scenes = load_manifest(str(self._manifest_path))
        mismatches = [s.scene_id for s in scenes if s.source != self._source_name]
        if mismatches:
            raise ValueError(
                f"ManifestSceneSource[{self._source_name!r}]: manifest at "
                f"{self._manifest_path} contains specs with mismatched source "
                f"field (first 5: {mismatches[:5]!r}). Source field must match "
                f"the source_name the source was constructed with."
            )
        self._scenes_cache = scenes
        self._by_id = {s.scene_id: s for s in scenes}
        return scenes

    # ── AssetSource: enumeration ──────────────────────────────────────

    def list_scene_ids(self, *, split: str | None = None) -> tuple[str, ...]:
        if split is not None:
            # Per-dataset splits would add a filter here; the base
            # implementation has no concept of splits. Subclasses that
            # know their dataset's split convention override this.
            raise ValueError(
                f"ManifestSceneSource[{self._source_name!r}]: splits are not "
                f"supported by the generic scene source. Subclass and override "
                f"list_scene_ids if the dataset defines a train/val split."
            )
        return tuple(s.scene_id for s in self._ensure_loaded())

    def list_object_ids(self, *, split: str | None = None) -> tuple[str, ...]:
        # Object-pool enumeration is not part of the scene-source
        # contract; object-based datasets live in sibling modules
        # (ycb.py, objaverse.py).
        return ()

    # ── AssetSource: lookup ───────────────────────────────────────────

    def get_scene(
        self, scene_id: str, *, request: AssetRequest | None = None
    ) -> SceneSpec:
        # Ensure the by-id cache is populated; `_ensure_loaded` builds
        # both the tuple and the dict cache.
        self._ensure_loaded()
        assert self._by_id is not None  # populated alongside _scenes_cache
        try:
            return self._by_id[scene_id]
        except KeyError:
            available = list(self._by_id)
            preview = available[:5]
            raise KeyError(
                f"ManifestSceneSource[{self._source_name!r}]: no scene with id "
                f"{scene_id!r}. Available (first 5 of {len(available)}): {preview!r}"
            ) from None

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
        raise NotImplementedError(
            f"ManifestSceneSource[{self._source_name!r}] is scene-only; "
            f"use the dataset's ObjectSource for object lookups"
        )

    # ── AssetSource: sampling ─────────────────────────────────────────

    def sample_scene_id(self, *, seed: int, split: str | None = None) -> str:
        scenes = self._ensure_loaded()
        if split is not None:
            # Same rationale as list_scene_ids — no default split semantics.
            raise ValueError(
                f"ManifestSceneSource[{self._source_name!r}]: splits are not "
                f"supported by the generic scene source."
            )
        if not scenes:
            raise RuntimeError(
                f"ManifestSceneSource[{self._source_name!r}]: scene pool is empty"
            )
        rng = random.Random(int(seed))
        return rng.choice(scenes).scene_id

    def sample_object_id(self, *, seed: int, split: str | None = None) -> str:
        raise NotImplementedError(
            f"ManifestSceneSource[{self._source_name!r}] is scene-only"
        )


# ── Per-dataset subclasses ───────────────────────────────────────────
# Manifest filenames must match what the generators in
# ``TyGrit.worlds.generators.<dataset>`` actually write — see each
# generator's ``DEFAULT_OUTPUT`` / ``MANIFEST_PATH`` constant.


class ReplicaCADSource(ManifestSceneSource):
    """Facebook ReplicaCAD apartments.

    Licensing is *uncertain*: the upstream Habitat dataset card lists
    CC-BY 4.0 for code; some asset variants are CC-BY-NC. Verify per
    asset before redistributing TyGrit-derived data.
    """

    def __init__(
        self,
        manifest_path: str | Path = MANIFEST_DIR / "replicacad.json",
    ) -> None:
        super().__init__("replicacad", manifest_path)


class ProcTHORSource(ManifestSceneSource):
    """AI2 ProcTHOR (~12k procedurally generated rooms).

    Code Apache-2.0; assets follow the parent AI2-THOR licence
    (Apache-2.0 according to the AI2-THOR repo's LICENSE file —
    re-verify before vendoring).

    The manifest is gzipped because plain JSON for ~12k scenes is
    ~50 MB; :func:`load_manifest` sniffs the ``.gz`` suffix
    transparently.
    """

    def __init__(
        self,
        manifest_path: str | Path = MANIFEST_DIR / "procthor.json.gz",
    ) -> None:
        super().__init__("procthor", manifest_path)


class IThorSource(ManifestSceneSource):
    """AI2 iTHOR (~150 hand-crafted rooms; Apache-2.0 per AI2-THOR repo)."""

    def __init__(
        self,
        manifest_path: str | Path = MANIFEST_DIR / "ithor.json",
    ) -> None:
        super().__init__("ithor", manifest_path)


class RoboTHORSource(ManifestSceneSource):
    """AI2 RoboTHOR (~75 navigation scenes; Apache-2.0)."""

    def __init__(
        self,
        manifest_path: str | Path = MANIFEST_DIR / "robothor.json",
    ) -> None:
        super().__init__("robothor", manifest_path)


class ArchitecTHORSource(ManifestSceneSource):
    """AI2 ArchitecTHOR (~10 architectural scenes; Apache-2.0)."""

    def __init__(
        self,
        manifest_path: str | Path = MANIFEST_DIR / "architecthor.json",
    ) -> None:
        super().__init__("architecthor", manifest_path)


class RoboCasaSource(ManifestSceneSource):
    """NVIDIA RoboCasa kitchens (code: MIT; assets: CC-BY 4.0)."""

    def __init__(
        self,
        manifest_path: str | Path = MANIFEST_DIR / "robocasa.json",
    ) -> None:
        super().__init__("robocasa", manifest_path)


class HolodeckSource(ManifestSceneSource):
    """Allen AI MolmoSpaces / Holodeck scenes.

    Code: Apache-2.0. Assets: mixed (rooms procedurally composed from
    Objaverse parts; per-asset licenses vary — call
    ``molmo_spaces.print_license_info`` to inspect a specific scene).

    The class is named ``HolodeckSource`` because the upstream engine
    is called Holodeck and TyGrit's ``SceneSpec.source`` field uses
    ``"holodeck"``. ``MolmoSpacesSource`` is a deprecated alias kept
    for back-compat.
    """

    def __init__(
        self,
        manifest_path: str | Path = MANIFEST_DIR / "holodeck.json.gz",
    ) -> None:
        super().__init__("holodeck", manifest_path)


#: Deprecated alias — :class:`HolodeckSource` is the canonical name now
#: (the source field is ``"holodeck"``; the dataset family is the
#: MolmoSpaces collection of Holodeck scenes). Will be removed once
#: external consumers migrate.
MolmoSpacesSource = HolodeckSource


# ── factory ──────────────────────────────────────────────────────────


#: Source-name → no-arg factory map. The factories are the per-dataset
#: subclass constructors; storing them as ``Callable[[], …]`` instead
#: of ``type[…]`` lets type-checkers see the no-arg call shape (each
#: subclass's ``__init__`` supplies the manifest-path default).
_SCENE_SOURCE_REGISTRY: dict[str, Callable[[], "ManifestSceneSource"]] = {
    "replicacad": ReplicaCADSource,
    "procthor": ProcTHORSource,
    "ithor": IThorSource,
    "robothor": RoboTHORSource,
    "architecthor": ArchitecTHORSource,
    "robocasa": RoboCasaSource,
    "holodeck": HolodeckSource,
}


def get_source(source_name: str) -> ManifestSceneSource:
    """Return a default-configured source for the given ``source_name``.

    The reverse lookup of :attr:`SceneSpec.source`. Useful when a
    caller has a spec and wants the source that produced it (to query
    metadata, sample a sibling scene, etc.).

    Raises :class:`KeyError` for unknown ``source_name``.
    """
    try:
        factory = _SCENE_SOURCE_REGISTRY[source_name]
    except KeyError:
        raise KeyError(
            f"get_source: unknown source_name {source_name!r}. Known: "
            f"{sorted(_SCENE_SOURCE_REGISTRY)!r}"
        ) from None
    return factory()


def register_source(
    source_name: str,
    factory: Callable[[], "ManifestSceneSource"],
    *,
    overwrite: bool = False,
) -> None:
    """Register a custom :class:`ManifestSceneSource` subclass with
    :func:`get_source`.

    Use this from outside-the-tree code that wants to add a new scene
    dataset without forking the registry.

    Parameters
    ----------
    source_name
        Identifier consumers will pass to :func:`get_source`. Must
        match the ``source`` field of the :class:`SceneSpec`\\ s the
        registered source produces.
    factory
        Zero-argument callable returning a configured
        :class:`ManifestSceneSource` instance — typically the
        subclass itself when its ``__init__`` supplies a default
        ``manifest_path``.
    overwrite
        If False (default) and ``source_name`` is already registered,
        raise :class:`KeyError` — collisions are usually a bug.
        Pass True to deliberately replace an existing entry (e.g. in
        a test).

    Raises
    ------
    ValueError
        If ``source_name`` is empty.
    KeyError
        If ``source_name`` is already registered and
        ``overwrite=False``.
    """
    if not source_name:
        raise ValueError("register_source: source_name must be non-empty")
    if source_name in _SCENE_SOURCE_REGISTRY and not overwrite:
        raise KeyError(
            f"register_source: source_name {source_name!r} is already "
            f"registered. Pass overwrite=True to replace deliberately."
        )
    _SCENE_SOURCE_REGISTRY[source_name] = factory


def unregister_source(source_name: str) -> None:
    """Remove a previously-registered source.

    Useful in test teardown to undo a :func:`register_source` call.
    Raises :class:`KeyError` if the source was not registered.
    """
    try:
        del _SCENE_SOURCE_REGISTRY[source_name]
    except KeyError:
        raise KeyError(
            f"unregister_source: source_name {source_name!r} is not registered"
        ) from None


__all__ = [
    "MANIFEST_DIR",
    "ManifestSceneSource",
    "ReplicaCADSource",
    "ProcTHORSource",
    "IThorSource",
    "RoboTHORSource",
    "ArchitecTHORSource",
    "RoboCasaSource",
    "HolodeckSource",
    "MolmoSpacesSource",  # deprecated alias
    "get_source",
    "register_source",
    "unregister_source",
]
