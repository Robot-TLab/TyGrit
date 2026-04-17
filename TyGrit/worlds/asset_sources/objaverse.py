"""Objaverse — ~10M mesh assets (ODC-By 1.0).

Objaverse is a *source dataset* rather than a sim-ready asset pack:

* Meshes are raw glTF / GLB files with no physics proxies, no
  collision meshes, no canonical orientation, and no unit scale.
* Per-object licenses vary; the dataset-level licence is ODC-By 1.0.
* Any real manipulation use requires a **curated subset** (bounding-
  box normalisation, VHACD/COACD collision mesh generation,
  category filtering). That curation is outside this source.

TyGrit treats Objaverse as a manifest-fed source: the
``TyGrit.worlds.generators.objaverse`` script produces a curated
manifest of :class:`ObjectSpec` entries with ``mesh_path`` pointing at
downloaded .glb files and per-object metadata (bbox, scale, category)
embedded in the spec. This source just loads that manifest and
enumerates it.

Consumers that need a scale normalisation cache / VHACD pass should
add that to the generator, not the source — the source stays thin.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from TyGrit.types.worlds import ObjectSpec, SceneSpec
from TyGrit.worlds.asset_sources.base import AssetRequest


class ObjaverseSource:
    """Object source backed by a manifest of curated Objaverse meshes.

    Parameters
    ----------
    manifest_path
        JSON file with a top-level ``"objects"`` array of objects
        serialised as ``{"name": …, "mesh_path": …, "scale": [x,y,z],
        "fix_base": bool}``. The generator at
        ``TyGrit.worlds.generators.objaverse`` produces this layout.
    """

    source_name: str = "objaverse"

    def __init__(
        self,
        manifest_path: str | Path | None = None,
    ) -> None:
        # Default mirrors the path
        # ``TyGrit.worlds.generators.objaverse`` writes to. Anchored at
        # the repo root via Path(__file__) so a source constructed
        # from any cwd resolves correctly.
        if manifest_path is None:
            project_root = Path(__file__).resolve().parents[3]
            manifest_path = (
                project_root / "resources" / "worlds" / "objects" / "objaverse.json"
            )
        self._manifest_path = Path(manifest_path)
        self._objects_cache: dict[str, ObjectSpec] | None = None

    @property
    def manifest_path(self) -> Path:
        return self._manifest_path

    def _ensure_loaded(self) -> dict[str, ObjectSpec]:
        if self._objects_cache is not None:
            return self._objects_cache
        if not self._manifest_path.exists():
            raise FileNotFoundError(
                f"ObjaverseSource: manifest not found at {self._manifest_path}. "
                f"Run `pixi run -e world generate-objaverse-objects` to produce it."
            )
        with self._manifest_path.open("r") as f:
            blob = json.load(f)
        out: dict[str, ObjectSpec] = {}
        for entry in blob.get("objects", []):
            name = entry["name"]
            mesh = entry["mesh_path"]
            scale = entry.get("scale", [1.0, 1.0, 1.0])
            fix_base = entry.get("fix_base", False)
            out[name] = ObjectSpec(
                name=name,
                mesh_path=mesh,
                scale=tuple(scale),
                fix_base=fix_base,
                is_articulated=False,
            )
        self._objects_cache = out
        return out

    # ── AssetSource: enumeration ──────────────────────────────────────

    def list_scene_ids(self, *, split: str | None = None) -> tuple[str, ...]:
        return ()

    def list_object_ids(self, *, split: str | None = None) -> tuple[str, ...]:
        if split is not None:
            raise ValueError(
                f"ObjaverseSource: split={split!r} not supported; curated "
                f"train/val splits would belong in the generator-produced manifest."
            )
        return tuple(sorted(self._ensure_loaded()))

    # ── AssetSource: lookup ───────────────────────────────────────────

    def get_scene(
        self, scene_id: str, *, request: AssetRequest | None = None
    ) -> SceneSpec:
        raise NotImplementedError("ObjaverseSource is object-only")

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
        objects = self._ensure_loaded()
        if object_id not in objects:
            raise KeyError(
                f"ObjaverseSource: unknown object_id {object_id!r}. "
                f"First 5 known: {sorted(objects)[:5]!r}"
            )
        template = objects[object_id]
        # The manifest spec is the canonical template; callers usually
        # want to place it with a fresh name + pose, so we rebuild the
        # spec with the caller-supplied placement and the template's
        # asset data.
        effective_scale = scale if scale != (1.0, 1.0, 1.0) else template.scale
        return ObjectSpec(
            name=name,
            mesh_path=template.mesh_path,
            position=position,
            orientation_xyzw=orientation_xyzw,
            scale=effective_scale,
            fix_base=fix_base,
            is_articulated=False,
        )

    # ── AssetSource: sampling ─────────────────────────────────────────

    def sample_scene_id(self, *, seed: int, split: str | None = None) -> str:
        raise NotImplementedError("ObjaverseSource is object-only")

    def sample_object_id(self, *, seed: int, split: str | None = None) -> str:
        ids = self.list_object_ids(split=split)
        if not ids:
            raise RuntimeError("ObjaverseSource: object pool is empty")
        rng = random.Random(int(seed))
        return rng.choice(ids)


__all__ = ["ObjaverseSource"]
