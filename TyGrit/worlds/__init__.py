"""World / scene construction logic.

This package holds the *logic* for constructing and sampling simulated
worlds from :class:`~TyGrit.types.worlds.SceneSpec` definitions. The
spec *types* themselves live in :mod:`TyGrit.types.worlds` (per repo
convention: pure dataclasses go in ``TyGrit/types/``).

Submodules
----------
Format-level (no simulator imports):

* :mod:`TyGrit.worlds.manifest` — load/save JSON manifests of SceneSpecs.
* :mod:`TyGrit.worlds.sampler` — seeded per-reset sampling (Step 3).

Per-simulator adapters (each imports only its own backend):

* :mod:`TyGrit.worlds.maniskill` — ManiSkill ``SceneBuilder`` adapter (Step 4).
* :mod:`TyGrit.worlds.genesis` — Genesis ``morph`` adapter (Step 12).

See :mod:`TyGrit.types.worlds` for the ``SceneSpec`` / ``ObjectSpec``
dataclass definitions.
"""

from __future__ import annotations

from TyGrit.worlds.manifest import MANIFEST_VERSION, load_manifest, save_manifest

__all__ = ["MANIFEST_VERSION", "load_manifest", "save_manifest"]
