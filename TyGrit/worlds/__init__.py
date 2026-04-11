"""World / scene construction logic.

This package holds the *logic* for constructing and sampling simulated
worlds from :class:`~TyGrit.types.worlds.SceneSpec` definitions. The
spec *types* themselves live in :mod:`TyGrit.types.worlds` (per repo
convention: pure dataclasses go in ``TyGrit/types/``).

Layout
------
Sim-agnostic modules at this level (importable in the default pixi env):

* :mod:`TyGrit.worlds.manifest` — load/save JSON manifests of SceneSpecs.
* :mod:`TyGrit.worlds.sampler` — seeded per-reset sampling (Step 3).

Per-simulator adapters live under :mod:`TyGrit.worlds.backends`, each
gated on its own pixi env:

* :mod:`TyGrit.worlds.backends.maniskill` — ManiSkill ``SceneBuilder``
  adapter. Needs the ``world`` / ``rl`` / ``thirdparty`` pixi env.
* :mod:`TyGrit.worlds.backends.genesis` — Genesis ``morph`` adapter
  (Step 12). Will need a ``genesis`` env.
* :mod:`TyGrit.worlds.backends.isaac_sim` — Isaac Lab adapter (future).

See :mod:`TyGrit.types.worlds` for the ``SceneSpec`` / ``ObjectSpec``
dataclass definitions.
"""

from __future__ import annotations

from TyGrit.worlds.manifest import MANIFEST_VERSION, load_manifest, save_manifest
from TyGrit.worlds.sampler import SceneSampler, create_sampler

# NOTE: Per-backend adapters live under TyGrit.worlds.backends.* and
# each imports its own simulator package. We intentionally do NOT
# re-export them here because a top-level import of TyGrit.worlds
# would then fail in pixi envs that don't have the chosen backend,
# breaking pure-Python tests in test_worlds_manifest.py /
# test_worlds_sampler.py that only need the spec + loader layer.
# Callers import adapters via the full path, e.g.
# ``from TyGrit.worlds.backends.maniskill import bind_specs, build_world``.

__all__ = [
    "MANIFEST_VERSION",
    "SceneSampler",
    "create_sampler",
    "load_manifest",
    "save_manifest",
]
