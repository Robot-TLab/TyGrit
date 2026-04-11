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
from TyGrit.worlds.sampler import SceneSampler, create_sampler

# NOTE: TyGrit.worlds.maniskill imports mani_skill and is therefore only
# available in the `world` / `rl` / `thirdparty` pixi envs, not the
# default env. We intentionally do NOT re-export its symbols here — a
# top-level import of TyGrit.worlds would then fail in the default env
# and break pure-Python tests in test_worlds_manifest.py /
# test_worlds_sampler.py that only touch the spec + loader layer. Code
# that needs the ManiSkill adapter should import it directly via
# ``from TyGrit.worlds.maniskill import bind_specs, build_world``.

__all__ = [
    "MANIFEST_VERSION",
    "SceneSampler",
    "create_sampler",
    "load_manifest",
    "save_manifest",
]
