"""Per-dataset asset sources.

An :class:`AssetSource` encapsulates *one* asset dataset (MolmoSpaces,
AI2-THOR / ProcTHOR / iTHOR / RoboTHOR / ArchitecTHOR, ReplicaCAD,
RoboCasa, ManiSkill YCB, Objaverse). It produces canonical
:class:`~TyGrit.types.worlds.SceneSpec` / :class:`~TyGrit.types.worlds.ObjectSpec`
values that the sim-agnostic handlers in :mod:`TyGrit.sim` consume.

Why this layer exists
---------------------

Before this refactor, per-dataset knowledge was split across:

* ``TyGrit/worlds/generators/<dataset>.py`` — one-shot scripts that
  emit a JSON manifest.
* Dispatch branches inside ``worlds/backends/<sim>.py`` adapters —
  "if spec.source == 'ycb' then use this ManiSkill builder".

That worked, but meant adding a new dataset required editing every
backend *and* its own generator, and there was no single class you
could point to and say "this is ManiSkill YCB". :class:`AssetSource`
gives each dataset one class that knows:

* the scene / object ids it provides,
* how to turn an id into an ``ObjectSpec`` / ``SceneSpec``,
* licensing / source URL / cache paths.

Backends consume the resulting specs; they don't need to know which
dataset each spec came from beyond the ``spec.source`` field used for
per-sim dispatch.

Usage
-----

>>> from TyGrit.worlds.asset_sources import ReplicaCADSource
>>> src = ReplicaCADSource()
>>> scene_id = src.sample_scene_id(seed=0)
>>> spec = src.get_scene(scene_id)
"""

from TyGrit.worlds.asset_sources.base import (
    SOURCE_SIM_COMPATIBILITY,
    AssetFormat,
    AssetRequest,
    AssetSource,
    compatible_sims,
)
from TyGrit.worlds.asset_sources.manifest_scene import (
    MolmoSpacesSource,  # deprecated alias for HolodeckSource
)
from TyGrit.worlds.asset_sources.manifest_scene import (
    MANIFEST_DIR,
    ArchitecTHORSource,
    HolodeckSource,
    IThorSource,
    ManifestSceneSource,
    ProcTHORSource,
    ReplicaCADSource,
    RoboCasaSource,
    RoboTHORSource,
    get_source,
    register_source,
    unregister_source,
)
from TyGrit.worlds.asset_sources.objaverse import ObjaverseSource
from TyGrit.worlds.asset_sources.ycb import YCB_FETCH_GRASPABLE, ManiSkillYCBSource

__all__ = [
    # Protocol + request
    "AssetFormat",
    "AssetRequest",
    "AssetSource",
    "SOURCE_SIM_COMPATIBILITY",
    "compatible_sims",
    # Manifest-backed scene sources
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
    # Object sources
    "ManiSkillYCBSource",
    "YCB_FETCH_GRASPABLE",
    "ObjaverseSource",
]
