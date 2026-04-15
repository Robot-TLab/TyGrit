"""Per-simulator handlers — one file per sim, robot-agnostic.

:class:`~TyGrit.sim.base.SimHandler` is the common protocol every per-sim
module (:mod:`TyGrit.sim.maniskill`, :mod:`TyGrit.sim.genesis`,
:mod:`TyGrit.sim.isaac_sim`) implements. A :class:`SimHandler` takes a
:class:`~TyGrit.types.robots.RobotCfg` plus a scene pool at construction
time, loads the robot + scene into its simulator, and exposes the
uniform ``step`` / ``reset`` / ``get_qpos`` / ``get_camera`` surface
consumed by the robot-specific cores in :mod:`TyGrit.envs`.

This package replaces the older split of per-robot per-sim files
(``envs/fetch/maniskill.py`` + ``worlds/backends/maniskill.py``) with
one robot-agnostic handler per simulator — so adding a new robot (e.g.
``AUTOLIFE_CFG``) requires **zero** changes here, and adding a new
simulator requires a single new module in this package.

Each concrete handler module imports its own sim SDK
(``mani_skill``, ``genesis``, ``isaacsim``/``isaaclab``). The
SDK imports are placed inside functions or guarded by
``if TYPE_CHECKING:`` so this package is importable from the pure-Python
default pixi env for type-checking and tests that only need the
:class:`SimHandler` Protocol itself.
"""

from TyGrit.sim.base import SimHandler

__all__ = ["SimHandler"]
