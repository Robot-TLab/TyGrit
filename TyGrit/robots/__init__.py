"""Robot descriptors (pure-data :class:`RobotCfg` values). Sim-agnostic.

:data:`FETCH_CFG` is the canonical Fetch descriptor. Every
:mod:`TyGrit.sim` handler consumes it. Adding a new robot is a new
package under this directory exporting its own ``<ROBOT>_CFG``.
"""

from TyGrit.robots.fetch import FETCH_CFG

__all__ = ["FETCH_CFG"]
