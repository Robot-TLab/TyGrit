"""Per-simulator handlers — one file per sim, robot-agnostic.

:class:`~TyGrit.sim.base.SimHandler` is the common protocol every per-sim
module (:mod:`TyGrit.sim.maniskill`, :mod:`TyGrit.sim.genesis`,
:mod:`TyGrit.sim.isaac_sim`) implements. A :class:`SimHandler` takes a
:class:`~TyGrit.types.robots.RobotCfg` plus a scene pool at construction
time, loads the robot + scene into its simulator, and exposes the
uniform ``step`` / ``reset`` / ``get_qpos`` / ``get_camera`` surface
consumed by the robot-specific cores in :mod:`TyGrit.envs`.

:class:`~TyGrit.sim.base.SimHandlerVec` is the batched counterpart used
when ``num_envs > 1`` — torch tensors on axis-0 ``num_envs`` batch.

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

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from TyGrit.sim.base import SimHandler, SimHandlerVec

if TYPE_CHECKING:
    from TyGrit.types.robots import RobotCfg
    from TyGrit.types.worlds import SceneSpec


def create_sim_handler(
    sim_name: str,
    robot_cfg: "RobotCfg",
    scenes: "Sequence[SceneSpec]",
    *,
    num_envs: int = 1,
    initial_scene_idx: int = 0,
    **sim_opts: Any,
) -> SimHandler | SimHandlerVec:
    """Dispatch on ``sim_name`` → the appropriate concrete handler.

    ``num_envs == 1`` returns a scalar :class:`SimHandler`; ``num_envs >
    1`` returns a :class:`SimHandlerVec`. The ``sim_opts`` kwargs are
    forwarded verbatim to the selected handler's constructor — callers
    pass per-sim settings (``obs_mode``, ``control_mode``,
    ``render_mode`` for ManiSkill; ``show_viewer`` for Genesis; etc.)
    via :attr:`FetchEnvConfig.sim_opts`.

    Parameters
    ----------
    sim_name
        One of ``"maniskill"``, ``"genesis"``, ``"isaac_sim"``.
    robot_cfg
        Robot descriptor; every handler validates its own required
        fields (``robot_cfg.sim_uids``, ``robot_cfg.urdf_path``,
        ``robot_cfg.usd_path``) and raises on missing data.
    scenes
        Scene pool to construct the handler against; see the
        :class:`SimHandler` protocol for the per-reset contract.

    Raises
    ------
    ValueError
        Unknown ``sim_name``.
    NotImplementedError
        ``num_envs > 1`` for a sim whose :class:`SimHandlerVec` has
        not been written yet. The per-sim vec handlers land with
        §7.5 in ``prompts/multi_sim_mobile_manip_refactor.md``;
        callers meanwhile can either drop to ``num_envs == 1`` or
        wait for the follow-up iteration.
    """
    if sim_name == "maniskill":
        if num_envs > 1:
            from TyGrit.sim.maniskill import ManiSkillSimHandlerVec

            return ManiSkillSimHandlerVec(
                robot_cfg,
                scenes,
                initial_scene_idx=initial_scene_idx,
                num_envs=num_envs,
                **sim_opts,
            )
        from TyGrit.sim.maniskill import ManiSkillSimHandler

        return ManiSkillSimHandler(
            robot_cfg,
            scenes,
            initial_scene_idx=initial_scene_idx,
            **sim_opts,
        )

    if sim_name == "genesis":
        if num_envs > 1:
            from TyGrit.sim.genesis import GenesisSimHandlerVec

            return GenesisSimHandlerVec(
                robot_cfg,
                scenes,
                initial_scene_idx=initial_scene_idx,
                num_envs=num_envs,
                **sim_opts,
            )
        from TyGrit.sim.genesis import GenesisSimHandler

        return GenesisSimHandler(
            robot_cfg,
            scenes,
            initial_scene_idx=initial_scene_idx,
            **sim_opts,
        )

    if sim_name == "isaac_sim":
        if num_envs > 1:
            from TyGrit.sim.isaac_sim import IsaacSimSimHandlerVec

            return IsaacSimSimHandlerVec(
                robot_cfg,
                scenes,
                initial_scene_idx=initial_scene_idx,
                num_envs=num_envs,
                **sim_opts,
            )
        from TyGrit.sim.isaac_sim import IsaacSimSimHandler

        return IsaacSimSimHandler(
            robot_cfg,
            scenes,
            initial_scene_idx=initial_scene_idx,
            **sim_opts,
        )

    raise ValueError(
        f"create_sim_handler: unknown sim_name {sim_name!r}; expected one of "
        f"'maniskill', 'genesis', 'isaac_sim'"
    )


__all__ = ["SimHandler", "SimHandlerVec", "create_sim_handler"]
