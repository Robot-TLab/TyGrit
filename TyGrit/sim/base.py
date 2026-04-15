"""The :class:`SimHandler` Protocol — one per simulator, robot-agnostic.

Every handler takes a :class:`~TyGrit.types.robots.RobotCfg` plus a
scene pool at construction and exposes a uniform sensing / actuation /
reset surface. The concrete implementations live in sibling modules:

* :mod:`TyGrit.sim.maniskill` — :class:`ManiSkillSimHandler`
* :mod:`TyGrit.sim.genesis` — :class:`GenesisSimHandler`
* :mod:`TyGrit.sim.isaac_sim` — :class:`IsaacSimSimHandler`

Design
------

**Robot-agnostic.** No method mentions Fetch-specific concepts. The
handler reads everything it needs about the robot from
``self.robot_cfg`` (joint names, controllers, cameras, URDF path,
sim-registered uid). Adding a second robot is a new ``RobotCfg`` value
+ a thin robot-specific core in :mod:`TyGrit.envs`; no changes here.

**Narrow, single-env at the protocol level.** All methods operate on
numpy (not torch) and work on one env or a batch indexed by
``env_idx``. Vectorised training wrappers that return batched torch
tensors compose *around* this protocol rather than extending it — the
cost of two slightly different shapes is lower than the cost of a
fat protocol that makes every single-env caller check ``num_envs``.

**Observation reads.** Per-step queries
(:meth:`get_qpos` / :meth:`get_link_pose` / :meth:`get_camera`) read
the simulator's current state — either from a cached observation
dict (ManiSkill's ``obs`` from the last ``step``) or from a live
state accessor (Genesis's ``entity.get_dofs_position()``, which is
itself a cheap lookup into resident buffers). The contract is
*post-step consistency*: after :meth:`apply_action` returns, all
per-step queries report the post-step state; after
:meth:`reset_to_scene_idx` returns, they report the reset state.
Callers must not interleave queries with external sim mutations.

**Camera frame convention.** All cameras in :attr:`RobotCfg.cameras`
follow the project quaternion convention (``[x, y, z, w]``); the
handler converts to its sim's native convention at the boundary when
spawning the camera and when reading back intrinsics. Callers see
metres / radians uniformly.

**Failure mode.** Methods raise on programming errors (unknown camera
id, unknown link name, mismatched action length) and return Result
types from :mod:`TyGrit.types.results` for domain-level failure. No
silent no-ops. See the project CLAUDE.md Rule 3.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from TyGrit.types.robots import RobotCfg


@runtime_checkable
class SimHandler(Protocol):
    """Uniform per-simulator handler. See module docstring."""

    # ── construction-time metadata ─────────────────────────────────────

    @property
    def robot_cfg(self) -> RobotCfg:
        """The robot this handler was constructed with.

        Callers read this to discover joint names, camera ids,
        controller layout, and so on — instead of the handler itself
        exposing robot-specific properties.
        """
        ...

    @property
    def num_envs(self) -> int:
        """Number of parallel envs the handler manages. ``1`` for the
        single-env path."""
        ...

    @property
    def total_action_dim(self) -> int:
        """Length of the flat action vector :meth:`apply_action` accepts.

        Equal to ``sum(cfg.action_dim for cfg in robot_cfg.actuators.values())``
        when the sim's native layout matches ``controller_order``;
        handlers that reorder controllers internally override this.
        """
        ...

    @property
    def action_slices(self) -> Mapping[str, slice]:
        """Per-controller slices into the flat action vector.

        Keys are :attr:`RobotCfg.controller_order` entries (``"arm"``,
        ``"gripper"``, ``"body"``, ``"base"`` for Fetch). Robot-specific
        cores use these to assemble a flat vector from per-controller
        sub-actions without knowing the underlying layout.
        """
        ...

    @property
    def joint_name_to_idx(self) -> Mapping[str, int]:
        """Map joint name → index into the qpos vector."""
        ...

    # ── per-step queries (from the obs cache, no sim round-trip) ──────

    def get_qpos(self, env_idx: int = 0) -> npt.NDArray[np.float64]:
        """Cached qpos vector for env ``env_idx``.

        Length matches ``len(joint_name_to_idx)``. Indexing is via
        :attr:`joint_name_to_idx`.
        """
        ...

    def get_link_pose(
        self, link_name: str, env_idx: int = 0
    ) -> npt.NDArray[np.float64]:
        """4×4 SE(3) world transform of the given link for env ``env_idx``.

        Used e.g. for the qpos↔world base-pose offset calibration.
        Raises :class:`KeyError` if ``link_name`` is not a link of the
        loaded robot.
        """
        ...

    def get_camera(self, camera_id: str, env_idx: int = 0) -> tuple[
        npt.NDArray[np.uint8],
        npt.NDArray[np.float32],
        npt.NDArray[np.int32] | None,
    ]:
        """Return ``(rgb, depth_m, segmentation)`` from camera ``camera_id``.

        * ``rgb`` is ``(H, W, 3)`` uint8.
        * ``depth_m`` is ``(H, W)`` float32 in metres.
        * ``segmentation`` is ``(H, W)`` int32 if the handler was
          configured with a segmentation channel, ``None`` otherwise.

        Raises :class:`KeyError` if ``camera_id`` is not in
        :attr:`RobotCfg.cameras`.
        """
        ...

    def get_intrinsics(self, camera_id: str) -> npt.NDArray[np.float64]:
        """Return the static 3×3 pinhole intrinsics for ``camera_id``.

        Intrinsics don't change after construction, so callers often
        cache the result. Raises :class:`KeyError` for unknown
        ``camera_id``.
        """
        ...

    # ── mutations ──────────────────────────────────────────────────────

    def apply_action(self, action: npt.NDArray[np.float32]) -> None:
        """Apply the flat action vector and advance one sim step.

        ``action`` has length :attr:`total_action_dim`. The handler
        updates its observation cache before returning so that the
        subsequent per-step queries read from post-step state.
        """
        ...

    def reset_to_scene_idx(self, idx: int, *, seed: int | None = None) -> None:
        """Switch to scene index ``idx`` and reset the robot.

        ``idx`` is an index into the scene pool the handler was
        constructed against. The handler rebuilds / re-randomises the
        sim as appropriate, re-homes the robot, and refreshes the
        observation cache.
        """
        ...

    def set_joint_positions(
        self,
        positions: Mapping[str, float],
        *,
        env_idx: int = 0,
    ) -> None:
        """Teleport specific joints to the given positions.

        Used for base-pose reset and for :meth:`control_gripper`-style
        instantaneous-grip tricks. Handlers may enforce sim-specific
        constraints (e.g. refuse to teleport an articulated drawer
        mid-action); those failures raise rather than silently skip.
        """
        ...

    def set_base_pose(
        self,
        x: float,
        y: float,
        theta: float,
        *,
        env_idx: int = 0,
    ) -> None:
        """Teleport a mobile robot's base to ``(x, y, θ)`` in world frame.

        Legal only when ``robot_cfg.is_mobile`` is ``True``.
        Fixed-base robots raise :class:`RuntimeError`. Equivalent to
        ``set_joint_positions`` on the base joints but expressed in
        world coordinates (handlers internally compute the qpos offset
        if the sim encodes the base as a floating joint).
        """
        ...

    # ── world hooks ────────────────────────────────────────────────────

    def get_navigable_positions(self) -> list:
        """Per-env navmesh objects.

        Length equals :attr:`num_envs`; each entry is a navmesh handle
        (typically a :class:`trimesh.Trimesh`) or ``None`` when the
        active scene has no navmesh. Robot-specific spawn
        randomisation samples vertices from here.
        """
        ...

    # ── lifecycle ──────────────────────────────────────────────────────

    def render(self) -> None:
        """Render one frame to the display the handler was configured
        with. No-op when render mode is ``None``."""
        ...

    def close(self) -> None:
        """Tear down the underlying sim env and release GPU resources."""
        ...


__all__ = ["SimHandler"]
