"""Simulator-facing protocol for the Fetch robot wrapper.

:class:`FetchSimBackend` defines the surface area that
:class:`~TyGrit.envs.fetch.core.FetchRobotCore` needs from any
simulator (ManiSkill, Genesis, ROS, …) to drive a Fetch robot. The
goal of the split is that Fetch-specific *logic* — joint indexing,
base-pose calibration, action assembly, head PD controller, MPC-based
trajectory execution — lives once in ``FetchRobotCore`` and reads the
sim through this narrow interface, while sim-specific code (env
construction, observation dict parsing, action plumbing) lives in a
backend implementation.

Backends own:
    * The underlying simulator handle (``gym.make`` env in the
      ManiSkill case, ``gs.Scene`` in Genesis, ROS topics on hardware).
    * Cached observations from the most recent ``step()`` /
      ``reset_to_idx()`` so the per-step queries below do NOT trigger
      fresh sim queries — same caching pattern the v1 single-env
      :class:`ManiSkillFetchRobot` already uses internally.
    * Construction-time setup: action slice computation, joint-name
      lookup, intrinsics caching.

The protocol is intentionally single-env (numpy, scalar-coded). The
batched/vectorised path used by RL training
(:class:`~TyGrit.envs.fetch.maniskill_vec.ManiSkillFetchRobotVec`)
does NOT yet implement this protocol — its dict-returning ``step`` /
``reset`` and torch-tensor types make it a separate kind of object.
A future ``FetchSimBackendBatched`` could parallel this one if the
duplication grows past a threshold worth abstracting.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt


class FetchSimBackend(Protocol):
    """Simulator-side surface area :class:`FetchRobotCore` depends on.

    All methods are synchronous (no background threads). All numpy.
    Implementations should cache the most recent observation so the
    per-step query methods don't trigger a fresh sim read.
    """

    # ── construction-time metadata ──────────────────────────────────────

    @property
    def num_envs(self) -> int:
        """Number of parallel envs the backend manages.

        Single-env backends return 1. The batched path is a future
        protocol; see module docstring.
        """
        ...

    @property
    def intrinsics(self) -> npt.NDArray[np.float64]:
        """Static 3×3 camera intrinsics for the configured head camera.

        The Fetch wrapper currently exposes one camera (``"head"``).
        Multi-camera support would extend this to a dict keyed by
        camera_id, mirroring :meth:`parse_camera`.
        """
        ...

    @property
    def total_action_dim(self) -> int:
        """Length of the low-level action vector the backend's
        :meth:`step` accepts."""
        ...

    @property
    def action_slices(self) -> dict[str, slice]:
        """Per-controller slices into the low-level action vector.

        Keys are a subset of ``{"arm", "gripper", "body", "base"}``.
        ``FetchRobotCore`` uses these to assemble the action vector
        from MPC output + head PD output + gripper target without
        knowing the underlying controller layout.
        """
        ...

    @property
    def joint_name_to_idx(self) -> dict[str, int]:
        """Map joint name → index into the qpos vector."""
        ...

    @property
    def base_joint_names(self) -> tuple[str, str, str]:
        """The (x, y, theta) joint names for the holonomic base.

        ManiSkill's Fetch agent exposes this on
        ``agent.base_joint_names``. Other backends should expose the
        equivalent triplet so :class:`FetchRobotCore` can compute the
        qpos→world base-pose offset at construction / post-reset.
        """
        ...

    # ── per-step queries ────────────────────────────────────────────────

    def get_qpos(self) -> npt.NDArray[np.float64]:
        """Return the most recent qpos vector (from cache, no sim query).

        Length matches the underlying robot model
        (``len(joint_name_to_idx)``). Indexing is via
        :attr:`joint_name_to_idx`.
        """
        ...

    def get_base_link_world_pose(self) -> npt.NDArray[np.float64]:
        """Return the 4×4 SE(3) transform of ``base_link`` in world frame.

        Used once at construction and after every reset to recalibrate
        the qpos→world offset for the holonomic base joints. Reading
        directly from the sim is fine since this is not a per-step
        query.
        """
        ...

    def parse_camera(self, camera_id: str) -> tuple[
        npt.NDArray[np.uint8],
        npt.NDArray[np.float32],
        npt.NDArray[np.int32] | None,
    ]:
        """Return ``(rgb, depth_m, segmentation)`` for the named camera.

        ``rgb`` is ``(H, W, 3)`` uint8. ``depth_m`` is ``(H, W)``
        float32 in metres. ``segmentation`` is ``(H, W)`` int32 if
        the backend was configured with a segmentation channel,
        ``None`` otherwise.
        """
        ...

    # ── mutations ───────────────────────────────────────────────────────

    def step(self, action: npt.NDArray[np.float32]) -> None:
        """Apply ``action`` (length :attr:`total_action_dim`) for one
        sim step.

        The backend updates its internal observation cache so the
        subsequent :meth:`get_qpos` / :meth:`parse_camera` calls
        return data from the post-step state.
        """
        ...

    def reset_to_idx(self, idx: int, seed: int | None = None) -> None:
        """Reconfigure the env to scene index ``idx`` and reset.

        ``idx`` is an index into the spec pool the backend was
        constructed against (``FetchEnvConfig.scene_sampler``'s
        manifest). Updates the obs cache in the same way as
        :meth:`step`.
        """
        ...

    def set_base_pose(self, x: float, y: float, theta: float, env_idx: int = 0) -> None:
        """Teleport the Fetch base to ``(x, y, theta)`` in world frame.

        Used by spawn randomisation in :meth:`FetchRobotCore.reset`.
        ``env_idx`` is for backends with parallel envs; single-env
        backends accept only ``0``.
        """
        ...

    # ── world hooks ─────────────────────────────────────────────────────

    def get_navigable_positions(self) -> list:
        """Return per-env navmesh objects (e.g. trimesh meshes), or
        ``[]`` when the active scene has no navmesh.

        :class:`FetchRobotCore` samples a free-space spawn position
        from the vertex set in :meth:`reset`. When the list is empty
        the core falls back to a fixed default pose.
        """
        ...

    # ── lifecycle ───────────────────────────────────────────────────────

    def render(self) -> None:
        """Render one frame to whatever display the backend was
        configured with. No-op when render mode is ``None``."""
        ...

    def close(self) -> None:
        """Tear down the underlying sim env."""
        ...
