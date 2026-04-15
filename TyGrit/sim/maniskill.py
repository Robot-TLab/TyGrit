"""ManiSkill handler — the robot-agnostic :class:`SimHandler` for ManiSkill3.

This module is the unified ManiSkill adapter: it builds the
``SceneManipulation-v1`` ``gym.make`` env with the right scene builder
bound (:mod:`TyGrit.worlds.backends.maniskill`
:class:`SpecBackedSceneBuilder`), loads *any* robot described by a
:class:`~TyGrit.types.robots.RobotCfg`, and exposes the uniform
:class:`~TyGrit.sim.base.SimHandler` surface consumed by robot cores
in :mod:`TyGrit.envs`.

Design
------

**Robot-agnostic.** Takes a :class:`RobotCfg` at construction and
reads its :attr:`sim_uids["maniskill"]` to pick the ManiSkill agent,
its :attr:`cameras` to drive the image-query API, and its
:attr:`controller_order` to decode the action vector. Adding a
second robot (Fetch + AutoLife) is just a new ``RobotCfg`` instance;
no code change here.

**Single-env.** The vectorised path lives in
:class:`TyGrit.envs.fetch.maniskill_vec.ManiSkillFetchRobotVec` for
now, with its own torch-tensor semantics that don't fit the numpy
``SimHandler`` surface. When that refactor lands we'll add a
companion ``ManiSkillSimHandlerVec`` and share the construction
helpers.

**Observation cache.** The base class stores ``(obs, info)`` from the
last ``step`` / ``reset`` call; per-step queries read exclusively
from it. This preserves the v1 pattern where one ``step`` maps to one
set of numpy arrays, and ``get_camera("head")`` is a free dict
lookup instead of a second sim round-trip.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from mani_skill.utils.structs.types import GPUMemoryConfig, SceneConfig, SimConfig

from TyGrit.sim.maniskill_helpers import (
    build_action_slices,
    build_joint_name_to_idx,
    extract_intrinsics,
    make_scene_manipulation_env,
)
from TyGrit.types.robots import RobotCfg
from TyGrit.types.worlds import SceneSpec
from TyGrit.utils.depth import integer_depth_to_metres
from TyGrit.utils.tensor import to_numpy

#: Default ManiSkill ``SimConfig`` used when the caller does not supply
#: one. Values come from the single-env Fetch tuning in the legacy
#: ``ManiSkillFetchSimBackend`` — retained so the uniform handler
#: reproduces the same sim behaviour byte-for-byte.
DEFAULT_SIM_CONFIG = SimConfig(
    gpu_memory_config=GPUMemoryConfig(
        found_lost_pairs_capacity=2**25,
        max_rigid_patch_count=2**18,
    ),
    scene_config=SceneConfig(contact_offset=0.001),
)


class ManiSkillSimHandler:
    """Robot-agnostic :class:`~TyGrit.sim.base.SimHandler` for ManiSkill3.

    Parameters
    ----------
    robot_cfg
        The robot descriptor. ``robot_cfg.sim_uids["maniskill"]`` must
        name a ManiSkill-registered agent (``"fetch"`` today; future
        robots must ship their agent class in ManiSkill or via the
        asset-path fallback we'll add when needed).
    scenes
        The scene pool this handler will draw from at reset. Must be
        non-empty; a single-scene handler is legal (pool of one).
    initial_scene_idx
        Index into ``scenes`` to build at construction. Defaults to 0.
    camera_resolution
        ``(width, height)`` used for every camera in the robot's
        :attr:`RobotCfg.cameras`. ManiSkill wants one resolution per
        sensor_configs dict; per-camera resolution is not used today
        and would need a richer sensor_configs assembly.
    obs_mode, control_mode, render_mode
        Forwarded to ``gym.make``. Keep the defaults for ``rgb+depth+segmentation``
        + ``pd_joint_vel`` + ``None`` (headless) unless the caller has
        a specific reason.
    sim_config
        Optional ManiSkill ``SimConfig`` override. Falls back to
        :data:`DEFAULT_SIM_CONFIG`.
    """

    # ── construction ───────────────────────────────────────────────────

    def __init__(
        self,
        robot_cfg: RobotCfg,
        scenes: Sequence[SceneSpec],
        *,
        initial_scene_idx: int = 0,
        camera_resolution: tuple[int, int] = (128, 128),
        obs_mode: str = "rgb+depth+segmentation",
        control_mode: str = "pd_joint_vel",
        render_mode: str | None = None,
        sim_config: SimConfig | None = None,
    ) -> None:
        self._robot_cfg = robot_cfg
        if "maniskill" not in robot_cfg.sim_uids:
            raise ValueError(
                f"ManiSkillSimHandler: RobotCfg {robot_cfg.name!r} has no "
                f"sim_uids['maniskill'] entry. Known sim_uids: "
                f"{dict(robot_cfg.sim_uids)!r}"
            )
        self._scenes: tuple[SceneSpec, ...] = tuple(scenes)
        if len(self._scenes) == 0:
            raise ValueError(
                "ManiSkillSimHandler: scene pool is empty; pass at least one SceneSpec"
            )
        if not 0 <= initial_scene_idx < len(self._scenes):
            raise IndexError(
                f"ManiSkillSimHandler: initial_scene_idx {initial_scene_idx} "
                f"out of range for scene pool of size {len(self._scenes)}"
            )

        self._render_mode = render_mode

        # Build the env once up front. The ManiSkill agent class, the
        # scene builder, and the controllers are all baked in here and
        # carried across resets.
        self._env = make_scene_manipulation_env(
            robot_cfg,
            self._scenes,
            build_config_idxs=[initial_scene_idx],
            sim_config=sim_config or DEFAULT_SIM_CONFIG,
            obs_mode=obs_mode,
            control_mode=control_mode,
            render_mode=render_mode,
            camera_resolution=camera_resolution,
        )

        obs, info = self._env.reset(
            options=dict(reconfigure=True, build_config_idxs=[initial_scene_idx])
        )
        self._obs: dict = obs
        self._info: dict = info

        agent = self.env_agent
        self._joint_name_to_idx: Mapping[str, int] = MappingProxyType(
            build_joint_name_to_idx(agent)
        )
        slices, total_dim = build_action_slices(agent, robot_cfg)
        self._action_slices: Mapping[str, slice] = MappingProxyType(slices)
        self._total_action_dim = total_dim

        # Intrinsics are static after construction — cache once per camera.
        self._intrinsics: dict[str, npt.NDArray[np.float64]] = {}
        for cam in robot_cfg.cameras:
            sensor_id = _maniskill_sensor_id(cam)
            self._intrinsics[cam.camera_id] = extract_intrinsics(self._env, sensor_id)

    # ── SimHandler: metadata ───────────────────────────────────────────

    @property
    def robot_cfg(self) -> RobotCfg:
        return self._robot_cfg

    @property
    def num_envs(self) -> int:
        # Single-env handler by construction. Vec path is a separate class.
        return 1

    @property
    def total_action_dim(self) -> int:
        return self._total_action_dim

    @property
    def action_slices(self) -> Mapping[str, slice]:
        return self._action_slices

    @property
    def joint_name_to_idx(self) -> Mapping[str, int]:
        return self._joint_name_to_idx

    # ── SimHandler: per-step queries ───────────────────────────────────

    def get_qpos(self, env_idx: int = 0) -> npt.NDArray[np.float64]:
        if env_idx != 0:
            raise IndexError(
                f"ManiSkillSimHandler is single-env (num_envs=1); got env_idx={env_idx}"
            )
        qpos = self.env_agent.robot.get_qpos()
        return to_numpy(qpos).reshape(-1).astype(np.float64)

    def get_link_pose(
        self, link_name: str, env_idx: int = 0
    ) -> npt.NDArray[np.float64]:
        if env_idx != 0:
            raise IndexError(
                f"ManiSkillSimHandler is single-env (num_envs=1); got env_idx={env_idx}"
            )
        link = self._find_link(link_name)
        # ManiSkill / Sapien pose → 4×4 SE(3) via to_transformation_matrix.
        T = link.pose.to_transformation_matrix()
        return to_numpy(T).reshape(4, 4).astype(np.float64)

    def get_camera(self, camera_id: str, env_idx: int = 0) -> tuple[
        npt.NDArray[np.uint8],
        npt.NDArray[np.float32],
        npt.NDArray[np.int32] | None,
    ]:
        if env_idx != 0:
            raise IndexError(
                f"ManiSkillSimHandler is single-env (num_envs=1); got env_idx={env_idx}"
            )
        # Validate against the robot's camera list so a typo raises
        # immediately instead of quietly returning the wrong sensor.
        cam = self._robot_cfg.camera_by_id(camera_id)
        sensor_id = _maniskill_sensor_id(cam)
        sensor_data = self._obs["sensor_data"][sensor_id]

        rgb = to_numpy(sensor_data["rgb"])
        depth_raw = to_numpy(sensor_data["depth"])
        seg = sensor_data.get("segmentation")
        seg_np: npt.NDArray[np.int32] | None
        if seg is not None:
            seg_np = to_numpy(seg)
        else:
            seg_np = None

        # ManiSkill returns (N, H, W, C); unwrap the batch dim.
        if rgb.ndim == 4:
            rgb = rgb[0]
        if depth_raw.ndim == 4:
            depth_raw = depth_raw[0]
        if seg_np is not None and seg_np.ndim == 4:
            seg_np = seg_np[0]

        # Depth normalisation lives in TyGrit.utils.depth — same helper
        # is consumed by Genesis / Isaac Sim handlers if they ever
        # produce integer depth. ManiSkill emits uint16 / int16
        # millimetres under rgb+depth observations.
        depth_m = integer_depth_to_metres(depth_raw)
        # Drop trailing single channel if present.
        if depth_m.ndim == 3 and depth_m.shape[-1] == 1:
            depth_m = depth_m[..., 0]

        rgb_u8 = rgb.astype(np.uint8)
        seg_i32 = seg_np.astype(np.int32) if seg_np is not None else None
        return rgb_u8, depth_m, seg_i32

    def get_intrinsics(self, camera_id: str) -> npt.NDArray[np.float64]:
        if camera_id not in self._intrinsics:
            self._robot_cfg.camera_by_id(camera_id)  # raises KeyError with nice msg
        return self._intrinsics[camera_id]

    # ── SimHandler: mutations ──────────────────────────────────────────

    def apply_action(self, action: npt.NDArray[np.float32]) -> None:
        if action.shape != (self._total_action_dim,):
            raise ValueError(
                f"ManiSkillSimHandler.apply_action: expected shape "
                f"({self._total_action_dim},), got {action.shape}"
            )
        self._obs, _reward, _terminated, _truncated, self._info = self._env.step(action)

    def reset_to_scene_idx(self, idx: int, *, seed: int | None = None) -> None:
        if not 0 <= idx < len(self._scenes):
            raise IndexError(
                f"ManiSkillSimHandler.reset_to_scene_idx: idx {idx} out of range "
                f"for scene pool of size {len(self._scenes)}"
            )
        options: dict[str, Any] = dict(reconfigure=True, build_config_idxs=[idx])
        self._obs, self._info = self._env.reset(seed=seed, options=options)

    def set_joint_positions(
        self,
        positions: Mapping[str, float],
        *,
        env_idx: int = 0,
    ) -> None:
        if env_idx != 0:
            raise IndexError(
                f"ManiSkillSimHandler is single-env (num_envs=1); got env_idx={env_idx}"
            )
        if not positions:
            return
        qpos = self.get_qpos().copy()
        for name, value in positions.items():
            if name not in self._joint_name_to_idx:
                raise KeyError(
                    f"ManiSkillSimHandler.set_joint_positions: unknown joint {name!r}. "
                    f"Known joints: {sorted(self._joint_name_to_idx)!r}"
                )
            qpos[self._joint_name_to_idx[name]] = value
        # Sapien wants a torch tensor with a leading batch dim.
        self.env_agent.robot.set_qpos(torch.as_tensor(qpos).reshape(1, -1))

    def set_base_pose(
        self,
        x: float,
        y: float,
        theta: float,
        *,
        env_idx: int = 0,
    ) -> None:
        if not self._robot_cfg.is_mobile:
            raise RuntimeError(
                f"ManiSkillSimHandler.set_base_pose: robot {self._robot_cfg.name!r} "
                f"is fixed-base; set_base_pose is not legal"
            )
        if env_idx != 0:
            raise IndexError(
                f"ManiSkillSimHandler is single-env (num_envs=1); got env_idx={env_idx}"
            )
        bj = self._robot_cfg.base_joint_names
        self.set_joint_positions({bj[0]: x, bj[1]: y, bj[2]: theta})

    # ── SimHandler: world hooks ────────────────────────────────────────

    def get_navigable_positions(self) -> list:
        # The ``SpecBackedSceneBuilder`` used by TyGrit populates a
        # ``navigable_positions`` attribute on itself during build.
        # ManiSkill's own ``SceneManipulation-v1`` exposes the active
        # scene builder under ``env.unwrapped.scene_builder``. Any
        # other scene builder (procedural ``SceneConfig`` etc.) would
        # omit that attribute; we raise in that case so callers learn
        # the handler is configured against an incompatible env
        # instead of silently getting an empty list (the previous
        # ``getattr(..., [])`` fallback hid that bug).
        unwrapped = self._env.unwrapped
        if not hasattr(unwrapped, "scene_builder"):
            raise RuntimeError(
                "ManiSkillSimHandler.get_navigable_positions: the underlying "
                "ManiSkill env has no scene_builder attribute. This handler "
                "requires an env that exposes SpecBackedSceneBuilder "
                "(SceneManipulation-v1). Check make_scene_manipulation_env."
            )
        sb = unwrapped.scene_builder
        if not hasattr(sb, "navigable_positions"):
            raise RuntimeError(
                f"ManiSkillSimHandler.get_navigable_positions: scene_builder "
                f"{type(sb).__name__} has no navigable_positions. Only "
                f"SpecBackedSceneBuilder populates it during build; other "
                f"SceneBuilder subclasses would need their own navmesh hook."
            )
        return list(sb.navigable_positions)

    # ── SimHandler: lifecycle ──────────────────────────────────────────

    def render(self) -> None:
        if self._render_mode is None:
            return
        self._env.render()

    def close(self) -> None:
        self._env.close()

    # ── ManiSkill-specific helpers (not part of SimHandler) ────────────

    @property
    def env_agent(self):
        """Live reference to the underlying ManiSkill ``BaseAgent``.

        Do NOT cache — ManiSkill destroys + recreates the agent on
        every ``reconfigure=True`` reset, and cached references point
        into freed memory. Query ``self._env.unwrapped.agent`` every
        time (this property does that implicitly).
        """
        return self._env.unwrapped.agent

    def _find_link(self, link_name: str):
        """Look up a Sapien link on the robot by name. Raises KeyError."""
        for link in self.env_agent.robot.get_links():
            if link.name == link_name:
                return link
        raise KeyError(
            f"ManiSkillSimHandler.get_link_pose: robot "
            f"{self._robot_cfg.name!r} has no link {link_name!r}; available: "
            f"{[l.name for l in self.env_agent.robot.get_links()]}"
        )


# ── private helpers ──────────────────────────────────────────────────


def _maniskill_sensor_id(cam) -> str:
    """Return the ManiSkill-internal sensor id for a :class:`CameraSpec`.

    Sims with built-in robot agents (ManiSkill's Fetch) register
    sensors under an internal name that doesn't necessarily match
    TyGrit's public ``camera_id``. The mapping lives on
    :attr:`CameraSpec.sim_sensor_ids`; if the robot's
    :class:`CameraSpec` doesn't declare a mapping for ManiSkill, we
    fall back to using ``cam.camera_id`` directly (sufficient for
    robots whose ManiSkill agent happens to register sensors under
    the same name).
    """
    return cam.sim_sensor_ids.get("maniskill", cam.camera_id)


__all__ = ["ManiSkillSimHandler", "DEFAULT_SIM_CONFIG"]
