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

**Single-env + vec.** :class:`ManiSkillSimHandler` is the scalar
numpy surface; :class:`ManiSkillSimHandlerVec` (below) is the batched
torch surface used by :class:`~TyGrit.envs.fetch.core_vec.FetchRobotCoreVec`
for parallel training / eval.

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


class ManiSkillSimHandlerVec:
    """Robot-agnostic :class:`~TyGrit.sim.base.SimHandlerVec` for ManiSkill3.

    Vectorised counterpart of :class:`ManiSkillSimHandler` — torch
    tensors batched on axis 0 with leading dimension ``num_envs``.

    Parameters
    ----------
    robot_cfg
        Robot descriptor; must have ``sim_uids["maniskill"]`` set.
    scenes
        Scene pool to draw from at reset.
    num_envs
        Number of parallel envs; must be ``> 1`` (callers with a
        single env should use :class:`ManiSkillSimHandler`).
    initial_scene_idx
        Single index used for every env at construction. Heterogeneous
        per-env indices are passed at :meth:`reset_to_scene_idx` time.
    camera_resolution, obs_mode, control_mode, render_mode
        Forwarded to the underlying ``gym.make`` env.
    sim_config
        Optional ManiSkill ``SimConfig`` override (vec-specific GPU
        defaults applied when ``None``).
    device
        Torch device hosting the GPU sim tensors. ``"cuda:0"`` default.
    """

    def __init__(
        self,
        robot_cfg: RobotCfg,
        scenes: Sequence[SceneSpec],
        *,
        num_envs: int,
        initial_scene_idx: int = 0,
        camera_resolution: tuple[int, int] = (128, 128),
        obs_mode: str = "rgbd",
        control_mode: str = "pd_joint_vel",
        render_mode: str | None = None,
        sim_config: SimConfig | None = None,
        device: str = "cuda:0",
    ) -> None:
        if num_envs <= 1:
            raise ValueError(
                f"ManiSkillSimHandlerVec: num_envs must be > 1; got {num_envs}. "
                f"Use ManiSkillSimHandler for the scalar path."
            )
        if "maniskill" not in robot_cfg.sim_uids:
            raise ValueError(
                f"ManiSkillSimHandlerVec: RobotCfg {robot_cfg.name!r} has no "
                f"sim_uids['maniskill'] entry."
            )
        self._robot_cfg = robot_cfg
        self._scenes = tuple(scenes)
        if len(self._scenes) == 0:
            raise ValueError("ManiSkillSimHandlerVec: scene pool is empty")
        if not 0 <= initial_scene_idx < len(self._scenes):
            raise IndexError(
                f"ManiSkillSimHandlerVec: initial_scene_idx {initial_scene_idx} "
                f"out of range for scene pool of size {len(self._scenes)}"
            )
        self._num_envs = int(num_envs)
        self._device = device
        self._render_mode = render_mode

        # Vec-friendly GPU sim config (matches the legacy
        # legacy ManiSkillFetchRobotVec defaults); callers can override.
        if sim_config is None:
            sim_config = SimConfig(
                spacing=50,
                gpu_memory_config=GPUMemoryConfig(
                    found_lost_pairs_capacity=2**26,
                    max_rigid_patch_count=2**19,
                ),
                scene_config=SceneConfig(contact_offset=0.002),
            )

        initial_idxs = [initial_scene_idx] * self._num_envs
        self._env = make_scene_manipulation_env(
            robot_cfg,
            self._scenes,
            build_config_idxs=initial_idxs,
            sim_config=sim_config,
            obs_mode=obs_mode,
            control_mode=control_mode,
            render_mode=render_mode,
            camera_resolution=camera_resolution,
            num_envs=self._num_envs,
            sim_backend="gpu",
        )

        obs, info = self._env.reset(
            options=dict(reconfigure=True, build_config_idxs=initial_idxs)
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

        self._intrinsics: dict[str, npt.NDArray[np.float64]] = {}
        for cam in robot_cfg.cameras:
            sensor_id = _maniskill_sensor_id(cam)
            self._intrinsics[cam.camera_id] = extract_intrinsics(self._env, sensor_id)

    # ── SimHandlerVec: metadata ────────────────────────────────────────

    @property
    def robot_cfg(self) -> RobotCfg:
        return self._robot_cfg

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def device(self) -> str:
        return self._device

    @property
    def total_action_dim(self) -> int:
        return self._total_action_dim

    @property
    def action_slices(self) -> Mapping[str, slice]:
        return self._action_slices

    @property
    def joint_name_to_idx(self) -> Mapping[str, int]:
        return self._joint_name_to_idx

    @property
    def env_agent(self):
        return self._env.unwrapped.agent

    # ── SimHandlerVec: batched queries ─────────────────────────────────

    def get_qpos(self) -> torch.Tensor:
        return self.env_agent.robot.get_qpos()

    def get_link_pose(self, link_name: str) -> torch.Tensor:
        for link in self.env_agent.robot.get_links():
            if link.name == link_name:
                pose = link.pose
                T = pose.to_transformation_matrix()
                return T.reshape(self._num_envs, 4, 4)
        raise KeyError(
            f"ManiSkillSimHandlerVec.get_link_pose: link {link_name!r} "
            f"not on robot {self._robot_cfg.name!r}"
        )

    def get_camera(self, camera_id: str) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        cam = self._robot_cfg.camera_by_id(camera_id)
        sensor_id = _maniskill_sensor_id(cam)
        sensor_data = self._obs["sensor_data"][sensor_id]
        rgb = sensor_data["rgb"]  # Tensor[N,H,W,3]
        depth_raw = sensor_data.get("depth")
        if depth_raw is not None and depth_raw.dtype == torch.uint16:
            # ManiSkill ships uint16 millimetre depth; the project's
            # tensor convention is float32 metres.
            depth = depth_raw.to(torch.float32) * 0.001
        else:
            depth = depth_raw
        seg = sensor_data.get("segmentation")
        return rgb, depth, seg

    def get_intrinsics(self, camera_id: str) -> npt.NDArray[np.float64]:
        if camera_id not in self._intrinsics:
            raise KeyError(
                f"ManiSkillSimHandlerVec.get_intrinsics: camera {camera_id!r} "
                f"not declared on RobotCfg {self._robot_cfg.name!r}"
            )
        return self._intrinsics[camera_id]

    # ── SimHandlerVec: batched mutations ───────────────────────────────

    def apply_action(self, action: torch.Tensor) -> None:
        if action.shape != (self._num_envs, self._total_action_dim):
            raise ValueError(
                f"ManiSkillSimHandlerVec.apply_action: action shape "
                f"{tuple(action.shape)} != ({self._num_envs}, "
                f"{self._total_action_dim})"
            )
        self._obs, _r, _term, _trunc, self._info = self._env.step(action)

    def reset_to_scene_idx(
        self,
        idxs,
        *,
        seed: int | None = None,
    ) -> None:
        # Coerce idxs into a per-env list. Length must match num_envs.
        if isinstance(idxs, torch.Tensor):
            idxs_list = idxs.detach().cpu().tolist()
        else:
            idxs_list = list(idxs)
        if len(idxs_list) != self._num_envs:
            raise ValueError(
                f"ManiSkillSimHandlerVec.reset_to_scene_idx: idxs length "
                f"{len(idxs_list)} != num_envs {self._num_envs}"
            )
        for i, idx in enumerate(idxs_list):
            if not 0 <= idx < len(self._scenes):
                raise IndexError(
                    f"reset_to_scene_idx: idxs[{i}] = {idx} out of range "
                    f"for scene pool of size {len(self._scenes)}"
                )
        self._obs, self._info = self._env.reset(
            seed=seed,
            options=dict(reconfigure=True, build_config_idxs=idxs_list),
        )

    def set_joint_positions(
        self,
        positions: Mapping[str, torch.Tensor],
        *,
        env_ids=None,
    ) -> None:
        if not positions:
            return
        # ManiSkill mutates qpos via agent.robot.set_qpos when needed;
        # for joint-name selective writes the simplest path is to read
        # the full qpos, modify the columns we own, write back.
        qpos = self.env_agent.robot.get_qpos().clone()
        if env_ids is None:
            env_ids = list(range(self._num_envs))
        elif isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.detach().cpu().tolist()
        else:
            env_ids = list(env_ids)
        for name, values in positions.items():
            j = self._joint_name_to_idx[name]
            for ei, ev in zip(env_ids, values.detach().cpu().tolist()):
                qpos[ei, j] = ev
        self.env_agent.robot.set_qpos(qpos)

    def set_base_pose(
        self,
        xy_theta: torch.Tensor,
        *,
        env_ids=None,
    ) -> None:
        if not self._robot_cfg.is_mobile:
            raise RuntimeError(
                f"ManiSkillSimHandlerVec.set_base_pose: robot "
                f"{self._robot_cfg.name!r} is fixed-base"
            )
        bj = self._robot_cfg.base_joint_names
        if len(bj) != 3:
            raise ValueError(
                f"ManiSkillSimHandlerVec.set_base_pose: expected 3 base "
                f"joints (x, y, theta); got {bj!r}"
            )
        positions = {
            bj[0]: xy_theta[:, 0],
            bj[1]: xy_theta[:, 1],
            bj[2]: xy_theta[:, 2],
        }
        self.set_joint_positions(positions, env_ids=env_ids)

    # ── SimHandlerVec: world hooks ─────────────────────────────────────

    def get_navigable_positions(self) -> list:
        sb = self._env.unwrapped.scene_builder
        if not hasattr(sb, "navigable_positions"):
            return [None] * self._num_envs
        return list(sb.navigable_positions)

    # ── SimHandlerVec: lifecycle ───────────────────────────────────────

    def render(self) -> None:
        if self._render_mode is None:
            return
        self._env.render()

    def close(self) -> None:
        self._env.close()


__all__ = ["ManiSkillSimHandler", "ManiSkillSimHandlerVec", "DEFAULT_SIM_CONFIG"]
