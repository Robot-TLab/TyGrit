"""ManiSkill3 backend for the Fetch robot.

This module hosts two pieces:

* :class:`ManiSkillFetchSimBackend` — the
  :class:`~TyGrit.envs.fetch.sim_backend.FetchSimBackend`
  implementation. Owns the underlying ``gym.make`` env, parses
  ManiSkill obs dicts into numpy arrays, and routes actions back into
  the env. All ManiSkill / Sapien / torch coupling lives here.
* :class:`ManiSkillFetchRobot` — a thin
  :class:`~TyGrit.envs.fetch.core.FetchRobotCore` subclass that
  constructs the backend and forwards. The Fetch-specific *logic*
  (joint indexing, base-pose calibration, action assembly, head PD,
  MPC trajectory execution, look_at) lives in ``FetchRobotCore`` so a
  Genesis or hardware backend can reuse it by composing with a
  different :class:`FetchSimBackend`.

Ported from ``grasp_anywhere/envs/maniskill/maniskill_env_mpc.py`` as
clean, single-threaded code.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SceneConfig, SimConfig

from TyGrit.controller.fetch.mpc import MPCConfig
from TyGrit.envs.fetch.config import FetchEnvConfig
from TyGrit.envs.fetch.core import FetchRobotCore
from TyGrit.envs.fetch.fetch import FetchRobot
from TyGrit.envs.fetch.maniskill_setup import (
    build_action_slices,
    build_joint_name_to_idx,
    extract_intrinsics,
    make_scene_manipulation_env,
)
from TyGrit.utils.tensor import to_numpy
from TyGrit.worlds.sampler import create_sampler


class ManiSkillFetchSimBackend:
    """ManiSkill3 implementation of :class:`FetchSimBackend`.

    Owns the ``gym.make`` env, the per-controller action slices, the
    joint-name lookup, and the cached observation. Single-env only —
    the batched path is :class:`ManiSkillFetchRobotVec` and currently
    does not implement this protocol (different semantics: torch
    tensors, dict-returning step/reset).
    """

    def __init__(self, config: FetchEnvConfig) -> None:
        self._config = config
        # Sampler ownership for scene-pool scope: the backend only
        # needs the spec list at gym.make time (for bind_specs) and
        # one initial idx. The core layer also constructs its own
        # sampler so it can advance the deterministic reset sequence
        # — sharing one sampler instance across the two layers would
        # couple them more tightly than necessary.
        sampler = create_sampler(config.scene_sampler)
        self._scenes = sampler.scenes
        initial_idx = sampler.sample_idx(env_idx=0, reset_count=0)

        self._env = make_scene_manipulation_env(
            config,
            self._scenes,
            build_config_idxs=[initial_idx],
            sim_config=SimConfig(
                gpu_memory_config=GPUMemoryConfig(
                    found_lost_pairs_capacity=2**25,
                    max_rigid_patch_count=2**18,
                ),
                scene_config=SceneConfig(contact_offset=0.001),
            ),
        )

        self._obs, _ = self._env.reset()

        self._action_slices, self._total_action_dim = build_action_slices(self._agent)
        self._joint_name_to_idx = build_joint_name_to_idx(self._agent)
        self._intrinsics: npt.NDArray[np.float64] = extract_intrinsics(
            self._env, "fetch_head"
        )

    # ── ManiSkill internals ────────────────────────────────────────────

    @property
    def _agent(self):
        """Always read the *current* agent.

        ManiSkill rebuilds it on every ``env.reset()`` reconfigure
        (e.g. ``num_envs=1`` paths), so caching the reference at
        ``__init__`` time would leave subsequent writes hitting the
        destroyed previous scene.
        """
        return self._env.unwrapped.agent  # type: ignore[attr-defined]

    # ── FetchSimBackend: metadata ─────────────────────────────────────

    @property
    def num_envs(self) -> int:
        return 1

    @property
    def intrinsics(self) -> npt.NDArray[np.float64]:
        return self._intrinsics

    @property
    def total_action_dim(self) -> int:
        return self._total_action_dim

    @property
    def action_slices(self) -> dict[str, slice]:
        return self._action_slices

    @property
    def joint_name_to_idx(self) -> dict[str, int]:
        return self._joint_name_to_idx

    @property
    def base_joint_names(self) -> tuple[str, str, str]:
        names = self._agent.base_joint_names
        return (names[0], names[1], names[2])

    # ── FetchSimBackend: per-step queries (cached obs) ────────────────

    def get_qpos(self) -> npt.NDArray[np.float64]:
        return to_numpy(self._obs["state"]).astype(np.float64)

    def get_base_link_world_pose(self) -> npt.NDArray[np.float64]:
        base_link = None
        for link in self._agent.robot.links:
            if link.name == "base_link":
                base_link = link
                break
        pose = base_link.pose if base_link is not None else self._agent.robot.pose
        T = pose.to_transformation_matrix()
        if isinstance(T, torch.Tensor):
            T = T.detach().cpu().numpy()
        else:
            T = np.array(T)
        if T.ndim == 3:
            T = T[0]
        return T.astype(np.float64)

    def parse_camera(self, camera_id: str) -> tuple[
        npt.NDArray[np.uint8],
        npt.NDArray[np.float32],
        npt.NDArray[np.int32] | None,
    ]:
        if camera_id != "head":
            raise ValueError(
                f"ManiSkillFetchSimBackend.parse_camera: unknown camera_id "
                f"{camera_id!r}. Currently only 'head' (fetch_head sensor) "
                f"is configured."
            )
        sensor = self._obs["sensor_data"]["fetch_head"]

        rgb = sensor["rgb"]
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.detach().cpu().numpy()
        rgb = rgb[0]  # remove batch dim

        depth = sensor["depth"]
        if isinstance(depth, torch.Tensor):
            depth = depth.detach().cpu().numpy()
        # mm → m, drop channel dim
        depth = depth[0, ..., 0].astype(np.float32) / 1000.0

        seg = sensor.get("segmentation")
        if seg is not None:
            if isinstance(seg, torch.Tensor):
                seg = seg.detach().cpu().numpy()
            seg = seg[0, ..., 0].astype(np.int32)

        return rgb, depth, seg

    # ── FetchSimBackend: mutations ────────────────────────────────────

    def step(self, action: npt.NDArray[np.float32]) -> None:
        self._obs, _, _, _, _ = self._env.step(action)

    def reset_to_idx(self, idx: int, seed: int | None = None) -> None:
        # Direct env.reset with reconfigure — equivalent to what
        # TyGrit.worlds.backends.maniskill.build_world does, but we
        # capture the returned obs inline rather than discarding it.
        self._obs, _ = self._env.reset(
            seed=seed,
            options={"reconfigure": True, "build_config_idxs": [idx]},
        )
        # Tell the interactive viewer (if any) to refresh its cached
        # frame on the next render call — the viewer skips
        # window.update_render when paused unless this flag is set.
        viewer = self._env.unwrapped.viewer  # type: ignore[attr-defined]
        if viewer is not None:
            viewer.notify_render_update()

    def set_base_pose(self, x: float, y: float, theta: float, env_idx: int = 0) -> None:
        if env_idx != 0:
            raise ValueError(
                f"ManiSkillFetchSimBackend.set_base_pose: this single-env "
                f"backend only accepts env_idx=0, got {env_idx}."
            )
        # 7-vector: position xyz + quaternion wxyz (Sapien convention).
        # z=0.02 matches the existing default spawn height.
        pose = np.array(
            [x, y, 0.02, np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)],
            dtype=np.float32,
        )
        pose_tensor = (
            torch.from_numpy(pose)
            .unsqueeze(0)
            .to(self._env.unwrapped.device)  # type: ignore[attr-defined]
        )
        self._agent.robot.set_pose(Pose.create(pose_tensor))

    # ── FetchSimBackend: world hooks ──────────────────────────────────

    def get_navigable_positions(self) -> list:
        nav = self._env.unwrapped.scene_builder.navigable_positions  # type: ignore[attr-defined]
        return list(nav) if nav is not None else []

    # ── FetchSimBackend: lifecycle ────────────────────────────────────

    def render(self) -> None:
        self._env.render()

    def close(self) -> None:
        self._env.close()


class ManiSkillFetchRobot(FetchRobotCore, FetchRobot):
    """Fetch robot driven by ManiSkill3 simulation.

    Composes :class:`FetchRobotCore` (sim-agnostic logic) with
    :class:`ManiSkillFetchSimBackend` (sim-specific glue). Most code
    lives in those two pieces; this class only wires construction.

    Inherits :class:`FetchRobot` for the public ``FetchRobot.create``
    factory dispatch path defined by :mod:`TyGrit.envs.fetch.fetch`.
    """

    def __init__(
        self,
        config: FetchEnvConfig | None = None,
        mpc_config: MPCConfig | None = None,
    ) -> None:
        cfg = config or FetchEnvConfig()
        backend = ManiSkillFetchSimBackend(cfg)
        FetchRobotCore.__init__(self, cfg, backend, mpc_config)
        # Initial render kicks the viewer once after the
        # construction-time reset so the first frame is visible.
        if cfg.render_mode == "human":
            backend.render()
