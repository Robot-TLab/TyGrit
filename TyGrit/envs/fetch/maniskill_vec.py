"""Vectorized ManiSkill3 Fetch robot for GPU-parallel RL training.

Subclass of :class:`ManiSkillFetchRobot` that handles ``num_envs > 1``.
The parent class assumes a single environment (hardcoded ``[0]`` indexing,
``float()`` calls, returning single ``SensorSnapshot``).  This override
keeps observations as batched tensors and returns batched results.

For ``num_envs == 1`` the behaviour is identical to the parent class
(the batch dimension is simply size 1).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from mani_skill.utils.structs.types import GPUMemoryConfig, SceneConfig, SimConfig
from torch import Tensor

from TyGrit.controller.fetch.mpc import MPCConfig
from TyGrit.envs.fetch.config import FetchEnvConfig
from TyGrit.envs.fetch.maniskill import ManiSkillFetchRobot
from TyGrit.kinematics.fetch.constants import HEAD_JOINT_NAMES, PLANNING_JOINT_NAMES
from TyGrit.types.geometry import SE2Pose
from TyGrit.types.robot import RobotState
from TyGrit.types.sensor import SensorSnapshot


def _gpu_memory_config(num_envs: int) -> GPUMemoryConfig:
    """Scale PhysX GPU buffers with the number of parallel environments.

    Base values match ReplicaCAD's ``_default_sim_config`` so that passing
    our own ``SimConfig`` to ``gym.make`` does not shrink critical buffers.
    """
    s = max(1, num_envs // 16)
    return GPUMemoryConfig(
        temp_buffer_capacity=2**24 * s,
        max_rigid_contact_count=2**23 * s,
        max_rigid_patch_count=2**21 * s,
        heap_capacity=2**26 * s,
        found_lost_pairs_capacity=2**25 * s,
        found_lost_aggregate_pairs_capacity=2**10 * s,
        total_aggregate_pairs_capacity=2**10 * s,
        collision_stack_size=2**24 * s,
    )


class ManiSkillFetchRobotVec(ManiSkillFetchRobot):
    """Vectorized Fetch robot for GPU-parallel training.

    Wraps ManiSkill3 with ``num_envs`` parallel environments.  Observations
    and actions are batched along the first dimension ``(N, ...)``.

    The :class:`RobotBase` single-env methods (``get_robot_state``,
    ``get_sensor_snapshot``, etc.) are **not available** — use the batched
    properties instead.
    """

    _gripper_targets: Tensor
    _head_targets: Tensor

    def __init__(
        self,
        config: FetchEnvConfig | None = None,
        mpc_config: MPCConfig | None = None,
    ) -> None:
        cfg = config or FetchEnvConfig()
        self._config = cfg
        self._mpc_config = mpc_config
        self._num_envs = cfg.num_envs

        # Create vectorized environment
        # Use GPU sim when num_envs > 1 for parallel stepping + rendering
        sim_backend = "gpu" if self._num_envs > 1 else "cpu"
        self._env = gym.make(
            cfg.env_id,
            robot_uids="fetch",
            obs_mode=cfg.obs_mode,
            control_mode=cfg.control_mode,
            render_mode=cfg.render_mode,
            num_envs=self._num_envs,
            sim_backend=sim_backend,
            sensor_configs={
                "width": cfg.camera_width,
                "height": cfg.camera_height,
            },
            sim_config=SimConfig(
                spacing=50,
                gpu_memory_config=_gpu_memory_config(self._num_envs),
                scene_config=SceneConfig(contact_offset=0.002),
            ),
        )

        self._agent = self._env.unwrapped.agent  # type: ignore[attr-defined]

        self._obs, _ = self._env.reset()

        # Build action slices — use shape[-1] because vectorized envs
        # have shape (num_envs, action_dim)
        self._action_slices: dict[str, slice] = {}
        idx = 0
        for name in ("arm", "gripper", "body", "base"):
            controller = self._agent.controller.controllers.get(name)
            if controller is None:
                continue
            dim = controller.action_space.shape[-1]
            self._action_slices[name] = slice(idx, idx + dim)
            idx += dim
        self._total_action_dim: int = self._env.action_space.shape[-1]  # type: ignore[union-attr]

        # Joint-name → index map
        self._joint_name_to_idx: dict[str, int] = {
            j.name: i for i, j in enumerate(self._agent.robot.active_joints)
        }

        # Base calibration (batched)
        self._qpos_base_indices: tuple[int, int, int] = (0, 0, 0)
        self._qpos_base_offset: Tensor = torch.zeros(
            self._num_envs, 3, dtype=torch.float64
        )
        self._init_qpos_world_offset()

        # Cache camera intrinsics (static, shared across envs)
        cam_params = self._env.unwrapped._sensors["fetch_head"].get_params()  # type: ignore[attr-defined]
        K = cam_params["intrinsic_cv"].detach().cpu().numpy()
        if K.ndim == 3:
            K = K[0]
        self._intrinsics = K.astype(np.float64)

        # Build joint index tensors for fast batched extraction
        self._planning_indices = torch.tensor(
            [self._joint_name_to_idx[n] for n in PLANNING_JOINT_NAMES],
            dtype=torch.long,
        )
        self._head_indices = torch.tensor(
            [self._joint_name_to_idx[n] for n in HEAD_JOINT_NAMES],
            dtype=torch.long,
        )

        # Batched actuator targets (separate from parent's scalar _gripper_target / _head_target)
        self._gripper_targets: Tensor = torch.zeros(self._num_envs, dtype=torch.float32)
        self._head_targets: Tensor = torch.full(
            (self._num_envs, 2), float("nan"), dtype=torch.float32
        )

        # No trajectory support in vec mode
        self._trajectory = None
        self._waypoint_idx = 0

        if cfg.render_mode == "human":
            self._env.render()

    @property
    def num_envs(self) -> int:
        return self._num_envs

    # ── Obs format helpers ───────────────────────────────────────────────

    def _extract_qpos_from_obs(self, obs: dict | Tensor) -> Tensor:
        """Extract qpos tensor from any ManiSkill obs format.

        Handles:
        - ``obs_mode="state"``: obs is a flat ``(N, D)`` tensor.
        - ``obs_mode="state_dict"``: obs is ``{"agent": {"qpos": ...}}``.
        - ``obs_mode="rgb+..."``: obs is ``{"state": ...}``.
        """
        if isinstance(obs, torch.Tensor):
            return obs
        if "state" in obs:
            return obs["state"]
        if "agent" in obs:
            return obs["agent"]["qpos"]
        raise ValueError(f"Cannot extract qpos from obs keys: {list(obs.keys())}")

    # ── Base-offset calibration (batched) ─────────────────────────────────

    def _init_qpos_world_offset(self) -> None:
        """Compute batched offset between qpos base joints and world base_link pose."""
        base_joint_names = self._agent.base_joint_names
        ix = self._joint_name_to_idx[base_joint_names[0]]
        iy = self._joint_name_to_idx[base_joint_names[1]]
        ith = self._joint_name_to_idx[base_joint_names[2]]
        self._qpos_base_indices = (ix, iy, ith)

        qpos = self._extract_qpos_from_obs(self._obs).detach().cpu().double()
        if qpos.ndim == 1:
            qpos = qpos.unsqueeze(0)  # (1, D)

        qx = qpos[:, ix]
        qy = qpos[:, iy]
        qth = qpos[:, ith]

        # World-frame base_link pose
        base_link = next(
            link for link in self._agent.robot.links if link.name == "base_link"
        )
        T = base_link.pose.to_transformation_matrix().detach().cpu().double()
        if T.ndim == 2:
            T = T.unsqueeze(0)  # (1, 4, 4)

        wx = T[:, 0, 3]
        wy = T[:, 1, 3]
        wth = torch.atan2(T[:, 1, 0], T[:, 0, 0])
        dth = torch.atan2(torch.sin(wth - qth), torch.cos(wth - qth))

        self._qpos_base_offset = torch.stack([wx - qx, wy - qy, dth], dim=1)

    # ── Batched state extraction ──────────────────────────────────────────

    def get_batched_qpos(self) -> Tensor:
        """Return ``(N, D)`` qpos tensor from the current observation."""
        qpos = self._extract_qpos_from_obs(self._obs)
        if qpos.ndim == 1:
            qpos = qpos.unsqueeze(0)
        return qpos

    def get_batched_base_pose(self, qpos: Tensor) -> Tensor:
        """Return ``(N, 3)`` tensor of ``[x, y, theta]`` in world frame."""
        ix, iy, ith = self._qpos_base_indices
        offset = self._qpos_base_offset.to(qpos.device)
        raw = torch.stack([qpos[:, ix], qpos[:, iy], qpos[:, ith]], dim=1)
        result = raw + offset
        # Wrap theta to [-pi, pi]
        result[:, 2] = torch.atan2(torch.sin(result[:, 2]), torch.cos(result[:, 2]))
        return result

    def get_batched_planning_joints(self, qpos: Tensor) -> Tensor:
        """Return ``(N, 8)`` planning joint positions."""
        return qpos[:, self._planning_indices.to(qpos.device)]

    def get_batched_head_joints(self, qpos: Tensor) -> Tensor:
        """Return ``(N, 2)`` head joint positions."""
        return qpos[:, self._head_indices.to(qpos.device)]

    def get_batched_robot_state(self) -> dict[str, Tensor]:
        """Return batched robot state as a dict of tensors.

        Keys: ``"base_pose"`` (N, 3), ``"planning_joints"`` (N, 8),
        ``"head_joints"`` (N, 2).
        """
        qpos = self.get_batched_qpos()
        return {
            "base_pose": self.get_batched_base_pose(qpos),
            "planning_joints": self.get_batched_planning_joints(qpos),
            "head_joints": self.get_batched_head_joints(qpos),
        }

    # ── Batched head PD controller ────────────────────────────────────────

    def _compute_head_pd_batched(self) -> Tensor:
        """Return ``(N, 2)`` head velocities ``[pan_vel, tilt_vel]``."""
        qpos = self.get_batched_qpos()
        current = self.get_batched_head_joints(qpos)  # (N, 2)
        target = self._head_targets.to(current.device)

        kp = self._config.gaze_kp
        max_vel = self._config.gaze_max_vel

        err = target - current
        # Where target is NaN, error is 0
        err = torch.where(torch.isnan(err), torch.zeros_like(err), err)
        vel = (kp * err).clamp(-max_vel, max_vel)
        return vel

    # ── Batched action assembly ───────────────────────────────────────────

    def _assemble_action_batched(self, mpc_action: Tensor) -> Tensor:
        """Map batched RL action to ManiSkill action ``(N, D)``.

        Supports two input formats:
        - ``(N, 10)``: ``[v, w, torso, arm0..arm6]`` — gripper via ``_gripper_targets``
        - ``(N, 11)``: ``[v, w, torso, arm0..arm6, gripper]`` — gripper as last dim
        """
        N = mpc_action.shape[0]
        device = mpc_action.device
        action = torch.zeros(
            N, self._total_action_dim, dtype=torch.float32, device=device
        )

        if "base" in self._action_slices:
            action[:, self._action_slices["base"]] = mpc_action[:, 0:2]

        if "arm" in self._action_slices:
            sl = self._action_slices["arm"]
            n = sl.stop - sl.start
            action[:, sl] = mpc_action[:, 3 : 3 + n]

        if "body" in self._action_slices:
            head_vel = self._compute_head_pd_batched().to(device)  # (N, 2)
            torso_vel = mpc_action[:, 2:3]  # (N, 1)
            body = torch.cat([head_vel, torso_vel], dim=1)  # (N, 3)
            action[:, self._action_slices["body"]] = body

        if "gripper" in self._action_slices:
            if mpc_action.shape[1] >= 11:
                # 11-dim: gripper action from last dim (RL mode)
                gripper_action = mpc_action[:, 10:11]
            else:
                # 10-dim: gripper from targets (MPC mode)
                gripper_action = (
                    2.0 * self._gripper_targets.to(device) - 1.0
                ).unsqueeze(1)
            action[:, self._action_slices["gripper"]] = gripper_action

        return torch.nan_to_num(action, nan=0.0)

    # ── Stepping (batched) ────────────────────────────────────────────────

    def step(self, action: torch.Tensor | np.ndarray) -> dict:  # type: ignore[override]
        """Step all envs with batched action ``(N, 10)`` and return raw obs dict.

        Unlike the parent class which returns ``SensorSnapshot``, the vec
        variant returns the raw ManiSkill observation dict with batched
        ``(N, ...)`` tensors for efficiency.
        """
        if isinstance(action, np.ndarray):
            action = torch.as_tensor(action, dtype=torch.float32)
        ms_action = self._assemble_action_batched(action)
        self._obs, reward, terminated, truncated, info = self._env.step(ms_action)
        if self._config.render_mode == "human":
            self._env.render()
        return {
            "obs": self._obs,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
        }

    def reset(self, settle_steps: int = 0, **kwargs) -> dict:  # type: ignore[override]
        """Reset all envs and return raw obs dict.

        Parameters
        ----------
        settle_steps : int
            Number of zero-action physics steps after reset to let objects
            settle before returning observations (prevents explosive forces
            from initial interpenetration).
        """
        self._obs, info = self._env.reset(**kwargs)
        self._init_qpos_world_offset()
        self._trajectory = None
        self._waypoint_idx = 0
        self._gripper_targets = torch.zeros(self._num_envs, dtype=torch.float32)
        self._head_targets = torch.full(
            (self._num_envs, 2), float("nan"), dtype=torch.float32
        )
        if settle_steps > 0:
            zero_action = torch.zeros(
                self._num_envs,
                self._total_action_dim,
                dtype=torch.float32,
                device=self._env.unwrapped.device,  # type: ignore[attr-defined]
            )
            for _ in range(settle_steps):
                self._obs, *_ = self._env.step(zero_action)
        if self._config.render_mode == "human":
            self._env.render()
        return {"obs": self._obs, "info": info}

    # ── Batched look_at ───────────────────────────────────────────────────

    def look_at_batched(
        self,
        targets: Tensor,
        link_poses: dict[str, Tensor] | None = None,
        base_pose: Tensor | None = None,
    ) -> None:
        """Point cameras at batched 3D targets ``(N, 3)`` in world frame.

        Sets internal head targets; actual head motion happens via the PD
        controller during action assembly.

        Pass pre-computed *link_poses* and *base_pose* to skip redundant FK.
        """
        qpos = self.get_batched_qpos()
        head = self.get_batched_head_joints(qpos)  # (N, 2)

        if link_poses is None:
            from TyGrit.kinematics.fetch.fk_torch import batch_forward_kinematics

            planning = self.get_batched_planning_joints(qpos)  # (N, 8)
            fk_input = torch.cat([planning, head], dim=1).float()  # (N, 10)
            link_poses = batch_forward_kinematics(fk_input)

        T_head_pan = link_poses["head_pan_link"]  # (N, 4, 4)

        if base_pose is None:
            base_pose = self.get_batched_base_pose(qpos)  # (N, 3)
        cos_th = torch.cos(base_pose[:, 2])
        sin_th = torch.sin(base_pose[:, 2])

        targets_dev = targets.to(qpos.device).float()
        dx = targets_dev[:, 0] - base_pose[:, 0].float()
        dy = targets_dev[:, 1] - base_pose[:, 1].float()
        target_base_x = cos_th.float() * dx + sin_th.float() * dy
        target_base_y = -sin_th.float() * dx + cos_th.float() * dy
        target_base_z = targets_dev[:, 2]
        target_base = torch.stack([target_base_x, target_base_y, target_base_z], dim=1)

        # Transform to head_pan_link frame
        T_inv = torch.linalg.inv(T_head_pan)
        target_base_h = torch.cat(
            [
                target_base,
                torch.ones(target_base.shape[0], 1, device=target_base.device),
            ],
            dim=1,
        )  # (N, 4)
        target_head_pan = torch.bmm(T_inv.float(), target_base_h.unsqueeze(2)).squeeze(
            2
        )[
            :, :3
        ]  # (N, 3)

        x, y, z = target_head_pan[:, 0], target_head_pan[:, 1], target_head_pan[:, 2]
        current_pan = head[:, 0].float()
        pan_rel = torch.atan2(y, x)

        # Tilt
        T_head_tilt = link_poses["head_tilt_link"]  # (N, 4, 4)
        T_pan_tilt = torch.bmm(T_inv.float(), T_head_tilt.float())
        tilt_origin_pan = T_pan_tilt[:, :3, 3]  # (N, 3)
        dist_xy = torch.sqrt(x**2 + y**2)
        v_target = torch.stack([dist_xy, torch.zeros_like(dist_xy), z], dim=1)
        v_tilt = v_target - tilt_origin_pan
        tilt_abs = torch.atan2(-v_tilt[:, 2], v_tilt[:, 0])

        pan = current_pan + pan_rel
        self._head_targets = torch.stack([pan, tilt_abs], dim=1)

    # ── Batched gripper control ───────────────────────────────────────────

    def control_gripper_batched(self, positions: Tensor) -> None:
        """Set gripper targets for all envs. ``positions``: ``(N,)`` in [0, 1]."""
        self._gripper_targets = positions.clamp(0.0, 1.0).float()

    # ── Single-env RobotBase methods (delegate to parent for compatibility) ──

    def get_robot_state(self) -> RobotState:
        """Return state for env 0 (for single-env compatibility)."""
        qpos = self.get_batched_qpos()
        bp = self.get_batched_base_pose(qpos)
        pj = self.get_batched_planning_joints(qpos)
        hj = self.get_batched_head_joints(qpos)
        return RobotState(
            base_pose=SE2Pose(
                x=float(bp[0, 0]),
                y=float(bp[0, 1]),
                theta=float(bp[0, 2]),
            ),
            planning_joints=tuple(float(v) for v in pj[0]),
            head_joints=tuple(float(v) for v in hj[0]),
        )

    def get_observation(self) -> SensorSnapshot:
        """Return observation for env 0 (for single-env compatibility)."""
        return self._parse_observation_single(self._obs)

    def get_sensor_snapshot(self, camera_id: str) -> SensorSnapshot:
        if camera_id != "head":
            raise ValueError(
                f"Unknown camera_id {camera_id!r}; available: {self.camera_ids}"
            )
        return self._parse_observation_single(self._obs)

    def _parse_observation_single(self, obs: dict) -> SensorSnapshot:
        """Parse obs dict for env 0 only (single-env compat)."""
        sensor = obs["sensor_data"]["fetch_head"]

        rgb = sensor["rgb"].detach().cpu().numpy()[0]

        depth = sensor["depth"].detach().cpu().numpy()
        depth = (depth[0, ..., 0].astype(np.float32)) / 1000.0

        seg = sensor.get("segmentation")
        if seg is not None:
            seg = seg.detach().cpu().numpy()[0, ..., 0].astype(np.int32)

        return SensorSnapshot(
            rgb=rgb,
            depth=depth,
            intrinsics=self._intrinsics,
            robot_state=self.get_robot_state(),
            segmentation=seg,
        )

    def settle(self, steps: int) -> None:
        """Run zero-action physics steps to let objects settle."""
        if steps <= 0:
            return
        zero = torch.zeros(
            self._num_envs,
            self._total_action_dim,
            dtype=torch.float32,
            device=self._env.unwrapped.device,  # type: ignore[attr-defined]
        )
        for _ in range(steps):
            self._obs, *_ = self._env.step(zero)

    # ── Trajectory execution not supported in vec mode ────────────────────

    def execute_trajectory(self, trajectory) -> bool:  # noqa: ARG002
        raise NotImplementedError(
            "Trajectory execution is not supported in vectorized mode. "
            "Use step() with batched actions instead."
        )

    def start_trajectory(self, trajectory) -> None:  # noqa: ARG002
        raise NotImplementedError("Trajectory execution not supported in vec mode.")
