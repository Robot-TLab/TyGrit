"""Vectorized ManiSkill3 Fetch robot for GPU-parallel RL training.

Standalone class (no longer a subclass of ``ManiSkillFetchRobot``)
that handles ``num_envs > 1``. The single-env class now composes a
:class:`FetchRobotCore` with a numpy/scalar :class:`FetchSimBackend`;
this vec class operates on torch tensors with dict-shaped step/reset
returns, so the two no longer share enough method bodies to justify
inheritance. Construction-time setup is shared via helpers in
:mod:`TyGrit.sim.maniskill_helpers` (gym.make wrapper, action
slices, joint-name map, intrinsics).

``step()`` and ``reset()`` return a dict built from the ManiSkill obs
(which already contains qpos, qvel, rgb, depth, camera matrices) plus
one lightweight TCP pose read — the only data not in the obs dict.
No redundant sim queries.
"""

from __future__ import annotations

import numpy as np
import torch
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SceneConfig, SimConfig
from torch import Tensor

from TyGrit.controller.fetch.mpc import MPCConfig
from TyGrit.envs.fetch import FetchRobot
from TyGrit.envs.fetch.config import FetchEnvConfig
from TyGrit.robots.fetch import FETCH_CFG
from TyGrit.sim.maniskill_helpers import (
    build_action_slices,
    build_joint_name_to_idx,
    extract_intrinsics,
    make_scene_manipulation_env,
)
from TyGrit.types.geometry import SE2Pose
from TyGrit.types.robots import RobotState
from TyGrit.types.sensors import SensorSnapshot
from TyGrit.worlds.sampler import create_sampler

# Resolve the ManiSkill-internal head sensor id once via the
# CameraSpec.sim_sensor_ids mapping (added 2026-04-15). Same lookup
# the single-env sim/maniskill.py handler uses.
_HEAD_SENSOR_ID = FETCH_CFG.camera_by_id("head").sim_sensor_ids.get("maniskill", "head")


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


class ManiSkillFetchRobotVec(FetchRobot):
    """Vectorized Fetch robot for GPU-parallel training.

    Wraps ManiSkill3 with ``num_envs`` parallel environments.  Observations
    and actions are batched along the first dimension ``(N, ...)``.

    ``step()`` and ``reset()`` return a dict containing the raw ManiSkill
    obs (which includes ``agent/qpos``, ``agent/qvel``, ``sensor_data``,
    ``sensor_param/cam2world_gl``) plus ``ee_pos`` and ``ee_forward``
    from a single TCP pose read — the only data not in the obs dict.

    Standalone (does NOT inherit from
    :class:`~TyGrit.envs.fetch.maniskill.ManiSkillFetchRobot`) because
    that single-env class now composes :class:`FetchRobotCore` with a
    pure-numpy :class:`FetchSimBackend`, while this vec class operates
    on torch tensors and returns dict-shaped results — the two no
    longer share enough method bodies to justify inheritance. A
    future ``FetchSimBackendBatched`` protocol could parallel
    :class:`~TyGrit.envs.fetch.sim_backend.FetchSimBackend` so vec
    classes follow the same composition pattern; deferred until a
    second batched backend (Genesis vec) actually arrives.
    """

    _gripper_targets: Tensor
    _head_targets: Tensor
    _qpos: Tensor

    @property
    def _agent(self):
        """Always read the *current* agent (rebuilt by ManiSkill on
        every reconfigure)."""
        return self._env.unwrapped.agent  # type: ignore[attr-defined]

    @property
    def camera_ids(self) -> list[str]:
        return ["head"]

    def __init__(
        self,
        config: FetchEnvConfig | None = None,
        mpc_config: MPCConfig | None = None,
    ) -> None:
        cfg = config or FetchEnvConfig()
        self._config = cfg
        self._mpc_config = mpc_config
        self._num_envs = cfg.num_envs

        # Load manifest + construct sampler. Sharing the sampler's scene
        # pool with bind_specs means sampler indices line up with the
        # SpecBackedSceneBuilder's build_config indices 1:1.
        self._sampler = create_sampler(cfg.scene_sampler)
        self._scenes = self._sampler.scenes
        self._reset_count = 0

        # Pick initial per-env scene indices deterministically at
        # reset_count=0. One draw per parallel worker so the starting
        # batch is already diverse.
        initial_idxs = [
            self._sampler.sample_idx(env_idx=i, reset_count=0)
            for i in range(self._num_envs)
        ]

        # Create vectorized environment. GPU sim when num_envs > 1
        # for parallel stepping + rendering; CPU otherwise.
        self._env = make_scene_manipulation_env(
            FETCH_CFG,
            self._scenes,
            build_config_idxs=initial_idxs,
            sim_config=SimConfig(
                spacing=50,
                gpu_memory_config=_gpu_memory_config(self._num_envs),
                scene_config=SceneConfig(contact_offset=0.002),
            ),
            obs_mode=cfg.sim_opts.get("obs_mode", "rgb+depth+state+segmentation"),
            control_mode=cfg.sim_opts.get("control_mode", "pd_joint_vel"),
            render_mode=cfg.sim_opts.get("render_mode", "human"),
            camera_resolution=(cfg.camera_width, cfg.camera_height),
            num_envs=self._num_envs,
            sim_backend="gpu" if self._num_envs > 1 else "cpu",
        )

        self._obs, _ = self._env.reset()

        self._action_slices, self._total_action_dim = build_action_slices(
            self._agent, FETCH_CFG
        )
        self._joint_name_to_idx = build_joint_name_to_idx(self._agent)
        self._intrinsics = extract_intrinsics(self._env, _HEAD_SENSOR_ID)

        # Base calibration (batched)
        self._qpos_base_indices: tuple[int, int, int] = (0, 0, 0)
        self._qpos_base_offset: Tensor = torch.zeros(
            self._num_envs, 3, dtype=torch.float64
        )
        self._init_qpos_world_offset()

        # Build joint index tensors for fast batched extraction
        self._planning_indices = torch.tensor(
            [self._joint_name_to_idx[n] for n in FETCH_CFG.planning_joint_names],
            dtype=torch.long,
        )
        self._head_indices = torch.tensor(
            [self._joint_name_to_idx[n] for n in FETCH_CFG.head_joint_names],
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

        # Cached qpos from obs dict — updated by step() and reset()
        self._qpos = self._obs["agent"]["qpos"].float()

        if cfg.sim_opts.get("render_mode") == "human":
            self._env.render()

    @property
    def num_envs(self) -> int:
        return self._num_envs

    # ── TCP pose (the only data not in the obs dict) ─────────────────────

    def _extract_tcp_pose(self) -> tuple[Tensor, Tensor]:
        """Read TCP pose — the only sim data not in the ManiSkill obs dict.

        Returns ``(ee_pos, ee_forward)`` each ``(N, 3)``.
        """
        tcp_pose = self._agent.tcp.pose
        ee_pos = tcp_pose.p.float()  # (N, 3)
        q = tcp_pose.q.float()  # (N, 4) [w,x,y,z] Sapien convention
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        ee_forward = torch.stack(
            [
                2 * (x * z + w * y),
                2 * (y * z - w * x),
                1 - 2 * (x * x + y * y),
            ],
            dim=1,
        )  # (N, 3)
        return ee_pos, ee_forward

    # ── Base-offset calibration (batched) ─────────────────────────────────

    def _init_qpos_world_offset(self) -> None:
        """Compute batched offset between qpos base joints and world base_link pose."""
        base_joint_names = self._agent.base_joint_names
        ix = self._joint_name_to_idx[base_joint_names[0]]
        iy = self._joint_name_to_idx[base_joint_names[1]]
        ith = self._joint_name_to_idx[base_joint_names[2]]
        self._qpos_base_indices = (ix, iy, ith)

        qpos = self._obs["agent"]["qpos"].detach().cpu().double()
        if qpos.ndim == 1:
            qpos = qpos.unsqueeze(0)  # (1, D)

        qx = qpos[:, ix]
        qy = qpos[:, iy]
        qth = qpos[:, ith]

        # World-frame base_link pose (not in obs dict — one-time-per-reset read)
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

    # ── Batched head PD controller ────────────────────────────────────────

    def _compute_head_pd_batched(self) -> Tensor:
        """Return ``(N, 2)`` head velocities ``[pan_vel, tilt_vel]``.

        Uses cached ``_qpos`` (pre-step state) for the current head position.
        """
        current = self._qpos[:, self._head_indices.to(self._qpos.device)]
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

        Supports three input formats:
        - ``(N, 10)``: ``[v, w, torso, arm0..arm6]`` — gripper via targets, head via PD
        - ``(N, 11)``: ``[v, w, torso, arm0..arm6, gripper]`` — head via PD
        - ``(N, 13)``: ``[v, w, torso, arm0..arm6, gripper, head_pan, head_tilt]``
        """
        N = mpc_action.shape[0]
        ndim = mpc_action.shape[1]
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
            if ndim >= 13:
                # 13-dim: head velocities from policy (free head control)
                head_vel = mpc_action[:, 11:13]
            else:
                # 10/11-dim: head via PD controller
                head_vel = self._compute_head_pd_batched().to(device)  # (N, 2)
            torso_vel = mpc_action[:, 2:3]  # (N, 1)
            body = torch.cat([head_vel, torso_vel], dim=1)  # (N, 3)
            action[:, self._action_slices["body"]] = body

        if "gripper" in self._action_slices:
            if ndim >= 11:
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
        """Step all envs with batched action ``(N, 10|11)`` and return result.

        Returns a dict containing the raw ManiSkill obs (with
        ``agent/qpos``, ``agent/qvel``, ``sensor_data``,
        ``sensor_param/cam2world_gl``) plus ``ee_pos`` and ``ee_forward``
        from the one TCP pose read that isn't in the obs dict.
        """
        if isinstance(action, np.ndarray):
            action = torch.as_tensor(action, dtype=torch.float32)
        ms_action = self._assemble_action_batched(action)
        self._obs, reward, terminated, truncated, info = self._env.step(ms_action)
        self._qpos = self._obs["agent"]["qpos"].float()
        ee_pos, ee_forward = self._extract_tcp_pose()
        if self._config.sim_opts.get("render_mode") == "human":
            self._env.render()
        return {
            "obs": self._obs,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
            "ee_pos": ee_pos,
            "ee_forward": ee_forward,
        }

    # ── Random in-room initial pose (batched) ─────────────────────────────

    def _randomize_robot_pose(self, seed: int | None = None) -> None:
        """Place the Fetch base of every parallel env at a random
        in-room collision-free pose.

        Samples one ``(x, y)`` per env from the active scene's navmesh
        (``*.fetch.navigable_positions.obj`` vertices, by construction
        in free space) plus a uniform yaw, and teleports all
        articulation roots in one batched ``set_pose`` call. Falls
        back to ``(-1, 0, 0.02)`` (ManiSkill's default Fetch spawn)
        for any env whose scene has no navmesh — Holodeck specifically
        ships none.

        This is the vec-side counterpart to
        :meth:`FetchRobotCore._randomize_robot_pose`; it can't share
        with that method because we need to issue a single batched
        ``set_pose`` rather than a per-env ``set_base_pose`` loop, and
        because the navmesh source on the vec env is the underlying
        scene_builder rather than a backend protocol method (vec
        doesn't compose a FetchSimBackend yet).
        """
        nav_meshes = self._env.unwrapped.scene_builder.navigable_positions  # type: ignore[attr-defined]
        rng = np.random.default_rng(seed)
        poses = np.zeros((self._num_envs, 7), dtype=np.float32)
        for i in range(self._num_envs):
            mesh = (
                nav_meshes[i]
                if nav_meshes is not None and i < len(nav_meshes)
                else None
            )
            if mesh is None or len(mesh.vertices) == 0:
                x, y = -1.0, 0.0
            else:
                verts = np.asarray(mesh.vertices)
                idx = int(rng.integers(0, len(verts)))
                x, y = float(verts[idx, 0]), float(verts[idx, 1])
            theta = float(rng.uniform(-np.pi, np.pi))
            poses[i] = [x, y, 0.02, np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)]

        pose_tensor = torch.from_numpy(poses).to(self._env.unwrapped.device)  # type: ignore[attr-defined]
        self._agent.robot.set_pose(Pose.create(pose_tensor))

    def reset(  # type: ignore[override]
        self,
        settle_steps: int = 0,
        randomize_init: bool = True,
        **kwargs,
    ) -> dict:
        """Reset all envs and return result with obs + TCP pose.

        Parameters
        ----------
        settle_steps : int
            Number of zero-action physics steps after reset to let objects
            settle before returning observations (prevents explosive forces
            from initial interpenetration).
        randomize_init : bool
            If True, randomize each env's Fetch base pose to a random
            in-room collision-free spawn sampled from the ReplicaCAD
            navmesh, instead of the hardcoded ``[-1, 0, 0.02]``.

        Scene selection per parallel worker is drawn from
        ``self._sampler`` using ``(env_idx, reset_count)`` so successive
        resets get fresh scenes without the caller having to vary
        their own seed — the v1 repeating-scene fix.
        """
        self._reset_count += 1
        idxs = [
            self._sampler.sample_idx(env_idx=i, reset_count=self._reset_count)
            for i in range(self._num_envs)
        ]
        # Override any reconfigure/build_config_idxs the caller may have
        # passed through kwargs — scene selection is owned by the
        # sampler, not the caller, to keep the v1 bug guard effective.
        options = kwargs.pop("options", None) or {}
        options["reconfigure"] = True
        options["build_config_idxs"] = idxs
        self._obs, info = self._env.reset(options=options, **kwargs)
        if randomize_init:
            self._randomize_robot_pose(seed=kwargs.get("seed"))
            # Tell the interactive viewer (if any) to refresh its
            # cached frame on the next render call.
            viewer = self._env.unwrapped.viewer  # type: ignore[attr-defined]
            if viewer is not None:
                viewer.notify_render_update()
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
        self._qpos = self._obs["agent"]["qpos"].float()
        ee_pos, ee_forward = self._extract_tcp_pose()
        if self._config.sim_opts.get("render_mode") == "human":
            self._env.render()
        return {
            "obs": self._obs,
            "info": info,
            "ee_pos": ee_pos,
            "ee_forward": ee_forward,
        }

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
        qpos = self._qpos
        if qpos.ndim == 1:
            qpos = qpos.unsqueeze(0)
        head = self.get_batched_head_joints(qpos)  # (N, 2)

        if link_poses is None:
            from TyGrit.robots.fetch.kinematics.fk_torch import batch_forward_kinematics

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

    # ── Single-env RobotBase methods (for eval compatibility) ─────────────

    def get_robot_state(self) -> RobotState:
        """Return state for env 0 (for single-env compatibility)."""
        qpos = self._qpos
        if qpos.ndim == 1:
            qpos = qpos.unsqueeze(0)
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
        sensor = obs["sensor_data"][_HEAD_SENSOR_ID]

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

    # ── Trajectory execution not supported in vec mode ────────────────────

    def execute_trajectory(self, trajectory) -> bool:  # noqa: ARG002
        raise NotImplementedError(
            "Trajectory execution is not supported in vectorized mode. "
            "Use step() with batched actions instead."
        )

    def start_trajectory(self, trajectory) -> None:  # noqa: ARG002
        raise NotImplementedError("Trajectory execution not supported in vec mode.")
