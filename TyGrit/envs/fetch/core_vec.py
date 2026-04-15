"""Sim-agnostic Fetch sensor + actuation adapter — vectorised path.

:class:`FetchRobotCoreVec` is the batched counterpart of
:class:`~TyGrit.envs.fetch.core.FetchRobotCore`. It composes a
:class:`~TyGrit.sim.base.SimHandlerVec` (one of
:class:`ManiSkillSimHandlerVec`, :class:`GenesisSimHandlerVec`,
:class:`IsaacSimSimHandlerVec`) and exposes the same sensor +
actuation surface but with all per-env data shaped as torch tensors
on axis 0.

Same scope rules as the scalar core (CLAUDE.md Rule 1):

* **Sensing** — read handler tensors and shape them into
  :class:`~TyGrit.types.robots.RobotStateVec` /
  :class:`~TyGrit.types.sensors.SensorSnapshotVec`.
* **Actuation** — assemble per-controller batched slices into the
  ``(num_envs, total_action_dim)`` tensor :meth:`SimHandlerVec.apply_action`
  accepts.
* **Reset / spawn** — advance the scene sampler per env, drive
  :meth:`SimHandlerVec.reset_to_scene_idx`, recalibrate the holonomic
  base offset, optionally randomise spawn against the navmesh.

Out of scope (lives elsewhere): MPC trajectory tracking
(:mod:`TyGrit.controller.fetch.trajectory`), head IK
(:mod:`TyGrit.gaze.fetch_head`), task goals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from TyGrit.envs.fetch.config import FetchEnvConfig
from TyGrit.robots.fetch import FETCH_CFG
from TyGrit.types.robots import RobotStateVec
from TyGrit.types.sensors import SensorSnapshotVec
from TyGrit.worlds.sampler import create_sampler

if TYPE_CHECKING:
    import torch

    from TyGrit.sim.base import SimHandlerVec


class FetchRobotCoreVec:
    """Sim-agnostic vectorised Fetch core composing a
    :class:`SimHandlerVec`.

    Mirrors :class:`FetchRobotCore` shape but every per-env quantity is
    a torch tensor with leading axis ``num_envs``.

    Parameters
    ----------
    config
        Fetch-specific env / scheduler configuration. ``config.num_envs``
        must match ``handler.num_envs``.
    handler
        Any :class:`SimHandlerVec` constructed against ``FETCH_CFG``.
    """

    def __init__(
        self,
        config: FetchEnvConfig,
        handler: "SimHandlerVec",
    ) -> None:
        if handler.robot_cfg.name != FETCH_CFG.name:
            raise ValueError(
                f"FetchRobotCoreVec: handler is configured for robot "
                f"{handler.robot_cfg.name!r}; expected {FETCH_CFG.name!r}."
            )
        if handler.num_envs != config.num_envs:
            raise ValueError(
                f"FetchRobotCoreVec: handler.num_envs={handler.num_envs} "
                f"!= config.num_envs={config.num_envs}"
            )
        self._config = config
        self._handler = handler
        self._num_envs = handler.num_envs
        self._device = handler.device

        # Sampler ownership: same as scalar core. We sample one idx per
        # env per reset so each env gets a deterministic deterministic
        # scene sequence (sampler.sample_idx is keyed by (env_idx,
        # reset_count)).
        self._sampler = create_sampler(config.scene_sampler)
        self._scenes = self._sampler.scenes
        self._reset_count = 0

        import torch as _torch

        # Base offset calibration. Filled by _init_qpos_world_offset.
        # Stored as torch tensors on the handler's device so action
        # assembly stays device-local.
        self._qpos_base_indices: tuple[int, int, int] = (0, 0, 0)
        self._qpos_base_offset: _torch.Tensor = _torch.zeros(
            self._num_envs, 3, dtype=_torch.float64, device=self._device
        )
        self._init_qpos_world_offset()

        # Per-env actuator state. nan == "no target set".
        self._gripper_target: _torch.Tensor = _torch.zeros(
            self._num_envs, dtype=_torch.float32, device=self._device
        )
        self._head_target: _torch.Tensor = _torch.full(
            (self._num_envs, 2),
            float("nan"),
            dtype=_torch.float32,
            device=self._device,
        )

    # ── handler / cfg accessors ────────────────────────────────────────

    @property
    def handler(self) -> "SimHandlerVec":
        return self._handler

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def device(self) -> str:
        return self._device

    @property
    def camera_ids(self) -> list[str]:
        return [cam.camera_id for cam in FETCH_CFG.cameras]

    # ── base-offset calibration ────────────────────────────────────────

    def _init_qpos_world_offset(self) -> None:
        """Recompute the qpos↔world offset for the holonomic base, per env."""
        import torch as _torch

        bj = FETCH_CFG.base_joint_names
        ix = self._handler.joint_name_to_idx[bj[0]]
        iy = self._handler.joint_name_to_idx[bj[1]]
        ith = self._handler.joint_name_to_idx[bj[2]]
        self._qpos_base_indices = (ix, iy, ith)

        qpos = self._handler.get_qpos()  # (N, dof)
        qx = qpos[:, ix].to(_torch.float64)
        qy = qpos[:, iy].to(_torch.float64)
        qth = qpos[:, ith].to(_torch.float64)

        T = self._handler.get_link_pose(FETCH_CFG.base_link_name)  # (N, 4, 4)
        wx = T[:, 0, 3]
        wy = T[:, 1, 3]
        wth = _torch.atan2(T[:, 1, 0], T[:, 0, 0])
        dth = _torch.atan2(_torch.sin(wth - qth), _torch.cos(wth - qth))
        self._qpos_base_offset = _torch.stack([wx - qx, wy - qy, dth], dim=-1).to(
            _torch.float64
        )

    # ── observation parsing (qpos → typed structures) ──────────────────

    def _build_robot_state(self, qpos: "torch.Tensor") -> RobotStateVec:
        return RobotStateVec(
            base_xy_theta=self._compute_base_xy_theta(qpos),
            planning_joints=self._extract_planning_joints(qpos),
            head_joints=self._extract_head_joints(qpos),
        )

    def _compute_base_xy_theta(self, qpos: "torch.Tensor") -> "torch.Tensor":
        import torch as _torch

        ix, iy, ith = self._qpos_base_indices
        dx = self._qpos_base_offset[:, 0]
        dy = self._qpos_base_offset[:, 1]
        dth = self._qpos_base_offset[:, 2]
        qx = qpos[:, ix].to(_torch.float64)
        qy = qpos[:, iy].to(_torch.float64)
        qth = qpos[:, ith].to(_torch.float64)
        x = qx + dx
        y = qy + dy
        # Wrap theta into (-pi, pi].
        th = qth + dth
        th = _torch.atan2(_torch.sin(th), _torch.cos(th))
        return _torch.stack([x, y, th], dim=-1)

    def _extract_planning_joints(self, qpos: "torch.Tensor") -> "torch.Tensor":
        import torch as _torch

        idxs = [
            self._handler.joint_name_to_idx[n] for n in FETCH_CFG.planning_joint_names
        ]
        idx_t = _torch.as_tensor(idxs, dtype=_torch.long, device=qpos.device)
        return qpos.index_select(1, idx_t)

    def _extract_head_joints(self, qpos: "torch.Tensor") -> "torch.Tensor":
        import torch as _torch

        idxs = [self._handler.joint_name_to_idx[n] for n in FETCH_CFG.head_joint_names]
        idx_t = _torch.as_tensor(idxs, dtype=_torch.long, device=qpos.device)
        return qpos.index_select(1, idx_t)

    def _build_sensor_snapshot(self, camera_id: str) -> SensorSnapshotVec:
        rgb, depth, seg = self._handler.get_camera(camera_id)
        qpos = self._handler.get_qpos()
        return SensorSnapshotVec(
            rgb=rgb,
            depth=depth,
            intrinsics=self._handler.get_intrinsics(camera_id),
            robot_state=self._build_robot_state(qpos),
            segmentation=seg,
        )

    # ── public reads ────────────────────────────────────────────────────

    def get_sensor_snapshot(self, camera_id: str) -> SensorSnapshotVec:
        return self._build_sensor_snapshot(camera_id)

    def get_robot_state(self) -> RobotStateVec:
        return self._build_robot_state(self._handler.get_qpos())

    def get_observation(self) -> SensorSnapshotVec:
        return self._build_sensor_snapshot("head")

    # ── head target sink (set externally — see TyGrit.gaze) ─────────────

    def set_head_target(self, pan: "torch.Tensor", tilt: "torch.Tensor") -> None:
        """Set per-env absolute head pan/tilt targets in radians.

        ``pan`` and ``tilt`` are length-``num_envs`` tensors. Vec gaze
        modules write here; the per-step PD in :meth:`_compute_head_pd`
        consumes it.
        """
        import torch as _torch

        if pan.shape != (self._num_envs,) or tilt.shape != (self._num_envs,):
            raise ValueError(
                f"FetchRobotCoreVec.set_head_target: pan/tilt must be "
                f"shape ({self._num_envs},); got pan={pan.shape}, "
                f"tilt={tilt.shape}"
            )
        self._head_target = _torch.stack([pan, tilt], dim=-1).to(
            dtype=_torch.float32, device=self._device
        )

    # ── head PD controller (used by action assembly) ───────────────────

    def _compute_head_pd(self) -> "torch.Tensor":
        """Per-env (pan_vel, tilt_vel) tensor of shape (N, 2)."""
        import torch as _torch

        state = self.get_robot_state()
        current = state.head_joints.to(_torch.float32)  # (N, 2)
        target = self._head_target  # (N, 2), nan when unset
        kp = float(self._config.gaze_kp)
        max_vel = float(self._config.gaze_max_vel)

        err = target - current  # nan propagates → zeroed below
        err = _torch.nan_to_num(err, nan=0.0)
        vel = _torch.clamp(kp * err, min=-max_vel, max=max_vel)
        return vel

    # ── action assembly ────────────────────────────────────────────────

    def _assemble_action(self, mpc_action: "torch.Tensor") -> "torch.Tensor":
        """Map per-env ``[v, w, torso_vel, *arm_vels]`` to the handler's
        per-controller layout.

        ``mpc_action`` shape: ``(num_envs, 10)``. Returned action shape:
        ``(num_envs, total_action_dim)``.
        """
        import torch as _torch

        slices = self._handler.action_slices
        dim = self._handler.total_action_dim
        action = _torch.zeros(
            (self._num_envs, dim), dtype=_torch.float32, device=self._device
        )

        if "base" in slices:
            action[:, slices["base"]] = mpc_action[:, 0:2]

        if "arm" in slices:
            sl = slices["arm"]
            n = sl.stop - sl.start
            action[:, sl] = mpc_action[:, 3 : 3 + n]

        if "body" in slices:
            head_pd = self._compute_head_pd()  # (N, 2)
            torso_vel = mpc_action[:, 2:3]  # (N, 1)
            body = _torch.cat([head_pd, torso_vel], dim=-1)
            action[:, slices["body"]] = body

        if "gripper" in slices:
            # [0, 1] → [-1, 1] mapping per env.
            gripper_action = 2.0 * self._gripper_target - 1.0
            sl = slices["gripper"]
            n = sl.stop - sl.start
            # Broadcast scalar-per-env across the slice.
            action[:, sl] = gripper_action.unsqueeze(-1).expand(-1, n)

        return _torch.nan_to_num(action, nan=0.0)

    # ── stepping ────────────────────────────────────────────────────────

    def step(self, action: "torch.Tensor") -> SensorSnapshotVec:
        low_level_action = self._assemble_action(action)
        self._handler.apply_action(low_level_action)
        if self._config.sim_opts.get("render_mode") == "human":
            self._handler.render()
        return self._build_sensor_snapshot("head")

    # ── end-effector ────────────────────────────────────────────────────

    def control_gripper(self, position: "torch.Tensor") -> None:
        """Per-env gripper target in [0, 1]. ``position`` is shape (N,)."""
        import torch as _torch

        if position.shape != (self._num_envs,):
            raise ValueError(
                f"FetchRobotCoreVec.control_gripper: position must be "
                f"shape ({self._num_envs},); got {position.shape}"
            )
        self._gripper_target = _torch.clamp(
            position.to(dtype=_torch.float32, device=self._device),
            min=0.0,
            max=1.0,
        )

    # ── lifecycle ────────────────────────────────────────────────────────

    def close(self) -> None:
        self._handler.close()

    # ── spawn randomisation ─────────────────────────────────────────────

    def _randomize_robot_pose(self, seed: int | None = None) -> None:
        """Sample one (x, y, θ) per env from the navmesh and teleport
        via :meth:`SimHandlerVec.set_base_pose`."""
        import torch as _torch

        nav_meshes = self._handler.get_navigable_positions()
        default_x, default_y, _theta = FETCH_CFG.default_spawn_pose

        rng = np.random.default_rng(seed)
        xy_theta_rows = []
        for env_idx in range(self._num_envs):
            mesh = nav_meshes[env_idx] if env_idx < len(nav_meshes) else None
            if mesh is None or len(mesh.vertices) == 0:
                x, y = float(default_x), float(default_y)
            else:
                verts = np.asarray(mesh.vertices)
                vidx = int(rng.integers(0, len(verts)))
                x, y = float(verts[vidx, 0]), float(verts[vidx, 1])
            theta = float(rng.uniform(-np.pi, np.pi))
            xy_theta_rows.append([x, y, theta])

        xy_theta = _torch.tensor(
            xy_theta_rows, dtype=_torch.float32, device=self._device
        )
        self._handler.set_base_pose(xy_theta)

    # ── reset ───────────────────────────────────────────────────────────

    def reset(
        self,
        seed: int | None = None,
        randomize_init: bool = True,
        settle_steps: int = 0,
    ) -> SensorSnapshotVec:
        """Reset every env to a fresh sampler-selected scene; recalibrate
        the holonomic base offset; optionally randomise spawn pose.

        The sampler picks one idx per env keyed by ``(env_idx,
        reset_count)`` so each env gets its own deterministic scene
        sequence (the v1 repeating-scene fix applied per env).

        ``settle_steps`` runs that many zero-action steps after reset
        so objects can settle out of any initial interpenetration
        before observations are returned. Callers doing randomised
        spawns against a cluttered scene typically pass 5-20.
        """
        import torch as _torch

        self._reset_count += 1
        idxs = [
            self._sampler.sample_idx(env_idx=ei, reset_count=self._reset_count)
            for ei in range(self._num_envs)
        ]
        self._handler.reset_to_scene_idx(idxs, seed=seed)
        if randomize_init:
            self._randomize_robot_pose(seed=seed)
        self._init_qpos_world_offset()
        self._gripper_target = _torch.zeros(
            self._num_envs, dtype=_torch.float32, device=self._device
        )
        self._head_target = _torch.full(
            (self._num_envs, 2),
            float("nan"),
            dtype=_torch.float32,
            device=self._device,
        )
        if settle_steps > 0:
            zero_action = _torch.zeros(
                self._num_envs,
                self._handler.total_action_dim,
                dtype=_torch.float32,
                device=self._device,
            )
            for _ in range(settle_steps):
                self._handler.apply_action(zero_action)
        if self._config.sim_opts.get("render_mode") == "human":
            self._handler.render()
        return self._build_sensor_snapshot("head")


__all__ = ["FetchRobotCoreVec"]
