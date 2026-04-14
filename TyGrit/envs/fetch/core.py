"""Sim-agnostic Fetch robot logic.

:class:`FetchRobotCore` implements the
:class:`~TyGrit.envs.base.RobotBase` protocol for a Fetch mobile
manipulator using a :class:`~TyGrit.envs.fetch.sim_backend.FetchSimBackend`
for everything sim-specific. Concrete robots (``ManiSkillFetchRobot``,
``GenesisFetchRobot``) are constructed by composing this core with
the backend appropriate to the sim.

What lives here (sim-agnostic):
    * Joint indexing into the qpos vector via the backend's
      ``joint_name_to_idx`` mapping.
    * Holonomic-base offset calibration (qpos↔world) using the
      backend's ``base_joint_names`` and ``get_base_link_world_pose``.
    * Fetch action assembly: maps the planning-layer 10-vector
      (``[v, w, torso, *arm_velocities]``) plus the gripper target
      and head PD output into the backend's per-controller slices.
    * :meth:`look_at` — pure FK math against
      :func:`~TyGrit.kinematics.fetch.fk_numpy.forward_kinematics`,
      sets the head target consumed by the PD step.
    * :meth:`execute_trajectory` — synchronous waypoint loop using
      :func:`~TyGrit.controller.fetch.mpc.compute_mpc_action`.
    * Scene sampler ownership: each :meth:`reset` advances the
      sampler's ``reset_count`` so the deterministic scene sequence
      stays consistent regardless of which backend is plugged in.
    * Spawn-pose randomisation against
      ``backend.get_navigable_positions()``.

What lives in the backend:
    * Sim env construction.
    * Observation parsing → cached numpy arrays.
    * Action plumbing into the sim.
    * Reset/reconfigure / teleport / render / close.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from TyGrit.controller.fetch.mpc import (
    MPCConfig,
    compute_mpc_action,
    robot_state_to_mpc_state,
)
from TyGrit.envs.fetch.config import FetchEnvConfig
from TyGrit.envs.fetch.sim_backend import FetchSimBackend
from TyGrit.kinematics.fetch.fk_numpy import forward_kinematics
from TyGrit.robots import FETCH_SPEC
from TyGrit.types.geometry import SE2Pose
from TyGrit.types.planning import Trajectory
from TyGrit.types.robot import RobotState
from TyGrit.types.sensor import SensorSnapshot
from TyGrit.worlds.sampler import create_sampler


class FetchRobotCore:
    """Sim-agnostic Fetch robot composing a :class:`FetchSimBackend`.

    Satisfies :class:`~TyGrit.envs.base.RobotBase` via duck typing —
    no inheritance because ``RobotBase`` is a Protocol.

    Constructed with the env config, the sim-specific backend, and an
    optional MPC config. The core does the once-per-reset base offset
    calibration via the backend during ``__init__`` and again at the
    end of every :meth:`reset`.
    """

    def __init__(
        self,
        config: FetchEnvConfig,
        backend: FetchSimBackend,
        mpc_config: MPCConfig | None = None,
    ) -> None:
        self._config = config
        self._backend = backend
        self._mpc_config = mpc_config

        # Sampler ownership lives at this layer because scene selection
        # (manifest → indices) is sim-agnostic. Backends just consume
        # the indices via reset_to_idx.
        self._sampler = create_sampler(config.scene_sampler)
        self._scenes = self._sampler.scenes
        self._reset_count = 0

        # Base offset calibration. Filled in by _init_qpos_world_offset.
        self._qpos_base_indices: tuple[int, int, int] = (0, 0, 0)
        self._qpos_base_offset: npt.NDArray[np.float64] = np.zeros(3, dtype=np.float64)
        self._init_qpos_world_offset()

        # Trajectory + actuator state (sim-agnostic).
        self._trajectory: Trajectory | None = None
        self._waypoint_idx: int = 0
        self._gripper_target: float = 0.0
        self._head_target: tuple[float, float] = (float("nan"), float("nan"))

    # ── base-offset calibration ────────────────────────────────────────

    def _init_qpos_world_offset(self) -> None:
        """Recompute the qpos↔world offset for the holonomic base.

        Run once at ``__init__`` and after every :meth:`reset`. The
        qpos base joints encode the base in their own frame; the
        world-frame ``base_link`` pose comes from the sim. The delta
        is the (x, y, theta) translation we add when reading the
        robot state.
        """
        bj = self._backend.base_joint_names
        ix = self._backend.joint_name_to_idx[bj[0]]
        iy = self._backend.joint_name_to_idx[bj[1]]
        ith = self._backend.joint_name_to_idx[bj[2]]
        self._qpos_base_indices = (ix, iy, ith)

        qpos = self._backend.get_qpos()
        qx, qy, qth = float(qpos[ix]), float(qpos[iy]), float(qpos[ith])
        T = self._backend.get_base_link_world_pose()
        wx = float(T[0, 3])
        wy = float(T[1, 3])
        wth = float(np.arctan2(T[1, 0], T[0, 0]))
        dth = float(np.arctan2(np.sin(wth - qth), np.cos(wth - qth)))
        self._qpos_base_offset = np.array([wx - qx, wy - qy, dth], dtype=np.float64)

    # ── observation parsing (qpos → typed structures) ──────────────────

    def _build_robot_state(self, qpos: npt.NDArray[np.float64]) -> RobotState:
        return RobotState(
            base_pose=self._compute_base_pose(qpos),
            planning_joints=self._extract_planning_joints(qpos),
            head_joints=self._extract_head_joints(qpos),
        )

    def _compute_base_pose(self, qpos: npt.NDArray[np.float64]) -> SE2Pose:
        ix, iy, ith = self._qpos_base_indices
        dx, dy, dth = self._qpos_base_offset
        x = float(qpos[ix]) + dx
        y = float(qpos[iy]) + dy
        th = float(
            np.arctan2(
                np.sin(float(qpos[ith]) + dth),
                np.cos(float(qpos[ith]) + dth),
            )
        )
        return SE2Pose(x=x, y=y, theta=th)

    def _extract_planning_joints(
        self, qpos: npt.NDArray[np.float64]
    ) -> tuple[float, ...]:
        return tuple(
            float(qpos[self._backend.joint_name_to_idx[n]])
            for n in FETCH_SPEC.planning_joint_names
        )

    def _extract_head_joints(self, qpos: npt.NDArray[np.float64]) -> tuple[float, ...]:
        return tuple(
            float(qpos[self._backend.joint_name_to_idx[n]])
            for n in FETCH_SPEC.head_joint_names
        )

    def _build_sensor_snapshot(self, camera_id: str) -> SensorSnapshot:
        rgb, depth, seg = self._backend.parse_camera(camera_id)
        qpos = self._backend.get_qpos()
        return SensorSnapshot(
            rgb=rgb,
            depth=depth,
            intrinsics=self._backend.intrinsics,
            robot_state=self._build_robot_state(qpos),
            segmentation=seg,
        )

    # ── RobotBase: sensing ─────────────────────────────────────────────

    @property
    def camera_ids(self) -> list[str]:
        return ["head"]

    def get_sensor_snapshot(self, camera_id: str) -> SensorSnapshot:
        if camera_id != "head":
            raise ValueError(
                f"Unknown camera_id {camera_id!r}; available: {self.camera_ids}"
            )
        return self._build_sensor_snapshot("head")

    def get_robot_state(self) -> RobotState:
        return self._build_robot_state(self._backend.get_qpos())

    def get_observation(self) -> SensorSnapshot:
        return self._build_sensor_snapshot("head")

    # ── RobotBase: active perception ──────────────────────────────────

    def look_at(self, target: npt.NDArray[np.float64], camera_id: str) -> None:
        """Aim the head camera at a 3-D world-frame ``target`` via FK + IK math.

        Sets ``self._head_target`` (pan, tilt). The actual head motion
        happens via the PD controller on each :meth:`step` call.
        """
        if camera_id != "head":
            raise NotImplementedError(f"Cannot steer camera {camera_id!r}")

        state = self.get_robot_state()

        # Build FK input: [torso, 7 arm, pan, tilt]
        fk_joints = np.array(
            [*state.planning_joints, *state.head_joints],
            dtype=np.float64,
        )
        link_poses = forward_kinematics(fk_joints)

        # Transform world target to base frame.
        bp = state.base_pose
        cos_th, sin_th = np.cos(bp.theta), np.sin(bp.theta)
        R_wb = np.array([[cos_th, -sin_th], [sin_th, cos_th]])
        t_wb = np.array([bp.x, bp.y])
        target_base_xy = R_wb.T @ (target[:2] - t_wb)
        target_base = np.array([target_base_xy[0], target_base_xy[1], target[2]])

        # Compute relative pan in head_pan_link frame.
        T_head_pan = link_poses["head_pan_link"]
        T_head_pan_inv = np.linalg.inv(T_head_pan)
        target_head_pan = (T_head_pan_inv @ np.append(target_base, 1.0))[:3]

        x, y, z = target_head_pan
        current_pan = float(state.head_joints[0])
        pan_rel = float(np.arctan2(y, x))

        # Tilt: vector from tilt joint origin to target in pan-aligned frame.
        T_head_tilt = link_poses["head_tilt_link"]
        T_pan_tilt = T_head_pan_inv @ T_head_tilt
        tilt_origin_pan = T_pan_tilt[:3, 3]
        dist_xy = np.sqrt(x**2 + y**2)
        v_tilt_target = np.array([dist_xy, 0.0, z]) - tilt_origin_pan
        tilt_abs = float(np.arctan2(-v_tilt_target[2], v_tilt_target[0]))

        pan = current_pan + pan_rel
        self._head_target = (pan, tilt_abs)

    # ── head PD controller (used by action assembly) ───────────────────

    def _compute_head_pd(self) -> tuple[float, float]:
        state = self.get_robot_state()
        current_pan, current_tilt = state.head_joints
        target_pan, target_tilt = self._head_target

        kp = self._config.gaze_kp
        max_vel = self._config.gaze_max_vel

        pan_err = 0.0 if np.isnan(target_pan) else target_pan - current_pan
        tilt_err = 0.0 if np.isnan(target_tilt) else target_tilt - current_tilt

        pan_vel = float(np.clip(kp * pan_err, -max_vel, max_vel))
        tilt_vel = float(np.clip(kp * tilt_err, -max_vel, max_vel))
        return pan_vel, tilt_vel

    # ── action assembly ────────────────────────────────────────────────

    def _assemble_action(
        self, mpc_action: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Map MPC output ``[v, w, torso_vel, *arm_vels]`` to the
        backend's low-level action layout.

        Reads :attr:`FetchSimBackend.action_slices` and
        :attr:`FetchSimBackend.total_action_dim` so the same logic
        works for any backend that exposes the standard four
        controllers (``arm``, ``gripper``, ``body``, ``base``).
        """
        slices = self._backend.action_slices
        dim = self._backend.total_action_dim
        action = np.zeros(dim, dtype=np.float32)

        # Base: [v, w]
        if "base" in slices:
            action[slices["base"]] = mpc_action[0:2]

        # Arm: 7 joint velocities
        if "arm" in slices:
            sl = slices["arm"]
            n = sl.stop - sl.start
            action[sl] = mpc_action[3 : 3 + n]

        # Body: [pan_vel, tilt_vel, torso_vel]
        if "body" in slices:
            pan_vel, tilt_vel = self._compute_head_pd()
            torso_vel = float(mpc_action[2])
            action[slices["body"]] = np.array(
                [pan_vel, tilt_vel, torso_vel], dtype=np.float32
            )

        # Gripper: map [0, 1] → [-1, 1]
        if "gripper" in slices:
            gripper_action = 2.0 * self._gripper_target - 1.0
            action[slices["gripper"]] = gripper_action

        return np.nan_to_num(action, nan=0.0)

    # ── RobotBase: stepping ────────────────────────────────────────────

    def step(self, action: npt.NDArray[np.float32]) -> SensorSnapshot:
        ms_action = self._assemble_action(action)
        self._backend.step(ms_action)
        if self._config.render_mode == "human":
            self._backend.render()
        return self._build_sensor_snapshot("head")

    # ── RobotBase: trajectory / motion ────────────────────────────────

    def start_trajectory(self, trajectory: Trajectory) -> None:
        self._trajectory = trajectory
        self._waypoint_idx = 0

    def is_motion_done(self) -> bool:
        return self._trajectory is None or self._waypoint_idx >= len(
            self._trajectory.arm_path
        )

    def stop_motion(self) -> None:
        self._trajectory = None
        self._waypoint_idx = 0

    def execute_trajectory(self, trajectory: Trajectory) -> bool:
        cfg = self._config
        for arm_wp, base_wp in zip(trajectory.arm_path, trajectory.base_configs):
            x_ref = np.array(
                [base_wp.x, base_wp.y, base_wp.theta, *arm_wp],
                dtype=np.float64,
            )
            for _ in range(cfg.max_steps_per_waypoint):
                state = self.get_robot_state()
                x = robot_state_to_mpc_state(state)
                error = float(np.linalg.norm(x_ref - x))
                if error < cfg.convergence_threshold:
                    break
                u = compute_mpc_action(x, x_ref, self._mpc_config)
                self.step(u)
        return True

    # ── RobotBase: end-effector ───────────────────────────────────────

    def control_gripper(self, position: float) -> None:
        self._gripper_target = float(np.clip(position, 0.0, 1.0))

    # ── RobotBase: lifecycle ──────────────────────────────────────────

    def close(self) -> None:
        self._backend.close()

    # ── spawn randomisation ────────────────────────────────────────────

    def _randomize_robot_pose(self, seed: int | None = None) -> None:
        """Sample one (x, y, θ) per parallel env from the navmesh and
        teleport.

        Called from :meth:`reset` after the backend has reset the env.
        Falls back to ``(-1, 0, *)`` (ManiSkill's default Fetch spawn)
        when the active scene has no navmesh — Holodeck specifically
        ships none, so this branch is exercised in real workloads.
        """
        nav_meshes = self._backend.get_navigable_positions()
        num_envs = self._backend.num_envs

        rng = np.random.default_rng(seed)
        for env_idx in range(num_envs):
            mesh = nav_meshes[env_idx] if env_idx < len(nav_meshes) else None
            if mesh is None or len(mesh.vertices) == 0:
                # Default fallback used by ManiSkill's Fetch spawn when
                # no navmesh is configured.
                x, y = -1.0, 0.0
            else:
                verts = np.asarray(mesh.vertices)
                vidx = int(rng.integers(0, len(verts)))
                x, y = float(verts[vidx, 0]), float(verts[vidx, 1])
            theta = float(rng.uniform(-np.pi, np.pi))
            self._backend.set_base_pose(x, y, theta, env_idx=env_idx)

    # ── RobotBase: reset ──────────────────────────────────────────────

    def reset(
        self, seed: int | None = None, randomize_init: bool = True
    ) -> SensorSnapshot:
        """Reset to a fresh sampler-selected scene; recalibrate offset.

        Increments ``_reset_count`` and asks the sampler for a new
        ``(env_idx=0, reset_count)`` selection — the deterministic
        per-reset switch that fixes the v1 repeating-scene bug.
        """
        self._reset_count += 1
        idx = self._sampler.sample_idx(env_idx=0, reset_count=self._reset_count)
        self._backend.reset_to_idx(idx, seed=seed)
        if randomize_init:
            self._randomize_robot_pose(seed=seed)
        self._init_qpos_world_offset()
        self._trajectory = None
        self._waypoint_idx = 0
        self._gripper_target = 0.0
        self._head_target = (float("nan"), float("nan"))
        if self._config.render_mode == "human":
            self._backend.render()
        return self._build_sensor_snapshot("head")
