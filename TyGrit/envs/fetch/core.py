"""Sim-agnostic Fetch robot logic.

:class:`FetchRobotCore` implements the
:class:`~TyGrit.envs.base.RobotBase` protocol for a Fetch mobile
manipulator using a :class:`~TyGrit.sim.base.SimHandler` for everything
sim-specific. Concrete robots (``ManiSkillFetchRobot``,
``GenesisFetchRobot``, future hardware Fetch) are constructed by
composing this core with the handler appropriate to the sim — adding a
new sim adds **zero** code here.

What lives here (sim-agnostic, Fetch-specific):
    * Joint indexing into qpos via the handler's
      :attr:`~SimHandler.joint_name_to_idx` mapping.
    * Holonomic-base offset calibration (qpos↔world) using
      :data:`~TyGrit.robots.fetch.FETCH_CFG`'s
      ``base_joint_names`` / ``base_link_name`` and the handler's
      :meth:`~SimHandler.get_link_pose`.
    * Fetch action assembly: maps the planning-layer 10-vector
      ``[v, w, torso, *arm_velocities]`` plus gripper target and head
      PD output into the handler's per-controller :attr:`action_slices`.
    * :meth:`look_at` — pure FK / IK math against
      :func:`~TyGrit.robots.fetch.kinematics.fk_numpy.forward_kinematics`,
      sets the head target consumed by the PD step.
    * :meth:`execute_trajectory` — synchronous waypoint loop using
      :func:`~TyGrit.controller.fetch.mpc.compute_mpc_action`.
    * Scene-sampler ownership: :meth:`reset` advances the sampler's
      ``reset_count`` so the deterministic scene sequence stays
      consistent regardless of which sim is plugged in.
    * Spawn-pose randomisation against
      ``handler.get_navigable_positions()``.

What lives in the handler (sim-specific):
    * Sim env construction.
    * Observation parsing → cached numpy arrays.
    * Action plumbing into the sim.
    * Reset / reconfigure / teleport / render / close.
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
from TyGrit.robots.fetch import FETCH_CFG
from TyGrit.robots.fetch.kinematics.fk_numpy import forward_kinematics
from TyGrit.sim.base import SimHandler
from TyGrit.types.geometry import SE2Pose
from TyGrit.types.planning import Trajectory
from TyGrit.types.robots import RobotState
from TyGrit.types.sensors import SensorSnapshot
from TyGrit.worlds.sampler import create_sampler


class FetchRobotCore:
    """Sim-agnostic Fetch robot composing a :class:`SimHandler`.

    Satisfies :class:`~TyGrit.envs.base.RobotBase` via duck typing —
    no inheritance because ``RobotBase`` is a Protocol.

    Parameters
    ----------
    config
        Fetch-specific env / scheduler configuration.
    handler
        Any :class:`SimHandler` constructed against ``FETCH_CFG``. The
        core asserts the handler's ``robot_cfg.name == "fetch"`` so a
        wiring bug surfaces immediately rather than via a confusing
        joint-name lookup failure later.
    mpc_config
        Optional MPC tuning override.

    The core does once-per-reset base-offset calibration via the
    handler during ``__init__`` and again at the end of every
    :meth:`reset`.
    """

    def __init__(
        self,
        config: FetchEnvConfig,
        handler: SimHandler,
        mpc_config: MPCConfig | None = None,
    ) -> None:
        if handler.robot_cfg.name != FETCH_CFG.name:
            raise ValueError(
                f"FetchRobotCore: handler is configured for robot "
                f"{handler.robot_cfg.name!r}; expected {FETCH_CFG.name!r}. "
                f"This is the wrong handler for a Fetch core."
            )
        self._config = config
        self._handler = handler
        self._mpc_config = mpc_config

        # Sampler ownership lives at this layer because scene selection
        # (manifest → indices) is sim-agnostic. Handlers just consume
        # the indices via reset_to_scene_idx.
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

    # ── handler / cfg accessors ────────────────────────────────────────

    @property
    def handler(self) -> SimHandler:
        """The underlying :class:`SimHandler`. Useful for tests and
        sim-specific extensions; production code should prefer the
        :class:`RobotBase` API."""
        return self._handler

    # ── base-offset calibration ────────────────────────────────────────

    def _init_qpos_world_offset(self) -> None:
        """Recompute the qpos↔world offset for the holonomic base.

        Run once at ``__init__`` and after every :meth:`reset`. The
        qpos base joints encode the base in their own frame; the
        world-frame ``base_link`` pose comes from the handler. The
        delta is the (x, y, theta) translation we add when reading
        the robot state.
        """
        bj = FETCH_CFG.base_joint_names
        ix = self._handler.joint_name_to_idx[bj[0]]
        iy = self._handler.joint_name_to_idx[bj[1]]
        ith = self._handler.joint_name_to_idx[bj[2]]
        self._qpos_base_indices = (ix, iy, ith)

        qpos = self._handler.get_qpos()
        qx, qy, qth = float(qpos[ix]), float(qpos[iy]), float(qpos[ith])
        T = self._handler.get_link_pose(FETCH_CFG.base_link_name)
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
            float(qpos[self._handler.joint_name_to_idx[n]])
            for n in FETCH_CFG.planning_joint_names
        )

    def _extract_head_joints(self, qpos: npt.NDArray[np.float64]) -> tuple[float, ...]:
        return tuple(
            float(qpos[self._handler.joint_name_to_idx[n]])
            for n in FETCH_CFG.head_joint_names
        )

    def _build_sensor_snapshot(self, camera_id: str) -> SensorSnapshot:
        rgb, depth, seg = self._handler.get_camera(camera_id)
        qpos = self._handler.get_qpos()
        return SensorSnapshot(
            rgb=rgb,
            depth=depth,
            intrinsics=self._handler.get_intrinsics(camera_id),
            robot_state=self._build_robot_state(qpos),
            segmentation=seg,
        )

    # ── RobotBase: sensing ─────────────────────────────────────────────

    @property
    def camera_ids(self) -> list[str]:
        return [c.camera_id for c in FETCH_CFG.cameras]

    def get_sensor_snapshot(self, camera_id: str) -> SensorSnapshot:
        # camera_by_id raises KeyError on typo; same effect as the
        # legacy ValueError but with a more useful message.
        FETCH_CFG.camera_by_id(camera_id)
        return self._build_sensor_snapshot(camera_id)

    def get_robot_state(self) -> RobotState:
        return self._build_robot_state(self._handler.get_qpos())

    def get_observation(self) -> SensorSnapshot:
        # Default observation is from the head camera. Multi-camera
        # observations are constructed per-call by the planner / task
        # layer via repeated get_sensor_snapshot calls.
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
        handler's low-level action layout.

        Reads :attr:`SimHandler.action_slices` and
        :attr:`SimHandler.total_action_dim` so the same logic works
        for any handler that exposes the standard four controllers
        (``arm``, ``gripper``, ``body``, ``base``).
        """
        slices = self._handler.action_slices
        dim = self._handler.total_action_dim
        action = np.zeros(dim, dtype=np.float32)

        # Base: [v, w] (Cartesian twist for kind="base_twist").
        if "base" in slices:
            action[slices["base"]] = mpc_action[0:2]

        # Arm: 7 joint velocities.
        if "arm" in slices:
            sl = slices["arm"]
            n = sl.stop - sl.start
            action[sl] = mpc_action[3 : 3 + n]

        # Body: [pan_vel, tilt_vel, torso_vel].
        if "body" in slices:
            pan_vel, tilt_vel = self._compute_head_pd()
            torso_vel = float(mpc_action[2])
            action[slices["body"]] = np.array(
                [pan_vel, tilt_vel, torso_vel], dtype=np.float32
            )

        # Gripper: map [0, 1] → [-1, 1] (one scalar; the actuator's
        # command_to_joint_mapping fans it across the two finger joints).
        if "gripper" in slices:
            gripper_action = 2.0 * self._gripper_target - 1.0
            action[slices["gripper"]] = gripper_action

        return np.nan_to_num(action, nan=0.0)

    # ── RobotBase: stepping ────────────────────────────────────────────

    def step(self, action: npt.NDArray[np.float32]) -> SensorSnapshot:
        low_level_action = self._assemble_action(action)
        self._handler.apply_action(low_level_action)
        if self._config.render_mode == "human":
            self._handler.render()
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
        self._handler.close()

    # ── spawn randomisation ────────────────────────────────────────────

    def _randomize_robot_pose(self, seed: int | None = None) -> None:
        """Sample one (x, y, θ) per parallel env from the navmesh and
        teleport.

        Called from :meth:`reset` after the handler has reset the env.
        Falls back to :attr:`FETCH_CFG.default_spawn_pose` when the
        active scene has no navmesh — Holodeck specifically ships
        none, so this branch is exercised in real workloads.
        """
        nav_meshes = self._handler.get_navigable_positions()
        num_envs = self._handler.num_envs
        default_x, default_y, _ = FETCH_CFG.default_spawn_pose

        rng = np.random.default_rng(seed)
        for env_idx in range(num_envs):
            mesh = nav_meshes[env_idx] if env_idx < len(nav_meshes) else None
            if mesh is None or len(mesh.vertices) == 0:
                x, y = float(default_x), float(default_y)
            else:
                verts = np.asarray(mesh.vertices)
                vidx = int(rng.integers(0, len(verts)))
                x, y = float(verts[vidx, 0]), float(verts[vidx, 1])
            theta = float(rng.uniform(-np.pi, np.pi))
            self._handler.set_base_pose(x, y, theta, env_idx=env_idx)

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
        # Genesis rejects seed (rebuilds scene; deterministic via idx);
        # ManiSkill honours it. Pass through and let the handler decide.
        self._handler.reset_to_scene_idx(idx, seed=seed)
        if randomize_init:
            self._randomize_robot_pose(seed=seed)
        self._init_qpos_world_offset()
        self._trajectory = None
        self._waypoint_idx = 0
        self._gripper_target = 0.0
        self._head_target = (float("nan"), float("nan"))
        if self._config.render_mode == "human":
            self._handler.render()
        return self._build_sensor_snapshot("head")
