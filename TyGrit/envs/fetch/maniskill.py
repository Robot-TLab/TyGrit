"""ManiSkill3-backed Fetch robot environment.

Concrete ``FetchRobot`` that drives a Fetch mobile manipulator inside a
ManiSkill3 simulation.  All control is synchronous — no background threads.

Ported from ``grasp_anywhere/envs/maniskill/maniskill_env_mpc.py`` as clean,
single-threaded code.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import torch
from mani_skill.utils.structs.types import GPUMemoryConfig, SceneConfig, SimConfig

from TyGrit.controller.fetch.mpc import (
    MPCConfig,
    compute_mpc_action,
    robot_state_to_mpc_state,
)
from TyGrit.envs.fetch.config import FetchEnvConfig
from TyGrit.envs.fetch.fetch import FetchRobot
from TyGrit.kinematics.fetch.constants import HEAD_JOINT_NAMES, PLANNING_JOINT_NAMES
from TyGrit.kinematics.fetch.fk_numpy import forward_kinematics
from TyGrit.types.geometry import SE2Pose
from TyGrit.types.planning import Trajectory
from TyGrit.types.robot import RobotState
from TyGrit.types.sensor import SensorSnapshot
from TyGrit.utils.tensor import to_numpy

# ── ManiSkill Fetch Robot ────────────────────────────────────────────────────


class ManiSkillFetchRobot(FetchRobot):
    """Fetch robot driven by ManiSkill3 simulation."""

    def __init__(
        self,
        config: FetchEnvConfig | None = None,
        mpc_config: MPCConfig | None = None,
    ) -> None:
        self._config = config or FetchEnvConfig()
        self._mpc_config = mpc_config

        # Create environment
        self._env = gym.make(
            self._config.env_id,
            robot_uids="fetch",
            obs_mode=self._config.obs_mode,
            control_mode=self._config.control_mode,
            render_mode=self._config.render_mode,
            sensor_configs={
                "width": self._config.camera_width,
                "height": self._config.camera_height,
            },
            sim_config=SimConfig(
                gpu_memory_config=GPUMemoryConfig(
                    found_lost_pairs_capacity=2**25,
                    max_rigid_patch_count=2**18,
                ),
                scene_config=SceneConfig(contact_offset=0.001),
            ),
        )

        self._agent = self._env.unwrapped.agent  # type: ignore[attr-defined]
        self._obs, _ = self._env.reset()

        # Build action slices: arm, gripper, body, base
        self._action_slices: dict[str, slice] = {}
        idx = 0
        for name in ("arm", "gripper", "body", "base"):
            controller = self._agent.controller.controllers.get(name)
            if controller is None:
                continue
            dim = controller.action_space.shape[0]
            self._action_slices[name] = slice(idx, idx + dim)
            idx += dim
        self._total_action_dim: int = self._env.action_space.shape[0]  # type: ignore[union-attr]

        # Build joint-name → index map
        self._joint_name_to_idx: dict[str, int] = {
            j.name: i for i, j in enumerate(self._agent.robot.active_joints)
        }

        # Calibrate base pose offset
        self._qpos_base_indices: tuple[int, int, int] = (0, 0, 0)
        self._qpos_base_offset = np.zeros(3, dtype=np.float64)
        self._init_qpos_world_offset()

        # Cache camera intrinsics (static)
        cam_params = self._env.unwrapped._sensors["fetch_head"].get_params()  # type: ignore[attr-defined]
        K = np.array(cam_params["intrinsic_cv"])
        if K.ndim == 3:
            K = K[0]
        self._intrinsics: npt.NDArray[np.float64] = K.astype(np.float64)

        # Trajectory state
        self._trajectory: Trajectory | None = None
        self._waypoint_idx: int = 0

        # Actuator targets
        self._gripper_target: float = 0.0
        self._head_target: tuple[float, float] = (float("nan"), float("nan"))

        # Initial render
        if self._config.render_mode == "human":
            self._env.render()

    # ── Base-offset calibration ───────────────────────────────────────────

    def _init_qpos_world_offset(self) -> None:
        """Compute offset between qpos base joints and world-frame base_link pose."""
        base_joint_names = self._agent.base_joint_names
        ix = self._joint_name_to_idx[base_joint_names[0]]
        iy = self._joint_name_to_idx[base_joint_names[1]]
        ith = self._joint_name_to_idx[base_joint_names[2]]
        self._qpos_base_indices = (ix, iy, ith)

        qpos = to_numpy(self._obs["state"])
        qx, qy, qth = float(qpos[ix]), float(qpos[iy]), float(qpos[ith])

        # Get world-frame pose of base_link
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

        wx = float(T[0, 3])
        wy = float(T[1, 3])
        wth = float(np.arctan2(T[1, 0], T[0, 0]))
        dth = float(np.arctan2(np.sin(wth - qth), np.cos(wth - qth)))
        self._qpos_base_offset = np.array([wx - qx, wy - qy, dth], dtype=np.float64)

    # ── Observation parsing ───────────────────────────────────────────────

    def _build_robot_state(self, qpos: np.ndarray) -> RobotState:
        """Build a RobotState from a qpos array (all from same observation)."""
        return RobotState(
            base_pose=self._compute_base_pose(qpos),
            planning_joints=self._extract_planning_joints(qpos),
            head_joints=self._extract_head_joints(qpos),
        )

    def _compute_base_pose(self, qpos: np.ndarray) -> SE2Pose:
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

    def _extract_planning_joints(self, qpos: np.ndarray) -> tuple[float, ...]:
        return tuple(
            float(qpos[self._joint_name_to_idx[n]]) for n in PLANNING_JOINT_NAMES
        )

    def _extract_head_joints(self, qpos: np.ndarray) -> tuple[float, ...]:
        return tuple(float(qpos[self._joint_name_to_idx[n]]) for n in HEAD_JOINT_NAMES)

    def _parse_observation(self, obs: dict) -> SensorSnapshot:
        """Parse a ManiSkill obs dict into a SensorSnapshot.

        All data (RGB, depth, robot state) is extracted from the same *obs*
        dict so they are guaranteed to be from the same simulation step.
        """
        sensor = obs["sensor_data"]["fetch_head"]

        rgb = sensor["rgb"]
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.detach().cpu().numpy()
        rgb = rgb[0]  # remove batch dim

        depth = sensor["depth"]
        if isinstance(depth, torch.Tensor):
            depth = depth.detach().cpu().numpy()
        depth = (depth[0, ..., 0].astype(np.float32)) / 1000.0  # mm → m, remove channel

        seg = sensor.get("segmentation")
        if seg is not None:
            if isinstance(seg, torch.Tensor):
                seg = seg.detach().cpu().numpy()
            seg = seg[0, ..., 0].astype(np.int32)

        qpos = to_numpy(obs["state"])

        return SensorSnapshot(
            rgb=rgb,
            depth=depth,
            intrinsics=self._intrinsics,
            robot_state=self._build_robot_state(qpos),
            segmentation=seg,
        )

    # ── RobotBase: sensing ────────────────────────────────────────────────

    @property
    def camera_ids(self) -> list[str]:
        return ["head"]

    def get_sensor_snapshot(self, camera_id: str) -> SensorSnapshot:
        if camera_id != "head":
            raise ValueError(
                f"Unknown camera_id {camera_id!r}; available: {self.camera_ids}"
            )
        return self._parse_observation(self._obs)

    def get_robot_state(self) -> RobotState:
        return self._build_robot_state(to_numpy(self._obs["state"]))

    def get_observation(self) -> SensorSnapshot:
        return self._parse_observation(self._obs)

    # ── RobotBase: active perception ──────────────────────────────────────

    def look_at(self, target: npt.NDArray[np.float64], camera_id: str) -> None:
        if camera_id != "head":
            raise NotImplementedError(f"Cannot steer camera {camera_id!r}")

        state = self.get_robot_state()

        # Build FK input: [torso, 7 arm, pan, tilt]
        fk_joints = np.array(
            [*state.planning_joints, *state.head_joints],
            dtype=np.float64,
        )
        link_poses = forward_kinematics(fk_joints)

        # Transform world target to base frame
        bp = state.base_pose
        cos_th, sin_th = np.cos(bp.theta), np.sin(bp.theta)
        R_wb = np.array([[cos_th, -sin_th], [sin_th, cos_th]])
        t_wb = np.array([bp.x, bp.y])
        target_base_xy = R_wb.T @ (target[:2] - t_wb)
        target_base = np.array([target_base_xy[0], target_base_xy[1], target[2]])

        # Compute relative pan in head_pan_link frame
        T_head_pan = link_poses["head_pan_link"]
        T_head_pan_inv = np.linalg.inv(T_head_pan)
        target_head_pan = (T_head_pan_inv @ np.append(target_base, 1.0))[:3]

        x, y, z = target_head_pan
        current_pan = float(state.head_joints[0])
        pan_rel = float(np.arctan2(y, x))

        # Tilt: vector from tilt joint origin to target in pan-aligned frame
        T_head_tilt = link_poses["head_tilt_link"]
        T_pan_tilt = T_head_pan_inv @ T_head_tilt
        tilt_origin_pan = T_pan_tilt[:3, 3]
        dist_xy = np.sqrt(x**2 + y**2)
        v_tilt_target = np.array([dist_xy, 0.0, z]) - tilt_origin_pan
        tilt_abs = float(np.arctan2(-v_tilt_target[2], v_tilt_target[0]))

        pan = current_pan + pan_rel
        self._head_target = (pan, tilt_abs)

    # ── Head PD controller ────────────────────────────────────────────────

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

    # ── Action assembly ───────────────────────────────────────────────────

    def _assemble_action(self, mpc_action: npt.NDArray[np.float32]) -> np.ndarray:
        """Map MPC output (10,) [v, w, torso_vel, 7 arm_vels] → ManiSkill action."""
        action = np.zeros(self._total_action_dim, dtype=np.float32)

        # Base: [v, w]
        if "base" in self._action_slices:
            action[self._action_slices["base"]] = mpc_action[0:2]

        # Arm: 7 joint velocities
        if "arm" in self._action_slices:
            sl = self._action_slices["arm"]
            n = sl.stop - sl.start
            action[sl] = mpc_action[3 : 3 + n]

        # Body: [pan_vel, tilt_vel, torso_vel]
        if "body" in self._action_slices:
            pan_vel, tilt_vel = self._compute_head_pd()
            torso_vel = float(mpc_action[2])
            action[self._action_slices["body"]] = np.array(
                [pan_vel, tilt_vel, torso_vel],
                dtype=np.float32,
            )

        # Gripper: map [0,1] → [-1,1]
        if "gripper" in self._action_slices:
            gripper_action = 2.0 * self._gripper_target - 1.0
            action[self._action_slices["gripper"]] = gripper_action

        return np.nan_to_num(action, nan=0.0)

    # ── RobotBase: stepping ───────────────────────────────────────────────

    def step(self, action: npt.NDArray[np.float32]) -> SensorSnapshot:
        ms_action = self._assemble_action(action)
        self._obs, _, _, _, _ = self._env.step(ms_action)
        if self._config.render_mode == "human":
            self._env.render()
        return self._parse_observation(self._obs)

    # ── RobotBase: trajectory / motion ────────────────────────────────────

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

    # ── RobotBase: end-effector ───────────────────────────────────────────

    def control_gripper(self, position: float) -> None:
        self._gripper_target = float(np.clip(position, 0.0, 1.0))

    # ── RobotBase: lifecycle ──────────────────────────────────────────────

    def close(self) -> None:
        self._env.close()

    # ── Extra: reset ──────────────────────────────────────────────────────

    def reset(self) -> SensorSnapshot:
        """Reset the environment and return a fresh observation."""
        self._obs, _ = self._env.reset()
        self._init_qpos_world_offset()
        self._trajectory = None
        self._waypoint_idx = 0
        self._gripper_target = 0.0
        self._head_target = (float("nan"), float("nan"))
        if self._config.render_mode == "human":
            self._env.render()
        return self._parse_observation(self._obs)
