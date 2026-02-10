"""Grasp pose sampler — generates a reachable 8-DOF arm goal for grasping.

Pipeline:  segmentation mask → object point cloud (camera frame)
           → GraspGen prediction → world-frame transform
           → IK (base_link frame) → collision validation → 8-DOF joints.

The entire FK→camera→grasp→IK chain uses TyGrit's own kinematics so the
frames are internally consistent regardless of the simulation backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from loguru import logger

from TyGrit.types.robot import RobotState, WholeBodyConfig
from TyGrit.types.sensor import SensorSnapshot
from TyGrit.utils.depth import pointcloud_from_mask
from TyGrit.utils.grasping import T_GRASPGEN_TO_FETCH_EE
from TyGrit.utils.transforms import se2_to_matrix

if TYPE_CHECKING:
    from TyGrit.kinematics.ik import IKSolverBase
    from TyGrit.perception.grasping.graspgen import GraspGenPredictor
    from TyGrit.planning.fetch.vamp_preview import VampPreviewPlanner


@dataclass(frozen=True)
class GraspSamplerConfig:
    """Parameters for the grasp sampler."""

    target_object_id: int = 1
    grasp_depth_offset: float = 0.11  # retract along approach axis (metres)
    max_ik_attempts: int = 20  # per grasp pose
    max_grasps_to_try: int = 10


class GraspSampler:
    """Sample a reachable 8-DOF arm configuration for grasping.

    Constructor dependencies are injected so this class stays testable
    without heavyweight simulation or GPU models.
    """

    def __init__(
        self,
        ik_solver: IKSolverBase,
        grasp_predictor: GraspGenPredictor,
        planner: VampPreviewPlanner,
        config: GraspSamplerConfig | None = None,
    ) -> None:
        self._ik = ik_solver
        self._grasps = grasp_predictor
        self._planner = planner
        self._cfg = config or GraspSamplerConfig()

    def sample(
        self,
        snapshot: SensorSnapshot,
        camera_pose: npt.NDArray[np.float64],
        robot_state: RobotState,
    ) -> npt.NDArray[np.float64] | None:
        """Try to find a reachable 8-DOF grasp configuration.

        Args:
            snapshot: Current sensor observation (must include segmentation).
            camera_pose: (4, 4) camera-to-world matrix (OpenCV convention).
            robot_state: Current robot state (for IK seed and base pose).

        Returns:
            (8,) planning-joint array on success, or *None* if no valid
            grasp was found.
        """
        cfg = self._cfg

        # 1. Segmentation mask
        if snapshot.segmentation is None:
            logger.warning("GraspSampler: no segmentation in snapshot")
            return None

        mask = (snapshot.segmentation == cfg.target_object_id).astype(np.uint8)
        if not np.any(mask):
            logger.warning(
                "GraspSampler: target id={} not found in segmentation",
                cfg.target_object_id,
            )
            return None

        # 2. Object point cloud in camera frame
        cloud_cam = pointcloud_from_mask(
            snapshot.depth, mask.astype(bool), snapshot.intrinsics
        )
        if cloud_cam.shape[0] < 10:
            logger.warning(
                "GraspSampler: object cloud too small ({} pts)", cloud_cam.shape[0]
            )
            return None

        logger.info("GraspSampler: object cloud has {} points", cloud_cam.shape[0])

        # 3. Predict grasps (in camera frame, sorted by score descending)
        grasp_poses = self._grasps.predict(cloud_cam)
        if not grasp_poses:
            logger.warning("GraspSampler: GraspGen returned no grasps")
            return None

        logger.info("GraspSampler: {} grasp candidates", len(grasp_poses))

        # 4. For each grasp, attempt IK + collision check
        bp = robot_state.base_pose
        T_world_base = se2_to_matrix(bp.x, bp.y, bp.theta)
        T_base_world = np.linalg.inv(T_world_base)
        seed = np.array(robot_state.planning_joints, dtype=np.float64)

        for i, grasp in enumerate(grasp_poses[: cfg.max_grasps_to_try]):
            # Camera frame → world frame
            T_grasp_world = camera_pose @ grasp.transform

            # GraspGen convention → Fetch EE convention
            T_ee_world = T_grasp_world @ T_GRASPGEN_TO_FETCH_EE

            # Apply depth offset: retract along EE approach axis (+X in gripper_link)
            T_ee_world[:3, 3] -= cfg.grasp_depth_offset * T_ee_world[:3, 0]

            # World → base_link (IKFast expects base_link frame)
            T_ee_base = T_base_world @ T_ee_world

            # Try IK multiple times (each call samples random free params)
            for _ in range(cfg.max_ik_attempts):
                try:
                    joints = self._ik.solve(T_ee_base, seed=seed)
                except ValueError:
                    continue

                # Validate with planner (collision check)
                wb = WholeBodyConfig(arm_joints=joints, base_pose=bp)
                if self._planner.validate_config(wb):
                    logger.info(
                        "GraspSampler: valid grasp found (candidate {}, score={:.3f})",
                        i,
                        grasp.score,
                    )
                    return joints

        logger.warning("GraspSampler: exhausted all candidates, no valid grasp")
        return None
