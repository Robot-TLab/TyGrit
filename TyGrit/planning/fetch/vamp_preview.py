"""VAMP-based motion planner for the Fetch robot.

Wraps the AdaCompNUS/vamp fork which provides whole-body planning
(``multilayer_rrtc``, ``fcit_wb``) on top of upstream KavrakiLab/vamp.

Satisfies the :class:`~TyGrit.planning.MotionPlanner` protocol and
exposes additional environment-management helpers (point clouds,
spheres, EEF attachments) used by the pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from loguru import logger as log

from TyGrit.planning.config import VampPlannerConfig
from TyGrit.planning.fetch.algorithms import (
    plan_arm_rrtc,
    plan_whole_body_fcit,
    plan_whole_body_multilayer_rrtc,
)
from TyGrit.types.failures import PlannerFailure
from TyGrit.types.geometry import SE2Pose
from TyGrit.types.results import PlanResult
from TyGrit.types.robot import WholeBodyConfig


class VampPreviewPlanner:
    """VAMP motion planner for the Fetch robot.

    Implements the :class:`~TyGrit.planning.MotionPlanner` protocol.

    Args:
        config: Planner configuration.
    """

    def __init__(self, config: VampPlannerConfig | None = None) -> None:
        import vamp_preview

        self._config = config or VampPlannerConfig()

        self._env = getattr(vamp_preview, "Environment")()
        (
            self._module,
            self._planner_func,
            self._plan_settings,
            self._simp_settings,
        ) = getattr(vamp_preview, "configure_robot_and_planner_with_kwargs")(
            "fetch",
            self._config.algorithm,
            sampler_name=self._config.sampler,
        )

        self._sampler = self._module.halton()
        self._sampler.skip(0)

        self._current_attachment: Any | None = None
        self._pc_bounds_xy: tuple[float, float, float, float] | None = None

        log.debug(
            "VampPreviewPlanner initialised (algorithm={})", self._config.algorithm
        )

    # ------------------------------------------------------------------
    # MotionPlanner protocol
    # ------------------------------------------------------------------

    def plan_arm(
        self,
        start: npt.NDArray[np.float64],
        goal: npt.NDArray[np.float64],
    ) -> PlanResult:
        """Plan an arm-only trajectory between two 8-DOF configurations."""
        start = np.asarray(start, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)

        return plan_arm_rrtc(
            start,
            goal,
            self._env,
            self._planner_func,
            self._plan_settings,
            self._simp_settings,
            self._sampler,
            self._module,
            simplify=self._config.simplify,
            interpolation_steps=self._config.interpolation_steps,
        )

    def plan_whole_body(
        self,
        start: WholeBodyConfig,
        goal: WholeBodyConfig,
    ) -> PlanResult:
        """Plan a whole-body (arm + base) trajectory."""
        # Validate start/goal
        start_arm = list(start.arm_joints)
        goal_arm = list(goal.arm_joints)
        start_base = [start.base_pose.x, start.base_pose.y, start.base_pose.theta]
        goal_base = [goal.base_pose.x, goal.base_pose.y, goal.base_pose.theta]

        if self._is_in_collision(start_arm, start_base):
            return PlanResult(
                success=False,
                failure=PlannerFailure.COLLISION_AT_START,
            )
        if self._is_in_collision(goal_arm, goal_base):
            return PlanResult(
                success=False,
                failure=PlannerFailure.COLLISION_AT_GOAL,
            )

        cfg = self._config
        if cfg.whole_body_algorithm == "fcit_wb":
            if self._pc_bounds_xy is None:
                log.warning(
                    "FCIT* selected but XY bounds unavailable — "
                    "call add_pointcloud first."
                )
                return PlanResult(
                    success=False,
                    failure=PlannerFailure.NO_PATH_FOUND,
                )
            return plan_whole_body_fcit(
                start_arm,
                goal_arm,
                start_base,
                goal_base,
                self._env,
                self._module,
                self._pc_bounds_xy,
                self._sampler,
                interpolation_density=cfg.interpolation_density,
                max_iterations=cfg.fcit_max_iterations,
                max_samples=cfg.fcit_max_samples,
                batch_size=cfg.fcit_batch_size,
                reverse_weight=cfg.fcit_reverse_weight,
                optimize=cfg.fcit_optimize,
            )

        # Default: multilayer RRT-Connect
        return plan_whole_body_multilayer_rrtc(
            start_arm,
            goal_arm,
            start_base,
            goal_base,
            self._env,
            self._module,
            self._plan_settings,
            self._simp_settings,
            self._sampler,
            interpolation_density=cfg.interpolation_density,
        )

    def validate_config(self, config: WholeBodyConfig) -> bool:
        """Return *True* if the configuration is collision-free."""
        arm = list(config.arm_joints)
        base = [config.base_pose.x, config.base_pose.y, config.base_pose.theta]
        return not self._is_in_collision(arm, base)

    # ------------------------------------------------------------------
    # Environment management (concrete-only, not in protocol)
    # ------------------------------------------------------------------

    def set_base_params(self, base_pose: SE2Pose) -> None:
        """Set the VAMP module's base parameters.

        Note: VAMP expects ``(theta, x, y)`` — not ``(x, y, theta)``.
        """
        self._module.set_base_params(base_pose.theta, base_pose.x, base_pose.y)

    def add_pointcloud(
        self,
        points: npt.NDArray[np.float64],
        point_radius: float | None = None,
    ) -> None:
        """Add a point cloud as collision geometry.

        Args:
            points: ``(N, 3)`` array of XYZ points.
            point_radius: Per-point collision radius.  Defaults to
                ``config.point_radius``.
        """
        import vamp_preview

        radius = point_radius if point_radius is not None else self._config.point_radius
        r_min, r_max = getattr(vamp_preview, "ROBOT_RADII_RANGES")["fetch"]

        pts = np.asarray(points, dtype=np.float64)
        pts_list = pts.tolist()
        self._env.add_pointcloud(pts_list, r_min, r_max, radius)

        # Update FCIT* XY bounds
        if pts.size > 0 and pts.ndim == 2 and pts.shape[1] >= 2:
            pad = self._config.bounds_padding
            min_xy = pts[:, :2].min(axis=0)
            max_xy = pts[:, :2].max(axis=0)
            self._pc_bounds_xy = (
                float(min_xy[0] - pad),
                float(max_xy[0] + pad),
                float(min_xy[1] - pad),
                float(max_xy[1] + pad),
            )

    def clear_pointclouds(self) -> None:
        """Remove all point-cloud collision geometry."""
        self._env.clear_pointclouds()

    def add_sphere(
        self,
        position: list[float],
        radius: float,
    ) -> None:
        """Add a sphere obstacle to the environment."""
        import vamp_preview

        sphere_cls = getattr(vamp_preview, "Sphere")
        self._env.add_sphere(sphere_cls(list(position), radius))

    def clear_spheres(self) -> None:
        """Remove all sphere obstacles."""
        self._env.clear_spheres()

    def attach_to_eef(
        self,
        spheres: list[dict[str, Any]],
        offset_pos: list[float] | None = None,
        offset_quat: list[float] | None = None,
    ) -> bool:
        """Attach collision spheres to the end-effector.

        Args:
            spheres: List of ``{"position": [x,y,z], "radius": float}`` dicts.
            offset_pos: ``[x, y, z]`` offset from EEF frame.
            offset_quat: ``[x, y, z, w]`` quaternion offset from EEF frame.

        Returns:
            ``True`` if attachment succeeded, ``False`` if already attached.
        """
        import vamp_preview

        if self._current_attachment is not None:
            log.warning("Cannot attach: object already attached — detach first.")
            return False

        if offset_pos is None:
            offset_pos = [0.0, 0.0, 0.0]
        if offset_quat is None:
            offset_quat = [0.0, 0.0, 0.0, 1.0]

        attachment_cls = getattr(vamp_preview, "Attachment")
        sphere_cls = getattr(vamp_preview, "Sphere")
        attachment = attachment_cls(offset_pos, offset_quat)
        vamp_spheres = [sphere_cls(list(s["position"]), s["radius"]) for s in spheres]
        attachment.add_spheres(vamp_spheres)
        self._env.attach(attachment)
        self._current_attachment = attachment
        return True

    def detach_from_eef(self) -> bool:
        """Detach collision object from the end-effector.

        Returns:
            ``True`` if detached, ``False`` if nothing was attached.
        """
        if self._current_attachment is None:
            log.warning("Cannot detach: nothing attached.")
            return False
        self._env.detach()
        self._current_attachment = None
        return True

    def check_trajectory_collisions(
        self,
        arm_path: list[list[float]],
        base_configs: list[list[float]],
    ) -> bool:
        """Check whether a trajectory has any collisions.

        Returns:
            ``True`` if a collision is detected.
        """
        return bool(
            self._module.check_whole_body_collisions(
                self._env,
                arm_path,
                base_configs,
            )
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _is_in_collision(
        self,
        arm_config: list[float],
        base_config: list[float],
    ) -> bool:
        """Return *True* if the whole-body config is in collision."""
        return not self._module.validate_whole_body_config(
            arm_config,
            base_config,
            self._env,
        )
