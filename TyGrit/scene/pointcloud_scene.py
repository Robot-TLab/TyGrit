"""Point-cloud scene — satisfies the ``Scene`` protocol.

Maintains three clouds (static map, dynamic observations, goal spec) and
supports five update modes (static, latest, accumulated, combine,
ray_casting).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import numpy.typing as npt

from TyGrit.scene.config import PointCloudSceneConfig
from TyGrit.types.sensor import SensorSnapshot
from TyGrit.utils.depth import depth_to_world_pointcloud
from TyGrit.utils.pointcloud import (
    crop_sphere,
    filter_ground,
    merge_dedup,
    points_in_frustum_mask,
    voxel_downsample,
)

# Type alias for the optional robot self-filter callback.
# Signature: (points_world, robot_state_from_snapshot) -> filtered_points
RobotFilterFn = Callable[
    [npt.NDArray[np.float32], SensorSnapshot],
    npt.NDArray[np.float32],
]

_VALID_MODES = frozenset({"static", "latest", "accumulated", "combine", "ray_casting"})


class PointCloudScene:
    """Point-cloud world model with 5 update modes.

    Maintains three clouds:
        _S : static map (set once, never modified by ``update``).
        _M : dynamic observations (evolves according to the selected mode).
        _G : goal specification cloud (set externally).

    Modes:
        static      – _M is never updated; belief = _S only.
        latest      – _M is replaced wholesale by each new observation.
        accumulated – Frustum carve + merge-dedup into _M.
        combine     – Stack + voxel downsample into _M.
        ray_casting – Remove free-space points via depth comparison, then merge.
    """

    def __init__(
        self,
        config: PointCloudSceneConfig | None = None,
        static_map: npt.NDArray[np.float32] | None = None,
        *,
        robot_filter_fn: RobotFilterFn | None = None,
    ) -> None:
        cfg = config or PointCloudSceneConfig()

        if cfg.mode not in _VALID_MODES:
            raise ValueError(
                f"mode must be one of {sorted(_VALID_MODES)}, got {cfg.mode!r}"
            )

        # Down-sample the static map once at init
        raw_s = (
            np.asarray(static_map, dtype=np.float32)
            if static_map is not None
            else np.empty((0, 3), dtype=np.float32)
        )
        self._S = (
            voxel_downsample(raw_s, cfg.downsample_voxel_size)
            if raw_s.shape[0] > 0
            else raw_s
        )
        self._M: npt.NDArray[np.float32] = np.empty((0, 3), dtype=np.float32)
        self._G: npt.NDArray[np.float32] = np.empty((0, 3), dtype=np.float32)

        self.mode = cfg.mode
        self.downsample_voxel_size = cfg.downsample_voxel_size
        self.merge_radius = cfg.merge_radius
        self.ground_z_threshold = cfg.ground_z_threshold
        self.enable_ground_filter = cfg.enable_ground_filter
        self.depth_range = cfg.depth_range
        self.crop_radius = cfg.crop_radius
        self._robot_filter_fn = robot_filter_fn
        self._depth_stride = cfg.depth_stride

    # ── Scene protocol ───────────────────────────────────────────────────

    def update(
        self,
        snapshot: SensorSnapshot,
        camera_pose: npt.NDArray[np.float64],
    ) -> None:
        """Integrate a new sensor observation."""
        if self.mode == "static":
            return

        # 1. Depth → world-frame point cloud
        pcd = depth_to_world_pointcloud(
            snapshot.depth,
            snapshot.intrinsics,
            camera_pose,
            z_range=self.depth_range,
            stride=self._depth_stride,
        )

        # 2. Crop around camera
        if pcd.shape[0] > 0:
            cam_pos = camera_pose[:3, 3].astype(np.float32)
            pcd = crop_sphere(pcd, cam_pos, self.crop_radius)

        # 3. Ground filter
        if self.enable_ground_filter and pcd.shape[0] > 0:
            pcd = filter_ground(pcd, self.ground_z_threshold)

        # 4. Robot self-filter
        if self._robot_filter_fn is not None and pcd.shape[0] > 0:
            pcd = self._robot_filter_fn(pcd, snapshot)

        # 5. Mode-specific integration
        K = np.asarray(snapshot.intrinsics).reshape(3, 3)
        T_wc = np.asarray(camera_pose).reshape(4, 4)
        z_range = self.depth_range

        if self.mode == "latest":
            self._M = pcd
        elif self.mode == "accumulated":
            self._update_accumulated(pcd, K, T_wc, z_range)
        elif self.mode == "combine":
            self._update_combine(pcd)
        elif self.mode == "ray_casting":
            self._update_ray_casting(pcd, snapshot.depth, K, T_wc, z_range)

    def get_pointcloud(self) -> npt.NDArray[np.float32]:
        """Return E(t) = S union M(t)."""
        return self._current_environment()

    def clear(self) -> None:
        """Reset dynamic observations, keeping the static map."""
        self._M = np.empty((0, 3), dtype=np.float32)

    # ── Extra API (beyond protocol) ──────────────────────────────────────

    def set_goal_pcd(self, pcd_world: npt.NDArray[np.float32]) -> None:
        self._G = np.asarray(pcd_world, dtype=np.float32)

    def get_goal_pcd(self) -> npt.NDArray[np.float32]:
        return self._G.copy()

    def current_observations(self) -> npt.NDArray[np.float32]:
        """Return M(t) only — useful for debugging."""
        return self._M.copy()

    # ── Internals ────────────────────────────────────────────────────────

    def _current_environment(
        self,
        roi_center: tuple[float, float, float] | None = None,
        roi_radius: float | None = None,
    ) -> npt.NDArray[np.float32]:
        if self.mode == "static":
            env = self._S.copy()
        elif self._S.shape[0] == 0:
            env = self._M.copy()
        elif self._M.shape[0] == 0:
            env = self._S.copy()
        else:
            env = merge_dedup(self._S, self._M, self.merge_radius)

        if roi_center is not None and roi_radius is not None and env.shape[0] > 0:
            env = crop_sphere(env, np.asarray(roi_center, dtype=np.float32), roi_radius)
        return env

    # -- accumulated mode --------------------------------------------------

    def _update_accumulated(
        self,
        pcd: npt.NDArray[np.float32],
        K: npt.NDArray[np.float64],
        T_wc: npt.NDArray[np.float64],
        z_range: tuple[float, float],
    ) -> None:
        if pcd.shape[0] == 0:
            return

        M = self._M

        # Carve out the current frustum
        if M.shape[0] > 0:
            in_frustum = points_in_frustum_mask(M, K, T_wc, z_range)
            if np.any(in_frustum):
                M = M[~in_frustum]

        # Merge new observation with dedup
        self._M = merge_dedup(M, pcd, self.merge_radius)

    # -- combine mode ------------------------------------------------------

    def _update_combine(self, pcd: npt.NDArray[np.float32]) -> None:
        if pcd.shape[0] == 0:
            return

        M = self._M
        if M.shape[0] == 0:
            M = pcd
        else:
            M = np.vstack((M, pcd))

        # Downsample to prevent unbounded growth
        self._M = voxel_downsample(M, self.downsample_voxel_size)

    # -- ray_casting mode --------------------------------------------------

    def _update_ray_casting(
        self,
        pcd: npt.NDArray[np.float32],
        depth: npt.NDArray[np.float32],
        K: npt.NDArray[np.float64],
        T_wc: npt.NDArray[np.float64],
        z_range: tuple[float, float],
    ) -> None:
        M = self._M

        if M.shape[0] > 0:
            R_mat = T_wc[:3, :3].astype(np.float32)
            t_vec = T_wc[:3, 3].astype(np.float32)

            # Transform M to camera frame
            Pc = (M - t_vec) @ R_mat
            zc = Pc[:, 2]

            valid_proj = zc > 1e-3
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]

            zc_safe = np.where(valid_proj, zc, 1.0)
            u = (Pc[:, 0] * fx / zc_safe) + cx
            v = (Pc[:, 1] * fy / zc_safe) + cy

            u_int = np.rint(u).astype(np.int32)
            v_int = np.rint(v).astype(np.int32)

            h, w = depth.shape
            in_img = (u_int >= 0) & (u_int < w) & (v_int >= 0) & (v_int < h)

            check = valid_proj & in_img
            if np.any(check):
                idx_check = np.where(check)[0]
                z_old = zc[idx_check]
                z_measured = depth[v_int[idx_check], u_int[idx_check]]

                valid_obs = (z_measured > 0) & np.isfinite(z_measured)
                is_free = z_old < (z_measured - self.merge_radius)

                remove_idx = idx_check[valid_obs & is_free]
                if len(remove_idx) > 0:
                    keep = np.ones(len(M), dtype=bool)
                    keep[remove_idx] = False
                    M = M[keep]

        # Merge new observation
        if pcd.shape[0] > 0:
            M = merge_dedup(M, pcd, self.merge_radius)

        self._M = M
