"""Pure utility functions for object placement on scene surfaces."""

from __future__ import annotations

from collections import deque

import numpy as np
import numpy.typing as npt
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def find_placement_surfaces(
    scene_mesh: trimesh.Trimesh,
    min_height: float = 0.1,
    max_height: float = 1.7,
    min_area: float = 0.04,
    height_bin_size: float = 0.02,
) -> list[npt.NDArray[np.float32]]:
    """Return point clouds of horizontal placement surfaces found on scene_mesh."""
    scene_points = 200_000
    up_dot_threshold = 0.95
    grid_size = 0.05
    min_cluster_points = 500

    pts, face_ids = trimesh.sample.sample_surface(scene_mesh, scene_points)
    pts = pts.astype(np.float32)
    face_normals = scene_mesh.face_normals
    normals = face_normals[np.asarray(face_ids, dtype=np.int64)]

    up_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    dot_up = (normals * up_vec).sum(axis=1)
    keep = dot_up > up_dot_threshold
    pts_keep = pts[keep]
    if pts_keep.shape[0] == 0:
        return []

    # Quantize to 5 cm XY grid and height bins.
    proj = pts_keep[:, :2]
    grid = np.floor(proj / grid_size).astype(np.int64)
    height_bins = np.floor(pts_keep[:, 2] / height_bin_size).astype(np.int64)

    cell_to_indices: dict[tuple[int, int, int], list[int]] = {}
    for i in range(pts_keep.shape[0]):
        key = (int(grid[i, 0]), int(grid[i, 1]), int(height_bins[i]))
        if key in cell_to_indices:
            cell_to_indices[key].append(i)
        else:
            cell_to_indices[key] = [i]

    # BFS clustering on 8-connected XY neighbors, allowing ±1 height bin adjacency.
    visited: set[tuple[int, int, int]] = set()
    clusters: list[list[tuple[int, int, int]]] = []
    xy_deltas = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    height_deltas = [-1, 0, 1]

    for seed_cell in cell_to_indices:
        if seed_cell in visited:
            continue
        queue: deque[tuple[int, int, int]] = deque([seed_cell])
        visited.add(seed_cell)
        cluster_cells: list[tuple[int, int, int]] = [seed_cell]
        while queue:
            cx, cy, cz = queue.popleft()
            for dx, dy in xy_deltas:
                for dz in height_deltas:
                    nc = (cx + dx, cy + dy, cz + dz)
                    if nc in cell_to_indices and nc not in visited:
                        visited.add(nc)
                        queue.append(nc)
                        cluster_cells.append(nc)
        clusters.append(cluster_cells)

    # For each (gx, gy) footprint cell keep only the topmost height bin.
    plane_clusters: list[npt.NDArray[np.float32]] = []
    for cluster_cells in clusters:
        # Deduplicate footprint: for each XY cell keep the highest Z bin.
        footprint_top: dict[tuple[int, int], int] = {}
        for gx, gy, gz in cluster_cells:
            xy = (gx, gy)
            if xy not in footprint_top or gz > footprint_top[xy]:
                footprint_top[xy] = gz

        idxs: list[int] = []
        for (gx, gy), gz in footprint_top.items():
            key = (gx, gy, gz)
            if key in cell_to_indices:
                idxs.extend(cell_to_indices[key])

        if len(idxs) < min_cluster_points:
            continue

        cluster_pts = pts_keep[np.asarray(idxs, dtype=np.int64)]
        avg_h = float(cluster_pts[:, 2].mean())
        if avg_h < min_height or avg_h > max_height:
            continue

        pts_2d = cluster_pts[:, :2]
        area = 0.0
        if pts_2d.shape[0] >= 3:
            centered = pts_2d - pts_2d.mean(axis=0, keepdims=True)
            _, s, _ = np.linalg.svd(centered, full_matrices=False)
            if s.shape[0] >= 2 and s[1] > 1e-8:
                hull = ConvexHull(pts_2d)
                area = float(hull.volume)

        if area >= min_area:
            plane_clusters.append(cluster_pts)

    return plane_clusters


def sample_object_placement(
    surface_pts: npt.NDArray[np.floating],
    existing_positions: list[npt.NDArray[np.floating]],
    min_distance: float = 0.20,
    rng: np.random.Generator | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None:
    """Sample a candidate pose on surface_pts, returning (position_xyz, orientation_xyzw) or None."""
    if rng is None:
        rng = np.random.default_rng()

    idx = int(rng.integers(0, surface_pts.shape[0]))
    position = surface_pts[idx].astype(np.float64).copy()

    for existing in existing_positions:
        if (
            np.linalg.norm(position - np.asarray(existing, dtype=np.float64))
            < min_distance
        ):
            return None

    yaw = float(rng.uniform(0.0, 2.0 * np.pi))
    # Rotation.from_euler returns [x, y, z, w] via as_quat() — TyGrit/SciPy convention.
    orientation: npt.NDArray[np.float64] = Rotation.from_euler("z", yaw).as_quat()

    # Raise Z by 0.2 m as a drop buffer so the object lands on the surface.
    position[2] += 0.2

    return position, orientation


def check_placement_stability(
    initial_pos: npt.NDArray[np.floating],
    final_pos: npt.NDArray[np.floating],
    target_surface_z: float,
    max_movement: float = 0.3,
    z_tolerance_low: float = 0.05,
    z_tolerance_high: float = 0.3,
) -> bool:
    """Return True iff the object settled within expected bounds after simulation."""
    final = np.asarray(final_pos, dtype=np.float64)
    initial = np.asarray(initial_pos, dtype=np.float64)

    if final[2] < target_surface_z - z_tolerance_low:
        return False
    if final[2] > target_surface_z + z_tolerance_high:
        return False
    if np.linalg.norm(final - initial) > max_movement:
        return False
    return True
