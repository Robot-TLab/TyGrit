"""Placement-surface detection from scene meshes.

Finds upward-facing horizontal surfaces (tables, counters, shelves) in a
triangulated scene mesh and returns them as point-cloud clusters. Each
cluster represents a contiguous placement region suitable for spawning
objects.

Adapted from grasp_anywhere v1's
``tools/generate_grasp_benchmark.py:BenchmarkGenerator.find_placement_surfaces``
with the following changes:

1. **Extracted as a pure function** (no class state, no simulator imports).
   Takes a trimesh mesh and returns numpy arrays. Sim backends call this
   after extracting meshes via their own actor-query APIs.
2. **Occupancy suppression** (``top_surface_only``). For each grid cell the
   algorithm keeps only the *highest* horizontal slab, suppressing lower
   shelves that would be shadowed by furniture above them. This prevents
   spawning objects inside closed drawers or on shelves hidden below a
   table top.
3. **No debug I/O**. Callers that need debug exports write the returned
   clusters themselves; this module only computes them.

Coordinate convention: Z-up world frame.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def find_placement_surfaces(
    mesh_vertices: npt.NDArray[np.float32],
    mesh_faces: npt.NDArray[np.int64],
    mesh_face_normals: npt.NDArray[np.float32],
    *,
    min_height: float = 0.1,
    max_height: float = 1.7,
    min_area: float = 0.04,
    up_dot_threshold: float = 0.95,
    grid_size: float = 0.05,
    height_bin_size: float = 0.02,
    min_cluster_points: int = 500,
    sample_count: int = 200_000,
    top_surface_only: bool = True,
    seed: int | None = None,
) -> list[npt.NDArray[np.float32]]:
    """Detect horizontal placement surfaces from a triangulated mesh.

    Parameters
    ----------
    mesh_vertices
        ``(V, 3)`` float32 vertex positions.
    mesh_faces
        ``(F, 3)`` int64 triangle face indices into *mesh_vertices*.
    mesh_face_normals
        ``(F, 3)`` float32 per-face unit normals.
    min_height, max_height
        Z-range filter on the average height of detected surface clusters.
        Clusters outside this band are discarded (e.g. floor or ceiling).
    min_area
        Minimum convex-hull area (m^2) of the 2-D projection (XY plane)
        of a surface cluster to be kept.
    up_dot_threshold
        Minimum dot product between a face normal and the world +Z axis
        for the face to be considered upward-facing.
    grid_size
        Side length (m) of the 2-D grid cells used for spatial clustering.
    height_bin_size
        Resolution (m) of the 1-D height binning. Two nearby face samples
        whose Z values fall within the same or adjacent bins are eligible
        to merge into one cluster.
    min_cluster_points
        Minimum number of surface samples a cluster must contain after
        BFS merging. Small clusters (isolated shelf edges) are discarded.
    sample_count
        Number of points to Poisson-sample from the mesh surface. Higher
        values improve coverage on large scenes at the cost of speed.
    top_surface_only
        If True, for each XY grid cell only the cluster whose average
        height is the tallest is retained. This prevents placing objects
        on lower shelves that are occluded from above. Matches v1's
        ``top_clusters`` logic.
    seed
        Optional RNG seed for reproducible surface sampling. ``None``
        uses numpy's default unseeded RNG.

    Returns
    -------
    list[NDArray[float32]]
        Each entry is an ``(N, 3)`` array of world-frame points lying on
        one detected placement surface. The list may be empty if no
        surfaces pass all filters.
    """
    import trimesh  # deferred: trimesh is heavy, keep importable cheaply

    # Early out: trimesh.sample.sample_surface crashes on empty meshes
    # (IndexError on the cumulative area array). Guard here so callers
    # can pass an empty mesh without a try/except.
    if len(mesh_faces) == 0 or len(mesh_vertices) == 0:
        return []

    mesh = trimesh.Trimesh(
        vertices=mesh_vertices.astype(np.float64),
        faces=mesh_faces,
        face_normals=mesh_face_normals.astype(np.float64),
        process=False,
    )

    rng = np.random.default_rng(seed)

    # ── 1. Sample surface points + inherit face normals ──────────────

    pts, face_ids = trimesh.sample.sample_surface(
        mesh, sample_count, seed=int(rng.integers(0, 2**31))
    )
    pts = pts.astype(np.float32)
    normals = mesh_face_normals[np.asarray(face_ids, dtype=np.int64)]

    # ── 2. Filter to upward-facing samples ───────────────────────────

    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    dot_up = (normals * up[np.newaxis, :]).sum(axis=1)
    mask = dot_up > up_dot_threshold
    pts_up = pts[mask]
    if pts_up.shape[0] == 0:
        return []

    # ── 3. Grid + height binning ─────────────────────────────────────

    xy = pts_up[:, :2]
    grid_ij = np.floor(xy / grid_size).astype(np.int64)
    h_bins = np.floor(pts_up[:, 2] / height_bin_size).astype(np.int64)

    cell_to_indices: dict[tuple[int, int, int], list[int]] = {}
    for i in range(len(pts_up)):
        key = (int(grid_ij[i, 0]), int(grid_ij[i, 1]), int(h_bins[i]))
        cell_to_indices.setdefault(key, []).append(i)

    # ── 4. BFS clustering in (grid_x, grid_y, height_bin) space ──────

    visited: set[tuple[int, int, int]] = set()
    xy_offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 0),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    h_offsets = [-1, 0, 1]

    raw_clusters: list[dict] = []
    for cell in list(cell_to_indices):
        if cell in visited:
            continue
        queue = [cell]
        visited.add(cell)
        cluster_cells = [cell]
        while queue:
            cx, cy, cz = queue.pop()
            for dx, dy in xy_offsets:
                for dz in h_offsets:
                    nc = (cx + dx, cy + dy, cz + dz)
                    if nc in cell_to_indices and nc not in visited:
                        visited.add(nc)
                        queue.append(nc)
                        cluster_cells.append(nc)

        idxs: list[int] = []
        footprint: set[tuple[int, int]] = set()
        for cc in cluster_cells:
            idxs.extend(cell_to_indices[cc])
            footprint.add((cc[0], cc[1]))
        if len(idxs) < min_cluster_points:
            continue
        raw_clusters.append(
            {"indices": np.array(idxs, dtype=np.int64), "footprint": footprint}
        )

    # ── 5. Area + height filtering ───────────────────────────────────

    from scipy.spatial import ConvexHull

    plane_clusters: list[dict] = []
    for cl in raw_clusters:
        cluster_pts = pts_up[cl["indices"]]
        avg_h = float(cluster_pts[:, 2].mean())
        pts_2d = cluster_pts[:, :2]
        area = 0.0
        if pts_2d.shape[0] >= 3:
            centered = pts_2d - pts_2d.mean(axis=0, keepdims=True)
            _, s, _ = np.linalg.svd(centered, full_matrices=False)
            # Only compute hull if the cluster has 2-D extent (not a
            # degenerate line of colinear samples).
            if s.shape[0] >= 2 and s[1] > 1e-8:
                try:
                    hull = ConvexHull(pts_2d)
                    area = float(hull.volume)  # 2-D ConvexHull.volume = area
                except Exception:
                    # scipy.spatial.ConvexHull raises QhullError on
                    # degenerate inputs (all points collinear). Skip.
                    pass
        if area >= min_area and min_height <= avg_h <= max_height:
            plane_clusters.append(
                {"points": cluster_pts, "avg_h": avg_h, "footprint": cl["footprint"]}
            )

    # ── 6. Top-surface filtering ─────────────────────────────────────

    if not top_surface_only or not plane_clusters:
        return [info["points"].astype(np.float32) for info in plane_clusters]

    # Per XY-cell, record the max average height across all clusters
    # whose footprint covers that cell.
    footprint_max_h: dict[tuple[int, int], float] = {}
    for info in plane_clusters:
        avg_h = info["avg_h"]
        for coord in info["footprint"]:
            if coord not in footprint_max_h or avg_h > footprint_max_h[coord]:
                footprint_max_h[coord] = avg_h

    height_epsilon = height_bin_size * 1.5
    top_clusters: list[npt.NDArray[np.float32]] = []
    for info in plane_clusters:
        avg_h = info["avg_h"]
        coords = list(info["footprint"])
        if not coords:
            continue
        # A cluster is "top" if the majority (>= 60 %) of its footprint
        # cells are within epsilon of that cell's global maximum height.
        num_top = sum(1 for c in coords if avg_h >= footprint_max_h[c] - height_epsilon)
        ratio = num_top / len(coords)
        if ratio >= 0.6:
            top_clusters.append(info["points"].astype(np.float32))

    return top_clusters
