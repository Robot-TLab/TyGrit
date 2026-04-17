"""Unit tests for TyGrit.worlds.placement.

Pure-Python tests (no simulator imports). Runnable in the default pixi env:
``pixi run test test/test_worlds_placement.py -v``

Requires ``trimesh`` and ``scipy`` which are available in the default env
via the base feature.
"""

from __future__ import annotations

import numpy as np


def _flat_quad_mesh(
    z: float = 0.8,
    xmin: float = -0.5,
    xmax: float = 0.5,
    ymin: float = -0.5,
    ymax: float = 0.5,
    *,
    nx: int = 10,
    ny: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a flat horizontal quad mesh at height *z*.

    Returns (vertices, faces, face_normals) as numpy arrays ready for
    :func:`find_placement_surfaces`.
    """
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.full_like(xx, z)
    vertices = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)

    faces = []
    for i in range(ny - 1):
        for j in range(nx - 1):
            v0 = i * nx + j
            v1 = v0 + 1
            v2 = v0 + nx
            v3 = v2 + 1
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    faces = np.array(faces, dtype=np.int64)

    # All normals point up for a flat horizontal surface.
    face_normals = np.zeros((len(faces), 3), dtype=np.float32)
    face_normals[:, 2] = 1.0

    return vertices, faces, face_normals


class TestFindPlacementSurfaces:
    """Tests for find_placement_surfaces."""

    def test_flat_table_returns_one_surface(self) -> None:
        from TyGrit.worlds.placement import find_placement_surfaces

        verts, faces, normals = _flat_quad_mesh(z=0.8)
        surfaces = find_placement_surfaces(
            verts,
            faces,
            normals,
            min_height=0.1,
            max_height=1.5,
            min_area=0.01,
            min_cluster_points=5,
            sample_count=5000,
            seed=42,
        )
        assert len(surfaces) >= 1
        # All points should be near z = 0.8.
        for s in surfaces:
            assert s.shape[1] == 3
            assert np.allclose(s[:, 2], 0.8, atol=0.05)

    def test_floor_level_is_filtered_out(self) -> None:
        from TyGrit.worlds.placement import find_placement_surfaces

        verts, faces, normals = _flat_quad_mesh(z=0.05)
        surfaces = find_placement_surfaces(
            verts,
            faces,
            normals,
            min_height=0.1,
            max_height=1.5,
            min_area=0.01,
            min_cluster_points=5,
            sample_count=5000,
            seed=42,
        )
        # Floor at z=0.05 is below min_height=0.1, should be filtered.
        assert len(surfaces) == 0

    def test_empty_mesh_returns_empty(self) -> None:
        from TyGrit.worlds.placement import find_placement_surfaces

        verts = np.zeros((0, 3), dtype=np.float32)
        faces = np.zeros((0, 3), dtype=np.int64)
        normals = np.zeros((0, 3), dtype=np.float32)
        surfaces = find_placement_surfaces(
            verts,
            faces,
            normals,
            min_cluster_points=5,
            sample_count=100,
            seed=42,
        )
        assert surfaces == []

    def test_vertical_wall_returns_no_surfaces(self) -> None:
        """A vertical wall has normals pointing sideways, not up."""
        from TyGrit.worlds.placement import find_placement_surfaces

        # Create a vertical mesh (XZ plane, normals along +Y).
        xs = np.linspace(-1, 1, 10)
        zs = np.linspace(0, 2, 10)
        xx, zz = np.meshgrid(xs, zs)
        yy = np.zeros_like(xx)
        vertices = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(
            np.float32
        )

        faces = []
        nx, ny = 10, 10
        for i in range(ny - 1):
            for j in range(nx - 1):
                v0 = i * nx + j
                faces.append([v0, v0 + 1, v0 + nx])
                faces.append([v0 + 1, v0 + nx + 1, v0 + nx])
        faces_arr = np.array(faces, dtype=np.int64)

        # Normals point along +Y (horizontal, not upward).
        normals = np.zeros((len(faces_arr), 3), dtype=np.float32)
        normals[:, 1] = 1.0

        surfaces = find_placement_surfaces(
            vertices,
            faces_arr,
            normals,
            min_cluster_points=5,
            sample_count=5000,
            seed=42,
        )
        assert surfaces == []

    def test_top_surface_only_filters_lower(self) -> None:
        """With two overlapping surfaces at different heights, only the top is kept."""
        from TyGrit.worlds.placement import find_placement_surfaces

        v1, f1, n1 = _flat_quad_mesh(z=0.5, xmin=-0.5, xmax=0.5, ymin=-0.5, ymax=0.5)
        v2, f2, n2 = _flat_quad_mesh(z=0.9, xmin=-0.5, xmax=0.5, ymin=-0.5, ymax=0.5)

        # Merge the two meshes.
        offset = len(v1)
        verts = np.vstack([v1, v2])
        faces = np.vstack([f1, f2 + offset])
        normals = np.vstack([n1, n2])

        surfaces = find_placement_surfaces(
            verts,
            faces,
            normals,
            min_height=0.1,
            max_height=1.5,
            min_area=0.01,
            min_cluster_points=5,
            sample_count=10000,
            top_surface_only=True,
            seed=42,
        )
        # Only the top surface (z ~ 0.9) should survive.
        assert len(surfaces) >= 1
        for s in surfaces:
            avg_z = s[:, 2].mean()
            assert avg_z > 0.7, f"Expected top surface (z~0.9), got avg_z={avg_z}"

    def test_deterministic_with_seed(self) -> None:
        from TyGrit.worlds.placement import find_placement_surfaces

        verts, faces, normals = _flat_quad_mesh(z=0.8)
        s1 = find_placement_surfaces(
            verts,
            faces,
            normals,
            min_cluster_points=5,
            sample_count=5000,
            seed=123,
        )
        s2 = find_placement_surfaces(
            verts,
            faces,
            normals,
            min_cluster_points=5,
            sample_count=5000,
            seed=123,
        )
        assert len(s1) == len(s2)
        for a, b in zip(s1, s2):
            np.testing.assert_array_equal(a, b)
