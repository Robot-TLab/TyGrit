"""Visualize GraspGen grasp pose predictions on an Objaverse object.

Loads a random Objaverse mesh, samples a point cloud from it, runs GraspGen,
and (by default) opens an interactive Open3D window showing the point cloud
plus predicted parallel-jaw gripper outlines (top-scoring grasp in blue,
others in red — matches grasp_anywhere v1's ``visualize_grasps_pcd``).

The interactive window is delegated to ``TyGrit.visualization.o3d`` so the
same primitives can be reused for any future grasp/point-cloud debugging.

Usage:
    pixi run -e thirdparty python scripts/viz_grasp_poses.py
    pixi run -e thirdparty python scripts/viz_grasp_poses.py --object-index 5 --top-k 10
    pixi run -e thirdparty python scripts/viz_grasp_poses.py --no-show  # headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_object_cloud(mesh_path: str, num_points: int = 2048) -> np.ndarray:
    """Load a mesh and sample a point cloud from it."""
    mesh = trimesh.load(mesh_path, force="mesh")
    pts, _ = trimesh.sample.sample_surface(mesh, num_points)
    return pts.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize GraspGen predictions on an Objaverse object."
    )
    parser.add_argument(
        "--object-index",
        type=int,
        default=0,
        help="Index into the Objaverse manifest (default: 0)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Number of top grasps to visualize (default: 15)",
    )
    parser.add_argument(
        "--num-points", type=int, default=2048, help="Point cloud sample count"
    )
    parser.add_argument(
        "--show",
        dest="show",
        action="store_true",
        default=True,
        help="Open an interactive Open3D window (default)",
    )
    parser.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        help="Skip the interactive window (headless mode)",
    )
    args = parser.parse_args()

    # ── Load object manifest ─────────────────────────────────────────
    from TyGrit.worlds.manifest import load_object_manifest

    manifest_path = PROJECT_ROOT / "resources" / "worlds" / "objects" / "objaverse.json"
    objects = load_object_manifest(manifest_path)
    if args.object_index >= len(objects):
        print(
            f"ERROR: object-index {args.object_index} out of range "
            f"(manifest has {len(objects)} objects)"
        )
        sys.exit(1)

    obj_spec = objects[args.object_index]
    mesh_path = str(PROJECT_ROOT / obj_spec.mesh_path)
    print(f"Object: {obj_spec.name}")
    print(f"Mesh:   {mesh_path}")
    print(f"Scale:  {obj_spec.scale}")

    # ── Sample point cloud ───────────────────────────────────────────
    cloud = load_object_cloud(mesh_path, args.num_points)
    # Apply the manifest scale so the cloud matches what the sim sees.
    # np.asarray ensures a plain ndarray — trimesh returns TrackedArray
    # (a subclass) which fails GraspGen's `type() ==` check and skips
    # the numpy→CUDA conversion.
    cloud = np.asarray(cloud * np.array(obj_spec.scale, dtype=np.float32))
    print(f"Cloud shape: {cloud.shape}, bbox: {cloud.min(0)} → {cloud.max(0)}")

    # ── Run GraspGen ─────────────────────────────────────────────────
    from TyGrit.perception.grasping.config import GraspGenConfig
    from TyGrit.perception.grasping.graspgen import GraspGenPredictor

    config_path = str(
        PROJECT_ROOT
        / "resources"
        / "weights"
        / "graspgen"
        / "graspgen_franka_panda.yml"
    )
    config = GraspGenConfig(
        checkpoint_config_path=config_path,
        num_grasps=200,
        topk_num_grasps=100,
        max_grasps=args.top_k,
    )
    predictor = GraspGenPredictor(config)
    print(f"Running GraspGen (top-{args.top_k})...")
    grasps = predictor.predict(cloud)
    print(f"Got {len(grasps)} grasp candidates")

    if not grasps:
        print(
            "WARNING: No grasps predicted. Object may be too small or cloud "
            "degenerate. Showing point cloud only."
        )
    else:
        scores = [g.score for g in grasps]
        print(
            f"Scores: min={min(scores):.3f}  max={max(scores):.3f}  "
            f"mean={np.mean(scores):.3f}"
        )

    # ── Visualize ────────────────────────────────────────────────────
    if args.show:
        from TyGrit.visualization.o3d import show_grasps

        show_grasps(
            cloud,
            grasps,
            window_name=f"GraspGen on Objaverse: {obj_spec.name}",
        )


if __name__ == "__main__":
    main()
