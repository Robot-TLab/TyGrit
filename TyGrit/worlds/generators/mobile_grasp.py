"""Generate a mobile-grasping dataset of (scene, object, robot_init_pose) tuples.

Each entry places an Objaverse object on a horizontal surface inside a
Holodeck scene, then samples a Fetch robot base pose from which the
object is IK-reachable for a top-down grasp.

Cross-backend: Holodeck scenes (MJCF) + Objaverse objects (GLB mesh)
are loadable by ManiSkill, Genesis, and Isaac Sim, so every entry in
the output manifest is backend-portable.

Surface detection is done geometrically (MuJoCo model → trimesh →
surface-normal clustering) without running a physics simulation.
Object Z is placed flush on the detected surface. IK reachability is
checked analytically via IKFast.

Pipeline::

    pixi run -e world generate-mobile-grasp-dataset
    pixi run -e world generate-mobile-grasp-dataset --num-scenes 10 --seed 42

Output: ``resources/benchmark/mobile_grasp.json``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from loguru import logger

from TyGrit.types.mobile_grasp import MobileGraspDatapoint, MobileGraspDataset
from TyGrit.types.worlds import ObjectSpec, SceneSpec
from TyGrit.worlds.manifest import load_manifest, load_object_manifest
from TyGrit.worlds.mobile_grasp_manifest import (
    save_mobile_grasp_manifest,
    validate_cross_backend,
)

SCENE_MANIFEST = Path("resources/worlds/holodeck.json.gz")
OBJECT_MANIFEST = Path("resources/worlds/objects/objaverse.json")
OUTPUT_PATH = Path("resources/benchmark/mobile_grasp.json")

DEFAULT_NUM_SCENES = 20
DEFAULT_OBJECTS_PER_SCENE = 3
DEFAULT_BASE_POSES_PER_OBJECT = 2
DEFAULT_SEED = 0


def _extract_scene_trimesh(mjcf_path: str):
    """Load a holodeck MJCF via MuJoCo and return a combined trimesh."""
    import mujoco
    import trimesh

    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    all_verts: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    offset = 0

    for geom_id in range(model.ngeom):
        if model.geom_type[geom_id] != 7:  # 7 = mesh
            continue
        mesh_id = model.geom_dataid[geom_id]
        if mesh_id < 0:
            continue

        vert_start = model.mesh_vertadr[mesh_id]
        vert_count = model.mesh_vertnum[mesh_id]
        face_start = model.mesh_faceadr[mesh_id]
        face_count = model.mesh_facenum[mesh_id]

        verts = model.mesh_vert[vert_start : vert_start + vert_count].copy()
        faces = model.mesh_face[face_start : face_start + face_count].copy()

        geom_xpos = data.geom_xpos[geom_id]
        geom_xmat = data.geom_xmat[geom_id].reshape(3, 3)
        verts_world = (geom_xmat @ verts.T).T + geom_xpos

        all_verts.append(verts_world)
        all_faces.append(faces + offset)
        offset += vert_count

    if not all_verts:
        return None

    return trimesh.Trimesh(
        vertices=np.concatenate(all_verts),
        faces=np.concatenate(all_faces),
    )


def _place_object_on_surface(
    surface_pts: np.ndarray,
    object_spec: ObjectSpec,
    existing_positions: list[np.ndarray],
    rng: np.random.Generator,
    max_trials: int = 50,
) -> ObjectSpec | None:
    """Place object_spec on a surface. Returns updated spec or None."""
    from TyGrit.worlds.generators._placement import sample_object_placement

    for _ in range(max_trials):
        result = sample_object_placement(
            surface_pts, existing_positions, min_distance=0.20, rng=rng
        )
        if result is None:
            continue

        position, orientation = result
        surface_z = float(surface_pts[:, 2].mean())
        position[2] = surface_z + 0.01

        return ObjectSpec(
            name=object_spec.name,
            urdf_path=object_spec.urdf_path,
            usd_path=object_spec.usd_path,
            mjcf_path=object_spec.mjcf_path,
            mesh_path=object_spec.mesh_path,
            builtin_id=object_spec.builtin_id,
            position=(float(position[0]), float(position[1]), float(position[2])),
            orientation_xyzw=tuple(float(v) for v in orientation),
            scale=object_spec.scale,
            fix_base=False,
            is_articulated=False,
        )
    return None


def _generate_for_scene(
    scene: SceneSpec,
    objects: tuple[ObjectSpec, ...],
    rng: np.random.Generator,
    objects_per_scene: int,
    base_poses_per_object: int,
) -> list[MobileGraspDatapoint]:
    """Generate datapoints for one holodeck scene."""
    from TyGrit.worlds.generators._placement import find_placement_surfaces
    from TyGrit.worlds.generators._reachability import (
        filter_reachable_base_poses,
        sample_base_poses,
    )

    mjcf_path = scene.background_mjcf
    if not mjcf_path:
        logger.warning("scene {} has no background_mjcf, skipping", scene.scene_id)
        return []

    if not Path(mjcf_path).exists():
        logger.warning("MJCF not found: {}, skipping", mjcf_path)
        return []

    mesh = _extract_scene_trimesh(mjcf_path)
    if mesh is None:
        logger.warning("no meshes extracted from {}, skipping", scene.scene_id)
        return []

    surfaces = find_placement_surfaces(mesh)
    if not surfaces:
        logger.warning("no placement surfaces in {}, skipping", scene.scene_id)
        return []

    logger.info("scene {}: {} surfaces found", scene.scene_id, len(surfaces))

    datapoints: list[MobileGraspDatapoint] = []
    placed_positions: list[np.ndarray] = []
    objects_placed = 0

    for surface_pts in surfaces:
        if objects_placed >= objects_per_scene:
            break

        obj_template = objects[int(rng.integers(0, len(objects)))]
        placed = _place_object_on_surface(
            surface_pts, obj_template, placed_positions, rng
        )
        if placed is None:
            continue

        placed_positions.append(np.array(placed.position))
        objects_placed += 1

        candidates = sample_base_poses(placed.position, num_candidates=32, rng=rng)
        reachable = filter_reachable_base_poses(candidates, placed.position)

        if not reachable:
            logger.debug(
                "no reachable base pose for {} in {}", placed.name, scene.scene_id
            )
            continue

        for base_pose, init_qpos, grasp_hint in reachable[:base_poses_per_object]:
            dp = MobileGraspDatapoint(
                scene=scene,
                object=placed,
                base_pose=base_pose,
                init_qpos=init_qpos,
                grasp_hint=grasp_hint,
            )
            datapoints.append(dp)

    logger.info("scene {}: {} datapoints generated", scene.scene_id, len(datapoints))
    return datapoints


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a mobile-grasping dataset manifest."
    )
    parser.add_argument(
        "--num-scenes",
        type=int,
        default=DEFAULT_NUM_SCENES,
        help=f"Number of holodeck scenes to sample (default: {DEFAULT_NUM_SCENES})",
    )
    parser.add_argument(
        "--objects-per-scene",
        type=int,
        default=DEFAULT_OBJECTS_PER_SCENE,
        help=f"Objects to place per scene (default: {DEFAULT_OBJECTS_PER_SCENE})",
    )
    parser.add_argument(
        "--base-poses-per-object",
        type=int,
        default=DEFAULT_BASE_POSES_PER_OBJECT,
        help=f"Base poses per placed object (default: {DEFAULT_BASE_POSES_PER_OBJECT})",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="RNG seed (default: 0)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_PATH),
        help=f"Output manifest path (default: {OUTPUT_PATH})",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    logger.info("loading scene manifest: {}", SCENE_MANIFEST)
    all_scenes = load_manifest(SCENE_MANIFEST)
    logger.info("loaded {} scenes", len(all_scenes))

    logger.info("loading object manifest: {}", OBJECT_MANIFEST)
    all_objects = load_object_manifest(OBJECT_MANIFEST)
    logger.info("loaded {} objects", len(all_objects))

    portable_objects = tuple(o for o in all_objects if o.mesh_path)
    if not portable_objects:
        logger.error(
            "no objects with mesh_path — cannot generate cross-backend dataset"
        )
        sys.exit(1)
    logger.info("{} portable objects (with mesh_path)", len(portable_objects))

    num_scenes = min(args.num_scenes, len(all_scenes))
    scene_indices = rng.choice(len(all_scenes), size=num_scenes, replace=False)
    selected_scenes = [all_scenes[i] for i in sorted(scene_indices)]
    logger.info("selected {} scenes for generation", num_scenes)

    all_datapoints: list[MobileGraspDatapoint] = []
    for i, scene in enumerate(selected_scenes):
        logger.info("[{}/{}] processing {}", i + 1, num_scenes, scene.scene_id)
        datapoints = _generate_for_scene(
            scene,
            portable_objects,
            rng,
            args.objects_per_scene,
            args.base_poses_per_object,
        )
        all_datapoints.extend(datapoints)

    dataset = MobileGraspDataset(
        entries=tuple(all_datapoints),
        metadata={
            "num_scenes": str(num_scenes),
            "objects_per_scene": str(args.objects_per_scene),
            "base_poses_per_object": str(args.base_poses_per_object),
            "seed": str(args.seed),
            "scene_source": "holodeck",
            "object_source": "objaverse",
        },
    )

    validate_cross_backend(dataset)

    output = Path(args.output)
    generator_str = (
        f"TyGrit.worlds.generators.mobile_grasp "
        f"--num-scenes {num_scenes} --objects-per-scene {args.objects_per_scene} "
        f"--base-poses-per-object {args.base_poses_per_object} --seed {args.seed}"
    )
    save_mobile_grasp_manifest(output, dataset, generator=generator_str)
    logger.info("wrote {} datapoints to {}", len(all_datapoints), output)


if __name__ == "__main__":
    main()
