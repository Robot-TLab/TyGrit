"""Load task suite from JSON — pure Python, no simulation deps."""

from __future__ import annotations

import json
import re
from pathlib import Path

from TyGrit.types.tasks import (
    DynamicObstacle,
    GraspTask,
    ObjectPose,
    TaskScene,
    TaskSuite,
)


def _natural_sort_key(s: str) -> list[int | str]:
    """Sort key that handles embedded numbers: scene_2 < scene_10."""
    return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", s)]


def _parse_object_pose(task_dict: dict) -> ObjectPose:
    pos = task_dict["position"]
    ori = task_dict["orientation"]
    return ObjectPose(
        model_id=task_dict["model_id"],
        position=(float(pos[0]), float(pos[1]), float(pos[2])),
        orientation_wxyz=(float(ori[0]), float(ori[1]), float(ori[2]), float(ori[3])),
    )


def _parse_dynamic_obstacle(obs_dict: dict) -> DynamicObstacle:
    sp = obs_dict["start_position"]
    so = obs_dict["start_orientation"]
    dim = obs_dict["dimension"]
    return DynamicObstacle(
        obstacle_type=obs_dict["type"],
        start_position=(float(sp[0]), float(sp[1]), float(sp[2])),
        start_orientation_wxyz=(float(so[0]), float(so[1]), float(so[2]), float(so[3])),
        dimension=(float(dim[0]), float(dim[1]), float(dim[2])),
    )


def _parse_scene(scene_id: str, scene_dict: dict) -> TaskScene:
    task_dicts = scene_dict["grasp_tasks"]

    # Each task dict defines one object; collect all as object_poses
    object_poses = tuple(_parse_object_pose(t) for t in task_dicts)

    grasp_tasks = []
    for idx, t in enumerate(task_dicts):
        obstacle = None
        if "dynamic_obstacle" in t and t["dynamic_obstacle"]:
            obstacle = _parse_dynamic_obstacle(t["dynamic_obstacle"])
        grasp_tasks.append(
            GraspTask(
                task_index=idx,
                object_pose=object_poses[idx],
                dynamic_obstacle=obstacle,
            )
        )

    return TaskScene(
        scene_id=scene_id,
        seed=int(scene_dict["seed"]),
        object_poses=object_poses,
        grasp_tasks=tuple(grasp_tasks),
        canonical_map_path=scene_dict.get("canonical_map_path", ""),
    )


def load_tasks(path: str | Path) -> TaskSuite:
    """Load a task suite from a JSON file.

    Parameters
    ----------
    path
        Path to a task suite JSON file (e.g. ``resources/benchmark/grasp_benchmark.json``).

    Returns
    -------
    TaskSuite
        Parsed suite with scenes sorted naturally (scene_0, ..., scene_19).
    """
    path = Path(path)
    with path.open() as f:
        data = json.load(f)

    scene_ids = sorted(data.keys(), key=_natural_sort_key)
    scenes = tuple(_parse_scene(sid, data[sid]) for sid in scene_ids)

    return TaskSuite(
        scenes=scenes,
        metadata={"source": str(path), "num_scenes": str(len(scenes))},
    )
