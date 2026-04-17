"""Spawn task objects into a ManiSkill/Sapien scene.

Depends on ``mani_skill`` — only import from code that already has
the ManiSkill environment initialised.
"""

from __future__ import annotations

from loguru import logger

from TyGrit.types.tasks import GraspTask, TaskScene


def spawn_scene_objects(
    sapien_scene: object,
    scene: TaskScene,
) -> dict[int, str]:
    """Spawn all objects from a :class:`TaskScene` into a Sapien scene.

    Parameters
    ----------
    sapien_scene
        The ``sapien.Scene`` (or ManiSkill scene) to spawn actors into.
    scene
        Task scene whose ``object_poses`` define what to spawn.

    Returns
    -------
    dict[int, str]
        Mapping from task index to actor name for each grasp task.
    """
    from mani_skill.utils.building.actors import get_actor_builder

    actor_names: dict[int, str] = {}

    for task in scene.grasp_tasks:
        obj = task.object_pose
        actor_name = f"task_{obj.model_id}_{task.task_index}"

        try:
            builder = get_actor_builder(sapien_scene, id=f"ycb:{obj.model_id}")
            builder.set_name(actor_name)
            builder.set_initial_pose(_make_pose(obj.position, obj.orientation_wxyz))
            builder.build()
            actor_names[task.task_index] = actor_name
        except FileNotFoundError as exc:
            # ManiSkill's ``get_actor_builder`` resolves the YCB model
            # id against the on-disk asset cache. Missing assets raise
            # FileNotFoundError; the benchmark should keep going with
            # the partial scene rather than abort, because a missing
            # asset is a data-availability problem (user hasn't run
            # ``pixi run -e world download-ycb``) — not a code bug.
            logger.warning(
                "scene_setup: skip {} (task {}): YCB asset missing — {}",
                obj.model_id,
                task.task_index,
                exc,
            )

    logger.info(
        "Spawned {}/{} objects for {}",
        len(actor_names),
        len(scene.grasp_tasks),
        scene.scene_id,
    )
    return actor_names


def get_target_actor_name(actor_names: dict[int, str], task: GraspTask) -> str:
    """Look up the actor name for a specific grasp task."""
    return actor_names[task.task_index]


def _make_pose(
    position: tuple[float, float, float],
    orientation_wxyz: tuple[float, float, float, float],
) -> object:
    """Create a ``sapien.Pose`` from position and wxyz quaternion."""
    import sapien

    return sapien.Pose(
        p=list(position),
        q=list(orientation_wxyz),
    )
