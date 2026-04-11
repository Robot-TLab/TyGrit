"""Task data types — pure frozen dataclasses, no simulation deps."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ObjectPose:
    """A single object placement in the scene."""

    model_id: str  # e.g. "063-a_marbles"
    position: tuple[float, float, float]  # world [x, y, z]
    orientation_wxyz: tuple[float, float, float, float]  # Sapien WXYZ


@dataclass(frozen=True)
class DynamicObstacle:
    """A dynamic obstacle paired with a grasp task."""

    obstacle_type: str  # e.g. "pedestrian_box"
    start_position: tuple[float, float, float]
    start_orientation_wxyz: tuple[float, float, float, float]
    dimension: tuple[float, float, float]  # [w, d, h] metres


@dataclass(frozen=True)
class GraspTask:
    """A single grasp task targeting one object in the scene."""

    task_index: int
    object_pose: ObjectPose
    dynamic_obstacle: DynamicObstacle | None = None


@dataclass(frozen=True)
class TaskScene:
    """A scene with objects to spawn and tasks to evaluate."""

    scene_id: str  # "scene_0"
    seed: int
    object_poses: tuple[ObjectPose, ...]  # ALL objects to spawn
    grasp_tasks: tuple[GraspTask, ...]  # each targets one object
    canonical_map_path: str = ""

    @property
    def num_tasks(self) -> int:
        return len(self.grasp_tasks)


@dataclass(frozen=True)
class TaskSuite:
    """A collection of task scenes with metadata."""

    scenes: tuple[TaskScene, ...]
    metadata: dict[str, str] = field(default_factory=dict)

    def iter_tasks(self) -> Iterator[tuple[TaskScene, GraspTask]]:
        """Yield ``(scene, task)`` pairs across all scenes."""
        for scene in self.scenes:
            for task in scene.grasp_tasks:
                yield scene, task

    @property
    def total_tasks(self) -> int:
        return sum(s.num_tasks for s in self.scenes)
