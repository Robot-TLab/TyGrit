"""Tests for task suite loader — pure Python, no ManiSkill needed."""

from __future__ import annotations

from pathlib import Path

import pytest

from TyGrit.tasks.loader import load_tasks
from TyGrit.types.tasks import (
    GraspTask,
    TaskScene,
    TaskSuite,
)

TASKS_PATH = Path("resources/benchmark/grasp_benchmark.json")


@pytest.fixture
def suite() -> TaskSuite:
    return load_tasks(TASKS_PATH)


class TestLoadTasks:
    def test_returns_task_suite(self, suite: TaskSuite) -> None:
        assert isinstance(suite, TaskSuite)

    def test_num_scenes(self, suite: TaskSuite) -> None:
        assert len(suite.scenes) == 20

    def test_total_tasks(self, suite: TaskSuite) -> None:
        assert suite.total_tasks == 400

    def test_scenes_sorted_naturally(self, suite: TaskSuite) -> None:
        ids = [s.scene_id for s in suite.scenes]
        assert ids == [f"scene_{i}" for i in range(20)]

    def test_each_scene_has_20_tasks(self, suite: TaskSuite) -> None:
        for scene in suite.scenes:
            assert scene.num_tasks == 20

    def test_scene_seeds_are_sequential(self, suite: TaskSuite) -> None:
        seeds = [s.seed for s in suite.scenes]
        assert seeds == list(range(20))


class TestTaskScene:
    def test_scene_id(self, suite: TaskSuite) -> None:
        assert suite.scenes[0].scene_id == "scene_0"

    def test_scene_seed(self, suite: TaskSuite) -> None:
        assert suite.scenes[0].seed == 0

    def test_canonical_map_path(self, suite: TaskSuite) -> None:
        path = suite.scenes[0].canonical_map_path
        assert "scene_0" in path

    def test_object_poses_count(self, suite: TaskSuite) -> None:
        assert len(suite.scenes[0].object_poses) == 20


class TestObjectPose:
    def test_model_id(self, suite: TaskSuite) -> None:
        obj = suite.scenes[0].grasp_tasks[0].object_pose
        assert isinstance(obj.model_id, str)
        assert len(obj.model_id) > 0

    def test_position_is_3_tuple(self, suite: TaskSuite) -> None:
        obj = suite.scenes[0].grasp_tasks[0].object_pose
        assert isinstance(obj.position, tuple)
        assert len(obj.position) == 3
        assert all(isinstance(v, float) for v in obj.position)

    def test_orientation_is_4_tuple(self, suite: TaskSuite) -> None:
        obj = suite.scenes[0].grasp_tasks[0].object_pose
        assert isinstance(obj.orientation_wxyz, tuple)
        assert len(obj.orientation_wxyz) == 4

    def test_orientation_is_unit_quaternion(self, suite: TaskSuite) -> None:
        obj = suite.scenes[0].grasp_tasks[0].object_pose
        norm = sum(v**2 for v in obj.orientation_wxyz) ** 0.5
        assert abs(norm - 1.0) < 1e-4

    def test_frozen(self, suite: TaskSuite) -> None:
        obj = suite.scenes[0].grasp_tasks[0].object_pose
        with pytest.raises(AttributeError):
            obj.model_id = "changed"  # type: ignore[misc]


class TestDynamicObstacle:
    def test_all_tasks_have_obstacle(self, suite: TaskSuite) -> None:
        for scene in suite.scenes:
            for task in scene.grasp_tasks:
                assert task.dynamic_obstacle is not None

    def test_obstacle_type(self, suite: TaskSuite) -> None:
        obs = suite.scenes[0].grasp_tasks[0].dynamic_obstacle
        assert obs is not None
        assert obs.obstacle_type == "pedestrian_box"

    def test_obstacle_dimension(self, suite: TaskSuite) -> None:
        obs = suite.scenes[0].grasp_tasks[0].dynamic_obstacle
        assert obs is not None
        assert isinstance(obs.dimension, tuple)
        assert len(obs.dimension) == 3
        assert all(d > 0 for d in obs.dimension)


class TestGraspTask:
    def test_task_indices_sequential(self, suite: TaskSuite) -> None:
        for scene in suite.scenes:
            indices = [t.task_index for t in scene.grasp_tasks]
            assert indices == list(range(20))

    def test_task_object_matches_scene_objects(self, suite: TaskSuite) -> None:
        scene = suite.scenes[0]
        for task in scene.grasp_tasks:
            assert task.object_pose is scene.object_poses[task.task_index]


class TestIterTasks:
    def test_yields_all_400_tasks(self, suite: TaskSuite) -> None:
        pairs = list(suite.iter_tasks())
        assert len(pairs) == 400

    def test_yields_scene_task_tuples(self, suite: TaskSuite) -> None:
        for scene, task in suite.iter_tasks():
            assert isinstance(scene, TaskScene)
            assert isinstance(task, GraspTask)

    def test_first_pair(self, suite: TaskSuite) -> None:
        scene, task = next(suite.iter_tasks())
        assert scene.scene_id == "scene_0"
        assert task.task_index == 0


class TestMetadata:
    def test_has_source(self, suite: TaskSuite) -> None:
        assert "source" in suite.metadata

    def test_has_num_scenes(self, suite: TaskSuite) -> None:
        assert suite.metadata["num_scenes"] == "20"
