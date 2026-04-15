"""Smoke tests for the Step 5 worlds-layer rewiring of ManiSkillFetchRobot.

Verifies:

1. FetchEnvConfig defaults to a valid SceneSamplerConfig.
2. ManiSkillFetchRobot.create can be instantiated via the new path
   (manifest + sampler + bind_specs) without needing a string env_id.
3. A sequence of reset() calls actually exercises multiple apartments
   — i.e. the v1 repeating-scene bug cannot recur end-to-end.

Requires the `world` pixi env (imports mani_skill). Skips cleanly in
the default env so `pixi run test` stays green there.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mani_skill", exc_type=ImportError)

from TyGrit.envs.fetch.config import FetchEnvConfig  # noqa: E402
from TyGrit.envs.fetch.maniskill import ManiSkillFetchRobot  # noqa: E402
from TyGrit.types.worlds import SceneSamplerConfig  # noqa: E402
from TyGrit.worlds.backends.maniskill import SpecBackedSceneBuilder  # noqa: E402


class TestFetchEnvConfigDefaults:
    def test_default_scene_sampler_points_at_baseline_manifest(self) -> None:
        cfg = FetchEnvConfig()
        assert isinstance(cfg.scene_sampler, SceneSamplerConfig)
        assert cfg.scene_sampler.manifest_path == "resources/worlds/replicacad.json"

    def test_no_env_id_field(self) -> None:
        # The old env_id field must be gone — confirms callers can't
        # regress by passing a string scene id instead of a manifest.
        with pytest.raises(TypeError, match="env_id"):
            FetchEnvConfig(env_id="some-string")  # type: ignore[call-arg]

    def test_custom_manifest_path_accepted(self) -> None:
        cfg = FetchEnvConfig(
            scene_sampler=SceneSamplerConfig(
                manifest_path="resources/worlds/replicacad.json",
                scene_ids=("replicacad/apt_0",),
                base_seed=42,
            ),
        )
        assert cfg.scene_sampler.scene_ids == ("replicacad/apt_0",)
        assert cfg.scene_sampler.base_seed == 42


class TestManiSkillFetchRobotWiring:
    """Single-env ManiSkillFetchRobot now routes through the worlds layer."""

    @pytest.fixture(scope="class")
    def robot(self):
        # Default scene_sampler picks up the 6-apt baseline manifest.
        # Use the wrapper's default dict obs_mode — this class's
        # _parse_observation expects sensor_data["fetch_head"]["rgb"]
        # and "depth" keys, which only exist when obs_mode includes
        # "rgb" + "depth".
        cfg = FetchEnvConfig(
            sim_opts={
                "obs_mode": "rgb+depth+state+segmentation",
                "control_mode": "pd_joint_vel",
                "render_mode": None,
            },
            camera_width=64,
            camera_height=64,
        )
        robot = ManiSkillFetchRobot(config=cfg)
        try:
            yield robot
        finally:
            robot.close()

    def test_scene_builder_is_spec_backed(self, robot) -> None:
        sb = robot._backend._env.unwrapped.scene_builder  # noqa: SLF001
        assert isinstance(sb, SpecBackedSceneBuilder)

    def test_sampler_and_scenes_are_populated(self, robot) -> None:
        # After Step 6, the baseline manifest covers all 90 ReplicaCAD
        # apts (6 main + 84 staging). The sampler pool must match.
        assert robot._sampler.scene_count == 90  # noqa: SLF001
        assert len(robot._scenes) == 90  # noqa: SLF001
        assert all(
            s.scene_id.startswith("replicacad/") for s in robot._scenes  # noqa: SLF001
        )

    def test_reset_count_starts_at_zero(self, robot) -> None:
        # Even after the constructor-time auto-reset, our reset_count
        # tracker stays at 0 until the user calls reset() explicitly.
        # The auto-reset uses build_config_idxs from gym.make kwargs,
        # not from self._reset_count.
        assert robot._reset_count == 0  # noqa: SLF001

    def test_reset_exercises_scene_variety(self, robot) -> None:
        # Call reset many times and collect which apt was loaded each
        # time. With 6 scenes and many resets, we must see more than
        # one apt — v1 bug regression guard at the env-wrapper level.
        seen: set[str] = set()
        for _ in range(30):
            robot.reset(randomize_init=False)
            sb = robot._backend._env.unwrapped.scene_builder  # noqa: SLF001
            # Infer loaded scene from the builder's internal
            # spec_to_delegate mapping — the first index selected
            # during build() is what this parallel env saw.
            last_delegate_idx = sb._delegate.build_config_idxs[0]  # noqa: SLF001
            apt_name = sb._delegate.build_configs[last_delegate_idx].split(".")[
                0
            ]  # noqa: SLF001
            seen.add(apt_name)
        # With 6 baseline apts and 30 draws under a uniform sampler,
        # we expect to hit at least 3 distinct scenes — much weaker
        # than expecting all 6, but strong enough to catch the v1
        # "same scene every time" failure mode.
        assert len(seen) >= 3, f"Scene variety too low: only saw {seen}"

    def test_reset_count_increments(self, robot) -> None:
        before = robot._reset_count  # noqa: SLF001
        robot.reset(randomize_init=False)
        after = robot._reset_count  # noqa: SLF001
        assert after == before + 1
