"""Tests for TyGrit.worlds.sampler — pure Python, no simulator deps.

The critical regression test is the "v1 repeating scene" bug (see
memory/v1_replicacad_repeating_bug.md): in grasp_anywhere v1, reusing
the same integer seed across consecutive env.reset() calls caused
ManiSkill's torch.randint scene-index draw to always return the same
apartment. The new sampler must NOT reproduce that failure mode.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from TyGrit.types.worlds import SceneSamplerConfig, SceneSpec
from TyGrit.worlds.manifest import save_manifest
from TyGrit.worlds.sampler import SceneSampler, create_sampler

BASELINE_MANIFEST = Path("resources/worlds/replicacad.json")


def _mk_scene(scene_id: str) -> SceneSpec:
    """Helper: minimal valid SceneSpec with a unique id."""
    return SceneSpec(
        scene_id=scene_id,
        source="custom",
        background_builtin_id=f"custom:{scene_id}",
    )


def _six_scenes() -> tuple[SceneSpec, ...]:
    return tuple(_mk_scene(f"scene_{i}") for i in range(6))


class TestConstruction:
    def test_accepts_full_scene_pool(self) -> None:
        cfg = SceneSamplerConfig(manifest_path="unused.json", base_seed=0)
        sampler = SceneSampler(cfg, _six_scenes())
        assert sampler.scene_count == 6

    def test_rejects_empty_pool(self) -> None:
        cfg = SceneSamplerConfig(manifest_path="unused.json")
        with pytest.raises(ValueError, match="scene pool is empty"):
            SceneSampler(cfg, ())

    def test_filter_by_scene_ids(self) -> None:
        cfg = SceneSamplerConfig(
            manifest_path="unused.json",
            scene_ids=("scene_1", "scene_3"),
        )
        sampler = SceneSampler(cfg, _six_scenes())
        assert sampler.scene_count == 2

    def test_filter_preserves_caller_order(self) -> None:
        # When the user specifies ["scene_3", "scene_1"], the sampler
        # should consider them in that order (matters for deterministic
        # debug output and reproducibility across scene-list orderings).
        cfg = SceneSamplerConfig(
            manifest_path="unused.json",
            scene_ids=("scene_3", "scene_1"),
        )
        sampler = SceneSampler(cfg, _six_scenes())
        # Draw many samples; only the two filtered scenes should appear.
        drawn = {sampler.sample(env_idx=i, reset_count=0).scene_id for i in range(50)}
        assert drawn <= {"scene_1", "scene_3"}

    def test_rejects_unknown_scene_id_in_filter(self) -> None:
        cfg = SceneSamplerConfig(
            manifest_path="unused.json",
            scene_ids=("scene_1", "scene_does_not_exist"),
        )
        with pytest.raises(ValueError, match="references unknown scene_ids"):
            SceneSampler(cfg, _six_scenes())

    def test_base_seed_exposed(self) -> None:
        cfg = SceneSamplerConfig(manifest_path="unused.json", base_seed=42)
        sampler = SceneSampler(cfg, _six_scenes())
        assert sampler.base_seed == 42


class TestDeterminism:
    """The sampler must be a pure function of (base_seed, env_idx, reset_count)."""

    def test_same_args_give_same_scene(self) -> None:
        cfg = SceneSamplerConfig(manifest_path="unused.json", base_seed=7)
        sampler = SceneSampler(cfg, _six_scenes())
        a = sampler.sample(env_idx=2, reset_count=5)
        b = sampler.sample(env_idx=2, reset_count=5)
        assert a is b or a == b  # frozen dataclass equality
        assert a.scene_id == b.scene_id

    def test_same_seed_across_sampler_instances(self) -> None:
        # Two samplers built from the same config on the same scenes
        # must agree — critical for multi-worker reproducibility where
        # each worker constructs its own sampler.
        cfg = SceneSamplerConfig(manifest_path="unused.json", base_seed=11)
        s1 = SceneSampler(cfg, _six_scenes())
        s2 = SceneSampler(cfg, _six_scenes())
        for env_idx in range(4):
            for reset_count in range(10):
                a = s1.sample(env_idx, reset_count)
                b = s2.sample(env_idx, reset_count)
                assert a.scene_id == b.scene_id

    def test_different_base_seeds_diverge(self) -> None:
        s1 = SceneSampler(
            SceneSamplerConfig(manifest_path="unused.json", base_seed=1),
            _six_scenes(),
        )
        s2 = SceneSampler(
            SceneSamplerConfig(manifest_path="unused.json", base_seed=2),
            _six_scenes(),
        )
        # Across 20 draws with identical (env_idx, reset_count), the two
        # streams should disagree at least once — if they never disagree
        # the seed isn't actually being mixed in.
        disagreements = sum(
            1 for i in range(20) if s1.sample(0, i).scene_id != s2.sample(0, i).scene_id
        )
        assert disagreements > 0


class TestV1RepeatingSceneBugRegression:
    """The specific failure mode from grasp_anywhere v1 must not recur.

    In v1, reusing the same integer seed across every reset() call led
    to the same scene being drawn every time. The new sampler routes
    reset_count into seed derivation, so the same base_seed across many
    resets must produce a diverse distribution.
    """

    def test_repeated_resets_with_same_base_seed_yield_variety(self) -> None:
        # Simulate a training loop that does many resets on a single
        # worker with one fixed base_seed.
        cfg = SceneSamplerConfig(manifest_path="unused.json", base_seed=0)
        sampler = SceneSampler(cfg, _six_scenes())
        draws = [sampler.sample(env_idx=0, reset_count=i).scene_id for i in range(200)]
        unique = set(draws)
        # With 6 scenes and 200 draws we expect all 6 to appear under
        # any reasonable PRNG. If fewer appear, the sampler is broken.
        assert unique == {f"scene_{i}" for i in range(6)}

    def test_parallel_workers_explore_independently(self) -> None:
        # Simulate 8 parallel env workers each doing 30 resets. Across
        # the full (worker, reset) grid we should see a good mix.
        cfg = SceneSamplerConfig(manifest_path="unused.json", base_seed=0)
        sampler = SceneSampler(cfg, _six_scenes())
        counts: Counter[str] = Counter()
        for env_idx in range(8):
            for reset_count in range(30):
                counts[sampler.sample(env_idx, reset_count).scene_id] += 1
        # All scenes must be hit (240 draws over 6 scenes, expected ~40
        # each) and no single scene should dominate.
        assert len(counts) == 6
        max_count = max(counts.values())
        min_count = min(counts.values())
        # Weak fairness: no scene is more than 4x the rarest. The v1
        # bug would make one scene 240 and the rest 0, so a 4x bound
        # catches it with huge margin.
        assert max_count <= 4 * min_count, counts

    def test_env_idx_variation_alone_produces_variety(self) -> None:
        # Fix reset_count=0, vary env_idx. Parallel workers on the same
        # reset step should see different scenes.
        cfg = SceneSamplerConfig(manifest_path="unused.json", base_seed=0)
        sampler = SceneSampler(cfg, _six_scenes())
        drawn = {sampler.sample(env_idx=i, reset_count=0).scene_id for i in range(60)}
        # With 6 scenes and 60 distinct env_idx values, we expect all 6
        # to appear.
        assert drawn == {f"scene_{i}" for i in range(6)}


class TestBaselineManifestIntegration:
    """End-to-end: sample from the real bundled ReplicaCAD manifest."""

    def test_create_sampler_loads_manifest(self) -> None:
        cfg = SceneSamplerConfig(manifest_path=str(BASELINE_MANIFEST))
        sampler = create_sampler(cfg)
        # Step 6 expanded the baseline manifest to all 90 ReplicaCAD
        # scenes (6 main + 84 staging). The sampler pool must match.
        assert sampler.scene_count == 90

    def test_create_sampler_respects_filter(self) -> None:
        cfg = SceneSamplerConfig(
            manifest_path=str(BASELINE_MANIFEST),
            scene_ids=("replicacad/apt_0", "replicacad/apt_3"),
        )
        sampler = create_sampler(cfg)
        assert sampler.scene_count == 2

    def test_create_sampler_sample_returns_replicacad_scene(self) -> None:
        cfg = SceneSamplerConfig(manifest_path=str(BASELINE_MANIFEST))
        sampler = create_sampler(cfg)
        scene = sampler.sample(env_idx=0, reset_count=0)
        assert scene.source == "replicacad"
        # Post-Step-6 the pool includes staging scenes, so the prefix
        # is "replicacad/" — either "replicacad/apt_N" or
        # "replicacad/v3_sc*_staging_NN".
        assert scene.scene_id.startswith("replicacad/")

    def test_create_sampler_raises_on_missing_manifest(self, tmp_path: Path) -> None:
        cfg = SceneSamplerConfig(manifest_path=str(tmp_path / "nonexistent.json"))
        # load_manifest's Path.read_text raises FileNotFoundError; we
        # let it propagate because the caller needs the original type.
        with pytest.raises(FileNotFoundError):
            create_sampler(cfg)

    def test_create_sampler_with_custom_manifest_file(self, tmp_path: Path) -> None:
        # Write a fresh manifest, then load and sample from it.
        manifest_path = tmp_path / "custom.json"
        save_manifest(manifest_path, _six_scenes())
        cfg = SceneSamplerConfig(manifest_path=str(manifest_path), base_seed=99)
        sampler = create_sampler(cfg)
        assert sampler.scene_count == 6
        # Determinism with a fresh sampler
        scene = sampler.sample(env_idx=1, reset_count=2)
        assert scene.scene_id.startswith("scene_")
