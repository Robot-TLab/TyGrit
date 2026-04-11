"""Tests for TyGrit.worlds.maniskill — requires the `world` pixi env.

These tests import ``mani_skill`` and instantiate a real ManiSkill env,
so they MUST be run via ``pixi run -e world test
test/test_worlds_maniskill.py`` — the default env has no torch /
mani-skill. Integration tests amortize the env-creation cost via a
module-scoped fixture.

The key regression under test is the grasp_anywhere v1 "repeating
scene" bug: SpecBackedSceneBuilder.sample_build_config_idxs must
raise, and the only supported way to select scenes is via explicit
build_config_idxs passed through env.reset(options=...).
"""

from __future__ import annotations

import pytest

# Skip the entire module cleanly in any env that doesn't have
# mani-skill installed — e.g. the default pixi env. pytest.importorskip
# raises Skipped at collection time, which is the pytest-native way to
# handle optional deps and is the pattern used by other sim-dependent
# tests in this repo (test_ik.py, test_cross_validation.py via
# pytracik). Passing exc_type=ImportError keeps us compatible with
# pytest 9.1's planned importorskip tightening.
pytest.importorskip("mani_skill", exc_type=ImportError)

import gymnasium as gym  # noqa: E402

from TyGrit.types.worlds import SceneSpec  # noqa: E402
from TyGrit.worlds.manifest import load_manifest  # noqa: E402
from TyGrit.worlds.maniskill import (  # noqa: E402
    SpecBackedSceneBuilder,
    bind_specs,
    build_world,
)

BASELINE_MANIFEST = "resources/worlds/replicacad.json"


# ─────────────────────────── unit-level ──────────────────────────


class _FakeEnv:
    """Minimal stand-in for a ManiSkill env.

    Only exposes the attributes SceneBuilder.__init__ touches. Cannot
    be used to drive a real ``build()`` call because that path reads
    ``env.num_envs`` and ``env.scene`` from Sapien. Good enough for
    testing set_specs validation in isolation.
    """

    num_envs: int = 1


class TestBindSpecsFactory:
    """bind_specs produces a working closure subclass."""

    def test_returns_a_subclass(self) -> None:
        cls = bind_specs([])
        assert issubclass(cls, SpecBackedSceneBuilder)

    def test_bound_specs_live_on_the_class(self) -> None:
        specs = load_manifest(BASELINE_MANIFEST)
        cls = bind_specs(specs)
        assert cls._cls_specs == specs

    def test_empty_binding_is_valid(self) -> None:
        # An empty bind is legal and should NOT auto-create a delegate
        # during __init__ — the caller can call set_specs() later.
        cls = bind_specs([])
        builder = cls(_FakeEnv())
        assert builder.build_configs == ()
        assert builder._delegate is None  # noqa: SLF001

    def test_different_bindings_are_independent_classes(self) -> None:
        # Two separate bindings should produce distinct classes so
        # re-registering wouldn't collide. Verified by identity of the
        # class objects, not by names (names match by design).
        a = bind_specs([])
        b = bind_specs([])
        assert a is not b


class TestSpecValidation:
    """set_specs rejects misconfigured inputs with clear messages."""

    def _minimal_valid_spec(self, apt_idx: int = 0) -> SceneSpec:
        return SceneSpec(
            scene_id=f"replicacad/apt_{apt_idx}",
            source="replicacad",
            background_builtin_id=f"replicacad:apt_{apt_idx}",
        )

    def test_rejects_mixed_sources(self) -> None:
        builder = SpecBackedSceneBuilder(_FakeEnv())
        mixed = (
            self._minimal_valid_spec(0),
            SceneSpec(
                scene_id="custom/other",
                source="custom",
                background_builtin_id="custom:other",
            ),
        )
        with pytest.raises(ValueError, match="mixed scene sources"):
            builder.set_specs(mixed)

    def test_rejects_unsupported_source(self) -> None:
        builder = SpecBackedSceneBuilder(_FakeEnv())
        specs = (
            SceneSpec(
                scene_id="hssd/108736824",
                source="hssd",
                background_builtin_id="hssd:108736824",
            ),
        )
        with pytest.raises(ValueError, match="hssd.* not yet supported"):
            builder.set_specs(specs)

    def test_rejects_unknown_scene_id(self) -> None:
        builder = SpecBackedSceneBuilder(_FakeEnv())
        specs = (
            SceneSpec(
                scene_id="replicacad/apt_notreal",
                source="replicacad",
                background_builtin_id="replicacad:apt_notreal",
            ),
        )
        with pytest.raises(ValueError, match="not in the ReplicaCAD build_configs"):
            builder.set_specs(specs)

    def test_empty_specs_resets_state(self) -> None:
        # Calling set_specs(()) should clear the delegate, not raise.
        builder = SpecBackedSceneBuilder(_FakeEnv())
        builder.set_specs([self._minimal_valid_spec(0)])
        assert builder._delegate is not None  # noqa: SLF001
        builder.set_specs(())
        assert builder._delegate is None  # noqa: SLF001
        assert builder.build_configs == ()

    def test_valid_replicacad_specs_populate_build_configs(self) -> None:
        builder = SpecBackedSceneBuilder(_FakeEnv())
        specs = tuple(self._minimal_valid_spec(i) for i in range(3))
        builder.set_specs(specs)
        assert builder.build_configs == specs
        assert builder._delegate is not None  # noqa: SLF001
        # Three specs → three delegate indices, one per spec
        assert len(builder._spec_to_delegate_idxs) == 3  # noqa: SLF001


class TestV1BugGuard:
    """The explicit guard against the v1 repeating-scene bug."""

    def test_sample_build_config_idxs_raises(self) -> None:
        # Implicit sampling via torch.randint is exactly what caused
        # the v1 bug — we must raise here so any accidental flow that
        # invokes sampling surfaces the error immediately instead of
        # silently reusing the same apartment.
        builder = SpecBackedSceneBuilder(_FakeEnv())
        with pytest.raises(RuntimeError, match="implicit sampling is disabled"):
            builder.sample_build_config_idxs()

    def test_build_rejects_none_idxs(self) -> None:
        builder = SpecBackedSceneBuilder(_FakeEnv())
        builder.set_specs(
            [
                SceneSpec(
                    scene_id="replicacad/apt_0",
                    source="replicacad",
                    background_builtin_id="replicacad:apt_0",
                )
            ]
        )
        with pytest.raises(ValueError, match="build_config_idxs is None"):
            builder.build(None)

    def test_build_rejects_mismatched_count(self) -> None:
        # env.num_envs = 1 but we pass 3 indices → error.
        builder = SpecBackedSceneBuilder(_FakeEnv())
        builder.set_specs(
            [
                SceneSpec(
                    scene_id="replicacad/apt_0",
                    source="replicacad",
                    background_builtin_id="replicacad:apt_0",
                )
            ]
        )
        with pytest.raises(ValueError, match="expected 1 indices"):
            builder.build([0, 0, 0])

    def test_build_without_specs_raises_runtime_error(self) -> None:
        # Bare construction with no specs and no bind_specs call should
        # fail loudly on build, not silently load nothing.
        builder = SpecBackedSceneBuilder(_FakeEnv())
        with pytest.raises(RuntimeError, match="no specs configured"):
            builder.build([0])


# ───────────────────── integration (real gym.make) ─────────────────


@pytest.fixture(scope="module")
def baseline_specs() -> tuple[SceneSpec, ...]:
    return load_manifest(BASELINE_MANIFEST)


@pytest.fixture(scope="module")
def maniskill_env(baseline_specs):
    """Build a ManiSkill env with our bound SpecBackedSceneBuilder.

    Module-scoped because gym.make + initial scene build is expensive
    (several seconds). Tests in this file mutate the env via
    env.reset(options=...) but share the underlying Sapien scene.

    build_config_idxs=[0] is required at gym.make so the constructor-
    time auto-reset (inside BaseEnv.__init__) has explicit indices and
    does not fall through to our sample_build_config_idxs guard.
    """
    env = gym.make(
        "SceneManipulation-v1",
        scene_builder_cls=bind_specs(baseline_specs),
        build_config_idxs=[0],
        num_envs=1,
    )
    try:
        yield env
    finally:
        env.close()


class TestReplicaCADIntegration:
    """End-to-end: real ManiSkill env + SpecBackedSceneBuilder."""

    def test_scene_builder_is_bound_subclass(self, maniskill_env) -> None:
        sb = maniskill_env.unwrapped.scene_builder
        assert isinstance(sb, SpecBackedSceneBuilder)

    def test_build_configs_match_baseline(self, maniskill_env, baseline_specs) -> None:
        sb = maniskill_env.unwrapped.scene_builder
        assert sb.build_configs == baseline_specs

    def test_navigable_positions_available(self, maniskill_env) -> None:
        # After the first reset inside gym.make, navigable_positions
        # must be populated (list of length num_envs).
        sb = maniskill_env.unwrapped.scene_builder
        nav = sb.navigable_positions
        assert nav is not None
        assert len(nav) == maniskill_env.unwrapped.num_envs

    def test_build_world_selects_specific_scene(
        self, maniskill_env, baseline_specs
    ) -> None:
        built = build_world(maniskill_env, baseline_specs, per_env_scene_idxs=[2])
        assert built.spec.scene_id == "replicacad/apt_2"
        # navigable_positions must change (or at least be present) after
        # reconfiguration to a new scene.
        assert built.navigable_positions is not None
        assert len(built.navigable_positions) == 1

    def test_build_world_different_idx_gives_different_scene(
        self, maniskill_env, baseline_specs
    ) -> None:
        first = build_world(maniskill_env, baseline_specs, per_env_scene_idxs=[0])
        second = build_world(maniskill_env, baseline_specs, per_env_scene_idxs=[5])
        assert first.spec.scene_id == "replicacad/apt_0"
        assert second.spec.scene_id == "replicacad/apt_5"
        assert first.spec != second.spec

    def test_build_world_rejects_wrong_scene_builder_type(self, baseline_specs) -> None:
        # An env whose scene_builder_cls is NOT SpecBackedSceneBuilder
        # should be rejected immediately. Use the default ReplicaCAD
        # builder — it's in ManiSkill's shipped registry and loads
        # without TyGrit glue.
        env = gym.make(
            "SceneManipulation-v1",
            scene_builder_cls="ReplicaCAD",
            num_envs=1,
        )
        try:
            with pytest.raises(TypeError, match="must be a SpecBackedSceneBuilder"):
                build_world(env, baseline_specs, per_env_scene_idxs=[0])
        finally:
            env.close()
