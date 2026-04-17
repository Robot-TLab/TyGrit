"""Tests for TyGrit.worlds.object_sampler and object manifest loader.

Pure Python, no simulator deps — runs in the default pixi env.

The sampling contract is symmetric with SceneSampler, so the test
layout mirrors test_worlds_sampler.py: construction validation,
determinism, v1-bug regression (same base_seed across many resets
must yield variety), and round-trip integration with the object
manifest file format.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from TyGrit.types.worlds import ObjectSamplerConfig, ObjectSpec
from TyGrit.worlds.manifest import (
    MANIFEST_VERSION,
    load_object_manifest,
    save_object_manifest,
)
from TyGrit.worlds.object_sampler import ObjectSampler, create_object_sampler


def _mk_obj(name: str) -> ObjectSpec:
    """Helper: minimal valid ObjectSpec with a unique name."""
    return ObjectSpec(name=name, builtin_id=f"ycb:{name}")


def _six_objects() -> tuple[ObjectSpec, ...]:
    return tuple(_mk_obj(f"obj_{i}") for i in range(6))


def _write_manifest(tmp_path: Path, payload: dict) -> Path:
    p = tmp_path / "objects.json"
    p.write_text(json.dumps(payload))
    return p


# ─────────────────────────── manifest loader ──────────────────────────


class TestLoadObjectManifest:
    def test_roundtrip_simple(self, tmp_path: Path) -> None:
        objects = _six_objects()
        path = tmp_path / "m.json"
        save_object_manifest(path, objects)
        restored = load_object_manifest(path)
        assert restored == objects

    def test_save_writes_version_and_metadata(self, tmp_path: Path) -> None:
        path = tmp_path / "m.json"
        save_object_manifest(path, _six_objects(), source="ycb", generator="test")
        payload = json.loads(path.read_text())
        assert payload["version"] == MANIFEST_VERSION
        assert payload["source"] == "ycb"
        assert payload["generator"] == "test"
        assert len(payload["objects"]) == 6

    def test_save_omits_unset_metadata(self, tmp_path: Path) -> None:
        path = tmp_path / "m.json"
        save_object_manifest(path, _six_objects())
        payload = json.loads(path.read_text())
        assert "source" not in payload
        assert "generator" not in payload

    def test_save_creates_parent_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "deep" / "m.json"
        save_object_manifest(path, _six_objects())
        assert path.exists()

    def test_output_ends_with_newline(self, tmp_path: Path) -> None:
        path = tmp_path / "m.json"
        save_object_manifest(path, _six_objects())
        assert path.read_text().endswith("\n")

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        path = tmp_path / "m.json"
        save_object_manifest(str(path), _six_objects())
        assert load_object_manifest(str(path)) == _six_objects()

    def test_rejects_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_object_manifest(tmp_path / "nope.json")

    def test_rejects_invalid_json(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text("{not valid")
        with pytest.raises(ValueError, match="invalid JSON"):
            load_object_manifest(p)

    def test_rejects_non_object_toplevel(self, tmp_path: Path) -> None:
        p = tmp_path / "list.json"
        p.write_text("[]")
        with pytest.raises(ValueError, match="top-level must be a JSON object"):
            load_object_manifest(p)

    def test_rejects_missing_version(self, tmp_path: Path) -> None:
        p = _write_manifest(tmp_path, {"objects": []})
        with pytest.raises(ValueError, match="unsupported schema version"):
            load_object_manifest(p)

    def test_rejects_wrong_version(self, tmp_path: Path) -> None:
        p = _write_manifest(tmp_path, {"version": 999, "objects": []})
        with pytest.raises(ValueError, match="unsupported schema version 999"):
            load_object_manifest(p)

    def test_rejects_missing_objects_key(self, tmp_path: Path) -> None:
        p = _write_manifest(tmp_path, {"version": MANIFEST_VERSION})
        with pytest.raises(ValueError, match="'objects' must be a list"):
            load_object_manifest(p)

    def test_object_validator_error_is_wrapped(self, tmp_path: Path) -> None:
        # ObjectSpec with no asset source triggers __post_init__.
        p = _write_manifest(
            tmp_path,
            {
                "version": MANIFEST_VERSION,
                "objects": [{"name": "orphan"}],
            },
        )
        with pytest.raises(ValueError) as info:
            load_object_manifest(p)
        msg = str(info.value)
        assert "'orphan'" in msg
        assert "must set one of" in msg


# ─────────────────────────── sampler construction ─────────────────────


class TestConstruction:
    def test_accepts_full_pool(self) -> None:
        cfg = ObjectSamplerConfig(manifest_path="unused.json")
        sampler = ObjectSampler(cfg, _six_objects())
        assert sampler.object_count == 6

    def test_rejects_empty_pool(self) -> None:
        cfg = ObjectSamplerConfig(manifest_path="unused.json")
        with pytest.raises(ValueError, match="object pool is empty"):
            ObjectSampler(cfg, ())

    def test_rejects_duplicate_names(self) -> None:
        cfg = ObjectSamplerConfig(
            manifest_path="unused.json",
            object_names=("obj_1",),
        )
        pool_with_dup = _six_objects() + (_mk_obj("obj_1"),)
        with pytest.raises(ValueError, match="duplicate names"):
            ObjectSampler(cfg, pool_with_dup)

    def test_filter_by_object_names(self) -> None:
        cfg = ObjectSamplerConfig(
            manifest_path="unused.json",
            object_names=("obj_1", "obj_3"),
        )
        sampler = ObjectSampler(cfg, _six_objects())
        assert sampler.object_count == 2

    def test_filter_preserves_caller_order(self) -> None:
        cfg = ObjectSamplerConfig(
            manifest_path="unused.json",
            object_names=("obj_3", "obj_1"),
        )
        sampler = ObjectSampler(cfg, _six_objects())
        names = tuple(o.name for o in sampler.objects)
        assert names == ("obj_3", "obj_1")

    def test_rejects_unknown_name_in_filter(self) -> None:
        cfg = ObjectSamplerConfig(
            manifest_path="unused.json",
            object_names=("obj_1", "does_not_exist"),
        )
        with pytest.raises(ValueError, match="references unknown objects"):
            ObjectSampler(cfg, _six_objects())

    def test_base_seed_exposed(self) -> None:
        cfg = ObjectSamplerConfig(manifest_path="unused.json", base_seed=42)
        sampler = ObjectSampler(cfg, _six_objects())
        assert sampler.base_seed == 42


# ─────────────────────────── determinism ──────────────────────────────


class TestDeterminism:
    def test_same_args_give_same_object(self) -> None:
        cfg = ObjectSamplerConfig(manifest_path="unused.json", base_seed=7)
        sampler = ObjectSampler(cfg, _six_objects())
        a = sampler.sample(env_idx=2, reset_count=5)
        b = sampler.sample(env_idx=2, reset_count=5)
        assert a == b

    def test_same_seed_across_sampler_instances(self) -> None:
        cfg = ObjectSamplerConfig(manifest_path="unused.json", base_seed=11)
        s1 = ObjectSampler(cfg, _six_objects())
        s2 = ObjectSampler(cfg, _six_objects())
        for env_idx in range(4):
            for reset_count in range(10):
                assert s1.sample(env_idx, reset_count) == s2.sample(
                    env_idx, reset_count
                )

    def test_different_base_seeds_diverge(self) -> None:
        s1 = ObjectSampler(
            ObjectSamplerConfig(manifest_path="unused.json", base_seed=1),
            _six_objects(),
        )
        s2 = ObjectSampler(
            ObjectSamplerConfig(manifest_path="unused.json", base_seed=2),
            _six_objects(),
        )
        disagreements = sum(1 for i in range(20) if s1.sample(0, i) != s2.sample(0, i))
        assert disagreements > 0


# ─────────────────────────── k-object sampling ────────────────────────


class TestSampleK:
    def test_sample_k_idxs_returns_k_distinct_without_replace(self) -> None:
        cfg = ObjectSamplerConfig(manifest_path="unused.json")
        sampler = ObjectSampler(cfg, _six_objects())
        idxs = sampler.sample_k_idxs(env_idx=0, reset_count=0, k=4)
        assert len(idxs) == 4
        assert len(set(idxs)) == 4  # distinct

    def test_sample_k_idxs_with_replace_may_duplicate(self) -> None:
        cfg = ObjectSamplerConfig(manifest_path="unused.json")
        sampler = ObjectSampler(cfg, _six_objects())
        # With replacement and k > pool size, we definitely get duplicates.
        idxs = sampler.sample_k_idxs(env_idx=0, reset_count=0, k=20, replace=True)
        assert len(idxs) == 20
        assert len(set(idxs)) < 20

    def test_sample_k_idxs_rejects_k_exceeds_pool_without_replace(self) -> None:
        cfg = ObjectSamplerConfig(manifest_path="unused.json")
        sampler = ObjectSampler(cfg, _six_objects())
        with pytest.raises(ValueError, match="exceeds pool size"):
            sampler.sample_k_idxs(env_idx=0, reset_count=0, k=10)

    def test_sample_k_idxs_rejects_negative_k(self) -> None:
        cfg = ObjectSamplerConfig(manifest_path="unused.json")
        sampler = ObjectSampler(cfg, _six_objects())
        with pytest.raises(ValueError, match="k must be >= 0"):
            sampler.sample_k_idxs(env_idx=0, reset_count=0, k=-1)

    def test_sample_k_zero_returns_empty(self) -> None:
        cfg = ObjectSamplerConfig(manifest_path="unused.json")
        sampler = ObjectSampler(cfg, _six_objects())
        assert sampler.sample_k(env_idx=0, reset_count=0, k=0) == ()

    def test_sample_k_determinism(self) -> None:
        cfg = ObjectSamplerConfig(manifest_path="unused.json", base_seed=3)
        sampler = ObjectSampler(cfg, _six_objects())
        a = sampler.sample_k_idxs(env_idx=0, reset_count=7, k=3)
        b = sampler.sample_k_idxs(env_idx=0, reset_count=7, k=3)
        assert a == b

    def test_sample_k_returns_objects_matching_idxs(self) -> None:
        cfg = ObjectSamplerConfig(manifest_path="unused.json")
        sampler = ObjectSampler(cfg, _six_objects())
        idxs = sampler.sample_k_idxs(env_idx=2, reset_count=3, k=3)
        objs = sampler.sample_k(env_idx=2, reset_count=3, k=3)
        assert tuple(sampler.objects[i] for i in idxs) == objs


# ─────────────────── v1 repeating-scene bug regression ────────────────


class TestV1BugRegression:
    """Same contract as the SceneSampler regression tests.

    The v1 grasp_anywhere bug was specifically about SCENES, but its
    root cause (reusing one seed across resets) applies equally to
    object sampling. These tests guard against a parallel bug in the
    object-sampling path.
    """

    def test_repeated_resets_with_same_base_seed_yield_variety(self) -> None:
        cfg = ObjectSamplerConfig(manifest_path="unused.json", base_seed=0)
        sampler = ObjectSampler(cfg, _six_objects())
        draws = [sampler.sample(env_idx=0, reset_count=i).name for i in range(200)]
        assert set(draws) == {f"obj_{i}" for i in range(6)}

    def test_parallel_workers_explore_independently(self) -> None:
        cfg = ObjectSamplerConfig(manifest_path="unused.json", base_seed=0)
        sampler = ObjectSampler(cfg, _six_objects())
        counts: Counter[str] = Counter()
        for env_idx in range(8):
            for reset_count in range(30):
                counts[sampler.sample(env_idx, reset_count).name] += 1
        assert len(counts) == 6
        max_count = max(counts.values())
        min_count = min(counts.values())
        assert max_count <= 4 * min_count, counts


# ─────────────────────────── integration ──────────────────────────────


class TestCreateObjectSampler:
    def test_loads_manifest_and_samples(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "ycb.json"
        save_object_manifest(manifest_path, _six_objects(), source="ycb")
        cfg = ObjectSamplerConfig(manifest_path=str(manifest_path))
        sampler = create_object_sampler(cfg)
        assert sampler.object_count == 6
        obj = sampler.sample(env_idx=0, reset_count=0)
        assert obj.name.startswith("obj_")

    def test_respects_filter(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "ycb.json"
        save_object_manifest(manifest_path, _six_objects(), source="ycb")
        cfg = ObjectSamplerConfig(
            manifest_path=str(manifest_path),
            object_names=("obj_0", "obj_2"),
        )
        sampler = create_object_sampler(cfg)
        assert sampler.object_count == 2

    def test_raises_on_missing_manifest(self, tmp_path: Path) -> None:
        cfg = ObjectSamplerConfig(manifest_path=str(tmp_path / "nope.json"))
        with pytest.raises(FileNotFoundError):
            create_object_sampler(cfg)
