"""Tests for TyGrit.perception.segmentation."""

from __future__ import annotations

import numpy as np
import pytest

from TyGrit.perception.segmentation import (
    SAM3Segmenter,
    SAM3SegmenterConfig,
    SegmenterConfig,
    SimSegmenter,
    create_segmenter,
)

# ── Helpers ──────────────────────────────────────────────────────────


def _make_segmentation(h: int = 4, w: int = 6) -> np.ndarray:
    """Return a synthetic (H, W, 1) segmentation map.

    Layout (4×6)::

        0 0 1 1 1 0
        0 0 1 1 1 0
        0 2 2 0 0 3
        0 2 2 0 0 3
    """
    seg = np.zeros((h, w, 1), dtype=np.int32)
    seg[0:2, 2:5, 0] = 1
    seg[2:4, 1:3, 0] = 2
    seg[2:4, 5:6, 0] = 3
    return seg


# ── SimSegmenter.segment ─────────────────────────────────────────────


class TestSimSegment:
    def test_returns_mask_for_object(self):
        seg = _make_segmentation()
        s = SimSegmenter()
        mask = s.segment(
            np.zeros((4, 6, 3), dtype=np.uint8), point=(2, 0), segmentation=seg
        )
        assert mask is not None
        assert mask.dtype == np.uint8
        assert mask.shape == (4, 6)
        # instance 1 occupies rows 0-1, cols 2-4
        assert mask[0, 2] == 1
        assert mask[1, 4] == 1
        assert mask[0, 0] == 0  # background
        assert mask[2, 1] == 0  # different instance

    def test_background_returns_none(self):
        seg = _make_segmentation()
        s = SimSegmenter()
        result = s.segment(
            np.zeros((4, 6, 3), dtype=np.uint8), point=(0, 0), segmentation=seg
        )
        assert result is None

    def test_oob_returns_none(self):
        seg = _make_segmentation()
        s = SimSegmenter()
        # u=10 is out of bounds (width=6)
        assert (
            s.segment(
                np.zeros((4, 6, 3), dtype=np.uint8), point=(10, 0), segmentation=seg
            )
            is None
        )
        # v=10 is out of bounds (height=4)
        assert (
            s.segment(
                np.zeros((4, 6, 3), dtype=np.uint8), point=(0, 10), segmentation=seg
            )
            is None
        )

    def test_no_segmentation_returns_none(self):
        s = SimSegmenter()
        result = s.segment(
            np.zeros((4, 6, 3), dtype=np.uint8), point=(2, 0), segmentation=None
        )
        assert result is None

    def test_wrong_shape_raises(self):
        s = SimSegmenter()
        bad = np.zeros((4, 6), dtype=np.int32)  # missing last dim
        with pytest.raises(ValueError, match="Expected segmentation shape"):
            s.segment(
                np.zeros((4, 6, 3), dtype=np.uint8), point=(0, 0), segmentation=bad
            )


# ── SimSegmenter.segment_by_id ───────────────────────────────────────


class TestSimSegmentById:
    def test_returns_mask_for_id(self):
        seg = _make_segmentation()
        mask = SimSegmenter.segment_by_id(seg, 2)
        assert mask.dtype == np.uint8
        assert mask.shape == (4, 6)
        assert mask[2, 1] == 1
        assert mask[3, 2] == 1
        assert mask[0, 0] == 0

    def test_missing_id_returns_zeros(self):
        seg = _make_segmentation()
        mask = SimSegmenter.segment_by_id(seg, 99)
        assert np.all(mask == 0)


# ── create_segmenter factory ─────────────────────────────────────────


class TestCreateSegmenter:
    def test_sim_backend(self):
        seg = create_segmenter(SegmenterConfig(backend="sim"))
        assert isinstance(seg, SimSegmenter)

    def test_sam3_backend(self):
        cfg = SAM3SegmenterConfig(text_prompt="cup")
        seg = create_segmenter(cfg)
        assert isinstance(seg, SAM3Segmenter)

    def test_sam3_requires_config_type(self):
        cfg = SegmenterConfig(backend="sam3")
        with pytest.raises(TypeError, match="SAM3SegmenterConfig"):
            create_segmenter(cfg)

    def test_unknown_backend_raises(self):
        cfg = SegmenterConfig(backend="unknown")
        with pytest.raises(ValueError, match="Unknown segmenter backend"):
            create_segmenter(cfg)


# ── Config construction ──────────────────────────────────────────────


class TestSegmenterConfig:
    def test_defaults(self):
        cfg = SegmenterConfig()
        assert cfg.backend == "sim"

    def test_sam3_defaults(self):
        cfg = SAM3SegmenterConfig()
        assert cfg.backend == "sam3"
        assert cfg.model_name == "facebook/sam3"
        assert cfg.text_prompt == ""
        assert cfg.threshold == 0.5
        assert cfg.device == "cuda"

    def test_sam3_custom(self):
        cfg = SAM3SegmenterConfig(text_prompt="bottle", threshold=0.8, device="cpu")
        assert cfg.text_prompt == "bottle"
        assert cfg.threshold == 0.8
        assert cfg.device == "cpu"
