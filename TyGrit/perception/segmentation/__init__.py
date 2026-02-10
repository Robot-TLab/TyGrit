"""Segmentation backends."""

from __future__ import annotations

from TyGrit.perception.segmentation.config import SAM3SegmenterConfig, SegmenterConfig
from TyGrit.perception.segmentation.sam3 import SAM3Segmenter
from TyGrit.perception.segmentation.sim import SimSegmenter

__all__ = [
    "SAM3Segmenter",
    "SAM3SegmenterConfig",
    "SegmenterConfig",
    "SimSegmenter",
    "create_segmenter",
]


def create_segmenter(config: SegmenterConfig) -> SimSegmenter | SAM3Segmenter:
    """Create a segmenter from *config*.

    Dispatches on ``config.backend``:

    - ``"sim"`` → :class:`SimSegmenter`
    - ``"sam3"`` → :class:`SAM3Segmenter` (requires :class:`SAM3SegmenterConfig`)
    """
    if config.backend == "sim":
        return SimSegmenter()

    if config.backend == "sam3":
        if not isinstance(config, SAM3SegmenterConfig):
            raise TypeError(
                "SAM3 backend requires SAM3SegmenterConfig, "
                f"got {type(config).__name__}"
            )
        return SAM3Segmenter(config)

    raise ValueError(f"Unknown segmenter backend: {config.backend!r}")
