"""Configuration for segmentation backends."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SegmenterConfig:
    """Base segmenter configuration.

    Attributes
    ----------
    backend:
        Which backend to use: ``"sim"`` or ``"sam3"``.
    """

    backend: str = "sim"


@dataclass(frozen=True)
class SAM3SegmenterConfig(SegmenterConfig):
    """Configuration for the SAM 3 segmenter.

    Attributes
    ----------
    model_name:
        HuggingFace model identifier for SAM 3.
    text_prompt:
        Text description of the target object (e.g. ``"cup"``, ``"bottle"``).
    threshold:
        Confidence threshold for instance predictions.
    mask_threshold:
        Threshold applied to raw mask logits.
    device:
        Torch device string.
    """

    backend: str = "sam3"
    model_name: str = "facebook/sam3"
    text_prompt: str = ""
    threshold: float = 0.5
    mask_threshold: float = 0.5
    device: str = "cuda"
