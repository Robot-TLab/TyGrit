"""SAM 3 segmenter — text-prompted segmentation via facebookresearch/sam3."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from PIL import Image

from TyGrit.perception.segmentation.config import SAM3SegmenterConfig

if TYPE_CHECKING:
    from sam3.model.sam3_image_processor import Sam3Processor


class SAM3Segmenter:
    """Text-prompted segmenter using Meta's SAM 3.

    Lazy-loads the model and processor on the first ``segment()`` call so
    module-level import works without torch / sam3 installed.
    """

    def __init__(self, config: SAM3SegmenterConfig) -> None:
        self._config = config
        self._processor: Sam3Processor | None = None

    def segment(
        self,
        rgb: npt.NDArray[np.uint8],
        point: tuple[int, int],
        segmentation: npt.NDArray[np.int32] | None = None,
    ) -> npt.NDArray[np.uint8] | None:
        """Segment the object described by *text_prompt* closest to *point*.

        Parameters
        ----------
        rgb:
            ``(H, W, 3)`` RGB image.
        point:
            ``(u, v)`` pixel used to disambiguate when multiple instances match.
        segmentation:
            Ignored — SAM 3 does not use ground-truth labels.

        Returns
        -------
        ``(H, W)`` uint8 binary mask, or ``None`` if no instance is found.
        """
        self._ensure_loaded()
        assert self._processor is not None

        cfg = self._config
        image = Image.fromarray(rgb)

        inference_state = self._processor.set_image(image)
        output = self._processor.set_text_prompt(
            state=inference_state,
            prompt=cfg.text_prompt,
        )

        masks = output["masks"]  # (N, H, W) tensor
        scores = output["scores"]  # (N,) tensor

        if masks is None or len(masks) == 0:
            return None

        # Filter by threshold.
        keep = scores >= cfg.threshold
        masks = masks[keep]
        if len(masks) == 0:
            return None

        # Pick the instance whose mask centre is closest to *point*.
        u, v = point
        best_mask: npt.NDArray[np.uint8] | None = None
        best_dist = float("inf")

        for mask_t in masks:
            binary = (mask_t > cfg.mask_threshold).cpu().numpy().astype(np.uint8)

            ys, xs = np.nonzero(binary)
            if len(ys) == 0:
                continue

            cx = float(xs.mean())
            cy = float(ys.mean())
            dist = (cx - u) ** 2 + (cy - v) ** 2

            if dist < best_dist:
                best_dist = dist
                best_mask = binary

        return best_mask

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._processor is not None:
            return

        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        cfg = self._config
        model = build_sam3_image_model(cfg.model_name, device=cfg.device)
        self._processor = Sam3Processor(model)
