"""GraspGen predictor — satisfies the ``GraspPredictor`` protocol."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from TyGrit.perception.grasping.config import GraspGenConfig
from TyGrit.types.grasp import GraspPose


class GraspGenPredictor:
    """GraspGen diffusion-based grasp predictor.

    Lazy-loads the model on first ``predict()`` call so module-level
    import works without torch / GraspGen deps installed.
    """

    def __init__(self, config: GraspGenConfig) -> None:
        self._config = config
        self._sampler: object | None = None

    # ------------------------------------------------------------------
    # GraspPredictor protocol
    # ------------------------------------------------------------------

    def predict(self, cloud: npt.NDArray[np.float32]) -> list[GraspPose]:
        """Generate grasp candidates for *cloud* (N, 3)."""
        self._ensure_loaded()

        from grasp_gen.grasp_server import GraspGenSampler

        cfg = self._config
        grasps_t, scores_t = GraspGenSampler.run_inference(
            object_pc=cloud,
            grasp_sampler=self._sampler,
            grasp_threshold=cfg.score_threshold if cfg.score_threshold > 0 else -1.0,
            num_grasps=cfg.num_grasps,
            topk_num_grasps=cfg.topk_num_grasps,
            min_grasps=cfg.min_grasps,
            max_tries=cfg.max_tries,
            remove_outliers=cfg.remove_outliers,
        )

        # torch (M, 4, 4) / (M,) → numpy
        transforms = grasps_t.cpu().numpy().astype(np.float64)
        scores = scores_t.cpu().numpy().astype(np.float64)

        # Build GraspPose list sorted by score (descending)
        order = np.argsort(-scores)
        results: list[GraspPose] = []
        for idx in order:
            s = float(scores[idx])
            if cfg.score_threshold > 0 and s < cfg.score_threshold:
                continue
            results.append(GraspPose(transform=transforms[idx], score=s))
            if len(results) >= cfg.max_grasps:
                break

        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._sampler is not None:
            return

        from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg

        grasp_cfg = load_grasp_cfg(self._config.checkpoint_config_path)
        self._sampler = GraspGenSampler(grasp_cfg)
