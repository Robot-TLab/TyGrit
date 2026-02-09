"""GraspGen predictor and grasp-selection utilities."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R

from TyGrit.perception.grasping.config import GraspGenConfig
from TyGrit.types.grasp import GraspPose


class GraspGenPredictor:
    """Concrete GraspPredictor wrapping GraspGen's GraspGenSampler.

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


# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------


def select_diverse_grasps(
    grasps: list[GraspPose],
    num_select: int,
) -> list[GraspPose]:
    """Select diverse grasps via greedy farthest-point sampling on rotation angle.

    Parameters
    ----------
    grasps : list[GraspPose]
        Grasps sorted by score (best first).
    num_select : int
        Number of diverse grasps to return.

    Returns
    -------
    list[GraspPose]
        The selected subset, preserving relative order.
    """
    if len(grasps) <= num_select:
        return list(grasps)

    rotations = [R.from_matrix(g.transform[:3, :3]) for g in grasps]

    selected = [0]
    for _ in range(num_select - 1):
        best_idx = -1
        best_min_dist = -1.0

        for i in range(len(grasps)):
            if i in selected:
                continue
            min_dist = min(
                (rotations[i].inv() * rotations[j]).magnitude() for j in selected
            )
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = i

        if best_idx >= 0:
            selected.append(best_idx)

    selected.sort()
    return [grasps[i] for i in selected]


def filter_by_score(
    grasps: list[GraspPose],
    threshold: float,
) -> list[GraspPose]:
    """Return grasps whose score meets *threshold*."""
    return [g for g in grasps if g.score >= threshold]
