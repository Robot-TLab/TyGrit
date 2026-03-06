"""Observation encoding for RL training with RGB-D input.

Pure function — reads all data from the step/reset result dict,
no sim re-queries.

The proprioceptive state vector is 36-dim:
    qpos(15) + qvel(15) + ee_pos(3) + target_pos(3)
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

STATE_DIM = 36
"""Dimensionality of the proprioceptive state vector."""


def build_obs_dict(
    step_result: dict,
    target_pos: Tensor,
) -> dict[str, Tensor]:
    """Build dict observation from a step/reset result.

    All data comes from the result dict returned by
    ``ManiSkillFetchRobotVec.step()`` or ``.reset()`` — no sim access.

    Parameters
    ----------
    step_result : dict
        Result from ``step()`` or ``reset()``.  Must contain:
        - ``obs`` with ``sensor_data/fetch_head/{rgb,depth}`` and
          ``agent/{qpos,qvel}``
        - ``ee_pos`` ``(N, 3)`` TCP position
    target_pos : Tensor
        ``(N, 3)`` target positions.

    Returns
    -------
    dict with keys ``"rgb"``, ``"depth"``, ``"state"``.
    """
    raw_obs = step_result["obs"]
    sensor_head = raw_obs["sensor_data"]["fetch_head"]

    rgb = sensor_head["rgb"]  # (N, H, W, 3)
    depth = sensor_head["depth"]  # (N, H, W, 1)

    qpos = raw_obs["agent"]["qpos"].float()
    qvel = raw_obs["agent"]["qvel"].float()
    ee_world = step_result["ee_pos"]

    n_joints = min(qpos.shape[-1], 15)
    dev = qpos.device
    state = torch.cat(
        [
            qpos[:, :n_joints],
            qvel[:, :n_joints],
            ee_world,
            target_pos.to(dev).float(),
        ],
        dim=1,
    )

    return {"rgb": rgb, "depth": depth, "state": state}


class DictArray:
    """Rollout buffer for dict observations, allocated on the sample's device."""

    def __init__(
        self,
        buffer_shape: tuple[int, ...],
        sample_element: dict[str, Tensor],
    ) -> None:
        self._data: dict[str, Tensor] = {}
        for key, val in sample_element.items():
            shape = buffer_shape + val.shape[1:]
            self._data[key] = torch.zeros(shape, dtype=val.dtype, device=val.device)

    def __setitem__(
        self,
        index: int | slice | tuple,
        value: dict[str, Tensor] | Mapping[str, Tensor],
    ) -> None:
        for key in self._data:
            self._data[key][index] = value[key]

    def __getitem__(self, index: int | slice | tuple | Tensor) -> dict[str, Tensor]:
        return {key: val[index] for key, val in self._data.items()}

    def flatten(self, n_leading: int = 2) -> dict[str, Tensor]:
        """Flatten first *n_leading* dims into one."""
        return {
            key: val.reshape((-1,) + val.shape[n_leading:])
            for key, val in self._data.items()
        }

    @property
    def data(self) -> dict[str, Tensor]:
        return self._data
