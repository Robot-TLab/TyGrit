"""Observation encoding for RL training with RGB-D input.

Uses ManiSkill's native sim poses (TCP, camera, links) — no custom FK.

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
    robot,
    target_pos: Tensor,
    raw_obs: dict,
) -> dict[str, Tensor]:
    """Build dict observation from robot sensor data + state.

    All returned tensors live on the same device as the ManiSkill sim.
    Uses sim-native poses — no custom FK needed.

    Parameters
    ----------
    raw_obs : dict
        Raw ManiSkill observation dict returned by ``step()`` or ``reset()``.

    Returns:
        Dict with keys ``"rgb"``, ``"depth"``, ``"state"``.
    """
    sensor_head = raw_obs["sensor_data"]["fetch_head"]

    rgb = sensor_head["rgb"]  # (N, H, W, 3)
    depth = sensor_head["depth"]  # (N, H, W, 1)

    # Proprioceptive state — all from sim, on GPU
    env_unwrapped = robot._env.unwrapped
    qpos = env_unwrapped.agent.robot.get_qpos().float()
    qvel = env_unwrapped.agent.robot.get_qvel().float()

    n_joints = min(qpos.shape[-1], 15)
    qpos_cut = qpos[:, :n_joints]
    qvel_cut = qvel[:, :n_joints]

    # EE position from sim's TCP — already in world frame, on GPU
    ee_world = env_unwrapped.agent.tcp.pose.p.float()  # (N, 3)

    dev = qpos.device
    state = torch.cat(
        [
            qpos_cut,
            qvel_cut,
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
