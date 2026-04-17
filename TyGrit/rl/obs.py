"""Observation encoding for RL training with RGB-D input.

Reads from a :class:`~TyGrit.envs.fetch.core_vec.FetchRobotCoreVec`
(plus the underlying ManiSkill obs dict for raw qpos/qvel/rgb/depth).
No sim re-queries — everything comes from the cached obs the handler
updated on the last ``step`` / ``reset``.

The proprioceptive state vector is 36-dim:
    qpos(15) + qvel(15) + ee_pos(3) + target_pos(3)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from TyGrit.envs.fetch.core_vec import FetchRobotCoreVec

STATE_DIM = 36
"""Dimensionality of the proprioceptive state vector."""


def build_obs_dict(
    robot: "FetchRobotCoreVec",
    target_pos: Tensor,
) -> dict[str, Tensor]:
    """Build the dict observation consumed by :mod:`TyGrit.rl.policy`.

    Reads the raw ManiSkill obs dict off the underlying
    :class:`~TyGrit.sim.maniskill.ManiSkillSimHandlerVec` (for RGB, depth,
    qpos, qvel) plus the TCP pose via the same handler's cached
    ``env_agent.tcp`` — the one piece of state ManiSkill does not ship
    in its obs dict.

    Parameters
    ----------
    robot
        :class:`FetchRobotCoreVec` constructed by
        ``FetchRobot.create(config)`` with ``config.num_envs > 1``.
    target_pos
        ``(num_envs, 3)`` per-env target position in world frame.

    Returns
    -------
    dict with keys ``"rgb"``, ``"depth"``, ``"state"``.
    """
    handler = robot.handler
    raw_obs = handler._obs  # noqa: SLF001 — raw ManiSkill obs is the only
    #                        source of RGB + qpos/qvel; the SimHandlerVec
    #                        surface returns already-shaped tensors that
    #                        drop qvel (not part of the protocol).
    sensor_head = raw_obs["sensor_data"]["fetch_head"]

    rgb = sensor_head["rgb"]  # (N, H, W, 3) uint8
    depth = sensor_head["depth"]  # (N, H, W, 1) uint16 in mm

    qpos = raw_obs["agent"]["qpos"].float()
    qvel = raw_obs["agent"]["qvel"].float()

    # TCP pose — ManiSkill does not ship this in the obs dict, so we
    # read it off the agent directly. This is ManiSkill-specific but
    # acceptable inside ``rl/`` which is tied to ManiSkill today.
    tcp_pose = handler.env_agent.tcp.pose  # type: ignore[attr-defined]
    ee_pos = tcp_pose.p.float()  # (N, 3)

    n_joints = min(qpos.shape[-1], 15)
    dev = qpos.device
    state = torch.cat(
        [
            qpos[:, :n_joints],
            qvel[:, :n_joints],
            ee_pos,
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
