"""Pure data types for the sensor layer.

A :class:`CameraSpec` describes a single camera mount on a robot
(or on a static fixture in the scene): which link it's attached to,
the offset/orientation from that link, its image resolution, and
which robot's public ``camera_id`` it satisfies.

Parallels :mod:`TyGrit.types.robots` and :mod:`TyGrit.types.worlds`:
pure data, no simulator imports, frozen. Per-sim backends in
:mod:`TyGrit.envs` consume a ``CameraSpec`` to instantiate the
native camera object (Sapien ``RenderCamera``, Genesis
``gs.Camera``, Isaac Sim ``USDRTCamera``, ROS ``Image`` subscriber).

Design notes
------------
* Offsets are stored as a ``(x, y, z)`` position + ``(x, y, z, w)``
  quaternion in the parent link's frame — the project-wide quaternion
  convention. Sim adapters convert at the backend boundary.
* Image dimensions are fixed per spec. Dynamic resolution changes
  happen by swapping specs, not mutating one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from TyGrit.types.robots import RobotState  # forward ref resolved at import time
from TyGrit.types.robots import RobotStateVec

if TYPE_CHECKING:
    import torch  # noqa: F401 — used in SensorSnapshotVec field annotations

_IDENTITY_QUAT_XYZW: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class CameraSpec:
    """Description of a single RGB-D camera mount.

    Parameters
    ----------
    camera_id
        Public identifier used by the robot's camera dict (e.g.
        ``"head"``, ``"wrist"``). Must be unique within a robot's
        ``RobotCfg.cameras`` tuple.
    parent_link
        Name of the robot link (or scene entity) the camera is
        attached to. For Fetch's head camera this is
        ``"head_camera_link"``. For a static fixture it's the name
        of the object in the :class:`SceneSpec`.
    position
        ``(x, y, z)`` offset from the parent link's origin, in metres.
    orientation_xyzw
        ``(x, y, z, w)`` rotation relative to the parent link.
    width
        Image width in pixels.
    height
        Image height in pixels.
    fovy_degrees
        Vertical field of view, in degrees. Horizontal FOV is derived
        from aspect ratio.
    near, far
        Near and far clip planes, in metres. Depth values outside
        ``[near, far]`` are clamped by the simulator.
    sim_sensor_ids
        Per-simulator mapping from TyGrit ``camera_id`` to the name
        the sim uses internally. Populated only for sims that ship a
        robot agent with pre-registered sensors under a non-matching
        name — e.g. ManiSkill's Fetch agent registers the head camera
        as ``"fetch_head"`` while TyGrit exposes ``"head"``. Sims that
        spawn cameras procedurally from the :class:`CameraSpec`
        itself (Genesis, Isaac Sim) don't need an entry.
    """

    camera_id: str
    parent_link: str
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation_xyzw: tuple[float, float, float, float] = _IDENTITY_QUAT_XYZW
    width: int = 640
    height: int = 480
    fovy_degrees: float = 45.0
    near: float = 0.01
    far: float = 10.0
    sim_sensor_ids: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"CameraSpec {self.camera_id!r}: width/height must be positive, "
                f"got {self.width}x{self.height}"
            )
        if self.near <= 0.0 or self.far <= self.near:
            raise ValueError(
                f"CameraSpec {self.camera_id!r}: require 0 < near < far, "
                f"got near={self.near}, far={self.far}"
            )
        if not 0.0 < self.fovy_degrees < 180.0:
            raise ValueError(
                f"CameraSpec {self.camera_id!r}: fovy_degrees must be in "
                f"(0, 180), got {self.fovy_degrees}"
            )
        object.__setattr__(
            self, "sim_sensor_ids", MappingProxyType(dict(self.sim_sensor_ids))
        )


# ────────────────────────────────────────────────────────────────
# Runtime sensor-data types (formerly types/sensor.py — merged here
# on 2026-04-15 to retire the parallel singular/plural hierarchy
# CLAUDE.md Rule 4 forbids).
# ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SensorSnapshot:
    """A single synchronised capture from the robot's sensors."""

    rgb: npt.NDArray[np.uint8]  # (H, W, 3)
    depth: npt.NDArray[np.float32]  # (H, W) in metres
    intrinsics: npt.NDArray[np.float64]  # (3, 3)
    robot_state: RobotState
    segmentation: npt.NDArray[np.int32] | None = None


@dataclass(frozen=True)
class SensorSnapshotVec:
    """Batched :class:`SensorSnapshot` — every per-env field is a torch
    tensor with leading axis ``num_envs``.

    Lives next to :class:`SensorSnapshot` (CLAUDE.md Rule 4: no
    parallel ``types/`` tree). Consumed by
    :class:`~TyGrit.envs.fetch.core_vec.FetchRobotCoreVec` and any vec
    controller / RL trainer.

    Shape convention:

    * ``rgb``: ``(num_envs, H, W, 3)`` uint8.
    * ``depth``: ``(num_envs, H, W)`` float32 in metres.
    * ``intrinsics``: ``(3, 3)`` float64 numpy — shared across envs
      (intrinsics don't vary per env in any handler today).
    * ``segmentation``: ``(num_envs, H, W)`` int32 or ``None``.
    * ``robot_state``: :class:`RobotStateVec` — already batched.

    Tensors live on the device the producing handler was constructed
    with. Callers must not cross the CPU↔GPU boundary implicitly —
    use ``.cpu()`` explicitly at observation-logging time.
    """

    rgb: "torch.Tensor"  # (N, H, W, 3) uint8
    depth: "torch.Tensor"  # (N, H, W) float32 metres
    intrinsics: npt.NDArray[np.float64]  # (3, 3) shared across envs
    robot_state: RobotStateVec
    segmentation: "torch.Tensor | None" = None

    @property
    def num_envs(self) -> int:
        return int(self.rgb.shape[0])


__all__ = ["CameraSpec", "SensorSnapshot", "SensorSnapshotVec"]
