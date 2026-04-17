"""Pure data types for the mobile-grasping dataset.

Each :class:`MobileGraspDatapoint` bundles a scene, an object placed
inside that scene, and a robot base pose + init qpos from which the
object is expected to be reachable. A dataset is a tuple of these.

Consumers pick one entry, spawn the scene + object via any backend
adapter under :mod:`TyGrit.worlds.backends`, teleport the robot to
``base_pose`` + ``init_qpos`` via the sim handler, and run a grasping
policy. The dataset is sim-agnostic: the entries reference only the
shared :mod:`TyGrit.types.worlds` types, so the same file is loadable
under ManiSkill, Genesis, and Isaac Sim without modification.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from TyGrit.types.worlds import ObjectSpec, SceneSpec


@dataclass(frozen=True)
class MobileGraspDatapoint:
    """A single (scene, object, robot init pose) datapoint.

    Parameters
    ----------
    scene
        Background world :class:`SceneSpec`. For cross-backend use the
        ``source`` should be one accepted by every sim adapter the
        dataset targets (currently only ``"holodeck"`` is loadable by
        all three of ManiSkill, Genesis, Isaac Sim).
    object
        Target graspable :class:`ObjectSpec`. ``position`` /
        ``orientation_xyzw`` are the settled world-frame pose after
        in-sim placement (not a zero pose). Must carry ``mesh_path``
        so every sim can load it — ``builtin_id="ycb:..."`` is
        ManiSkill-only and breaks cross-backend compatibility.
    base_pose
        Robot base world-frame pose ``(x, y, theta)``. ``theta = 0``
        points along world +x (CCW positive). Matches the holonomic
        base joint convention used by :class:`FetchRobotCore`.
    init_qpos
        Initial joint positions keyed by joint name (e.g.
        ``torso_lift_joint``, ``head_pan_joint``, each of the 7 arm
        joints). Missing keys default to the sim's rest pose.
    grasp_hint
        Optional end-effector pose hint in **world frame**, stored as
        ``(x, y, z, qx, qy, qz, qw)``. When present, the generator
        has verified the pose is IK-reachable from ``base_pose`` +
        ``init_qpos``. ``None`` means consumers must run their own
        grasp predictor.
    """

    scene: SceneSpec
    object: ObjectSpec
    base_pose: tuple[float, float, float]
    init_qpos: Mapping[str, float] = field(default_factory=dict)
    grasp_hint: tuple[float, float, float, float, float, float, float] | None = None


@dataclass(frozen=True)
class MobileGraspDataset:
    """A collection of :class:`MobileGraspDatapoint` entries plus metadata.

    Parameters
    ----------
    entries
        All datapoints in file order.
    metadata
        Free-form string metadata (generator CLI args, asset versions,
        commit hash, etc.). Not load-bearing — consumers may inspect
        but should not depend on specific keys.
    """

    entries: tuple[MobileGraspDatapoint, ...]
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)
