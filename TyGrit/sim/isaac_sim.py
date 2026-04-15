"""Isaac Sim / Isaac Lab handler — robot-agnostic :class:`SimHandler`.

Target APIs (as of Isaac Sim 5.0 + Isaac Lab 2.3.x):

* ``isaaclab.sim.SimulationContext`` — headless simulation loop.
* ``isaaclab.scene.InteractiveScene`` — the container holding the
  robot articulation + scene assets.
* ``isaaclab.assets.ArticulationCfg`` — declarative robot
  instantiation from a USD stage (or URDF via
  ``UrdfConverterCfg``).
* ``isaaclab.sensors.CameraCfg`` — camera attached to a link via a
  prim path.

Status
------

**SKELETON.** Every real Isaac Lab API call is marked with a
``TODO(isaac_sim)`` comment. The handler validates its
:class:`RobotCfg` + scene pool the same way as the ManiSkill /
Genesis handlers so constructor-time errors are caught uniformly.
When the Isaac Lab pixi env actually solves (see
``pixi.toml`` note on NVIDIA's wheel metadata bugs), replace the
``TODO`` bodies with real calls against the Isaac Lab SDK.

Import strategy: Isaac Lab's SDK is massive and not importable
outside its pixi env. All SDK imports happen lazily inside methods
guarded by the per-method ``TODO`` docstring. The class definition
itself is importable from the default pixi env so the Protocol
conformance check runs in tests.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType

import numpy as np
import numpy.typing as npt

from TyGrit.types.robots import RobotCfg
from TyGrit.types.worlds import SceneSpec


class IsaacSimSimHandler:
    """Robot-agnostic :class:`~TyGrit.sim.base.SimHandler` for Isaac Sim.

    Parameters
    ----------
    robot_cfg
        Robot descriptor. ``robot_cfg.usd_path`` is preferred (Isaac
        Sim native); ``robot_cfg.urdf_path`` is the fallback via
        :class:`UrdfConverterCfg`.
    scenes
        Scene pool drawn from at reset.
    initial_scene_idx
        Index into ``scenes`` loaded at construction.
    device
        Torch device for Isaac Lab tensors. ``"cuda:0"`` is the
        common case; CPU is legal for headless testing.
    """

    def __init__(
        self,
        robot_cfg: RobotCfg,
        scenes: Sequence[SceneSpec],
        *,
        initial_scene_idx: int = 0,
        device: str = "cuda:0",
    ) -> None:
        if robot_cfg.usd_path is None and robot_cfg.urdf_path is None:
            raise ValueError(
                f"IsaacSimSimHandler: RobotCfg {robot_cfg.name!r} has neither "
                f"usd_path nor urdf_path. Isaac Sim needs at least one."
            )
        self._robot_cfg = robot_cfg
        self._scenes: tuple[SceneSpec, ...] = tuple(scenes)
        if len(self._scenes) == 0:
            raise ValueError(
                "IsaacSimSimHandler: scene pool is empty; pass at least one SceneSpec"
            )
        if not 0 <= initial_scene_idx < len(self._scenes):
            raise IndexError(
                f"IsaacSimSimHandler: initial_scene_idx {initial_scene_idx} "
                f"out of range for scene pool of size {len(self._scenes)}"
            )
        self._device = device

        # Action layout derived from RobotCfg, same contract as the
        # other handlers.
        self._action_slices: Mapping[str, slice] = robot_cfg.action_slices_from_order()
        self._total_action_dim = robot_cfg.total_action_dim()

        # Placeholders populated by _build_scene.
        self._sim_context = None  # SimulationContext
        self._scene = None  # InteractiveScene
        self._robot = None  # Articulation
        self._cameras: dict[str, object] = {}
        self._joint_name_to_idx: Mapping[str, int] = MappingProxyType({})
        self._active_scene_idx: int = -1

        self._build_scene(initial_scene_idx)

    # ── scene build / teardown ────────────────────────────────────────

    def _build_scene(self, idx: int) -> None:
        """(Re)instantiate the Isaac Lab scene for the ``idx``-th spec.

        Isaac Lab's :class:`InteractiveScene` plus
        :class:`SimulationContext` share a global Omniverse stage —
        scene switching typically goes through ``stage_utils.close``
        + ``SimulationContext.initialize``. Exact surgery TBD when we
        actually wire this up.
        """
        # TODO(isaac_sim): use isaaclab.sim.SimulationCfg /
        # SimulationContext + isaaclab.scene.InteractiveSceneCfg to
        # (re)create the stage. Populate the scene via
        # TyGrit.worlds.backends.isaac_sim (to be written — mirrors
        # worlds.backends.genesis for per-source dispatch into Isaac
        # Sim's asset loaders).
        raise NotImplementedError(
            "IsaacSimSimHandler._build_scene: Isaac Lab integration is a skeleton "
            "until the pixi isaacsim env solves (NVIDIA wheel metadata bugs). "
            "See TyGrit.sim.isaac_sim module docstring for the target API."
        )

    # ── SimHandler: metadata ───────────────────────────────────────────

    @property
    def robot_cfg(self) -> RobotCfg:
        return self._robot_cfg

    @property
    def num_envs(self) -> int:
        return 1

    @property
    def total_action_dim(self) -> int:
        return self._total_action_dim

    @property
    def action_slices(self) -> Mapping[str, slice]:
        return self._action_slices

    @property
    def joint_name_to_idx(self) -> Mapping[str, int]:
        return self._joint_name_to_idx

    # ── SimHandler: per-step queries ───────────────────────────────────

    def get_qpos(self, env_idx: int = 0) -> npt.NDArray[np.float64]:
        # TODO(isaac_sim): articulation.data.joint_pos is a torch
        # tensor of shape (num_envs, num_dof). Select [env_idx] and
        # .cpu().numpy() it. Keep ordering aligned with joint_name_to_idx.
        raise NotImplementedError("IsaacSimSimHandler.get_qpos — skeleton")

    def get_link_pose(
        self, link_name: str, env_idx: int = 0
    ) -> npt.NDArray[np.float64]:
        # TODO(isaac_sim): articulation.find_bodies([link_name]) ->
        # body_idx; articulation.data.body_pos_w / body_quat_w give
        # world pose. Assemble the 4×4 matrix same as other handlers.
        raise NotImplementedError("IsaacSimSimHandler.get_link_pose — skeleton")

    def get_camera(self, camera_id: str, env_idx: int = 0) -> tuple[
        npt.NDArray[np.uint8],
        npt.NDArray[np.float32],
        npt.NDArray[np.int32] | None,
    ]:
        # TODO(isaac_sim): Camera sensor outputs are in
        # camera.data.output["rgb"] / ["depth"] / ["semantic_segmentation"].
        # Shape (N, H, W, *); index [env_idx] and cast to project dtypes.
        raise NotImplementedError("IsaacSimSimHandler.get_camera — skeleton")

    def get_intrinsics(self, camera_id: str) -> npt.NDArray[np.float64]:
        # TODO(isaac_sim): camera.data.intrinsic_matrices has the
        # per-env 3×3. Use [env_idx=0] for single-env.
        raise NotImplementedError("IsaacSimSimHandler.get_intrinsics — skeleton")

    # ── SimHandler: mutations ──────────────────────────────────────────

    def apply_action(self, action: npt.NDArray[np.float32]) -> None:
        # TODO(isaac_sim): Set joint targets via
        # articulation.set_joint_velocity_target / _position_target
        # per actuator's control_mode, then self._sim_context.step().
        raise NotImplementedError("IsaacSimSimHandler.apply_action — skeleton")

    def reset_to_scene_idx(self, idx: int, *, seed: int | None = None) -> None:
        # TODO(isaac_sim): Isaac Lab resets via scene.reset() + optional
        # stage rebuild when the scene pool has distinct backgrounds.
        # For fixed scenes (single background + randomised object pose),
        # a scene.reset() is enough.
        raise NotImplementedError("IsaacSimSimHandler.reset_to_scene_idx — skeleton")

    def set_joint_positions(
        self,
        positions: Mapping[str, float],
        *,
        env_idx: int = 0,
    ) -> None:
        # TODO(isaac_sim): articulation.write_joint_state_to_sim(
        # position=q, velocity=0, joint_ids=…, env_ids=[env_idx]).
        raise NotImplementedError("IsaacSimSimHandler.set_joint_positions — skeleton")

    def set_base_pose(
        self,
        x: float,
        y: float,
        theta: float,
        *,
        env_idx: int = 0,
    ) -> None:
        if not self._robot_cfg.is_mobile:
            raise RuntimeError(
                f"IsaacSimSimHandler.set_base_pose: robot "
                f"{self._robot_cfg.name!r} is fixed-base; set_base_pose is not legal"
            )
        # TODO(isaac_sim): For a holonomic base, either teleport the
        # base joints via set_joint_positions(base_joint_names) or
        # write_root_state_to_sim(root_state) if the URDF models the
        # base as a floating joint. Fetch uses three prismatic/revolute
        # joints, so the joint-state path applies.
        raise NotImplementedError("IsaacSimSimHandler.set_base_pose — skeleton")

    # ── SimHandler: world hooks ────────────────────────────────────────

    def get_navigable_positions(self) -> list:
        # TODO(isaac_sim): Habitat .navigable_positions.obj loading
        # lives in TyGrit.worlds.backends.isaac_sim alongside scene
        # loading. Plumb through.
        return []

    # ── SimHandler: lifecycle ──────────────────────────────────────────

    def render(self) -> None:
        # TODO(isaac_sim): sim_context.render() when a viewer is attached.
        pass

    def close(self) -> None:
        # TODO(isaac_sim): sim_context.close() + stage_utils.close_stage().
        self._sim_context = None
        self._scene = None
        self._robot = None
        self._cameras = {}


__all__ = ["IsaacSimSimHandler"]
