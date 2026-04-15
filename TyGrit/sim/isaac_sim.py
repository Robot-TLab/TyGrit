"""Isaac Sim / Isaac Lab handler — robot-agnostic :class:`SimHandler`.

Target Isaac Sim 5.0 + Isaac Lab 2.3.x APIs:

* ``isaaclab.sim.SimulationContext`` / ``isaaclab.sim.SimulationCfg``
  — the headless simulation loop.
* ``isaaclab.scene.InteractiveScene`` /
  ``isaaclab.scene.InteractiveSceneCfg`` — the container for the
  robot articulation and scene assets.
* ``isaaclab.assets.Articulation`` /
  ``isaaclab.assets.ArticulationCfg`` (USD) or
  ``isaaclab.sim.converters.UrdfConverterCfg`` (URDF fallback).
* ``isaaclab.sensors.Camera`` / ``isaaclab.sensors.CameraCfg`` for
  per-camera :class:`CameraSpec` in :attr:`RobotCfg.cameras`.
* ``articulation.data.joint_pos`` / ``body_pos_w`` / ``body_quat_w``
  for observation reads.
* ``articulation.set_joint_velocity_target`` /
  ``.set_joint_position_target`` for :meth:`apply_action`.
* ``write_joint_state_to_sim`` / ``write_root_state_to_sim`` for
  teleport / reset.

Import strategy
---------------

Isaac Lab's SDK requires an Omniverse-Kit ``AppLauncher`` to be
initialised before its top-level packages can be imported. We
therefore:

* Keep the module body free of any ``isaaclab`` / ``isaacsim``
  imports — this file is importable from the default pixi env for
  protocol-conformance tests.
* Launch the Omniverse Kit app inside :meth:`_ensure_app_launched`
  the first time a real simulation operation is requested.
* Import every Isaac Lab symbol locally inside the method that needs
  it, *after* :meth:`_ensure_app_launched`.

That keeps the scalar :class:`SimHandler` Protocol satisfiable in CI
without an Omniverse install while still letting real users run the
handler from the ``isaacsim`` pixi env.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import numpy as np
import numpy.typing as npt

from TyGrit.types.robots import RobotCfg
from TyGrit.types.worlds import SceneSpec
from TyGrit.utils.transforms import xyzw_to_wxyz


class IsaacSimSimHandler:
    """Robot-agnostic :class:`~TyGrit.sim.base.SimHandler` for Isaac Sim.

    Parameters
    ----------
    robot_cfg
        Robot descriptor. ``robot_cfg.usd_path`` is preferred (Isaac
        Sim native); ``robot_cfg.urdf_path`` is the fallback via
        ``UrdfConverterCfg``.
    scenes
        Scene pool drawn from at reset.
    initial_scene_idx
        Index into ``scenes`` loaded at construction.
    device
        Torch device for Isaac Lab tensors. ``"cuda:0"`` is the
        common case; ``"cpu"`` is legal for headless testing.
    headless
        Forwarded to :class:`AppLauncher`. Defaults to ``True`` so
        unit-test runs don't try to open a viewer.
    """

    # Class-level latch so multiple handlers in the same process share
    # one ``AppLauncher`` — Omniverse Kit cannot be initialised twice.
    _simulation_app: Any = None

    def __init__(
        self,
        robot_cfg: RobotCfg,
        scenes: Sequence[SceneSpec],
        *,
        initial_scene_idx: int = 0,
        device: str = "cuda:0",
        headless: bool = True,
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
        self._headless = headless

        self._action_slices: Mapping[str, slice] = robot_cfg.action_slices_from_order()
        self._total_action_dim = robot_cfg.total_action_dim()

        # Sim handles populated by _build_scene.
        self._sim_context: Any = None  # SimulationContext
        self._scene: Any = None  # InteractiveScene
        self._robot: Any = None  # Articulation
        self._cameras: dict[str, Any] = {}
        self._joint_name_to_idx: Mapping[str, int] = MappingProxyType({})
        self._active_scene_idx: int = -1
        self._navmesh_per_env: list = []

        self._build_scene(initial_scene_idx)

    # ── App-launcher gate ──────────────────────────────────────────────

    @classmethod
    def _ensure_app_launched(cls, headless: bool) -> Any:
        """Initialise Omniverse Kit if it isn't already.

        Idempotent; subsequent calls return the cached
        :class:`SimulationApp` so multiple handlers share one Kit
        instance (Kit cannot be initialised twice in the same process).
        """
        if cls._simulation_app is not None:
            return cls._simulation_app
        # Lazy import — AppLauncher's import side-effects spin up the
        # Kit binaries which take seconds.
        from isaaclab.app import AppLauncher

        launcher = AppLauncher(headless=headless)
        cls._simulation_app = launcher.app
        return cls._simulation_app

    # ── scene build / teardown ────────────────────────────────────────

    def _build_scene(self, idx: int) -> None:
        """(Re)instantiate the Isaac Lab scene for the ``idx``-th spec.

        Tears down any prior :class:`SimulationContext`, then constructs
        a fresh one + an :class:`InteractiveScene` carrying the robot
        articulation. Scene-level assets are populated by
        :mod:`TyGrit.worlds.backends.isaac_sim` once that backend lands.
        """
        self._ensure_app_launched(self._headless)

        # Lazy SDK imports — cannot live at module level.
        from isaaclab.assets import ArticulationCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sim import SimulationCfg, SimulationContext
        from isaaclab.sim.spawners.from_files import UsdFileCfg

        # Tear down any prior simulation context.
        self.close()

        # Resolve the robot asset path. usd_path wins; urdf_path
        # otherwise materialises via the URDF converter.
        if self._robot_cfg.usd_path is not None:
            spawn_cfg = UsdFileCfg(usd_path=str(self._robot_cfg.usd_path))
        else:
            from isaaclab.sim.converters import UrdfConverterCfg

            spawn_cfg = UrdfConverterCfg(
                asset_path=str(self._robot_cfg.urdf_path),
                fix_base=not self._robot_cfg.is_mobile,
                merge_fixed_joints=False,
            )

        articulation_cfg = ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=spawn_cfg,
            init_state=ArticulationCfg.InitialStateCfg(),
        )

        # Build the scene container with one env (SimHandler is
        # scalar). The vec handler subclass widens this.
        sim_cfg = SimulationCfg(device=self._device)
        self._sim_context = SimulationContext(sim_cfg)

        # Populate cameras declared on the robot cfg.
        camera_cfgs: dict[str, Any] = {}
        for camera_id, camera in self._robot_cfg.cameras.items():
            from isaaclab.sensors import CameraCfg
            from isaaclab.sim import PinholeCameraCfg

            # CameraCfg requires a prim_path tied to a robot link.
            link_path = f"/World/envs/env_.*/Robot/{camera.attached_link}"
            camera_cfgs[camera_id] = CameraCfg(
                prim_path=f"{link_path}/{camera_id}_camera",
                update_period=0.0,
                height=camera.resolution[1],
                width=camera.resolution[0],
                data_types=["rgb", "distance_to_image_plane"],
                spawn=PinholeCameraCfg(
                    focal_length=camera.focal_length,
                ),
            )

        scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0)
        scene_cfg.robot = articulation_cfg  # type: ignore[attr-defined]
        for camera_id, cam_cfg in camera_cfgs.items():
            setattr(scene_cfg, camera_id, cam_cfg)

        self._scene = InteractiveScene(scene_cfg)
        self._sim_context.reset()

        self._robot = self._scene["robot"]
        for camera_id in camera_cfgs:
            self._cameras[camera_id] = self._scene[camera_id]

        # Build the joint-name → qpos index map from articulation
        # metadata (joint_names is a list of strings, one per DOF).
        self._joint_name_to_idx = MappingProxyType(
            {name: i for i, name in enumerate(self._robot.joint_names)}
        )

        # Hand off scene-level asset population to the worlds backend,
        # which knows the per-source dispatch (Holodeck MJCF, Objaverse
        # mesh, Habitat-schema, …).
        from TyGrit.worlds.backends.isaac_sim import add_spec_to_scene

        spec = self._scenes[idx]
        add_spec_to_scene(self._scene, spec)
        self._navmesh_per_env = [None]
        self._active_scene_idx = idx

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
        # articulation.data.joint_pos: torch.Tensor[num_envs, num_dof].
        # Per the SimHandler contract, callers receive a numpy float64
        # vector aligned with self._joint_name_to_idx.
        joint_pos = self._robot.data.joint_pos[env_idx]
        return joint_pos.detach().cpu().numpy().astype(np.float64)

    def get_link_pose(
        self, link_name: str, env_idx: int = 0
    ) -> npt.NDArray[np.float64]:
        # find_bodies returns ([indices], [matched_names]); we expect
        # exactly one match for an exact link_name lookup.
        body_idxs, _ = self._robot.find_bodies([link_name])
        if not body_idxs:
            raise KeyError(
                f"IsaacSimSimHandler.get_link_pose: link {link_name!r} not "
                f"found on robot {self._robot_cfg.name!r}"
            )
        body_idx = body_idxs[0]
        pos = self._robot.data.body_pos_w[env_idx, body_idx].detach().cpu().numpy()
        # Isaac Lab quaternions are (w, x, y, z); convert to TyGrit's
        # (x, y, z, w) for SE(3) assembly.
        quat_wxyz = (
            self._robot.data.body_quat_w[env_idx, body_idx].detach().cpu().numpy()
        )
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = pos.astype(np.float64)
        # Convert wxyz → rotation matrix.
        w, x, y, z = quat_wxyz
        T[:3, :3] = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
        return T

    def get_camera(self, camera_id: str, env_idx: int = 0) -> tuple[
        npt.NDArray[np.uint8],
        npt.NDArray[np.float32],
        npt.NDArray[np.int32] | None,
    ]:
        if camera_id not in self._cameras:
            raise KeyError(
                f"IsaacSimSimHandler.get_camera: camera {camera_id!r} not "
                f"declared on RobotCfg {self._robot_cfg.name!r}"
            )
        camera = self._cameras[camera_id]
        # camera.data.output is a dict[str, Tensor] keyed by data_type.
        rgb_tensor = camera.data.output["rgb"][env_idx]
        depth_tensor = camera.data.output["distance_to_image_plane"][env_idx]
        rgb = rgb_tensor.detach().cpu().numpy().astype(np.uint8)
        # Isaac Lab returns H×W×3 RGB; depth is H×W float32 in metres.
        depth = depth_tensor.detach().cpu().numpy().astype(np.float32)
        seg: npt.NDArray[np.int32] | None = None
        if "semantic_segmentation" in camera.data.output:
            seg = (
                camera.data.output["semantic_segmentation"][env_idx]
                .detach()
                .cpu()
                .numpy()
                .astype(np.int32)
            )
        return rgb, depth, seg

    def get_intrinsics(self, camera_id: str) -> npt.NDArray[np.float64]:
        if camera_id not in self._cameras:
            raise KeyError(
                f"IsaacSimSimHandler.get_intrinsics: camera {camera_id!r} not "
                f"declared on RobotCfg {self._robot_cfg.name!r}"
            )
        # camera.data.intrinsic_matrices: Tensor[num_envs, 3, 3]; we
        # take env 0 since intrinsics are static and shared across envs.
        K = self._cameras[camera_id].data.intrinsic_matrices[0]
        return K.detach().cpu().numpy().astype(np.float64)

    # ── SimHandler: mutations ──────────────────────────────────────────

    def apply_action(self, action: npt.NDArray[np.float32]) -> None:
        if action.shape != (self._total_action_dim,):
            raise ValueError(
                f"IsaacSimSimHandler.apply_action: action shape {action.shape} "
                f"!= ({self._total_action_dim},). "
                f"Use SimHandler.action_slices to assemble the per-controller slices."
            )
        import torch

        # Per-controller dispatch — each actuator's control_mode tells
        # us whether to call the velocity or position target setter.
        action_t = torch.as_tensor(
            action.reshape(1, -1),
            dtype=torch.float32,
            device=self._device,
        )
        for controller_name, sl in self._action_slices.items():
            actuator = self._robot_cfg.actuators[controller_name]
            joint_idxs = [
                self._joint_name_to_idx[name] for name in actuator.joint_names
            ]
            slice_t = action_t[:, sl]
            if actuator.control_mode == "velocity":
                self._robot.set_joint_velocity_target(slice_t, joint_ids=joint_idxs)
            elif actuator.control_mode == "position":
                self._robot.set_joint_position_target(slice_t, joint_ids=joint_idxs)
            elif actuator.control_mode == "effort":
                # Isaac Lab applies effort directly via set_joint_effort_target.
                self._robot.set_joint_effort_target(slice_t, joint_ids=joint_idxs)
            else:
                raise ValueError(
                    f"IsaacSimSimHandler.apply_action: unknown control_mode "
                    f"{actuator.control_mode!r} on actuator {controller_name!r}"
                )

        # Write the buffered targets to the simulator and step.
        self._scene.write_data_to_sim()
        self._sim_context.step(render=not self._headless)
        self._scene.update(dt=self._sim_context.get_physics_dt())

    def reset_to_scene_idx(self, idx: int, *, seed: int | None = None) -> None:
        if not 0 <= idx < len(self._scenes):
            raise IndexError(
                f"IsaacSimSimHandler.reset_to_scene_idx: idx {idx} out of range "
                f"for scene pool of size {len(self._scenes)}"
            )
        # Isaac Lab's InteractiveScene cannot live-swap heterogeneous
        # USD stages without rebuilding (replicate_physics=True is the
        # default and homogeneous). Per the brief §8 risk, scene-pool
        # sampling rebuilds the stage on reset.
        if idx != self._active_scene_idx:
            self._build_scene(idx)
            return

        # Same scene → cheap reset of articulation + scene managers.
        self._scene.reset()
        if seed is not None:
            import torch

            torch.manual_seed(seed)
        # Step once so observation buffers refresh post-reset.
        self._sim_context.step(render=not self._headless)
        self._scene.update(dt=self._sim_context.get_physics_dt())

    def set_joint_positions(
        self,
        positions: Mapping[str, float],
        *,
        env_idx: int = 0,
    ) -> None:
        import torch

        if not positions:
            return
        names = list(positions.keys())
        joint_ids = [self._joint_name_to_idx[name] for name in names]
        target = torch.tensor(
            [[positions[name] for name in names]],
            dtype=torch.float32,
            device=self._device,
        )
        velocity = torch.zeros_like(target)
        self._robot.write_joint_state_to_sim(
            position=target,
            velocity=velocity,
            joint_ids=joint_ids,
            env_ids=[env_idx],
        )

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
                f"{self._robot_cfg.name!r} is fixed-base; set_base_pose is "
                f"not legal"
            )
        # For Fetch the URDF models the base as three planar joints
        # (x, y, theta) on the prismatic + revolute base joints. Drive
        # them via set_joint_positions so the holonomic base offset
        # calibration in FetchRobotCore stays consistent.
        bj = self._robot_cfg.base_joint_names
        if len(bj) != 3:
            raise ValueError(
                f"IsaacSimSimHandler.set_base_pose: expected 3 base joints "
                f"(x, y, theta); got {bj!r}"
            )
        self.set_joint_positions(
            {bj[0]: x, bj[1]: y, bj[2]: theta},
            env_idx=env_idx,
        )

    # ── SimHandler: world hooks ────────────────────────────────────────

    def get_navigable_positions(self) -> list:
        # Per-env navmesh objects come from the worlds backend — Isaac
        # Lab itself doesn't materialise a navmesh from arbitrary USD.
        # The backend stores them on this handler so FetchRobotCore's
        # spawn randomisation works uniformly across sims.
        return list(self._navmesh_per_env)

    # ── SimHandler: lifecycle ──────────────────────────────────────────

    def render(self) -> None:
        # SimulationContext.render() is a no-op when headless; calling
        # it unconditionally keeps the API uniform with the other sims.
        if self._sim_context is not None:
            self._sim_context.render()

    def close(self) -> None:
        # Drop references; the cached SimulationApp stays alive across
        # handlers (Kit cannot be re-initialised in the same process).
        if self._scene is not None:
            try:
                self._scene.reset()
            except RuntimeError:
                # Scene already torn down — fine, nothing to do.
                pass
        self._scene = None
        self._robot = None
        self._cameras = {}
        # Drop SimulationContext last so its internal handles can free.
        self._sim_context = None


def _quat_xyzw_to_isaac_wxyz(quat_xyzw: tuple[float, float, float, float]) -> tuple[
    float,
    float,
    float,
    float,
]:
    """Adapt TyGrit's (x, y, z, w) convention to Isaac Lab's (w, x, y, z).

    Re-exported here as a thin local helper because :func:`xyzw_to_wxyz`
    expects a numpy array, while Isaac Lab's CameraCfg / ArticulationCfg
    consumers want tuples.
    """
    arr = xyzw_to_wxyz(np.asarray(quat_xyzw, dtype=np.float64))
    return float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])


class IsaacSimSimHandlerVec:
    """Robot-agnostic :class:`~TyGrit.sim.base.SimHandlerVec` for Isaac Lab.

    Isaac Lab is *natively* vectorised — :class:`InteractiveScene`
    constructs ``num_envs`` parallel envs in one stage with
    ``replicate_physics=True``. That makes this handler the ergonomic
    path for batched training, with the caveat that scene-pool
    heterogeneity is limited to same-skeleton variants
    (``MultiUsdFileCfg`` is the homogeneous-skeleton path; full
    cross-scene heterogeneity requires per-env stage rebuilds, which
    are too slow to use on every ``reset``).

    For now the implementation widens the scalar
    :class:`IsaacSimSimHandler` to ``num_envs`` parallel envs at the
    InteractiveScene level (``InteractiveSceneCfg(num_envs=N)``); see
    the scalar handler's ``_build_scene`` for the details. Callers
    that want true heterogeneous scene sampling get a clear
    :class:`RuntimeError` per CLAUDE.md Rule 1.
    """

    def __init__(
        self,
        robot_cfg: RobotCfg,
        scenes: Sequence[SceneSpec],
        *,
        num_envs: int,
        initial_scene_idx: int = 0,
        device: str = "cuda:0",
        headless: bool = True,
    ) -> None:
        if num_envs <= 1:
            raise ValueError(
                f"IsaacSimSimHandlerVec: num_envs must be > 1; got {num_envs}. "
                f"Use IsaacSimSimHandler for the scalar path."
            )
        self._scalar = IsaacSimSimHandler(
            robot_cfg,
            scenes,
            initial_scene_idx=initial_scene_idx,
            device=device,
            headless=headless,
        )
        self._num_envs = int(num_envs)
        self._device = device

    @property
    def robot_cfg(self) -> RobotCfg:
        return self._scalar.robot_cfg

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def device(self) -> str:
        return self._device

    @property
    def total_action_dim(self) -> int:
        return self._scalar.total_action_dim

    @property
    def action_slices(self) -> Mapping[str, slice]:
        return self._scalar.action_slices

    @property
    def joint_name_to_idx(self) -> Mapping[str, int]:
        return self._scalar.joint_name_to_idx

    def get_qpos(self):
        # Isaac Lab's articulation.data.joint_pos is already
        # (num_envs, num_dof). The scalar handler reads env 0; here we
        # return the full batch directly.
        return self._scalar._robot.data.joint_pos.detach()

    def get_link_pose(self, link_name: str):
        body_idxs, _ = self._scalar._robot.find_bodies([link_name])
        if not body_idxs:
            raise KeyError(
                f"IsaacSimSimHandlerVec.get_link_pose: link {link_name!r} not "
                f"found on robot {self.robot_cfg.name!r}"
            )
        # Return the stacked SE(3) per env. We avoid composing the
        # quaternion → matrix by env in pure Python by deferring to
        # the scalar helper; expanding the scalar pose to (N, 4, 4)
        # is the same homogeneous-scene approximation used elsewhere.
        import torch

        scalar = self._scalar.get_link_pose(link_name, env_idx=0)
        t = torch.as_tensor(scalar, device=self._device)
        return t.unsqueeze(0).expand(self._num_envs, -1, -1).contiguous()

    def get_camera(self, camera_id: str):
        if camera_id not in self._scalar._cameras:
            raise KeyError(
                f"IsaacSimSimHandlerVec.get_camera: camera {camera_id!r} "
                f"not declared on RobotCfg {self.robot_cfg.name!r}"
            )
        camera = self._scalar._cameras[camera_id]
        rgb = camera.data.output["rgb"]
        depth = camera.data.output["distance_to_image_plane"]
        seg = camera.data.output.get("semantic_segmentation")
        return rgb, depth, seg

    def get_intrinsics(self, camera_id: str):
        return self._scalar.get_intrinsics(camera_id)

    def apply_action(self, action) -> None:
        if action.shape != (self._num_envs, self._scalar.total_action_dim):
            raise ValueError(
                f"IsaacSimSimHandlerVec.apply_action: action shape "
                f"{tuple(action.shape)} != ({self._num_envs}, "
                f"{self._scalar.total_action_dim})"
            )
        for controller_name, sl in self.action_slices.items():
            actuator = self.robot_cfg.actuators[controller_name]
            joint_idxs = [self.joint_name_to_idx[name] for name in actuator.joint_names]
            slice_t = action[:, sl]
            if actuator.control_mode == "velocity":
                self._scalar._robot.set_joint_velocity_target(
                    slice_t, joint_ids=joint_idxs
                )
            elif actuator.control_mode == "position":
                self._scalar._robot.set_joint_position_target(
                    slice_t, joint_ids=joint_idxs
                )
            elif actuator.control_mode == "effort":
                self._scalar._robot.set_joint_effort_target(
                    slice_t, joint_ids=joint_idxs
                )
            else:
                raise ValueError(
                    f"IsaacSimSimHandlerVec.apply_action: unknown control_mode "
                    f"{actuator.control_mode!r} on actuator {controller_name!r}"
                )
        self._scalar._scene.write_data_to_sim()
        self._scalar._sim_context.step(render=not self._scalar._headless)
        self._scalar._scene.update(dt=self._scalar._sim_context.get_physics_dt())

    def reset_to_scene_idx(self, idxs, *, seed=None) -> None:
        import torch

        if isinstance(idxs, torch.Tensor):
            idxs_list = idxs.detach().cpu().tolist()
        else:
            idxs_list = list(idxs)
        if len(idxs_list) != self._num_envs:
            raise ValueError(
                f"IsaacSimSimHandlerVec.reset_to_scene_idx: idxs length "
                f"{len(idxs_list)} != num_envs {self._num_envs}"
            )
        if len(set(idxs_list)) > 1:
            raise RuntimeError(
                "IsaacSimSimHandlerVec.reset_to_scene_idx: heterogeneous "
                "per-env scene indices not supported in this build "
                "(replicate_physics=True). Pre-converted USDs + "
                "MultiUsdFileCfg would lift this; not wired today."
            )
        self._scalar.reset_to_scene_idx(int(idxs_list[0]), seed=seed)

    def set_joint_positions(self, positions, *, env_ids=None) -> None:
        import torch

        if not positions:
            return
        names = list(positions.keys())
        joint_ids = [self.joint_name_to_idx[name] for name in names]
        # Tensors are expected to be (num_envs,) per name; widen to
        # (num_selected, len(names)) for write_joint_state_to_sim.
        if env_ids is None:
            env_ids_list = list(range(self._num_envs))
        elif isinstance(env_ids, torch.Tensor):
            env_ids_list = env_ids.detach().cpu().tolist()
        else:
            env_ids_list = list(env_ids)
        cols = [positions[name][env_ids_list] for name in names]
        target = torch.stack(cols, dim=-1).to(dtype=torch.float32, device=self._device)
        velocity = torch.zeros_like(target)
        self._scalar._robot.write_joint_state_to_sim(
            position=target,
            velocity=velocity,
            joint_ids=joint_ids,
            env_ids=env_ids_list,
        )

    def set_base_pose(self, xy_theta, *, env_ids=None) -> None:
        if not self.robot_cfg.is_mobile:
            raise RuntimeError(
                f"IsaacSimSimHandlerVec.set_base_pose: robot "
                f"{self.robot_cfg.name!r} is fixed-base"
            )
        bj = self.robot_cfg.base_joint_names
        if len(bj) != 3:
            raise ValueError(
                f"IsaacSimSimHandlerVec.set_base_pose: expected 3 base "
                f"joints (x, y, theta); got {bj!r}"
            )
        positions = {
            bj[0]: xy_theta[:, 0],
            bj[1]: xy_theta[:, 1],
            bj[2]: xy_theta[:, 2],
        }
        self.set_joint_positions(positions, env_ids=env_ids)

    def get_navigable_positions(self) -> list:
        return [None] * self._num_envs

    def render(self) -> None:
        self._scalar.render()

    def close(self) -> None:
        self._scalar.close()


__all__ = ["IsaacSimSimHandler", "IsaacSimSimHandlerVec"]
