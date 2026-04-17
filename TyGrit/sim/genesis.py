"""Genesis handler — the robot-agnostic :class:`SimHandler` for Genesis.

Genesis is a pure-Python physics engine with a single-build scene
contract: :meth:`genesis.Scene.build` is called exactly once, entities
cannot be added or removed afterward. That shapes the implementation
below:

* **Robot loading.** :class:`RobotCfg.urdf_path` is required
  (Genesis has no robot registry equivalent to ManiSkill's ``sim_uids``).
  We load it with :class:`gs.morphs.URDF` before the one-shot
  :meth:`Scene.build`.
* **Scene loading.** Delegates to
  :mod:`TyGrit.worlds.backends.genesis` which already supports
  Holodeck (MJCF) and Habitat-schema datasets (ReplicaCAD +
  AI2THOR variants). RoboCasa and YCB builtin ids raise — see the
  :mod:`TyGrit.worlds.backends.genesis` module docstring for why.
* **Scene switching on reset.** Because ``build`` is one-shot, a
  reset-time scene change requires tearing down the scene and
  rebuilding. The handler encapsulates that — callers see a uniform
  :meth:`reset_to_scene_idx` like any other :class:`SimHandler`.
* **Observation cache.** Genesis returns fresh state every query,
  so our "cache" is really a thin wrapper that ensures we
  ``scene.step()`` once per :meth:`apply_action` and read everything
  from the post-step state.

Genesis is imported inside methods so this module can be typechecked
and lightly introspected from the default pixi env without the
``genesis-world`` dep present. At runtime, only the ``genesis`` pixi
env is expected to exercise this path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from TyGrit.types.robots import ActuatorCfg, RobotCfg
from TyGrit.types.worlds import SceneSpec
from TyGrit.utils.transforms import (
    pose_from_pos_quat_wxyz,
    xyzw_to_wxyz,
)

if TYPE_CHECKING:  # pragma: no cover - typing-only imports
    import genesis as gs


class GenesisSimHandler:
    """Robot-agnostic :class:`~TyGrit.sim.base.SimHandler` for Genesis.

    Parameters
    ----------
    robot_cfg
        Robot descriptor. ``robot_cfg.urdf_path`` must be set.
    scenes
        Scene pool drawn from at reset. Must be non-empty.
    initial_scene_idx
        Index into ``scenes`` built at construction.
    show_viewer
        Pass-through to :class:`gs.Scene`.
    """

    def __init__(
        self,
        robot_cfg: RobotCfg,
        scenes: Sequence[SceneSpec],
        *,
        initial_scene_idx: int = 0,
        show_viewer: bool = False,
        num_envs: int = 0,
    ) -> None:
        if robot_cfg.urdf_path is None:
            raise ValueError(
                f"GenesisSimHandler: RobotCfg {robot_cfg.name!r} has no "
                f"urdf_path. Genesis loads robots from URDF; set RobotCfg.urdf_path."
            )
        self._robot_cfg = robot_cfg
        self._scenes: tuple[SceneSpec, ...] = tuple(scenes)
        if len(self._scenes) == 0:
            raise ValueError(
                "GenesisSimHandler: scene pool is empty; pass at least one SceneSpec"
            )
        if not 0 <= initial_scene_idx < len(self._scenes):
            raise IndexError(
                f"GenesisSimHandler: initial_scene_idx {initial_scene_idx} "
                f"out of range for scene pool of size {len(self._scenes)}"
            )
        self._show_viewer = show_viewer
        # Genesis uses n_envs=0 for the "no batching dimension" path
        # (scalar) and n_envs>=1 for batched. GenesisSimHandlerVec
        # passes n_envs=N here; scalar callers leave the default 0.
        self._num_envs_build: int = int(num_envs)

        # Build once at construction; rebuild on every scene switch.
        self._scene: "gs.Scene" | None = None
        self._robot_entity: Any = None
        self._cameras: dict[str, Any] = {}
        self._joint_name_to_idx: Mapping[str, int] = MappingProxyType({})
        self._active_scene_idx: int = -1

        self._build_scene(initial_scene_idx)

        # Cache derived action layout from RobotCfg. Genesis doesn't
        # expose controllers as gym-like action_space entries; the
        # handler assembles per-actuator writes inside apply_action()
        # from this layout.
        self._action_slices: Mapping[str, slice] = robot_cfg.action_slices_from_order()
        self._total_action_dim = robot_cfg.total_action_dim()

    # ── scene build / teardown ────────────────────────────────────────

    def _build_scene(self, idx: int) -> None:
        """(Re)build the sim scene for the ``idx``-th :class:`SceneSpec`.

        Tears down any existing scene and constructs a fresh one.
        Genesis's single-build API forces this for every scene switch.
        """
        import genesis as gs

        from TyGrit.worlds.backends.genesis import add_spec_to_scene

        if self._scene is not None:
            # Genesis lacks an explicit teardown; deleting the last
            # reference lets the GC reclaim the scene. We clear our
            # attribute references so that GC can proceed.
            self._scene = None
            self._robot_entity = None
            self._cameras = {}

        scene = gs.Scene(show_viewer=self._show_viewer)

        # Scene population (background + objects) first.
        add_spec_to_scene(scene, self._scenes[idx])

        # Robot second so its spawn pose doesn't get stomped by the
        # background's pose setters.
        #
        # ``RobotCfg.__post_init__`` enforces ``default_spawn_pose``
        # is set whenever ``is_mobile`` is True, and the GenesisSimHandler
        # constructor validates that ``urdf_path`` is set — so for the
        # mobile-robot case we always have a real spawn pose. The
        # fixed-base branch reads None and defaults to the URDF origin,
        # which is the documented behaviour and the only meaningful
        # spawn for non-mobile robots.
        if self._robot_cfg.is_mobile:
            spawn = self._robot_cfg.default_spawn_pose
            assert spawn is not None  # guaranteed by RobotCfg validator
            spawn_pos = (float(spawn[0]), float(spawn[1]), 0.0)
        else:
            spawn_pos = (0.0, 0.0, 0.0)
        robot_entity = scene.add_entity(
            gs.morphs.URDF(file=self._robot_cfg.urdf_path, pos=spawn_pos),
            name=f"robot__{self._robot_cfg.name}",
        )

        # Cameras third. Genesis cameras are world-frame entities — to
        # follow a robot link we have to write the camera pose every
        # step in :meth:`_update_attached_cameras`. We construct each
        # camera at the identity pose; the first :meth:`apply_action`
        # immediately overwrites it.
        cameras: dict[str, Any] = {}
        for cam in self._robot_cfg.cameras:
            cam_handle = scene.add_camera(
                res=(cam.width, cam.height),
                pos=(0.0, 0.0, 0.0),
                quat=(1.0, 0.0, 0.0, 0.0),  # wxyz identity
                fov=cam.fovy_degrees,
            )
            cameras[cam.camera_id] = cam_handle

        # n_envs=0 → no batching dimension (scalar path); n_envs>=1 →
        # batched scene with that many parallel envs.
        scene.build(n_envs=self._num_envs_build)

        self._scene = scene
        self._robot_entity = robot_entity
        self._cameras = cameras
        self._active_scene_idx = idx

        # Rebuild joint name → qpos index map from the freshly-loaded
        # URDF. Genesis orders joints as declared in the URDF file.
        self._joint_name_to_idx = MappingProxyType(
            {name: i for i, name in enumerate(self._joint_dof_names())}
        )

    def _joint_dof_names(self) -> list[str]:
        """Joint names in DOF order from the loaded URDF.

        Consolidated here so that future fixes for Genesis's joint
        indexing quirks land in one place. Genesis numbers DOFs inside
        ``Entity.dofs_info``; this helper filters to the robot entity's
        active joints.
        """
        # Public API: robot_entity.dofs_info is a list of DofInfo; each
        # exposes a .name attribute. The exact list depends on whether
        # Genesis treats the mobile base as a free joint (6-DOF prefix)
        # or as three prismatic/revolute joints. We trust the URDF as
        # the source of truth and expose exactly what's there.
        return [dof.name for dof in self._robot_entity.dofs_info]

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
        self._ensure_single_env(env_idx)
        qpos = self._robot_entity.get_dofs_position()
        return np.asarray(qpos, dtype=np.float64).reshape(-1)

    def get_link_pose(
        self, link_name: str, env_idx: int = 0
    ) -> npt.NDArray[np.float64]:
        self._ensure_single_env(env_idx)
        link = self._robot_entity.get_link(link_name)
        pos = np.asarray(link.get_pos(), dtype=np.float64).reshape(3)
        quat_wxyz = np.asarray(link.get_quat(), dtype=np.float64).reshape(4)
        return pose_from_pos_quat_wxyz(pos, quat_wxyz)

    def get_camera(self, camera_id: str, env_idx: int = 0) -> tuple[
        npt.NDArray[np.uint8],
        npt.NDArray[np.float32],
        npt.NDArray[np.int32] | None,
    ]:
        self._ensure_single_env(env_idx)
        self._robot_cfg.camera_by_id(camera_id)  # raises on typo
        if camera_id not in self._cameras:
            raise RuntimeError(
                f"GenesisSimHandler.get_camera: camera {camera_id!r} is in the "
                f"RobotCfg but was not constructed on this scene. This is a "
                f"programming error in _build_scene."
            )
        cam = self._cameras[camera_id]
        # Genesis camera render API: cam.render() returns (rgb, depth).
        # Segmentation is a separate channel — not wired yet; return None.
        rgb, depth = cam.render()
        rgb_u8 = np.asarray(rgb, dtype=np.uint8)
        depth_m = np.asarray(depth, dtype=np.float32)
        return rgb_u8, depth_m, None

    def get_intrinsics(self, camera_id: str) -> npt.NDArray[np.float64]:
        cam = self._robot_cfg.camera_by_id(camera_id)
        # Compute the pinhole intrinsic matrix from the CameraSpec.
        # Genesis exposes FOV as vertical degrees; we build K manually
        # because querying the sim for intrinsics is not guaranteed.
        fy = 0.5 * cam.height / np.tan(np.deg2rad(cam.fovy_degrees) / 2.0)
        fx = fy  # square pixels assumed
        cx = cam.width / 2.0
        cy = cam.height / 2.0
        return np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )

    # ── SimHandler: mutations ──────────────────────────────────────────

    def apply_action(self, action: npt.NDArray[np.float32]) -> None:
        if action.shape != (self._total_action_dim,):
            raise ValueError(
                f"GenesisSimHandler.apply_action: expected shape "
                f"({self._total_action_dim},), got {action.shape}"
            )
        # Refresh attached-camera poses BEFORE stepping so the rendered
        # frame returned by :meth:`get_camera` reflects the post-step
        # robot configuration the caller will see.
        self._update_attached_cameras()
        for actuator_name, sl in self._action_slices.items():
            actuator = self._robot_cfg.actuators[actuator_name]
            sub = np.asarray(action[sl], dtype=np.float64)

            if actuator.kind == "base_twist":
                self._apply_base_twist(actuator, sub)
                continue

            dof_idxs = [self._joint_name_to_idx[n] for n in actuator.joint_names]
            if actuator.command_to_joint_mapping is None:
                # 1-to-1 direct joint command. The ActuatorCfg
                # validator already ensures action_dim == len(joints)
                # for this path.
                values = sub
            else:
                # Broadcast according to the explicit mapping (e.g.
                # gripper (0, 0) fans one scalar to two fingers).
                values = np.array(
                    [sub[i] for i in actuator.command_to_joint_mapping],
                    dtype=np.float64,
                )

            self._apply_actuator(actuator.control_mode, dof_idxs, values)
        self._scene.step()

    def _apply_base_twist(
        self,
        actuator: ActuatorCfg,
        command: npt.NDArray[np.float64],
        dt: float | None = None,
    ) -> None:
        if command.shape != (2,):
            raise ValueError(
                "GenesisSimHandler._apply_base_twist: expected command shape "
                f"(2,), got {command.shape}"
            )
        if dt is None:
            dt = float(self._scene.dt)

        v = float(command[0])
        w = float(command[1])
        dof_indices = [self._joint_name_to_idx[name] for name in actuator.joint_names]
        theta = float(
            np.asarray(
                self._robot_entity.get_dofs_position(), dtype=np.float64
            ).reshape(-1)[dof_indices[2]]
        )

        # ManiSkill's holonomic base controller rotates the ego-frame
        # planar command into the world-frame base DOFs. Writing the
        # equivalent delta-pose (v*dt, 0, w*dt) and dividing by dt
        # yields the same joint velocity targets.
        base_delta = np.array([v * dt, 0.0, w * dt], dtype=np.float64)
        cos_theta = float(np.cos(theta))
        sin_theta = float(np.sin(theta))
        world_velocities = np.array(
            [
                (base_delta[0] * cos_theta) / dt,
                (base_delta[0] * sin_theta) / dt,
                base_delta[2] / dt,
            ],
            dtype=np.float64,
        )
        self._robot_entity.control_dofs_velocity(world_velocities, dof_indices)

    def _apply_actuator(
        self,
        control_mode: str,
        dof_idxs: list[int],
        values: npt.NDArray[np.float64],
    ) -> None:
        if control_mode == "velocity":
            self._robot_entity.control_dofs_velocity(values, dof_idxs)
        elif control_mode == "position":
            self._robot_entity.control_dofs_position(values, dof_idxs)
        elif control_mode == "effort":
            self._robot_entity.control_dofs_force(values, dof_idxs)
        else:
            raise ValueError(
                f"GenesisSimHandler: unknown control_mode {control_mode!r} "
                f"on actuator; expected one of velocity / position / effort"
            )

    def reset_to_scene_idx(self, idx: int, *, seed: int | None = None) -> None:
        if not 0 <= idx < len(self._scenes):
            raise IndexError(
                f"GenesisSimHandler.reset_to_scene_idx: idx {idx} out of range "
                f"for scene pool of size {len(self._scenes)}"
            )
        if seed is not None:
            # Genesis exposes gs.init(seed=...) at *simulator* init, not
            # per-reset; there is no per-scene-build reproducible-seed
            # hook. Rather than silently ignore the argument (which
            # would mask determinism bugs in callers), we reject it
            # explicitly. Deterministic per-reset variety in TyGrit is
            # driven by idx itself (SceneSampler hashes (base_seed,
            # env_idx, reset_count) into idx); no extra per-reset
            # randomness lives under seed.
            raise NotImplementedError(
                "GenesisSimHandler.reset_to_scene_idx: seed-based reproducible "
                "per-reset randomness is not yet wired through Genesis. The "
                "deterministic scene selection already encoded in `idx` "
                "(see TyGrit.worlds.sampler.SceneSampler) is the supported "
                "variety source; pass seed=None."
            )
        if idx == self._active_scene_idx and self._scene is not None:
            # Fast path: same scene, only entity poses need to reset.
            # Genesis's :meth:`Scene.reset` rewinds entity state without
            # re-loading geometry, which is the majority cost of the
            # full :meth:`_build_scene` rebuild. Verified via Codex
            # against Genesis ``scene.py:959``.
            self._scene.reset()
            return
        # Different scene: tear down + rebuild. Genesis's one-shot
        # ``Scene.build`` makes this unavoidable for geometry changes.
        self._build_scene(idx)

    def set_joint_positions(
        self,
        positions: Mapping[str, float],
        *,
        env_idx: int = 0,
    ) -> None:
        self._ensure_single_env(env_idx)
        if not positions:
            return
        qpos = self.get_qpos().copy()
        for name, value in positions.items():
            if name not in self._joint_name_to_idx:
                raise KeyError(
                    f"GenesisSimHandler.set_joint_positions: unknown joint "
                    f"{name!r}. Known: {sorted(self._joint_name_to_idx)!r}"
                )
            qpos[self._joint_name_to_idx[name]] = float(value)
        self._robot_entity.set_dofs_position(qpos)

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
                f"GenesisSimHandler.set_base_pose: robot "
                f"{self._robot_cfg.name!r} is fixed-base; set_base_pose is not legal"
            )
        self._ensure_single_env(env_idx)
        bj = self._robot_cfg.base_joint_names
        self.set_joint_positions({bj[0]: x, bj[1]: y, bj[2]: theta})

    # ── SimHandler: world hooks ────────────────────────────────────────

    def get_navigable_positions(self) -> list:
        # Genesis scene builder integration in
        # TyGrit.worlds.backends.genesis does not currently emit a
        # navmesh — the Habitat dataset navmeshes could be loaded via
        # their .navigable_positions.obj files, but that wiring is
        # outside the first Genesis landing. Report empty here so
        # robot cores fall back to their default spawn pose, which is
        # the documented no-navmesh behaviour.
        return []

    # ── SimHandler: lifecycle ──────────────────────────────────────────

    def render(self) -> None:
        # Genesis renders via the viewer if enabled; nothing to do in
        # headless mode.
        if self._show_viewer and self._scene is not None:
            self._scene.viewer_render()

    def close(self) -> None:
        self._scene = None
        self._robot_entity = None
        self._cameras = {}

    # ── internals ──────────────────────────────────────────────────────

    def _ensure_single_env(self, env_idx: int) -> None:
        if env_idx != 0:
            raise IndexError(
                f"GenesisSimHandler is single-env (num_envs=1); got env_idx={env_idx}"
            )

    def _update_attached_cameras(self) -> None:
        """Refresh world-frame poses of every robot-mounted camera.

        Genesis's :class:`gs.Camera` is a world-frame entity — there is
        no native ``attach(link)`` API as of April 2026. Following a
        robot link therefore requires writing the camera pose every
        step from the link's current world transform, composed with
        the per-camera offset declared in
        :attr:`CameraSpec.position` / :attr:`orientation_xyzw`.

        This is a no-op when the robot has no cameras (skips the
        per-step Sapien link query for fixed-base / no-camera robots).
        """
        for cam in self._robot_cfg.cameras:
            cam_handle = self._cameras.get(cam.camera_id)
            if cam_handle is None:
                continue
            link = self._robot_entity.get_link(cam.parent_link)
            link_pos = np.asarray(link.get_pos(), dtype=np.float64).reshape(3)
            link_quat_wxyz = np.asarray(link.get_quat(), dtype=np.float64).reshape(4)

            # Compose world = link ∘ camera_offset. Camera offset is
            # stored in TyGrit (xyzw) convention; convert before
            # composition. Using a matrix product keeps rotations and
            # translations consistent without re-deriving quaternion
            # multiplication here.
            T_link = pose_from_pos_quat_wxyz(link_pos, link_quat_wxyz)
            offset_quat_wxyz = np.array(xyzw_to_wxyz(cam.orientation_xyzw))
            T_offset = pose_from_pos_quat_wxyz(
                np.asarray(cam.position, dtype=np.float64), offset_quat_wxyz
            )
            T_world = T_link @ T_offset
            world_pos = T_world[:3, 3]
            # scipy returns xyzw; convert to wxyz for Genesis.
            from scipy.spatial.transform import Rotation as _R

            quat_xyzw = _R.from_matrix(T_world[:3, :3]).as_quat()
            world_quat_wxyz = xyzw_to_wxyz(tuple(quat_xyzw))
            cam_handle.set_pose(
                pos=tuple(float(v) for v in world_pos),
                quat=world_quat_wxyz,
            )


class GenesisSimHandlerVec(GenesisSimHandler):
    """Robot-agnostic :class:`~TyGrit.sim.base.SimHandlerVec` for Genesis.

    Genesis's :meth:`gs.Scene.build` accepts ``n_envs=N`` for native
    batched simulation. Every :meth:`Entity.get_dofs_position` /
    :meth:`Link.get_pos` / :meth:`Scene.step` is already batched on
    axis 0 in that mode. This subclass threads ``num_envs`` through
    :class:`GenesisSimHandler`'s ``_build_scene`` (via the
    ``num_envs`` ctor kwarg that passes to ``scene.build(n_envs=...)``)
    and overrides the :class:`SimHandlerVec` reads to skip the
    scalar's env-0 flatten.

    Scene pool heterogeneity: Genesis's ``Scene.build`` is one-shot —
    swapping the backing stage per env is not possible within a build
    group. Per-env scene indices must all match; mismatched
    ``reset_to_scene_idx(idxs)`` raises (CLAUDE.md Rule 1).
    """

    def __init__(
        self,
        robot_cfg: RobotCfg,
        scenes: Sequence[SceneSpec],
        *,
        num_envs: int,
        initial_scene_idx: int = 0,
        show_viewer: bool = False,
        device: str = "cuda:0",
    ) -> None:
        if num_envs <= 1:
            raise ValueError(
                f"GenesisSimHandlerVec: num_envs must be > 1; got {num_envs}. "
                f"Use GenesisSimHandler for the scalar path."
            )
        super().__init__(
            robot_cfg,
            scenes,
            initial_scene_idx=initial_scene_idx,
            show_viewer=show_viewer,
            num_envs=num_envs,
        )
        self._num_envs = int(num_envs)
        self._device = device

    @property
    def num_envs(self) -> int:  # type: ignore[override]
        return self._num_envs

    @property
    def device(self) -> str:
        return self._device

    # ── SimHandlerVec Protocol: batched reads ──────────────────────────

    def get_qpos(self):  # type: ignore[override]
        import torch

        # Genesis returns (N, dof) when built with n_envs>=1.
        qpos = self._robot_entity.get_dofs_position()
        return torch.as_tensor(qpos, device=self._device)

    def get_link_pose(self, link_name: str):  # type: ignore[override]
        import torch

        link = self._robot_entity.get_link(link_name)
        # Genesis batched API: link.get_pos() → (N, 3); get_quat() →
        # (N, 4) in wxyz.
        pos = torch.as_tensor(link.get_pos(), device=self._device).float()
        q = torch.as_tensor(link.get_quat(), device=self._device).float()
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        R = torch.stack(
            [
                1 - 2 * (yy + zz),
                2 * (xy - wz),
                2 * (xz + wy),
                2 * (xy + wz),
                1 - 2 * (xx + zz),
                2 * (yz - wx),
                2 * (xz - wy),
                2 * (yz + wx),
                1 - 2 * (xx + yy),
            ],
            dim=-1,
        ).reshape(-1, 3, 3)
        T = torch.eye(4, dtype=pos.dtype, device=pos.device).repeat(
            self._num_envs, 1, 1
        )
        T[:, :3, :3] = R
        T[:, :3, 3] = pos
        return T

    def get_camera(self, camera_id: str):  # type: ignore[override]
        import torch

        # Genesis renders a single viewpoint per call regardless of
        # n_envs — the camera is a world-frame entity, not per-env.
        # To satisfy the SimHandlerVec shape contract we broadcast
        # the one render across all envs and document it here.
        self._robot_cfg.camera_by_id(camera_id)  # raises on typo
        if camera_id not in self._cameras:
            raise RuntimeError(
                f"GenesisSimHandlerVec.get_camera: camera {camera_id!r} is in "
                f"the RobotCfg but was not constructed on this scene."
            )
        rgb_np, depth_np = self._cameras[camera_id].render()
        rgb = (
            torch.as_tensor(rgb_np, device=self._device)
            .unsqueeze(0)
            .expand(self._num_envs, -1, -1, -1)
            .contiguous()
        )
        depth = (
            torch.as_tensor(depth_np, device=self._device)
            .float()
            .unsqueeze(0)
            .expand(self._num_envs, -1, -1)
            .contiguous()
        )
        return rgb, depth, None

    # ── SimHandlerVec Protocol: batched writes ─────────────────────────

    def apply_action(self, action) -> None:  # type: ignore[override]
        import torch

        if action.shape != (self._num_envs, self._total_action_dim):
            raise ValueError(
                f"GenesisSimHandlerVec.apply_action: action shape "
                f"{tuple(action.shape)} != ({self._num_envs}, "
                f"{self._total_action_dim})"
            )
        self._update_attached_cameras()
        action_np = action.detach().cpu().numpy()
        for actuator_name, sl in self._action_slices.items():
            actuator = self._robot_cfg.actuators[actuator_name]
            sub = np.asarray(action_np[:, sl], dtype=np.float64)  # (N, dim)

            if actuator.kind == "base_twist":
                # base_twist is an SE(2) velocity command; Genesis'
                # base twist applier expects a per-env (N, 2) array.
                self._apply_base_twist(actuator, sub)
                continue

            dof_idxs = [self._joint_name_to_idx[n] for n in actuator.joint_names]
            if actuator.command_to_joint_mapping is None:
                values = sub  # (N, len(joints))
            else:
                values = np.stack(
                    [sub[:, i] for i in actuator.command_to_joint_mapping],
                    axis=-1,
                )  # (N, n_joints)

            self._apply_actuator(actuator.control_mode, dof_idxs, values)
        self._scene.step()
        _ = torch  # shape validation used torch above

    def reset_to_scene_idx(self, idxs, *, seed=None) -> None:  # type: ignore[override]
        import torch

        if isinstance(idxs, torch.Tensor):
            idxs_list = idxs.detach().cpu().tolist()
        else:
            idxs_list = list(idxs)
        if len(idxs_list) != self._num_envs:
            raise ValueError(
                f"GenesisSimHandlerVec.reset_to_scene_idx: idxs length "
                f"{len(idxs_list)} != num_envs {self._num_envs}"
            )
        if len(set(idxs_list)) > 1:
            raise RuntimeError(
                f"GenesisSimHandlerVec.reset_to_scene_idx: heterogeneous "
                f"per-env scene indices not supported; Genesis enforces a "
                f"single backing scene per build group. Got idxs={idxs_list}."
            )
        super().reset_to_scene_idx(int(idxs_list[0]), seed=seed)

    def set_joint_positions(self, positions, *, env_ids=None) -> None:  # type: ignore[override]
        import torch

        if not positions:
            return
        # Genesis's set_dofs_position accepts a batched tensor +
        # envs_idx list for per-env selection. Assemble a (len(env_ids),
        # len(dof_idxs)) tensor.
        if env_ids is None:
            env_ids_list = list(range(self._num_envs))
        elif isinstance(env_ids, torch.Tensor):
            env_ids_list = env_ids.detach().cpu().tolist()
        else:
            env_ids_list = list(env_ids)

        dof_idxs = [self._joint_name_to_idx[name] for name in positions]
        # positions[name] is (num_envs,); select env_ids rows, stack
        # into columns matching dof_idxs ordering.
        cols = []
        for name in positions:
            v = positions[name]
            if hasattr(v, "detach"):
                v = v.detach().cpu().numpy()
            cols.append(np.asarray(v)[env_ids_list])
        batched_positions = np.stack(cols, axis=-1)  # (len(env_ids), n_dofs)
        self._robot_entity.set_dofs_position(
            batched_positions, dof_idxs, envs_idx=env_ids_list
        )

    def set_base_pose(self, xy_theta, *, env_ids=None) -> None:  # type: ignore[override]
        if not self._robot_cfg.is_mobile:
            raise RuntimeError(
                f"GenesisSimHandlerVec.set_base_pose: robot "
                f"{self._robot_cfg.name!r} is fixed-base"
            )
        bj = self._robot_cfg.base_joint_names
        if len(bj) != 3:
            raise ValueError(
                f"GenesisSimHandlerVec.set_base_pose: expected 3 base "
                f"joints (x, y, theta); got {bj!r}"
            )
        positions = {
            bj[0]: xy_theta[:, 0],
            bj[1]: xy_theta[:, 1],
            bj[2]: xy_theta[:, 2],
        }
        self.set_joint_positions(positions, env_ids=env_ids)

    def get_navigable_positions(self) -> list:  # type: ignore[override]
        # Single-scene build group (Genesis one-shot); every env
        # shares the same navmesh. Base impl returns a list length
        # num_envs with the active scene's navmesh repeated.
        base = super().get_navigable_positions()
        if not base:
            return [None] * self._num_envs
        return [base[0]] * self._num_envs


__all__ = ["GenesisSimHandler", "GenesisSimHandlerVec"]
