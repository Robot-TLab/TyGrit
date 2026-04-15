"""Shared ManiSkill construction helpers.

Both :class:`~TyGrit.sim.maniskill.ManiSkillSimHandler` (single-env,
numpy/scalar) and :class:`~TyGrit.sim.maniskill.ManiSkillSimHandlerVec`
(vectorised, torch/batched) need the same construction-time setup:
calling ``gym.make`` on ``SceneManipulation-v1`` with the right
sensor / scene-builder kwargs, computing per-controller action slices
off the agent, building the joint-name→index map, and reading the
camera intrinsics. The two paths only diverge after that — when they
start parsing observations and assembling actions in scalar vs
batched form.

Robot-agnostic by construction: every helper consumes
:class:`~TyGrit.types.robots.RobotCfg`, never a Fetch-specific
descriptor or env-config.
"""

from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from mani_skill.utils.structs.types import SimConfig

from TyGrit.types.robots import RobotCfg
from TyGrit.types.worlds import SceneSpec
from TyGrit.worlds.backends.maniskill import bind_specs

#: ManiSkill env id used by the worlds-backed scene wrapper. Scene
#: selection comes from the caller-supplied scene pool;
#: ``"SceneManipulation-v1"`` is the only piece ManiSkill needs to
#: dispatch the env class.
SCENE_MANIPULATION_ENV_ID = "SceneManipulation-v1"


def make_scene_manipulation_env(
    robot: RobotCfg,
    scenes: Sequence[SceneSpec],
    *,
    build_config_idxs: list[int],
    sim_config: SimConfig,
    obs_mode: str,
    control_mode: str,
    render_mode: str | None,
    camera_resolution: tuple[int, int],
    num_envs: int | None = None,
    sim_backend: str | None = None,
):
    """Build a ``SceneManipulation-v1`` env via ``gym.make``.

    Parameters
    ----------
    robot
        Robot descriptor. ``robot.sim_uids["maniskill"]`` selects the
        ManiSkill agent class. Must have an entry — raises
        :class:`KeyError` otherwise.
    scenes
        Scene pool the bound :class:`SpecBackedSceneBuilder` will draw
        from at reconfigure.
    build_config_idxs
        One scene-pool index per parallel env (length 1 in the
        single-env case, length ``num_envs`` in the vec case).
    sim_config
        ManiSkill ``SimConfig`` (GPU memory, contact offset, …).
        Caller-provided because single and vec paths use different
        tunings.
    obs_mode, control_mode, render_mode
        ``gym.make`` pass-through.
    camera_resolution
        ``(width, height)`` applied uniformly to every sensor in the
        env. Per-camera resolution would need a richer
        ``sensor_configs`` assembly — not used today.
    num_envs, sim_backend
        Optional ``gym.make`` kwargs for the vectorised path. Omitted
        for single-env so ManiSkill picks its CPU defaults.
    """
    width, height = camera_resolution
    kwargs: dict = dict(
        robot_uids=robot.sim_uids["maniskill"],
        scene_builder_cls=bind_specs(scenes),
        build_config_idxs=build_config_idxs,
        obs_mode=obs_mode,
        control_mode=control_mode,
        render_mode=render_mode,
        sensor_configs={"width": width, "height": height},
        sim_config=sim_config,
    )
    if num_envs is not None:
        kwargs["num_envs"] = num_envs
    if sim_backend is not None:
        kwargs["sim_backend"] = sim_backend
    return gym.make(SCENE_MANIPULATION_ENV_ID, **kwargs)


def build_action_slices(agent, robot: RobotCfg) -> tuple[dict[str, slice], int]:
    """Compute per-controller slices of the low-level action vector.

    Iterates :attr:`RobotCfg.controller_order` so the concatenation
    layout is robot-specific — the same helper works for any robot
    whose cfg names its controllers. Only controllers the underlying
    agent actually exposes get a slice; absent ones are skipped.

    Returns ``(slices, total_dim)`` where ``total_dim`` is the length
    of the full action vector.

    Reads ``action_space.shape[-1]`` so the helper works for both
    single-env (shape ``(D,)``) and vec (shape ``(N, D)``) without
    branching.
    """
    slices: dict[str, slice] = {}
    cursor = 0
    for name in robot.controller_order:
        controller = agent.controller.controllers.get(name)
        if controller is None:
            continue
        dim = controller.action_space.shape[-1]
        slices[name] = slice(cursor, cursor + dim)
        cursor += dim
    return slices, cursor


def build_joint_name_to_idx(agent) -> dict[str, int]:
    """Map joint name → index into the qpos vector for the active joints.

    Iterates ``agent.robot.active_joints`` in the order ManiSkill
    registers them; this is the same order the qpos vector uses.
    """
    return {j.name: i for i, j in enumerate(agent.robot.active_joints)}


def extract_intrinsics(env, sensor_id: str) -> npt.NDArray[np.float64]:
    """Read the static 3×3 intrinsic matrix for ``sensor_id``.

    Sensor configs don't change once ``gym.make`` returns, so callers
    cache this once at construction time. Handles both numpy
    (single-env) and torch (vec) intrinsic_cv tensors, and unwraps
    the leading batch dim when the env is vectorised.

    The ``_sensors`` attribute access is into a private ManiSkill
    attribute — there is no public API for "give me the camera
    intrinsics by sensor id" as of mani-skill 3.x. If ManiSkill
    eventually publishes one, switch this single call site.
    """
    cam_params = env.unwrapped._sensors[sensor_id].get_params()
    K = cam_params["intrinsic_cv"]
    if hasattr(K, "detach"):
        K = K.detach().cpu().numpy()
    K = np.array(K)
    if K.ndim == 3:
        K = K[0]
    return K.astype(np.float64)
