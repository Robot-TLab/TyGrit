"""Shared ManiSkill construction helpers for Fetch backends.

Both :class:`~TyGrit.envs.fetch.maniskill.ManiSkillFetchSimBackend`
(single-env, numpy/scalar) and
:class:`~TyGrit.envs.fetch.maniskill_vec.ManiSkillFetchRobotVec`
(vectorised, torch/batched) need the same construction-time setup:
calling ``gym.make`` on ``SceneManipulation-v1`` with the right
sensor/scene-builder kwargs, computing per-controller action slices
off the agent, building the joint-name→index map, and reading the
camera intrinsics. The two classes only diverge after that — when
they start parsing observations and assembling actions in scalar vs
batched form.

Keeping these helpers in one module means a change to (e.g.) the
sensor layout or a new env kwarg only has to be made once.
"""

from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from mani_skill.utils.structs.types import SimConfig

from TyGrit.envs.fetch.config import FetchEnvConfig
from TyGrit.types.robots import RobotSpec
from TyGrit.types.worlds import SceneSpec
from TyGrit.worlds.backends.maniskill import bind_specs

#: ManiSkill env id used by the worlds-backed Fetch wrapper. Scene
#: selection comes from ``config.scene_sampler``; this string is the
#: only piece ManiSkill needs to dispatch the env class.
SCENE_MANIPULATION_ENV_ID = "SceneManipulation-v1"


def make_scene_manipulation_env(
    config: FetchEnvConfig,
    robot: RobotSpec,
    scenes: Sequence[SceneSpec],
    build_config_idxs: list[int],
    sim_config: SimConfig,
    *,
    num_envs: int | None = None,
    sim_backend: str | None = None,
):
    """Build a ``SceneManipulation-v1`` Fetch env via ``gym.make``.

    ``build_config_idxs`` must have one entry per parallel env (length
    1 in the single-env case, length ``num_envs`` in the vec case);
    it's the construction-time scene selection that
    :class:`~TyGrit.worlds.backends.maniskill.SpecBackedSceneBuilder`
    consumes during its first reconfigure.

    ``sim_config`` is caller-provided because the single and vec
    paths use different ``SceneConfig.contact_offset``,
    ``GPUMemoryConfig`` settings, and (vec only) ``spacing`` —
    parameterising these inside this helper would just hide them
    behind another layer of plumbing.

    ``num_envs`` and ``sim_backend`` are optional so the single-env
    caller can omit them and let ManiSkill pick its CPU defaults.
    """
    kwargs: dict = dict(
        robot_uids=robot.sim_uids["maniskill"],
        scene_builder_cls=bind_specs(scenes),
        build_config_idxs=build_config_idxs,
        obs_mode=config.obs_mode,
        control_mode=config.control_mode,
        render_mode=config.render_mode,
        sensor_configs={
            "width": config.camera_width,
            "height": config.camera_height,
        },
        sim_config=sim_config,
    )
    if num_envs is not None:
        kwargs["num_envs"] = num_envs
    if sim_backend is not None:
        kwargs["sim_backend"] = sim_backend
    return gym.make(SCENE_MANIPULATION_ENV_ID, **kwargs)


def build_action_slices(agent, robot: RobotSpec) -> tuple[dict[str, slice], int]:
    """Compute per-controller slices of the low-level action vector.

    Iterates ``robot.controller_order`` so the concatenation layout is
    robot-specific — the same helper works for any robot whose spec
    names its controllers. Only controllers the underlying agent
    actually exposes get a slice; absent ones are skipped.

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
    """
    cam_params = env.unwrapped._sensors[sensor_id].get_params()  # type: ignore[attr-defined]
    K = cam_params["intrinsic_cv"]
    if hasattr(K, "detach"):
        K = K.detach().cpu().numpy()
    K = np.array(K)
    if K.ndim == 3:
        K = K[0]
    return K.astype(np.float64)
