# TyGrit — Claude Code instructions

These rules are load-bearing. Follow them on every change.

## 1. No hacks, only necessary fallbacks

Fix root causes. Do not paper over problems.

**Never:**
- Add a `try/except` that silently swallows an error "to make the test pass."
- Add a fallback branch "just in case" for a scenario you haven't actually seen.
- Monkey-patch, reach into private attributes, or rename things to avoid fixing a broken assumption.
- Use `# type: ignore`, `cast`, or `Any` to silence a type error you don't understand.
- Add a default value to an argument to work around a caller that isn't passing it.
- Introduce a new config flag or feature toggle to avoid modifying existing behavior.

**A fallback is only acceptable when all three are true:**
1. You have identified a specific, real condition under which the primary path cannot work (e.g. "robot hardware is not connected during unit tests", "the ManiSkill env hasn't been reset yet and `agent` is `None`").
2. You have documented that condition in a comment at the fallback site, naming the exact cause.
3. The fallback is the behavior the user would want in that case — not silent no-ops, not fabricated data.

If you're tempted to add a workaround, stop and diagnose the root cause instead. If you're stuck, tell the user what you've tried and ask — don't reach for a hack.

## 2. Always use pixi

TyGrit uses pixi for all dependency and task management. **Never run Python, pytest, pip, or any project tool against the system interpreter.**

### Environments

TyGrit's `pixi.toml` defines **one env per purpose**, composed from minimal features — no "kitchen sink" env. Pick the smallest env that has what you need:

| Env          | Features composed         | When to use                                             |
|--------------|---------------------------|---------------------------------------------------------|
| `default`    | base only                 | Pure-Python unit tests, most daily work                 |
| `world`      | maniskill + world         | World/scene generation, `SpecBackedSceneBuilder` (Step 4+) |
| `rl`         | maniskill + rl            | RL training (`TyGrit.rl.train`)                         |
| `thirdparty` | maniskill + thirdparty    | One-shot install tasks (`install-vamp`, `install-graspgen`, …) |
| `ci`         | ci                        | GitHub Actions slim test suite                          |
| `lint`       | lint (no default feature) | `pre-commit`, `black`, `ruff`                           |
| `docs`       | docs                      | `sphinx-build`                                          |
| *(no `robot` env yet)* | robot feature defined   | `ros-humble-desktop` for real-robot work — add env when needed |

The `maniskill` feature (`torch`, `mani-skill`, `cuda-toolkit`) is shared by `rl`, `world`, and `thirdparty` via feature composition — torch is declared in exactly one place.

**Do:**

- `pixi run test test/test_specific.py -v` — run a single test file in the default env. Most pure-Python tests (types, tasks, rl rewards, rl policy) run here.
- `pixi run -e world test test/test_maniskill_world.py` — run a world-layer test that needs `mani_skill` imports.
- `pixi run -e rl test` — run the full test suite including RL and sim-dependent tests.
- `pixi run -e rl python -m TyGrit.rl.train ...` — run the RL baseline.
- `pixi run -e thirdparty install-vamp` — run a one-shot install task.
- `pixi run <task>` — any default-env task defined in `pixi.toml`.
- `pixi add <package>` — add a conda dependency to the default env. `pixi add --feature world <package>` for a feature. `pixi add --pypi <package>` for pypi deps.
- `pixi shell -e world` — only when you genuinely need an interactive shell.

**Do not:**

- `python ...`, `python3 ...`, `pytest ...`, `pip ...` — these hit system Python and will load the wrong dependencies or none at all.
- `pixi run pytest ...` — use `pixi run test ...` instead. Pixi tasks are the canonical entry points.
- Hand-edit `pixi.toml` to **add dependencies**. Use `pixi add` so `pixi.lock` stays in sync with `pixi.toml` atomically. (Structural refactors — creating new features, changing `[environments]` composition — are fine to hand-edit, followed by `pixi install`.)
- Reach for a heavier env than you need. Pure-Python tests belong in `default`, not `rl`.
- Re-introduce a kitchen-sink `dev` env. If two envs need the same deps, extract a shared feature and compose both envs from it (like `maniskill`).

If a pixi task is missing for something you need repeatedly, propose adding it to `pixi.toml` rather than working around it.

## 3. Error handling — catch exactly what you know

Every `except` clause must specify exactly what failure it is catching and why.

**Rules:**

- **Never use bare `except:` or `except Exception:`.** Catch the most specific exception type that matches the case you're handling — `FileNotFoundError`, `ValueError`, `KeyError`, `ImportError`, a specific library exception class, etc.
- **Document the catch with a comment** naming the exact failure scenario: which line can raise, which library raises it, under what condition, and what we are doing in response. "Just in case" is not a reason.
- **If you don't know what can raise, find out before writing the handler.** Read the upstream function, check the library's source, or run the failing path and read the actual traceback. Do not guess.
- **Prefer Result types for expected failures.** The project already uses this pattern — see `TyGrit/types/results.py` (`PlanResult`, `StageResult`, `SchedulerResult`) and per-subsystem failure types (`PlannerFailure`, `IKFailure`, `GraspFailure`, `PerceptionFailure`, `ExecutionFailure`). Return a failure result; don't raise and re-catch.
- **Exceptions are for programming errors**, not expected control flow. A `ValueError` from a frozen dataclass validator is a programming error — let it propagate. A failed IK solution is expected — return an `IKFailure`.
- **Never catch an exception just to log-and-continue.** If the failure is recoverable, the recovery is part of the contract and must be documented. If it isn't recoverable, let it propagate.
- **Do not use `except` to suppress import errors** for optional dependencies without a clear reason (e.g. ROS only available on the robot). If you do, use `ImportError` specifically and document which module and why.

**Example — bad:**

```python
try:
    builder.build()
except Exception:
    logger.warning("failed to build")  # which failure? why warn instead of raise?
```

**Example — good:**

```python
try:
    builder.build()
except sapien.pysapien_core.SapienError as exc:
    # ManiSkill raises SapienError when an asset GLB is missing or malformed.
    # We log and skip the object because the benchmark should run with a
    # partial scene rather than abort; a missing asset is a data problem,
    # not a code bug.
    logger.warning("skip {}: {}", spec.name, exc)
    return None
```

If you cannot write a comment at that level of specificity, you don't yet understand the failure well enough to handle it — go investigate.

## 4. Match the existing project structure when adding new code

Before creating any new file, module, or directory, survey the project and place new code alongside conceptually similar existing code. Do not invent parallel hierarchies or new top-level homes for concerns that already have one.

**The established conventions in this repo:**

- `TyGrit/types/` — pure frozen dataclasses, no simulator imports. Every subsystem's data types live here as `types/<subsystem>.py` (e.g. `types/tasks.py`, `types/geometry.py`, `types/robot.py`, `types/planning.py`, `types/worlds.py`). If you're writing a pure dataclass, it belongs in `types/`, not anywhere else.
- `TyGrit/utils/` — pure functions, data in / data out, no class state.
- `TyGrit/envs/` — robot + simulator wrapper interface (`RobotBase` protocol and concrete implementations). Nothing about scene assets belongs here.
- `TyGrit/worlds/` — scene / world construction logic. Sim-agnostic modules (`manifest.py`, `sampler.py`) live at the top level and stay importable in the default pixi env. Per-simulator adapters live under `TyGrit/worlds/backends/` (one file per backend: `backends/maniskill.py`, later `backends/genesis.py`, `backends/isaac_sim.py`) and each imports its own sim package. **Never mix sim-agnostic modules with per-backend adapters at the same directory level** — that hides the env-dependency boundary. Precedent for this split: `TyGrit/planning/` puts sim-agnostic protocol/factory/config at the parent level and robot-specific impls in a subpackage.
- `TyGrit/tasks/` — task/goal definitions that reference worlds.
- `TyGrit/belief_state/` — perception representation (point clouds, voxels). Not sim-scene configuration.
- `TyGrit/planning/`, `TyGrit/perception/`, `TyGrit/controller/`, `TyGrit/kinematics/`, `TyGrit/gaze/`, `TyGrit/core/` — one directory per subsystem.
- `test/test_<module>.py` — one test file per module, mirroring the import path (e.g. `test/test_worlds_maniskill.py` for `TyGrit/worlds/backends/maniskill.py`).

**Before adding a new file, answer these in order:**

1. Does a sibling module already exist for this concern? If yes, use it.
2. Is what I'm writing a pure dataclass with no sim imports? If yes, it goes in `TyGrit/types/`, full stop.
3. Is it pure functions on data? → `TyGrit/utils/`.
4. Does it belong to an existing subsystem (planning, perception, envs, worlds, …)? Put it inside that subsystem's directory.
5. Only create a brand-new top-level directory when you genuinely have a new subsystem that nothing existing covers — and flag it to the user first.

**Concrete example (a mistake I actually made on 2026-04-11):** Writing pure dataclasses `ObjectSpec`, `SceneSpec`, `BuiltWorld` into `TyGrit/worlds/base.py` instead of `TyGrit/types/worlds.py`. The `types/` directory was the existing home for every other pure dataclass in the project; introducing `worlds/base.py` as a parallel "pure data" home violated the convention for no reason. The correct placement: pure types in `types/worlds.py`, logic (samplers, adapters, manifest loaders) in `worlds/`.

If in doubt, list the existing top-level modules under `TyGrit/` with `Glob` before creating anything new. Consistency over cleverness.

---

## Quick reference

- **Run one test file (default env):** `pixi run test test/test_foo.py -v`
- **Run world/sim test:** `pixi run -e world test test/test_foo.py -v`
- **Run full suite incl. RL + sim:** `pixi run -e rl test`
- **Add a conda dep:** `pixi add <package>` (or `pixi add --feature world <package>`)
- **Add a pypi dep:** `pixi add --pypi <package>`
- **Project memory (user-local):** `~/.claude/projects/-media-run-Work-TyGrit/memory/` — check `MEMORY.md` for pointers to architecture notes, v1 prior art, and user preferences.
- **v1 source for prior art:** `/media/run/Work/paper/mobile_grasping_uncertainty/version_1/grasp_anywhere_v1/`
