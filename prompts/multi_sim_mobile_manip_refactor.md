# Autoresearch brief — multi-sim mobile-manipulation refactor

**Audience:** a fresh Claude Code agent, no prior conversation context. Read
this top to bottom before touching code. Follow `CLAUDE.md` (project root)
exactly — the rules there override anything you infer from this brief.

**How to launch:** in a new session, `@prompts/multi_sim_mobile_manip_refactor.md start`.

---

## 1. Mission

Finish the 2026-04-15 three-layer sim refactor so that TyGrit cleanly
supports **ManiSkill + Genesis + Isaac Sim** across every asset source
already wired (ReplicaCAD, ProcTHOR, iTHOR, RoboTHOR, ArchitecTHOR,
RoboCasa, Holodeck, Objaverse, YCB) for **mobile-manipulation only**.

Four objectives, in priority order:

1. **Clean `TyGrit/envs/`.** No per-simulator scripts in the envs layer.
   `envs/fetch/` is the **direct sim-interface adapter for one robot**
   and nothing more:
   * **Actuation:** the low-level joint interface (velocity / position
     commands) — assemble a per-controller action slice into the flat
     vector the `SimHandler` accepts.
   * **Sensing:** the direct sensor-reading interface — turn the
     handler's raw `qpos` / link poses / camera outputs into the
     project's `RobotState` + `SensorSnapshot` dataclasses.
   * Reset / teleport plumbing that belongs to the robot (base-offset
     calibration, spawn-pose randomisation against the navmesh).

   **Not in `envs/`:** MPC, end-effector tracking, gaze control,
   trajectory execution loops, task-level goals. Those are a layer
   above and already live in `TyGrit/controller/`, `TyGrit/gaze/`,
   `TyGrit/tasks/`. If you find them tangled into `envs/fetch/core.py`
   (they are, as of today's 431-line file), split them out as part of
   this refactor.

   Adding a new simulator must require zero edits under `envs/`.
2. **RoboVerse-style cross-sim spec, scoped to mobile manipulation.**
   Mirror MetaSim's pattern (`RoboVerseOrg/RoboVerse` — `ScenarioCfg` /
   `SceneCfg` / `_FileBasedMixin` with per-sim `file_type` dispatch) only
   for scene-level assets + mobile manipulators. **Exclude tabletop and
   tool-use cases.** See §5 for the scope filter.
3. **Maximise Codex subagent usage.** Every non-trivial phase is driven
   by the Claude↔Codex mutual-review loop. See §6.
4. **Full Isaac Sim support for every current asset source.** The
   `IsaacSimSimHandler` is a skeleton today (every method raises
   `NotImplementedError`). Make it real, and add a
   `TyGrit/worlds/backends/isaac_sim.py` that dispatches every source in
   `SOURCE_SIM_COMPATIBILITY` into Isaac Lab's loaders.

---

## 2. Current state inventory (verify before trusting — code moves fast)

Verify each claim with `Read` / `Grep` before acting on it. Line counts
are as of 2026-04-15; treat as approximate.

### 2.1 Sim layer (`TyGrit/sim/`) — in good shape

* `sim/base.py` (238 lines) — `SimHandler` Protocol. Stable contract.
* `sim/maniskill.py` (415 lines) + `sim/maniskill_helpers.py` (154
  lines) — working. Numpy / single-env.
* `sim/genesis.py` (514 lines) — working.
* `sim/isaac_sim.py` (252 lines) — **skeleton**. Every method raises
  `NotImplementedError("… — skeleton")`. Constructor validates
  RobotCfg + scene pool.

### 2.2 Envs layer (`TyGrit/envs/`) — not yet clean

* `envs/base.py`, `envs/config.py` — Protocol + base config. Fine.
* `envs/__init__.py` — dispatches on `config.robot`. Fine.
* `envs/fetch/__init__.py` — `FetchRobot.create(config)` dispatches on
  `config.backend` ∈ `{"maniskill", "ros"}`. **Problem:** it imports
  `maniskill.py` / `maniskill_vec.py` by name. Adding Genesis/Isaac
  forces edits here.
* `envs/fetch/core.py` (431 lines) — `FetchRobotCore`, sim-agnostic.
  **Mixed concern:** holds the direct sim-interface (joint indexing,
  action-slice assembly, base-offset calibration, qpos→`RobotState`,
  camera→`SensorSnapshot`) — good, that's where it belongs — **plus**
  higher-level logic (`execute_trajectory` calling
  `compute_mpc_action`, `look_at` doing IK math). The MPC/IK/gaze
  pieces should move out to `TyGrit/controller/` and `TyGrit/gaze/`
  (where siblings already exist) or to a thin orchestration layer
  above the env. Core stays as the actuation + sensing adapter.
* `envs/fetch/config.py` (50 lines) — **has ManiSkill-specific fields**
  (`obs_mode`, `control_mode`, `render_mode`) hard-wired into
  `FetchEnvConfig`. Must move.
* `envs/fetch/maniskill.py` (77 lines) — thin glue:
  `ManiSkillFetchRobot = FetchRobotCore + ManiSkillSimHandler`. Should
  collapse into a generic factory.
* `envs/fetch/maniskill_vec.py` (610 lines) — vectorised GPU-parallel
  ManiSkill path for RL. Has its own torch-tensor semantics that don't
  fit the numpy `SimHandler`. **Required action:** lift the useful
  pieces (env construction, action-slice mapping, obs parsing) into a
  new `ManiSkillSimHandlerVec` in `TyGrit/sim/maniskill.py` that
  implements the new `SimHandlerVec` Protocol (§3.1). Delete the envs-
  layer file. Genesis and Isaac Sim ship vec handlers at the same
  time — parallel training is the default, not an opt-in.
* `envs/fetch/ros.py` (62 lines) — placeholder. Leave alone.

### 2.3 Worlds layer (`TyGrit/worlds/`) — partial

* `worlds/manifest.py`, `worlds/sampler.py`, `worlds/object_sampler.py`
  — sim-agnostic. Fine.
* `worlds/asset_sources/` — per-dataset `AssetSource` implementations.
  `ReplicaCADSource`, `ProcTHORSource`, `IThorSource`, `RoboTHORSource`,
  `ArchitecTHORSource`, `RoboCasaSource`, `HolodeckSource`,
  `ObjaverseSource`, `ManiSkillYCBSource`. Stable.
* `worlds/asset_sources/base.py::SOURCE_SIM_COMPATIBILITY` — per-source
  × per-sim matrix. Currently lists `maniskill` and `genesis`. **Must
  add `isaac_sim` entries** once each source's USD/URDF/MJCF path is
  verified.
* `worlds/backends/maniskill.py` (576 lines) — full 7-source dispatch
  via `SpecBackedSceneBuilder`.
* `worlds/backends/genesis.py` (235 lines) + helper
  `_genesis_habitat.py` (462 lines) — working.
* `worlds/backends/isaac_sim.py` — **does not exist.** Must be created
  mirroring `maniskill.py`'s per-source dispatch.

### 2.4 Robot layer (`TyGrit/robots/`)

* `robots/fetch/` package. `FETCH_CFG: RobotCfg` with `sim_uids`,
  `urdf_path`, and (currently) `usd_path = None`. Verify
  — if `usd_path` is missing, Isaac Sim work will need a URDF→USD
  materialisation step (see `UrdfConverterCfg`, or a one-shot
  `pixi run setup-fetch-usd` task mirroring `setup-fetch-sim`).

### 2.5 Pixi envs (`pixi.toml`)

Per `architecture_sim_refactor_2026_04_15.md`, the `isaacsim` pixi env
*was* configured to solve with Isaac Sim 5.0 + Isaac Lab via git-source
install. **Verify it still solves today** before assuming. If it
doesn't, the first phase is unblocking it (see §8 Risks).

---

## 3. Target architecture

### 3.1 Envs layer — single generic composition over scalar + vec handlers

Replace the per-sim classes in `envs/fetch/` with one generic composition:

```
FetchRobot.create(config) →
    handler = create_sim_handler(
        config.sim_backend, FETCH_CFG, scenes,
        num_envs=config.num_envs, **sim_opts,
    )
    return FetchRobotCore(config, handler)   # sim-interface only
```

`create_sim_handler(..., num_envs=n)` dispatches internally:

* `num_envs == 1` → single-env handler (numpy, current `SimHandler`
  Protocol).
* `num_envs > 1` → batched handler implementing the new
  **`SimHandlerVec` Protocol** (torch tensors on GPU, shape
  `(num_envs, ...)`).

**`SimHandlerVec` is a first-class contract, not an escape hatch.** It
lives in `TyGrit/sim/base.py` alongside `SimHandler`. Every simulator
ships both:

| Sim        | Scalar handler                 | Vec handler                       |
|------------|--------------------------------|-----------------------------------|
| ManiSkill  | `ManiSkillSimHandler`          | `ManiSkillSimHandlerVec` (new, lifted from `envs/fetch/maniskill_vec.py`) |
| Genesis    | `GenesisSimHandler`            | `GenesisSimHandlerVec` (new, uses `envs_idx`) |
| Isaac Sim  | `IsaacSimSimHandler`           | `IsaacSimSimHandlerVec` (new — Isaac Lab is natively vectorised, this is the ergonomic path) |

`SimHandlerVec` mirrors `SimHandler` but:

* All reads return torch tensors batched on axis 0: `get_qpos() →
  Tensor[num_envs, dof]`, `get_link_pose(link) → Tensor[num_envs, 4,
  4]`, `get_camera(id) → (rgb: uint8[N,H,W,3], depth: float32[N,H,W],
  seg: int32[N,H,W] | None)`.
* `apply_action(action: Tensor[num_envs, total_action_dim])` — full
  batch per step. No per-env action dispatch.
* `reset_to_scene_idx(idxs: Sequence[int] | Tensor)` takes one scene
  index per env (ManiSkill's heterogeneous-GPU path; Genesis/Isaac
  rebuild or replicate per their constraints — see §8).
* `set_joint_positions` / `set_base_pose` accept an optional
  `env_ids: Sequence[int] | Tensor | None` (None = all envs).
* Device is a construction-time argument (`"cuda:0"` default). Tensors
  returned share that device; callers do not cross the CPU↔GPU
  boundary except at observation-logging time.

`FetchRobotCore`'s responsibility after the cleanup:

* `apply_velocity(...)` / `apply_joint_position(...)` — build the flat
  action vector (or batched tensor, when backed by
  `SimHandlerVec`) from per-controller slices and call
  `handler.apply_action`.
* `read_state() → RobotState` and `read_sensors() → SensorSnapshot` —
  for the scalar path; for the vec path the analogous
  `read_state_vec() → RobotStateVec` / `read_sensors_vec()` return
  batched tensor-backed dataclasses. Design the batched dataclasses as
  frozen types in `TyGrit/types/` co-located with the scalar versions
  (`types/robots.py`, `types/sensors.py`) — do **not** invent a
  parallel `types/` tree.
* `reset(...)` — advance sampler, call
  `handler.reset_to_scene_idx(idx)` or `handler.reset_to_scene_idx(idxs)`
  depending on the handler kind, re-calibrate the base offset,
  randomise spawn against navmesh.
* **Nothing else.** No MPC, no IK, no trajectory loops, no gaze.

(MPC / IK / trajectory-loops / gaze are strictly above `envs/` — they
consume a `FetchRobotCore` (or the `RobotBase` Protocol) and live in
`TyGrit/controller/`, `TyGrit/gaze/`, `TyGrit/core/scheduler.py`.)

* `create_sim_handler(sim_name, robot_cfg, scenes, **sim_opts)` lives
  in `TyGrit/sim/__init__.py`. Dispatches on `sim_name` ∈
  `{"maniskill", "genesis", "isaac_sim"}` (+ `"ros"` stays in envs as
  a hardware backend, not a sim handler).
* `envs/fetch/maniskill.py` deleted. `envs/fetch/__init__.py` no longer
  imports any sim module by name.
* `FetchEnvConfig` splits: robot-facing fields stay; ManiSkill-specific
  fields (`obs_mode`, `control_mode`, `render_mode`) move to an
  opaque `sim_opts: Mapping[str, Any]` (or a per-sim config dataclass
  co-located with each handler, e.g.
  `TyGrit.sim.maniskill.ManiSkillOpts`). Rule-of-3: don't over-engineer
  — pick the simplest shape that keeps `envs/fetch/` clean.
* `envs/fetch/maniskill_vec.py` — **decision required with Codex**. Two
  paths: (a) design a `SimHandlerVec` Protocol that Isaac Lab and
  Genesis can implement, then fold the vec class into `sim/maniskill.py`
  as a second handler; (b) mark it an RL-specialised ManiSkill escape
  hatch, keep it under `TyGrit/rl/`, and document that
  `FetchRobot.create(...)` with `num_envs > 1` still routes there. Do
  not guess — bring evidence.

### 3.2 Worlds layer — add Isaac Sim backend

Create `TyGrit/worlds/backends/isaac_sim.py` mirroring
`worlds/backends/maniskill.py`'s structure:

* Per-source dispatch inside `_SUPPORTED_SOURCES` for
  `replicacad`, `procthor`, `ithor`, `robothor`, `architecthor`,
  `robocasa`, `holodeck`, `objaverse` (verify each actually has a
  USD/URDF/MJCF path reachable by Isaac Lab — may need per-source
  conversion tasks).
* Entry points consumed by `IsaacSimSimHandler._build_scene`.
* Extend `SOURCE_SIM_COMPATIBILITY` in lock-step.

### 3.3 Isaac Sim handler — fill in the skeleton

Implement every `NotImplementedError` method in
`TyGrit/sim/isaac_sim.py`. Target APIs (verify against the installed
Isaac Lab — versions drift):

* `isaaclab.sim.SimulationContext`, `isaaclab.scene.InteractiveScene`.
* `isaaclab.assets.ArticulationCfg` (USD) or `UrdfConverterCfg` (URDF
  fallback).
* `isaaclab.sensors.CameraCfg` for per-camera `CameraSpec` in
  `RobotCfg.cameras`.
* `articulation.data.joint_pos` / `body_pos_w` / `body_quat_w` for
  observation reads.
* `articulation.set_joint_velocity_target` /
  `.set_joint_position_target` for `apply_action`.
* `write_joint_state_to_sim` / `write_root_state_to_sim` for
  teleport / reset.

Respect the contract in `sim/base.py`: post-step consistency, camera
frame convention (`[x, y, z, w]`), numpy / metres / radians at the
boundary, raise on programming errors, `Result` types for domain
failure.

---

## 4. Follow RoboVerse, but…

Read the MetaSim handler docs:
<https://roboverse.wiki/metasim/tutorial/handler>. Useful structural
precedents:

* `ScenarioCfg` + `SceneCfg` with per-sim `file_type` dispatch —
  TyGrit already has the moral equivalent in `SceneSpec` +
  `SOURCE_SIM_COMPATIBILITY` + backend modules. Confirm naming is
  consistent, do not rename for the sake of matching them.
* Per-sim handler surface — matches our `SimHandler` Protocol.
* Their Isaac Sim / IsaacGym / Genesis / MuJoCo handlers are worth
  reading for API-call ergonomics; **do not copy code** (licence +
  style mismatch). Borrow the shape, write it ourselves.

**Gaps in RoboVerse that TyGrit needs (and RoboVerse doesn't cover):**
mobile-base control, navmesh-driven spawn, scene-level heterogeneous
asset loading at the `SceneSpec` level, ManiSkill3 backend. Keep these.

---

## 5. Scope filter — mobile manipulation only

**In scope:**

* Scene-level assets: full apartments / houses / kitchens
  (ReplicaCAD, Holodeck, ProcTHOR family, RoboCasa, HSSD when wired).
* Objects placeable within scene-level assets (Objaverse,
  `ManiSkillYCB` kept for backward-compat only if already used).
* Mobile manipulators (Fetch today; Spot / AutoLife / Stretch later).
  Must have a navigable base + arm.

**Out of scope — do not carry work for these:**

* Tabletop-only benchmarks (`PickCube`, `PushCube`, `Peg-Insertion`,
  etc.). Any ManiSkill tabletop task id beyond
  `SceneManipulation-v1` is out.
* Fixed-base manipulators (Franka Panda, UR5, etc.).
* Tool-use / contact-rich manipulation benchmarks (Meta-World,
  LIBERO tool variants).
* Dexterous / humanoid bodies.

If you encounter existing TyGrit code that's only for
tabletop/tool-use, flag it to the user before deleting — do not
assume.

---

## 6. Codex usage playbook

Per `~/.claude/projects/-media-run-Work-TyGrit/memory/feedback_codex_mutual_review.md`:
run a **mutual review loop** between Claude and Codex for every
non-trivial design or diff in this refactor.

**Required Codex delegations in this refactor (at minimum):**

1. **Isaac Lab API survey.** Delegate to `codex:codex-rescue` a focused
   research task: *"Produce a concrete per-method implementation plan
   for `TyGrit/sim/isaac_sim.py::IsaacSimSimHandler` against Isaac Lab
   2.3.x. For each `NotImplementedError` method, name the exact Isaac
   Lab API to call, argument shapes, units, and failure modes. Verify
   against installed Isaac Lab source under `thirdparty/IsaacLab`."*
2. **RoboVerse/MetaSim handler diff.** Codex reads RoboVerse's Isaac
   Sim handler and summarises which patterns TyGrit should borrow vs
   skip.
3. **Per-source Isaac Sim loader dispatch.** For each source
   (`replicacad`, `procthor`, `ithor`, `robothor`, `architecthor`,
   `robocasa`, `holodeck`, `objaverse`), Codex drafts the
   `add_spec_to_scene`-equivalent function. Claude reviews each
   against `CLAUDE.md` rules 1 + 3 + 4.
4. **Envs cleanup diff.** Claude drafts the envs/ cleanup (delete
   `maniskill.py`, move ManiSkill opts out of `FetchEnvConfig`, new
   `create_sim_handler` factory); Codex reviews before commit.
5. **`SimHandlerVec` Protocol design.** Codex drafts the `SimHandlerVec`
   Protocol (torch tensors, `(num_envs, ...)` shape, `env_ids` semantics,
   device contract, reset-with-per-env-idx semantics). Claude reviews
   against the existing `SimHandler` shape and the real APIs of
   ManiSkill3, Genesis, and Isaac Lab so the Protocol fits all three
   without per-sim leakage. The *vec path is committed scope* — this
   delegation is about the interface shape, not whether to build it.
6. **Per-sim vec handler implementation.** One Codex-then-Claude review
   cycle per sim (`ManiSkillSimHandlerVec`, `GenesisSimHandlerVec`,
   `IsaacSimSimHandlerVec`).

**Apply every Codex suggestion critically** — check against
`CLAUDE.md` Rule 1 (no hacks), Rule 3 (catch exactly what you know),
Rule 4 (match existing structure). Reject suggestions that regress
those.

Use `Agent` with `subagent_type: "general-purpose"` for parallelisable
reads that don't need Codex. Use `Agent` with `subagent_type:
"codex:codex-rescue"` for the items above. Prefer parallel Codex +
Explore queries whenever they're independent.

---

## 7. Success criteria

A PR (or stacked PRs, rule-of-3-bounded) lands when **all** of:

* [ ] `TyGrit/envs/fetch/` has no module whose filename contains a
      simulator name. `FetchRobot.create(config)` routes via
      `TyGrit.sim.create_sim_handler(...)` — confirmed by
      `Grep "from mani_skill" TyGrit/envs/` returning nothing.
* [ ] `envs/fetch/core.py` has no import from
      `TyGrit.controller.*` / `TyGrit.gaze.*` and defines no MPC or
      IK methods. Sensor/actuation adapter only. Verifiable via
      `Grep "compute_mpc_action|forward_kinematics" TyGrit/envs/` → zero.
* [ ] `FetchEnvConfig` has no ManiSkill-specific fields. Adding Genesis
      or Isaac Sim does not touch `envs/fetch/config.py`.
* [ ] `TyGrit/sim/isaac_sim.py` — every `NotImplementedError` removed.
      `Grep "raise NotImplementedError" TyGrit/sim/isaac_sim.py`
      returns zero matches.
* [ ] `SimHandlerVec` Protocol exists in `TyGrit/sim/base.py` and
      **all three** sims implement it: `ManiSkillSimHandlerVec`,
      `GenesisSimHandlerVec`, `IsaacSimSimHandlerVec`. Verifiable via
      `test_sim_base.py` Protocol-conformance assertions.
* [ ] `TyGrit/envs/fetch/maniskill_vec.py` is **deleted** and its
      useful pieces are folded into `ManiSkillSimHandlerVec`.
* [ ] `FetchRobot.create(config)` with `config.num_envs > 1` routes to
      `SimHandlerVec` uniformly across all three sims; no sim name
      appears in `TyGrit/envs/` or in `TyGrit/rl/` dispatch.
* [ ] `TyGrit/worlds/backends/isaac_sim.py` exists and dispatches every
      source in `SOURCE_SIM_COMPATIBILITY` that lists `"isaac_sim"`.
* [ ] `SOURCE_SIM_COMPATIBILITY` updated with `"isaac_sim"` entries
      wherever a USD/URDF/MJCF path is reachable. If a source cannot
      be loaded in Isaac Sim without upstream changes, it must
      **not** list `isaac_sim` — `CLAUDE.md` Rule 1 forbids silent
      stubs.
* [ ] Tests:
  * `pixi run -e default test test/test_sim_base.py` — Protocol
    conformance for Isaac handler (tested without importing Isaac
    Lab).
  * `pixi run -e default test test/test_types_robot_cfg.py
    test/test_worlds_asset_sources.py` — existing suite still passes.
  * `pixi run -e isaacsim test test/test_sim_isaac.py` (new) — at
    least a smoke test per handler method on a minimal scene.
  * `pixi run -e world test test/test_envs_fetch_factory.py` (new or
    renamed) — exercises the new `create_sim_handler` dispatch for
    ManiSkill (+ Genesis if the test suite supports it).
  * Full pure-Python suite (default env) still passes — 299+ tests as
    of 2026-04-15.
* [ ] Both `maniskill` and `genesis` backends still work unchanged
      end-to-end; no regression on RL baseline (`pixi run -e rl
      python -m TyGrit.rl.train …` smoke).
* [ ] Codex has reviewed each of: (a) the envs cleanup diff, (b) the
      Isaac handler implementation, (c) the worlds/backends/isaac_sim.py
      dispatch, (d) the final `SimHandlerVec` decision. Disagreements
      are surfaced in the PR description.

---

## 8. Risks & blockers

* **Isaac Lab pixi env.** Per the 2026-04-15 memory the env solves with
  Isaac Sim 5.0 + Isaac Lab via git-source. Verify `pixi install -e
  isaacsim` succeeds today. If NVIDIA has shipped a new pillow / numpy
  pin that breaks it, stop and report — do not paper over with
  `pyproject` overrides (`CLAUDE.md` Rule 1).
* **Fetch USD.** If `FETCH_CFG.usd_path is None`, Isaac Sim falls back
  to URDF→USD via `UrdfConverterCfg`. First-run conversion is slow;
  add a `pixi run setup-fetch-usd` task and cache under
  `resources/fetch/usd/` if it hurts the dev loop.
* **Heterogeneous scenes in Isaac Sim.** Unlike ManiSkill3,
  `InteractiveScene` is homogeneous by default. For scene-pool
  sampling we either (a) rebuild the stage on `reset_to_scene_idx`
  (slow but correct) or (b) use `MultiUsdFileCfg` +
  `replicate_physics=False` (limited). Investigate with Codex before
  committing.
* **Vec heterogeneity across sims.** ManiSkill3 does true heterogeneous
  GPU sim (`set_scene_idxs(env_idx)` per actor build). Genesis has
  `envs_idx` for per-env ops but is narrower — confirm scope of
  scene-level heterogeneity before promising it. Isaac Lab defaults to
  `replicate_physics=True` (homogeneous); `MultiUsdFileCfg` +
  `replicate_physics=False` is the heterogeneous path but is limited
  to same-skeleton variants. Codex must produce a written decision for
  *each sim* on what per-env scene heterogeneity `SimHandlerVec`
  actually guarantees; do not promise a stronger contract in the
  Protocol than the weakest sim can deliver. When a sim cannot support
  per-env scene variety, say so in its handler docstring and raise on
  mismatched `idxs` rather than silently replicating (CLAUDE.md Rule 1).
* **Source compatibility lies.** `SOURCE_SIM_COMPATIBILITY` is the
  source of truth; per-backend frozensets assert on it. Keep them in
  lockstep, do not let them drift.

---

## 9. Phasing (suggested; adjust with the user before starting)

1. **Survey + plan.** Spawn two Codex agents in parallel: one for Isaac
   Lab API survey (§6 item 1), one for the RoboVerse handler diff (§6
   item 2). Claude runs `Grep`/`Read` over the current envs/sim/worlds
   trees to confirm §2 inventory. Report to user, get green-light.
2. **Envs cleanup.** Draft the `create_sim_handler` factory + config
   split. Delete `envs/fetch/maniskill.py`. Codex review. Commit as a
   dedicated PR; tests still green for ManiSkill + Genesis.
3. **Isaac Sim handler.** Implement one method at a time against a
   failing test, in the order: `robot_cfg` / `joint_name_to_idx`
   (trivial) → `reset_to_scene_idx` + `_build_scene` → `get_qpos` +
   `get_link_pose` → `apply_action` → cameras → teleport → navmesh.
   Mutual review per method.
4. **Worlds/backends/isaac_sim.py.** One source per commit, each with
   its own smoke test. Extend `SOURCE_SIM_COMPATIBILITY` per commit.
5. **Cross-sim smoke + PR.** Run the full test matrix; have Codex do a
   final diff review; open PR.

---

## 10. Rules of the road (recap — `CLAUDE.md` is authoritative)

* **No hacks.** No silent fallbacks, no broad `except`, no `type:
  ignore` / `cast` / `Any` to silence type errors.
* **Always pixi.** Never `python`, `python3`, `pytest`, `pip`. Use
  `pixi run test …`, `pixi run -e <env> test …`, `pixi add <pkg>`.
* **Match existing structure.** Pure dataclasses → `TyGrit/types/`.
  Pure functions → `TyGrit/utils/`. Per-sim handlers → `TyGrit/sim/`.
  Per-sim scene adapters → `TyGrit/worlds/backends/`. Do not invent
  parallel hierarchies.
* **Small modules.** Split past ~400 lines. Rule-of-3 before
  extracting helpers.
* **Commit cadence.** Atomic commits, each green. Never a
  rewrite-everything pass.

---

## 11. Reporting cadence

After each phase, post a short status to the user: what landed, what's
next, any open Codex disagreements. Keep it tight — a diff summary and
one blocker per report.

When done, update
`~/.claude/projects/-media-run-Work-TyGrit/memory/architecture_sim_refactor_2026_04_15.md`
with the end-state: file counts, new pixi tasks, new test files,
resolved / unresolved risks.
