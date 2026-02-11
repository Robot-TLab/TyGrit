# Architecture

## Core idea

Mobile manipulation in unknown environments is hard because the robot cannot plan everything upfront — it discovers the world as it acts. TyGrit addresses this with a two-level hierarchical policy:

- **Subgoal generator** (high-level) — decides *what* to do next: where to look, where to move, what to grasp. It re-evaluates every time the low-level policy finishes or the world model changes.
- **Low-level policy** (planning + control) — decides *how* to get there: collision-free motion planning, trajectory tracking, gripper control.

The key insight is that progression is **emergent, not scripted**. The subgoal generator does not commit to a fixed sequence. It observes, picks the most ambitious feasible goal, and lets the low-level policy execute. When new information arrives (a better grasp candidate, an obstacle, a failed plan), the subgoal generator simply re-decides. This makes the system robust to the unknown — partial observability, moving objects, failed grasps — without explicit replanning logic.

## Design principles

| Principle | How |
|-----------|-----|
| Protocol-first | Every module defines a `Protocol`; concrete classes satisfy it structurally |
| Config-driven dispatch | TOML config selects the concrete implementation via factory functions |
| Pure functions in `utils/` | No class state — data in, data out |
| Result types for expected failures | Exceptions only for programming errors |
| No threads, no sleep | Scheduler is strictly sequential — deterministic replay |

## Module overview

| Module | Responsibility |
|--------|---------------|
| `types/` | Frozen dataclasses — geometry, robot state, sensor snapshots, planning types |
| `utils/` | Pure functions — math, transforms, pointcloud, depth |
| `envs/` | Robot environment layer with two-level dispatch (robot → backend) |
| `kinematics/` | IK and FK solvers (IKFast analytical, TRAC-IK numerical, batch FK) |
| `planning/` | Motion planning (`MotionPlanner` protocol, VAMP-based implementation) |
| `perception/` | Grasp prediction (`GraspPredictor` protocol) and segmentation (`Segmenter` protocol) |
| `scene/` | World model / belief state (`Scene` protocol, point-cloud implementation) |
| `subgoal_generator/` | High-level goal selection (`SubGoalGenerator` protocol, grasp task) |
| `controller/` | Tracking control (MPC) and gripper commands |
| `gaze/` | Active gaze — where the head should look during trajectory execution |
| `checker/` | Collision and self-occlusion checks |
| `core/` | Receding-horizon scheduler |
| `config.py` | `SystemConfig` aggregator + TOML loader |

## Factory pattern

All modules with multiple implementations follow the same pattern:

1. **Protocol** in `protocol.py` — defines the interface
2. **Config hierarchy** in `config.py` — base config with discriminator field, concrete subclass per implementation
3. **Factory function** in `__init__.py` — dispatches on the discriminator to build the right implementation
4. **TOML section** — config loader auto-selects the concrete config subclass
