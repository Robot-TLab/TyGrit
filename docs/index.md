# TyGrit

**A Unified Platform for Mobile Manipulation Research**

One platform. Multiple robots, simulators, and control frameworks.
Standardized benchmarks for comparing classical, learning-based, and foundation-model approaches to mobile manipulation.

## What is TyGrit?

Mobile manipulation is one of the hardest open problems in robotics — a robot that can navigate a home, understand a scene, and grasp what it needs would be transformative. But research in this space is deeply fragmented: every group uses a different robot, a different simulator, and a different evaluation setup. Results don't compare. Insights don't transfer.

TyGrit is a single research platform where you can run the same tasks across multiple robots, multiple simulators, and multiple control frameworks — all sharing the same perception, planning, and evaluation interfaces. Swap in a learned policy for a classical planner and measure the difference on identical benchmarks.

The goal: stop reinventing the glue code and start answering the questions that matter — what actually works, where the real gaps are, and how different paradigms (classical planning, RL, VLAs, world models) compare when the task and the metrics are held constant.

## Key Features

- **Whole-body motion planning** — VAMP-based planner with an extensible `MotionPlanner` protocol
- **Neural grasp prediction** — GraspGen 6-DOF grasp synthesis with `GraspPredictor` protocol
- **Segmentation** — Ground-truth (sim) and SAM 3 backends with `Segmenter` protocol
- **Multi-backend environments** — ManiSkill 3 simulation via `RobotBase` protocol
- **Receding-horizon control loop** — scheduler with config-driven subgoal generation
- **Vendored C++ IK solvers** — IKFast (analytical) + TRAC-IK (numerical)
- **Visualization toolkit** — MomaViz: Blender renders, ManiSkill replays, video

::::{grid} 2
:gutter: 3

:::{grid-item-card} Getting Started
:link: setup
:link-type: doc

Prerequisites, installation, and environment setup.
:::

:::{grid-item-card} Architecture
:link: architecture
:link-type: doc

Extensible multi-robot, multi-sim design and data flow.
:::

:::{grid-item-card} Configuration
:link: configuration
:link-type: doc

All TOML sections and parameters.
:::

:::{grid-item-card} Visualization
:link: visualization
:link-type: doc

MomaViz: Blender renders, ManiSkill replays, video.
:::

::::

```{toctree}
:hidden:
:maxdepth: 2

setup
architecture
configuration
visualization
api/index
```
