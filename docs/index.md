# TyGrit

**A Research Platform for Mobile Manipulation in Unknown Environments**

## The problem

Mobile manipulation in unknown, dynamic environments is one of the most important unsolved problems in robotics. A robot that can enter an unfamiliar room, understand what it sees, and physically interact with objects would be transformative. Yet there is not even a clear formulation for this problem — let alone a solution.

Existing mobile manipulation environments are mostly known and static: the robot knows the scene layout, the object positions, and the goal specification upfront. Navigation benchmarks introduce uncertainty but lack the complex physical interaction that manipulation demands. Neither captures the real challenge: acting under partial observability, in a world that changes as the robot interacts with it.

## What is TyGrit?

TyGrit is a research platform built around this problem. It provides the infrastructure to study mobile manipulation under uncertainty — from classical model-based approaches to reinforcement learning, from data generation for policy training to systematic comparison of different architectures.

This is not just an engineering assembly of known components. The hierarchical structure (high-level subgoal generation + low-level policy) is designed specifically for the unknown: the robot observes, commits to the most ambitious feasible goal, executes, and re-decides when the world model changes. Progression is emergent, not scripted.

TyGrit enables researchers to:

- **Study model-based approaches** — classical planning and control under partial observability
- **Train and evaluate learned policies** — RL, VLAs, world models on realistic manipulation tasks
- **Generate data** — produce diverse trajectories for offline policy learning
- **Compare architectures** — measure how different paradigms perform on identical tasks and metrics

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

Hierarchical policy design and module overview.
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
