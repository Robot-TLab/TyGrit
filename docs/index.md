# TyGrit — Tying Grit

**A Research Platform for Mobile Manipulation in Unknown Environments**

TyGrit provides infrastructure to study mobile manipulation under uncertainty — where the robot discovers the world through interaction rather than receiving a complete model upfront. See {doc}`why-new-framework` for the problem formulation.

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

:::{grid-item-card} Why a New Framework
:link: why-new-framework
:link-type: doc

Why standard frameworks fail and what makes this problem fundamentally different.
:::

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

why-new-framework
setup
architecture
configuration
visualization
api/index
```
