<div align="center">

# TyGrit

**A Unified Platform for Mobile Manipulation Research**

[![CI](https://github.com/Robot-TLab/TyGrit/actions/workflows/ci.yml/badge.svg)](https://github.com/Robot-TLab/TyGrit/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-live-brightgreen?logo=readthedocs&logoColor=white)](https://robot-tlab.github.io/TyGrit/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![pixi](https://img.shields.io/badge/env-pixi-orange.svg)](https://pixi.sh/)

One platform. Multiple robots, simulators, and control frameworks.
Standardized benchmarks for comparing classical, learning-based, and foundation-model approaches to mobile manipulation.

</div>

---

## Why this project

Mobile manipulation is one of the hardest open problems in robotics — and one of the most exciting. A robot that can navigate a home, understand a scene, and grasp what it needs would be transformative. But research in this space is deeply fragmented: every group uses a different robot, a different simulator, and a different evaluation setup. Results don't compare. Insights don't transfer.

TyGrit exists to fix that.

It's a single research platform where you can run the same tasks across multiple robots (Fetch today, AutoLife next), multiple simulators (ManiSkill 3 now, Isaac Sim planned), and multiple control frameworks (receding-horizon, plan-to-goal, purely reactive) — all sharing the same perception, planning, and evaluation interfaces. Swap in a learned policy for a classical planner and measure the difference on identical benchmarks.

The goal is simple: build the infrastructure so researchers can stop reinventing the glue code and start answering the questions that matter — what actually works, where the real gaps are, and how different paradigms (classical planning, RL, VLAs, world models) compare when the task and the metrics are held constant.

## Quick start

```bash
git clone --recursive git@github.com:Robot-TLab/TyGrit.git
cd TyGrit
bash scripts/setup.sh
pixi run test
```

After setup, download ManiSkill assets:

```bash
pixi run python -m mani_skill.utils.download_asset ReplicaCAD
pixi run python -m mani_skill.utils.download_asset ycb
```

## Overview

| Module | What it does |
|--------|-------------|
| `envs/` | Robot environment layer — ManiSkill 3 + ROS (Fetch); Isaac Sim & more robots planned |
| `planning/` | Motion planning — currently VAMP whole-body; `MotionPlanner` protocol supports any planner |
| `perception/` | GraspGen 6-DOF neural grasp prediction |
| `kinematics/` | IKFast (analytical) + TRAC-IK (numerical) solvers |
| `controller/` | MPC trajectory tracking + gripper control |
| `gaze/` | Robot-agnostic head tracking |
| `core/` | System frameworks — receding-horizon (implemented); plan-to-goal & reactive (planned) |

## Roadmap

| Area | Current | Planned |
|------|---------|---------|
| Robots | Fetch | AutoLife |
| Simulators | ManiSkill 3 | Isaac Sim |
| Deployment | ROS (Fetch) | — |
| Frameworks | Receding-horizon | Plan-to-goal, purely reactive |
| Paradigms | Classical planning + neural grasping | Learned policies, VLAs, world models |
| Evaluation | Per-run metrics | Standardized benchmarks across difficulty levels & approaches |

## Documentation

<div align="center">
<h3>
<a href="https://robot-tlab.github.io/TyGrit/">
robot-tlab.github.io/TyGrit
</a>
</h3>
</div>

| | |
|-|-|
| [**Architecture**](docs/architecture.md) | Module design, data flow, key decisions |
| [**Setup**](docs/setup.md) | Prerequisites, installation, environment details |
| [**Configuration**](docs/configuration.md) | All TOML sections and parameters |
| [**Visualization**](docs/visualization.md) | MomaViz: Blender renders, ManiSkill replays, video |

## License

[MIT](LICENSE)
