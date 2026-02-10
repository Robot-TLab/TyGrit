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
| ------ | ----------- |
| `envs/` | Robot environment layer — ManiSkill 3 + ROS (Fetch); Isaac Sim & more robots planned |
| `planning/` | Motion planning — currently VAMP whole-body; `MotionPlanner` protocol supports any planner |
| `perception/` | GraspGen 6-DOF neural grasp prediction |
| `kinematics/` | IKFast (analytical) + TRAC-IK (numerical) solvers |
| `controller/` | MPC trajectory tracking + gripper control |
| `gaze/` | Robot-agnostic head tracking |
| `core/` | System frameworks — receding-horizon (implemented); plan-to-goal & reactive (planned) |

## Documentation

See the [documentation](https://robot-tlab.github.io/TyGrit/) for the full vision, architecture, and roadmap.

| | |
| - | - |
| [**Setup**](docs/setup.md) | Prerequisites, installation, environment details |
| [**Architecture**](docs/architecture.md) | Module design, data flow, key decisions |
| [**Configuration**](docs/configuration.md) | All TOML sections and parameters |
| [**Visualization**](docs/visualization.md) | MomaViz: Blender renders, ManiSkill replays, video |

## License

[MIT](LICENSE)
