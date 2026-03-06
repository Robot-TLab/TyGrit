<div align="center">

# TyGrit — Tying Grit

**A Research Platform for Mobile Manipulation in Unknown Environments**

[![CI](https://img.shields.io/github/actions/workflow/status/Robot-TLab/TyGrit/ci.yml?branch=main&style=for-the-badge&label=CI&logo=github)](https://github.com/Robot-TLab/TyGrit/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue?style=for-the-badge&logo=materialformkdocs)](https://robot-tlab.github.io/TyGrit/)
[![Build Docs](https://img.shields.io/github/actions/workflow/status/Robot-TLab/TyGrit/docs.yml?branch=main&style=for-the-badge&label=docs%20build&logo=github)](https://github.com/Robot-TLab/TyGrit/actions/workflows/docs.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)](LICENSE)
[![pixi](https://img.shields.io/badge/env-pixi-orange?style=for-the-badge)](https://pixi.sh/)

Mobile manipulation in unknown, dynamic environments is one of the most important unsolved problems in robotics. TyGrit provides the infrastructure to study this problem — from classical model-based approaches to reinforcement learning, from data generation to systematic comparison of different architectures.

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
| `envs/` | Robot environment layer — ManiSkill 3 (Fetch) via `RobotBase` protocol |
| `planning/` | Whole-body motion planning — VAMP-based, `MotionPlanner` protocol |
| `perception/` | Grasp prediction (`GraspPredictor`) and segmentation (`Segmenter`) |
| `subgoal_generator/` | High-level goal selection — `SubGoalGenerator` protocol |
| `kinematics/` | IKFast (analytical) + TRAC-IK (numerical) solvers |
| `controller/` | MPC trajectory tracking + gripper control |
| `scene/` | World model / belief state — `Scene` protocol |
| `core/` | Receding-horizon scheduler |

## Documentation

See the [documentation](https://robot-tlab.github.io/TyGrit/) for architecture and configuration details.

| | |
| - | - |
| [**Setup**](docs/setup.md) | Prerequisites, installation, environment details |
| [**Architecture**](docs/architecture.md) | Hierarchical policy design and module overview |
| [**Configuration**](docs/configuration.md) | All TOML sections and parameters |
| [**Visualization**](docs/visualization.md) | MomaViz: Blender renders, ManiSkill replays, video |

## License

[MIT](LICENSE)
