<div align="center">

# TyGrit

**Autonomous Mobile Manipulation for the Fetch Robot**

[![CI](https://github.com/Robot-TLab/TyGrit/actions/workflows/ci.yml/badge.svg)](https://github.com/Robot-TLab/TyGrit/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-live-brightgreen?logo=readthedocs&logoColor=white)](https://robot-tlab.github.io/TyGrit/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![pixi](https://img.shields.io/badge/env-pixi-orange.svg)](https://pixi.sh/)

Whole-body planning, grasping, and reactive control on the Fetch mobile manipulator — with ManiSkill 3 simulation and ROS deployment backends.

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
|--------|-------------|
| `envs/` | ManiSkill 3 + ROS backends for the Fetch robot |
| `planning/` | VAMP-based whole-body motion planning (base + torso + 7-DOF arm) |
| `perception/` | GraspGen 6-DOF neural grasp prediction |
| `kinematics/` | IKFast (analytical) + TRAC-IK (numerical) solvers |
| `controller/` | MPC trajectory tracking + gripper control |
| `gaze/` | Lookahead-based head tracking |
| `core/` | Receding-horizon scheduler (observe → plan → control → step) |

## Documentation

<div align="center">
<h3>
<a href="https://robot-tlab.github.io/TyGrit/">
📖 robot-tlab.github.io/TyGrit
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
