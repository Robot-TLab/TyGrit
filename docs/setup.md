# Setup

## Prerequisites

- **OS**: Linux x86_64
- **Python**: 3.11
- **GPU**: CUDA 12.x (for GraspGen, ManiSkill RT rendering, torch)
- **pixi**: [Install pixi](https://pixi.sh/) package manager

## Installation

```bash
git clone --recursive git@github.com:Robot-TLab/TyGrit.git
cd TyGrit
bash scripts/setup.sh
```

### What `setup.sh` does

| Step | Command | What it installs |
|------|---------|-----------------|
| 1 | `git submodule update --init --recursive` | VAMP, GraspGen, MomaViz |
| 2 | `pixi install` | Full conda + pypi environment |
| 3 | `pixi run install-vamp` | VAMP motion planner (C++ build) |
| 4 | `pixi run install-momaviz` | MomaViz visualization toolkit |
| 5 | Symlink gripper config | Register Fetch gripper with GraspGen |
| 6 | `pixi run install-graspgen` | pointnet2_ops + GraspGen package |
| 7 | `pixi run download-graspgen-weights` | Pretrained GraspGen model weights |

### ManiSkill assets

After setup, download simulation assets:

```bash
pixi run python -m mani_skill.utils.download_asset ReplicaCAD
pixi run python -m mani_skill.utils.download_asset ycb
```

### Verify

```bash
pixi run test
```

## Environment details

The environment is managed by [pixi](https://pixi.sh/) via [`pixi.toml`](../pixi.toml).

### Conda dependencies

`scipy`, `numpy`, `loguru`, `cmake`, `scikit-build-core`, `nanobind`, `pybind11`, `eigen`, `nlopt`, `orocos-kdl`, `cuda-toolkit 12.8`, `ros-humble-desktop`

### PyPI dependencies

`py-trees`, `torch 2.7`, `mani-skill 3`, `torchvision`, `hydra-core`, `trimesh`, `timm`, and more — see `pixi.toml` for the full list.

### Vendored C extensions

| Extension | Source | Build system |
|-----------|--------|-------------|
| `ikfast_fetch` | `ext/ikfast_fetch/` | setuptools (via `setup.py`) |
| `pytracik` | `ext/trac_ik/` | pybind11, deps: eigen, nlopt, orocos-kdl |

### Git submodules

| Submodule | Path | Purpose |
|-----------|------|---------|
| [VAMP](https://github.com/AdaCompNUS/vamp) | `thirdparty/vamp_preview/` | Motion planning (`import vamp_preview`) |
| [GraspGen](https://github.com/Robot-TLab/GraspGen) | `thirdparty/GraspGen/` | 6-DOF grasp prediction |
| [MomaViz](https://github.com/Robot-TLab/MomaViz) | `thirdparty/MomaViz/` | Episode visualization |

## Environments (pixi)

| Environment | Purpose | Command |
|-------------|---------|---------|
| `default` | Full dev environment (GPU + ROS) | `pixi run test` |
| `lint` | Pre-commit hooks only | `pixi run -e lint pre-commit run` |
| `ci` | CI (no GPU, headless) | `pixi run -e ci test` |
