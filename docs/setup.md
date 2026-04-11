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
| 2 | `pixi install` | Base conda + pypi environment for the default feature |
| 3 | `pixi run -e thirdparty install-vamp` | VAMP motion planner (C++ build) |
| 4 | `pixi run -e thirdparty install-momaviz` | MomaViz visualization toolkit |
| 5 | Symlink gripper config | Register Fetch gripper with GraspGen |
| 6 | `pixi run -e thirdparty install-graspgen` | pointnet2_ops + GraspGen package |
| 7 | `pixi run -e thirdparty download-graspgen-weights` | Pretrained GraspGen model weights |

All install-* tasks live in the `thirdparty` pixi feature because they pull
heavy build dependencies (cuda, torch, pybind11, C++ compilers). The default
feature stays minimal so `pixi run test` runs fast for pure-Python unit tests.

### ManiSkill scene assets

Scene datasets (ReplicaCAD, AI2THOR, RoboCasa) are cached at a **project-local**
path — `assets/maniskill/` — via the `MS_ASSET_DIR` env var set by the
`maniskill` pixi feature's activation. That way every sim's asset footprint
stays inside the repo tree and fresh clones show their missing-asset state
via `ls`.

```bash
# Downloads go to assets/maniskill/data/scene_datasets/<dataset>/
pixi run -e world download-replicacad   # ~500 MB, 6 main + 84 staging apts
pixi run -e world download-ai2thor      # ~15.6 GB, 12,235 THOR-family scenes
pixi run -e world download-robocasa     # ~8.2 GB,   120 kitchen layouts
pixi run -e world download-ycb          # ~25 MB,    78 tabletop objects
```

### Object meshes from Objaverse (optional)

For scale-up training, download a sampled slice of NVIDIA's GraspGen-curated
Objaverse-LVIS subset:

```bash
pixi run -e world generate-objaverse-objects --count 200
```

This grabs 200 UUIDs from `nvidia/PhysicalAI-Robotics-GraspGen`'s
`robotiq_2f_140/train.txt` split, downloads each mesh via the `objaverse` pip
package, and emits `resources/worlds/objects/objaverse.json`. Mesh cache
lives at `assets/objaverse/meshes/` (gitignored). See
`TyGrit/worlds/generators/objaverse.py` for `--gripper`/`--count`/`--seed`
options.

### Verify

```bash
# Default env — pure-Python unit tests (types, samplers, manifests)
pixi run test

# Full suite including sim-dependent tests (needs torch + mani-skill)
pixi run -e rl test
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

TyGrit uses a **one-env-per-purpose** layout composed from minimal features.
No kitchen-sink "dev" env — each task has its own home, and the shared
`maniskill` feature (torch + mani-skill + cuda-toolkit) is composed into
every env that needs the simulator.

| Environment | Composed features | Purpose |
| --- | --- | --- |
| `default` | base (python, numpy, scipy, loguru, pytest) | Pure-Python unit tests, daily work |
| `world` | `maniskill` + `world` (trimesh, objaverse) | `SpecBackedSceneBuilder`, manifest generators, world-layer tests |
| `rl` | `maniskill` + `rl` (wandb, tensordict) | RL training via `TyGrit.rl.train` |
| `thirdparty` | `maniskill` + `thirdparty` (C++ build stack, heavy pypi) | One-shot install-* tasks (VAMP, GraspGen, MomaViz, SAM3) |
| `lint` | `lint` (no default feature) | `pre-commit`, `black`, `ruff` |
| `docs` | `docs` (sphinx, furo, myst) | `sphinx-build` |
| `ci` | `ci` (slim matplotlib + C++ libs) | GitHub Actions slim test suite |

Never run `python`, `pytest`, `pip` against system Python — always go through
pixi. See the [CLAUDE.md](../CLAUDE.md) project rules for the full do/don't
list.
