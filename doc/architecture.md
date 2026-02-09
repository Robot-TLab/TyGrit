# Architecture

## Project structure

```
TyGrit/                  # Main Python package
├── types/               #   Frozen dataclasses (geometry, robot, sensor, planning)
├── utils/               #   Pure functions: math, transforms, pointcloud, depth
├── envs/                #   Robot environment layer
│   ├── base.py          #     RobotBase ABC
│   ├── config.py        #     EnvConfig ABC
│   └── fetch/           #     FetchRobot + ManiSkill / ROS backends
├── kinematics/
│   ├── trac_ik.py       #     TRAC-IK solver (vendored C extension)
│   └── fetch/           #     IKFast solver + FK + frame transforms
├── planning/
│   ├── motion_planner.py #    MotionPlanner protocol
│   ├── config.py        #     PlannerConfig, VampPlannerConfig
│   └── fetch/           #     VAMP-based planner (RRT-C, multi-layer, FCIT)
├── perception/
│   └── grasping/        #     GraspGenPredictor + config
├── controller/
│   └── fetch/           #     MPC tracking + gripper control (pure functions)
├── gaze/                #   Robot-agnostic gaze controller
├── scene/               #   Pointcloud scene representation
├── core/
│   └── scheduler.py     #     Receding-horizon loop
└── config.py            #   SystemConfig + TOML loader

ext/                     # Vendored C++ extensions
├── ikfast_fetch/        #   IKFast analytical IK (compiled via setup.py)
└── trac_ik/             #   TRAC-IK numerical IK (pybind11, deps: eigen, nlopt, kdl)

thirdparty/              # Git submodules
├── vamp_preview/        #   VAMP motion planner (AdaCompNUS/vamp fork)
├── GraspGen/            #   Neural grasp generation (Robot-TLab/GraspGen)
└── MomaViz/             #   Visualization toolkit (Robot-TLab/MomaViz)
```

## Data flow

```
observe          plan              control           step
  │                │                  │                 │
  │  pointcloud    │  trajectory      │  joint cmds     │
  │  + grasps      │  (T, 11)        │  via MPC        │
  ▼                ▼                  ▼                 ▼
┌──────┐     ┌──────────┐     ┌────────────┐     ┌──────────┐
│ Scene │────▶│ Planner  │────▶│ Controller │────▶│   Env    │
│      │     │ (VAMP)   │     │  (MPC)     │     │(ManiSkill│
│      │     │          │     │            │     │  or ROS) │
└──────┘     └──────────┘     └────────────┘     └──────────┘
    ▲              ▲                                   │
    │              │         ┌──────────┐              │
    │              └─────────│   Gaze   │◀─────────────┘
    │                        │Controller│         qpos feedback
    └────────────────────────└──────────┘
```

The scheduler runs this loop sequentially — no threads, no sleep.

## Key design decisions

| Decision | Rationale |
|----------|-----------|
| Quaternions `[x, y, z, w]` | SciPy / ROS convention throughout the codebase |
| Pure functions in `utils/` | No class state — data in, data out; easy to test |
| Result types for expected failures | Exceptions only for programming errors |
| Two-level env dispatch | `config.robot` selects robot, `config.backend` selects sim/real |
| `FetchRobot.create(config)` factory | Single public API, backend hidden behind ABC |
| No threads, no sleep | Scheduler is strictly sequential — deterministic replay |
| Vendored C extensions via setuptools | IKFast and TRAC-IK compiled at install time, no system deps |
| VAMP `set_base_params(theta, x, y)` | Quirky arg order — documented to avoid bugs |

## Environments

The `envs/` layer uses two-level dispatch:

```
create_env(config)
  └── config.robot == "fetch"
        └── FetchRobot.create(config)
              ├── config.backend == "maniskill" → ManiSkillFetchRobot
              └── config.backend == "ros"       → ROSFetchRobot
```

`FetchRobot` is the public API — all code operates against this abstract class.

## Kinematics

Two IK solvers are available, both compiled as C extensions:

| Solver | Type | Speed | Robustness | Extension |
|--------|------|-------|------------|-----------|
| IKFast | Analytical | ~10 μs | May miss solutions | `ikfast_fetch` |
| TRAC-IK | Numerical | ~1 ms | High success rate | `pytracik` |

Factory functions in `kinematics/fetch/ik.py` handle frame transforms between world, base, and torso frames.

## Planning

The planner uses [VAMP](https://github.com/AdaCompNUS/vamp) (from the AdaCompNUS fork) for collision-aware whole-body planning:

- **RRT-Connect** — fast single-query planning
- **Multi-layer RRT-C** — layered planning with progressive constraint tightening
- **FCIT** — feasibility-checked inverse trajectory

The trajectory config is 11-dimensional: `[x, y, θ, torso, shoulder_pan, shoulder_lift, upperarm_roll, elbow_flex, forearm_roll, wrist_flex, wrist_roll]`.
