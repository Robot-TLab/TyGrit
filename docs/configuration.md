# Configuration

TyGrit uses a single TOML file to configure all subsystems. See [`config/grasp_example.toml`](../config/grasp_example.toml) for an example.

## Loading

```python
from TyGrit.config import load_config

config = load_config("config/my_config.toml")
# config.env, config.scene, config.gaze, config.grasping,
# config.segmentation, config.planner, config.subgoal,
# config.mpc, config.scheduler
```

Only sections present in the TOML are overridden; missing sections use defaults.

## Sections

### `[scene]` — Pointcloud scene processing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ground_z_threshold` | float | 0.3 | Points below this Z are ground |
| `depth_range` | [float, float] | [0.2, 3.0] | Valid depth range (metres) |
| `enable_ground_filter` | bool | true | Remove ground points |
| `merge_radius` | float | 0.03 | Merge radius for scene updates |
| `downsample_voxel_size` | float | 0.05 | Voxel grid downsample size |
| `crop_radius` | float | 2.5 | Keep points within this radius |

### `[gaze]` — Head tracking controller

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lookahead_window` | int | 80 | Number of trajectory steps to look ahead |
| `decay_rate` | float | 0.99 | Temporal decay for joint priorities |
| `velocity_weight` | float | 1.0 | Weight for velocity-based attention |

#### `[gaze.joint_priorities]`

Per-joint attention weights:

| Joint | Default |
|-------|---------|
| `base` | 3.0 |
| `torso` | 2.0 |
| `shoulder_lift` | 1.3 |
| `elbow` | 1.1 |
| `wrist_flex` | 1.0 |
| `gripper` | 1.0 |

### `[grasping]` — GraspGen predictor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `checkpoint_config_path` | str | — | Path to GraspGen weights YAML |
| `num_grasps` | int | 200 | Number of grasp candidates to generate |
| `topk_num_grasps` | int | 100 | Top-K grasps to keep |
| `min_grasps` | int | 40 | Minimum viable grasps |
| `max_tries` | int | 6 | Retry attempts |
| `remove_outliers` | bool | true | Filter outlier grasps |
| `score_threshold` | float | 0.0 | Minimum grasp score |
| `max_grasps` | int | 50 | Final output cap |
| `device` | str | "cuda" | Torch device |

### `[segmentation]` — Segmentation backend

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | str | "sim" | Backend: `"sim"` (ground-truth) or `"sam3"` (SAM 3) |

Additional parameters when `backend = "sam3"`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | "sam3_hiera_large" | SAM 3 model variant |
| `text_prompt` | str | "object" | Text prompt for segmentation |
| `threshold` | float | 0.5 | Score threshold |
| `mask_threshold` | float | 0.5 | Mask binarization threshold |
| `device` | str | "cuda" | Torch device |

### `[subgoal]` — Subgoal generator

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task` | str | "grasp" | Task type: `"grasp"` |

Additional parameters when `task = "grasp"`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `approach_threshold` | float | 0.15 | Joint-space distance to consider approach complete |
| `lift_height` | float | 0.15 | Torso raise for lift phase |
| `lift_threshold` | float | 0.15 | Joint-space distance to consider lift complete |

### `[planner]` — VAMP motion planner

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `timeout` | float | 5.0 | Planning timeout (seconds) |
| `point_radius` | float | 0.03 | Collision sphere radius for pointcloud obstacles |

### `[mpc]` — Model predictive controller

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gain` | float | 2.5 | Overall tracking gain |
| `v_max` | float | 5.0 | Max linear velocity |
| `w_max` | float | 5.0 | Max angular velocity |
| `state_weights` | [float; 11] | [20, 20, 15, 12, ...] | Per-DOF state error weights |
| `control_weights` | [float; 10] | [0.5, 0.8, 1, ...] | Per-DOF control effort weights |
| `joint_vel_max` | [float; 8] | [2, 7, 7, ...] | Joint velocity limits (torso + 7 arm) |

### `[scheduler]` — Receding-horizon loop

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `steps_per_iteration` | int | 10 | Simulation steps per control iteration |
| `waypoint_lookahead` | int | 2 | Waypoints to look ahead on the path |
