# Visualization

TyGrit uses [MomaViz](https://github.com/Robot-TLab/MomaViz) for all visualization — a standalone package with no TyGrit dependencies.

## Installation

MomaViz is installed automatically by `scripts/setup.sh`, or manually:

```bash
pixi run install-momaviz
```

## Workflow

```
npy + json data ──▶ H5 episode ──▶ rendered frames ──▶ video
    (convert)           (blender / maniskill)        (video)
```

### 1. Convert recorded data to H5

```bash
momaviz convert --data-dir ./data/replanning_data -o episode.h5
```

Optionally include an external camera pose:

```bash
momaviz convert --data-dir ./data/replanning_data -o episode.h5 --camera-pose camera_pose.json
```

### 2. Inspect the H5 file

```bash
momaviz info episode.h5
```

### 3. Render stage snapshots (Blender)

Publication-quality renders with robot mesh, belief-map pointcloud, target object, and FOV cone:

```bash
momaviz blender \
    --input episode.h5 \
    --output-dir ./rendered_stages \
    --mesh-dir ./resources/fetch_ext/meshes \
    --width 1920 --height 1080 \
    --samples 64
```

Requires [Blender](https://www.blender.org/) on PATH.

### 4. Render trajectory replay (ManiSkill)

Frame-by-frame replay with external + first-person cameras:

```bash
momaviz maniskill \
    --input episode.h5 \
    --output-dir ./rendered_frames \
    --rt \
    --width 1920 --height 1080
```

Outputs `external/`, `first_person_rgb/`, and `first_person_depth/` subdirectories.

### 5. Compose video

```bash
momaviz video --frame-dir ./rendered_frames_rt/external -o trajectory.mp4 --fps 20
```

## Python API

```python
from momaviz import read_episode, write_episode

episode = read_episode("episode.h5")
print(episode.metadata.object_id)
print(episode.trajectory.config.shape)

for stage in episode.stages:
    print(stage.name, stage.belief_map.shape if stage.belief_map is not None else "—")
```

## H5 schema

See the [MomaViz README](https://github.com/Robot-TLab/MomaViz#h5-episode-schema-v10) for the full H5 schema specification.
