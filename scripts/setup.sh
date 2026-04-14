#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# ── 1. Submodules ──────────────────────────────────────────────────
echo "Initialising git submodules..."
git submodule update --init --recursive

# ── 2. Pixi environment ───────────────────────────────────────────
echo "Installing pixi environment..."
pixi install

# ── 3. VAMP ────────────────────────────────────────────────────────
echo "Building and installing vamp_preview..."
pixi run install-vamp

# ── 4. MomaViz ────────────────────────────────────────────────────
echo "Installing MomaViz..."
pixi run install-momaviz

# ── 5. GraspGen gripper registration ──────────────────────────────
# GraspGen discovers grippers from thirdparty/GraspGen/config/grippers/.
# Symlink the Fetch gripper definition so GraspGen can find it.
GRIPPER_DST="thirdparty/GraspGen/config/grippers"
for ext in py yaml; do
    src="$ROOT_DIR/resources/grippers/fetch/fetch.$ext"
    dst="$GRIPPER_DST/fetch.$ext"
    if [ -e "$src" ] && [ ! -e "$dst" ]; then
        ln -sfv "$(realpath "$src")" "$dst"
    fi
done

# ── 6. GraspGen (pointnet2_ops + package) ─────────────────────────
echo "Building and installing GraspGen..."
pixi run install-graspgen

# ── 7. GraspGen weights ───────────────────────────────────────────
echo "Downloading GraspGen weights..."
pixi run download-graspgen-weights

# ── 8. AllenAI MJCF→Sapien loader (for Holodeck scenes) ──────────
# Pure-Python; --no-deps in the task skips the upstream's stated
# mani-skill-nightly requirement, which is a project-wide AllenAI
# preference and not an actual API dependency of the loader.
echo "Installing molmo_spaces_maniskill..."
pixi run -e world install-molmo-spaces-maniskill

echo "Done! Run 'pixi run test' to verify."
