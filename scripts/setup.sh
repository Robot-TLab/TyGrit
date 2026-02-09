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

echo "Done! Run 'pixi run test' to verify."
