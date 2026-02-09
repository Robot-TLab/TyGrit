#!/usr/bin/env bash
set -euo pipefail

DEST="resources/weights/graspgen"
BASE_URL="https://huggingface.co/adithyamurali/GraspGenModels/resolve/main/checkpoints"

mkdir -p "$DEST"

FILES=(
    graspgen_franka_panda.yml
    graspgen_franka_panda_dis.pth
    graspgen_franka_panda_gen.pth
)

for f in "${FILES[@]}"; do
    if [ -f "$DEST/$f" ]; then
        echo "Already exists: $DEST/$f"
    else
        echo "Downloading $f ..."
        wget -q --show-progress -O "$DEST/$f" "$BASE_URL/$f"
    fi
done

echo "Done. Weights saved to $DEST/"
