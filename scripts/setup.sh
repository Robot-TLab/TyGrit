#!/usr/bin/env bash
set -euo pipefail

echo "Installing pixi environment..."
pixi install

echo "Building and installing vamp_preview..."
pixi run install-vamp

echo "Done! Run 'pixi run test' to verify."
