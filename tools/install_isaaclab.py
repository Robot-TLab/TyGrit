"""One-shot Isaac Lab install driven from the isaacsim pixi env.

Runs as ``pixi run -e isaacsim install-isaaclab``. The pixi task points
at this script so we get real Python control flow (conditionals, error
handling) that the pixi task DSL can't express — the previous bash
``if/then/fi`` form in pixi.toml was rejected by pixi's task parser.

What it does, in order:

1. Clone ``isaac-sim/IsaacLab @ v2.3.1`` under ``thirdparty/IsaacLab``
   if absent. Idempotent: skipped on subsequent runs.
2. Bootstrap pip into ``$CONDA_PREFIX`` (the isaacsim pixi env)
   because the env intentionally doesn't declare conda's pip — see
   the comment in pixi.toml's ``[feature.isaacsim.dependencies]``.
3. Install ``warp-lang`` and ``isaaclab`` from the cloned source with
   ``pip install --no-build-isolation --no-deps``. ``--no-deps``
   sidesteps the upstream ``pillow==11.3.0`` conflict (isaacsim 5.0
   pins ``11.2.1``); ``--no-build-isolation`` lets ``flatdict==4.0.1``
   build using the env's setuptools instead of failing on its
   undeclared ``pkg_resources`` import.

Exits non-zero on any subprocess failure.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ISAACLAB_REPO = "https://github.com/isaac-sim/IsaacLab.git"
ISAACLAB_TAG = "v2.3.1"
ISAACLAB_DIR = Path("thirdparty/IsaacLab")


def run(cmd: list[str], **kwargs: object) -> None:
    """Run ``cmd`` and exit on non-zero status, printing the command."""
    print("→", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, **kwargs)
    if rc != 0:
        sys.exit(f"command failed (exit {rc}): {' '.join(cmd)}")


def main() -> None:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix is None:
        sys.exit(
            "install_isaaclab.py: CONDA_PREFIX is unset — run via "
            "`pixi run -e isaacsim install-isaaclab` so the pixi env's "
            "interpreter is active."
        )
    python = Path(conda_prefix) / "bin" / "python"
    if not python.exists():
        sys.exit(f"install_isaaclab.py: python not found at {python}")

    if not ISAACLAB_DIR.exists():
        run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                ISAACLAB_TAG,
                ISAACLAB_REPO,
                str(ISAACLAB_DIR),
            ]
        )
    else:
        print(f"→ {ISAACLAB_DIR} already present, skipping clone", flush=True)

    # Bootstrap pip — the isaacsim pixi env intentionally omits conda's
    # pip to avoid resolver conflicts (see pixi.toml).
    try:
        subprocess.check_call([str(python), "-m", "pip", "--version"])
    except subprocess.CalledProcessError:
        run([str(python), "-m", "ensurepip", "--upgrade"])

    # warp-lang is declared as a pypi-dep in pixi.toml so it should
    # already be present, but install defensively in case the env was
    # built before that line landed.
    run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "warp-lang>=1.12,<2",
        ]
    )

    isaaclab_pkg = ISAACLAB_DIR / "source" / "isaaclab"
    if not isaaclab_pkg.is_dir():
        sys.exit(f"isaaclab source dir missing at {isaaclab_pkg}")

    run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "--no-deps",
            "-e",
            str(isaaclab_pkg),
        ]
    )

    print("✓ isaaclab installed into", conda_prefix, flush=True)


if __name__ == "__main__":
    main()
