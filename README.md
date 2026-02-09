# TyGrit

[![CI](https://github.com/Robot-TLab/TyGrit/actions/workflows/ci.yml/badge.svg)](https://github.com/Robot-TLab/TyGrit/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Pixi](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Robotics playground for mobile manipulator.

## Installation

The project is managed by [pixi](https://prefix.dev/). Ensure you have `pixi` installed. Clone the repository and run:

```bash
pixi install
```

## ManiSkill Assets

After installation, download the required ManiSkill scene and object assets:

```bash
pixi run python -m mani_skill.utils.download_asset ReplicaCAD
pixi run python -m mani_skill.utils.download_asset ycb
```

## Usage

This project is installed as an editable package. You can import it in python:

```python
import TyGrit
```

## Development

Run tests using the defined task:

```bash
pixi run test
```
