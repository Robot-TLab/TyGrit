#!/usr/bin/env python3
import tomli
import tomli_w

# Read pixi.toml
with open("pixi.toml", "rb") as f:
    pixi = tomli.load(f)

# Read pyproject.toml
with open("pyproject.toml", "rb") as f:
    pyproject = tomli.load(f)

# Extract dependencies from pixi (excluding python and dev deps)
deps = []
for pkg, version in pixi.get("dependencies", {}).items():
    if pkg != "python":
        # Convert pixi version format to pip format
        if version == "*":
            deps.append(pkg)
        else:
            deps.append(f"{pkg}{version}")

# Update pyproject.toml
pyproject["project"]["dependencies"] = deps

# Write back
with open("pyproject.toml", "wb") as f:
    tomli_w.dump(pyproject, f)

print(f"Synced {len(deps)} dependencies to pyproject.toml")
