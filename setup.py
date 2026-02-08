"""Build script for C++ extensions (IKFast + TRAC-IK)."""

import glob
import os

from setuptools import Extension, setup

# Resolve conda prefix for include/lib paths (works under pixi)
conda_prefix = os.environ.get("CONDA_PREFIX", "")
conda_include = os.path.join(conda_prefix, "include") if conda_prefix else ""
conda_lib = os.path.join(conda_prefix, "lib") if conda_prefix else ""

# Collect all .cpp files under ext/trac_ik/ (including urdf/)
trac_ik_sources = sorted(glob.glob("ext/trac_ik/**/*.cpp", recursive=True))

# pybind11 include path
try:
    import pybind11

    pybind11_include = pybind11.get_include()
except ImportError:
    pybind11_include = ""

trac_ik_include_dirs = [
    "ext/trac_ik",
    "ext/trac_ik/urdf",
    pybind11_include,
]
trac_ik_library_dirs = []

if conda_include:
    trac_ik_include_dirs.append(conda_include)
    trac_ik_include_dirs.append(os.path.join(conda_include, "eigen3"))
if conda_lib:
    trac_ik_library_dirs.append(conda_lib)

setup(
    ext_modules=[
        # IKFast: CPython C API, no extra deps
        Extension(
            "ikfast_fetch",
            sources=["ext/ikfast_fetch/ikfast_robot.cpp"],
            include_dirs=["ext/ikfast_fetch"],
            language="c++",
        ),
        # TRAC-IK: pybind11, needs KDL/NLopt/Eigen
        Extension(
            "pytracik",
            sources=trac_ik_sources,
            include_dirs=trac_ik_include_dirs,
            library_dirs=trac_ik_library_dirs,
            libraries=["orocos-kdl", "nlopt"],
            language="c++",
            extra_compile_args=["-std=c++17"],
        ),
    ],
)
