"""Build script for C++ extensions (IKFast + TRAC-IK).

Extensions are skipped when build dependencies are missing (e.g. docs env).
"""

import glob
import os

from setuptools import Extension, setup

# Resolve conda prefix for include/lib paths (works under pixi)
conda_prefix = os.environ.get("CONDA_PREFIX", "")
conda_include = os.path.join(conda_prefix, "include") if conda_prefix else ""
conda_lib = os.path.join(conda_prefix, "lib") if conda_prefix else ""

ext_modules = []

# IKFast: CPython C API, no extra deps — always buildable
ext_modules.append(
    Extension(
        "ikfast_fetch",
        sources=["ext/ikfast_fetch/ikfast_robot.cpp"],
        include_dirs=["ext/ikfast_fetch"],
        language="c++",
    )
)

# TRAC-IK: pybind11, needs KDL/NLopt/Eigen — skip when headers are missing
_kdl_header = os.path.join(conda_include, "kdl", "tree.hpp") if conda_include else ""
if _kdl_header and os.path.isfile(_kdl_header):
    trac_ik_sources = sorted(glob.glob("ext/trac_ik/**/*.cpp", recursive=True))

    try:
        import pybind11

        pybind11_include = pybind11.get_include()
    except ImportError:
        pybind11_include = ""

    trac_ik_include_dirs = [
        "ext/trac_ik",
        "ext/trac_ik/urdf",
        pybind11_include,
        conda_include,
        os.path.join(conda_include, "eigen3"),
    ]
    trac_ik_library_dirs = [conda_lib] if conda_lib else []

    ext_modules.append(
        Extension(
            "pytracik",
            sources=trac_ik_sources,
            include_dirs=trac_ik_include_dirs,
            library_dirs=trac_ik_library_dirs,
            libraries=["orocos-kdl", "nlopt"],
            language="c++",
            extra_compile_args=["-std=c++17"],
        )
    )

setup(ext_modules=ext_modules)
