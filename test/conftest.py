"""Shared pytest configuration.

Sets OMP_NUM_THREADS=1 to prevent LAPACK/MKL thread deadlocks when
PyTorch orthogonal init runs alongside ManiSkill's GPU workers.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
