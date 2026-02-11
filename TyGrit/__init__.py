"""TyGrit — mobile manipulation research platform."""

import os as _os
import sys as _sys

# Compiled C extensions (ikfast_fetch, pytracik) live in build/.
_build_dir = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "build"
)
if _os.path.isdir(_build_dir) and _build_dir not in _sys.path:
    _sys.path.insert(0, _build_dir)
