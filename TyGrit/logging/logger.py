"""Centralised logging for TyGrit.

Wraps **loguru** with:
- A custom ``VIZ`` level (15) between DEBUG (10) and INFO (20) for
  visualisation messages (point cloud renders, screenshots, plots).
- A ``configure()`` function that controls the global minimum level,
  per-module overrides, and VIZ toggle from one place.

Usage::

    from TyGrit.logging import log

    log.info("Scene updated with {} points", n)
    log.viz("Saved belief cloud to {}", path)    # only shown when level ≤ VIZ

    # At startup, set the desired verbosity once:
    from TyGrit.logging import configure
    configure(level="INFO")                       # hides DEBUG + VIZ
    configure(level="VIZ")                        # shows VIZ but hides DEBUG
    configure(level="DEBUG")                      # shows everything
    configure(level="INFO", modules={"TyGrit.perception": "DEBUG"})
"""

from __future__ import annotations

import sys
from typing import Callable

from loguru import logger

# ── Custom VIZ level ────────────────────────────────────────────────────────
# Sits between DEBUG (10) and INFO (20).  At ``level="INFO"`` the VIZ
# messages are suppressed, keeping production output clean.  Set
# ``level="VIZ"`` to opt-in during development.

VIZ_LEVEL_NO = 15
VIZ_LEVEL_NAME = "VIZ"

logger.level(VIZ_LEVEL_NAME, no=VIZ_LEVEL_NO, color="<magenta>", icon="~")

# ── Format string ───────────────────────────────────────────────────────────

_FORMAT = (
    "<green>{time:HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

# ── Internal state ──────────────────────────────────────────────────────────

_handler_ids: list[int] = []


def _make_filter(
    min_level: str,
    modules: dict[str, str] | None,
) -> Callable:
    """Build a loguru filter function.

    The filter enforces *min_level* globally, with optional per-module
    overrides (e.g. ``{"TyGrit.perception": "DEBUG"}`` to make one
    module more verbose while the rest stays at INFO).
    """
    # Pre-resolve numeric thresholds so the hot path is fast.
    global_no = logger.level(min_level.upper()).no

    module_nos: dict[str, int] = {}
    if modules:
        for mod, lvl in modules.items():
            module_nos[mod] = logger.level(lvl.upper()).no

    def _filter(record: dict) -> bool:
        name: str = record["name"] or ""
        record_no: int = record["level"].no

        # Check per-module overrides (longest prefix match).
        for mod, threshold in module_nos.items():
            if name == mod or name.startswith(mod + "."):
                return record_no >= threshold

        return record_no >= global_no

    return _filter


def configure(
    level: str = "DEBUG",
    modules: dict[str, str] | None = None,
) -> None:
    """(Re)configure the global logging output.

    Call this **once** at application startup.  Calling it again replaces
    the previous configuration entirely.

    Args:
        level: Minimum level for all modules.  One of
               ``"DEBUG"``, ``"VIZ"``, ``"INFO"``, ``"WARNING"``,
               ``"ERROR"``, ``"CRITICAL"``.
        modules: Optional per-module overrides.
                 Keys are dotted module names (e.g. ``"TyGrit.perception"``),
                 values are level strings.
    """
    # Remove previous handlers that we own.
    for hid in _handler_ids:
        logger.remove(hid)
    _handler_ids.clear()

    filt = _make_filter(level, modules)
    hid = logger.add(sys.stderr, format=_FORMAT, filter=filt, level=0)
    _handler_ids.append(hid)


# ── Bootstrap ───────────────────────────────────────────────────────────────
# Remove loguru's default handler and install ours so importing the module
# is enough to get nicely formatted output.

logger.remove()
configure(level="DEBUG")

# ── Public API ──────────────────────────────────────────────────────────────

log = logger
"""The logger instance.  Use ``log.debug / .viz / .info / .warning / .error``."""
