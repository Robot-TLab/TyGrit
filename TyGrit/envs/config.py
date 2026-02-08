"""Abstract base configuration for all robot environments."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass


@dataclass(frozen=True)
class EnvConfig(ABC):
    """Abstract base for all environment configurations.

    Concrete robots define their own config subclass (e.g. ``FetchEnvConfig``)
    with robot-specific and backend-specific fields.

    Attributes
    ----------
    robot : str
        Which robot platform (``"fetch"``, later ``"spot"``, etc.).
    backend : str
        Which driver / simulator (``"maniskill"``, ``"ros"``, etc.).
    """

    robot: str = "fetch"
    backend: str = "maniskill"
    camera_width: int = 640
    camera_height: int = 480
