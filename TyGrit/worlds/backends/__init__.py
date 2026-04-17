"""Per-simulator adapters for the world/scene layer.

Each module in this package translates sim-agnostic
:class:`~TyGrit.types.worlds.SceneSpec` entries into a specific
simulator's native scene representation. Every adapter imports its
own backend package (``mani_skill``, ``genesis``, ``isaaclab``), so
each is only importable in the pixi env that has that backend
installed.

This subpackage has **no top-level re-exports** on purpose: importing
``TyGrit.worlds.backends`` should not force a choice of backend, and
any re-export here would trigger an ``ImportError`` in pixi envs that
don't have the chosen sim. Callers import adapters via their full
path::

    from TyGrit.worlds.backends.maniskill import bind_specs, build_world
    # Later:
    from TyGrit.worlds.backends.genesis import build_world
    from TyGrit.worlds.backends.isaac_sim import build_world

The sim-agnostic pieces (:mod:`~TyGrit.worlds.manifest`,
:mod:`~TyGrit.worlds.sampler`) live one level up and stay importable
in the default env.
"""
