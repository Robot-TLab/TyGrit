"""One-shot manifest generators.

Each submodule produces a JSON manifest under ``resources/worlds/``
from an upstream dataset (ReplicaCAD, HSSD, RoboCasa, ProcTHOR,
MolmoSpaces, …). Generators are **not** part of the runtime import
path of :mod:`TyGrit.worlds` — they are invoked manually as modules::

    pixi run -e world python -m TyGrit.worlds.generators.replicacad

and write their output to the repo tree, where it's committed as
a data artifact alongside the code. Regenerating a manifest is a
deliberate act: the output file is versioned and reviewers see the
diff.

Generators import their upstream package (e.g. :mod:`mani_skill`),
so they must run in the pixi env that has that package installed —
usually ``-e world`` for ManiSkill-shipped scene data.
"""
