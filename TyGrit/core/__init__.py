"""Core scheduler — the receding-horizon control loop.

The :class:`Scheduler` class is intentionally NOT re-exported at this
package level: it transitively imports
:mod:`TyGrit.subgoal_generator.tasks.grasp`, which pulls in optional
heavy dependencies (Pillow / GraspGen / segmentation models) that are
not present in the default pixi env. Importing :class:`Scheduler` at
package init would force every test that touches
``TyGrit.controller.fetch`` (which imports
:class:`TyGrit.core.config.ControllerFn`) to load the segmentation
stack — slow and broken in CI.

Callers wanting :class:`Scheduler` import it directly:

    from TyGrit.core.scheduler import Scheduler
"""
