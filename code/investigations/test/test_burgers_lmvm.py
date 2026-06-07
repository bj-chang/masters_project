"""LMVM on Burgers control: does H^1 alone (no history hack) survive
on a non-quadratic problem?

On Poisson (quadratic in m) H^1 + default LMVM history converges in 2
iterations across the whole grading sweep - but that's potentially
misleading because any well-set-up quasi-Newton method handles a
quadratic in a few steps regardless of metric or history (Josh and
David's "too easy" observation about NLS applies just as well here).

Burgers is the harder, genuinely non-quadratic test:

  - L^2 + default history, tight gatol: the original failure case on
    Poisson. Does it stay failing on Burgers?
  - H^1 + default history, loose gatol: the alternative Josh suggested.
    Does it stay mesh-independent on Burgers?

The 1D Burgers control problem itself lives in
``meshdep.problems.burgers_control.solve_burgers_control``, exactly
paralleling ``meshdep.problems.poisson_control``.

Run from ~/masters_project/.
"""
import os
import sys

# Make ``meshdep`` importable when this script is run directly without
# PYTHONPATH being set: add the project's ``code/`` directory.
_PROJECT_CODE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
if _PROJECT_CODE not in sys.path:
    sys.path.insert(0, _PROJECT_CODE)

import numpy as np

from firedrake import *

from meshdep.problems.burgers_control import solve_burgers_control


def graded_interval(n=32, stretch=0.0):
    """1D analogue of graded_square: stretch the unit interval so cells
    cluster near x = 0."""
    mesh = UnitIntervalMesh(n)
    if stretch == 0.0:
        return mesh
    new = Function(mesh.coordinates.function_space())
    xs = mesh.coordinates.dat.data_ro
    new.dat.data[:] = (np.exp(stretch * xs) - 1.0) / (np.exp(stretch) - 1.0)
    return Mesh(new)


def h_ratio(mesh):
    DG0 = FunctionSpace(mesh, "DG", 0)
    h = Function(DG0).interpolate(CellDiameter(mesh)).dat.data_ro
    return h.max() / h.min()


def run(mesh, riesz_map, gatol, max_funcs=500):
    """Return a short string with iteration count + a converged/failed flag."""
    try:
        result = solve_burgers_control(
            mesh,
            riesz_map=riesz_map,
            tao_gatol=gatol,
            tao_max_funcs=max_funcs,
        )
        marker = "" if result['converged'] else " (no-conv)"
        return f"{result['iterations']:>4d}{marker}"
    except Exception as e:
        return f"FAIL: {e.__class__.__name__}"


STRETCHES = (0.0, 2.0, 3.0)


def sweep(label, riesz_map, gatol):
    print(f"\n=== {label} (riesz_map='{riesz_map}', gatol={gatol:.0e}) ===")
    print(f"{'stretch':>7}  {'h_ratio':>7}  {'iters':>20}")
    for stretch in STRETCHES:
        mesh = graded_interval(stretch=stretch)
        print(f"{stretch:7.1f}  {h_ratio(mesh):7.1f}  "
              f"{run(mesh, riesz_map, gatol):>20}", flush=True)


sweep("L2 + default LMVM history (the original failure case on Poisson)",
      riesz_map="L2", gatol=1.0e-7)
sweep("L2 + default LMVM history, looser gatol",
      riesz_map="L2", gatol=1.0e-5)
sweep("H1 + default LMVM history (Josh's alternative)",
      riesz_map="H1", gatol=1.0e-3)
