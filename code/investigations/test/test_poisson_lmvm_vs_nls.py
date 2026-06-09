"""Poisson control: LMVM (with the history fix) vs NLS across a grading sweep.

Side-by-side check that both TAO solvers, set up the way the
dissertation actually recommends, are mesh-independent on the Poisson
control problem.

  - LMVM is run with the sec. 10.4 recipe: register the option
    ``-tao_lmvm_mat_lmvm_hist_size 100`` on the global PETSc options
    database before constructing the solver. Without this, the default
    history of five gives the mesh-dependent blow-up documented in
    Table 10.x.
  - NLS converges in roughly two iterations on every mesh because the
    Poisson control problem is quadratic in ``m`` - one exact-Newton
    step lands the minimum.

Expected outcome: both columns roughly flat across the grading sweep,
with NLS at ~2 iters and LMVM at ~30 iters. The point is not that
LMVM matches NLS (it cannot, on a quadratic), but that both are
mesh-independent at the dissertation's stopping threshold once LMVM is
configured correctly.

Run from ~/masters_project/.
"""
import numpy as np

from firedrake import *
from firedrake.adjoint import *
from firedrake.petsc import PETSc
from pyadjoint.optimization.tao_solver import TAOSolver

# Sec. 10.4 recipe: pyadjoint's parameters dict does not propagate down
# to the LMVM sub-Mat, so the history size has to be set on the global
# PETSc options database before any TAOSolver is constructed.
PETSc.Options().setValue('-tao_lmvm_mat_lmvm_hist_size', 100)

ALPHA = 1.0e-4
GATOL = 1.0e-7
MAX_IT = 2000


def graded_square(n=32, stretch=3.0):
    """Unit square mesh with cells bunched up toward ``(0, 0)``."""
    mesh = UnitSquareMesh(n, n)
    if stretch == 0.0:
        return mesh
    new = Function(mesh.coordinates.function_space())
    xy = mesh.coordinates.dat.data_ro
    new.dat.data[:] = (np.exp(stretch * xy) - 1.0) / (np.exp(stretch) - 1.0)
    return Mesh(new)


def h_ratio(mesh):
    DG0 = FunctionSpace(mesh, "DG", 0)
    h = Function(DG0).interpolate(CellDiameter(mesh)).dat.data_ro
    return h.max() / h.min()


def build_jhat(mesh):
    """Tape the forward Poisson solve and return the reduced functional."""
    x, y = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 1)
    u, v = Function(V), TestFunction(V)
    m = Function(V)
    F = inner(grad(u), grad(v)) * dx - m * v * dx
    bc = DirichletBC(V, 0.0, "on_boundary")
    d = Function(V).interpolate(sin(pi * x) * sin(pi * y))

    # LU forward solve so the gradient is computed to machine precision
    # and the gatol target isn't blocked by Krylov-solver noise.
    fwd_params = {'ksp_type': 'preonly', 'pc_type': 'lu'}

    m.zero()
    continue_annotation()
    with set_working_tape() as tape:
        solve(F == 0, u, bcs=bc, solver_parameters=fwd_params)
        J = assemble(0.5 * (u - d) ** 2 * dx + 0.5 * ALPHA * m ** 2 * dx)
        Jhat = ReducedFunctional(J, Control(m), tape=tape)
    pause_annotation()
    return Jhat


def run(Jhat, tao_type):
    """Run ``tao_type`` on ``Jhat`` and return the iteration count string."""
    params = {'tao_type': tao_type,
              'tao_gatol': GATOL,
              'tao_grtol': 0.0, 'tao_gttol': 0.0,
              'tao_max_it': MAX_IT}
    solver = TAOSolver(MinimizationProblem(Jhat), parameters=params)
    try:
        solver.solve()
        its = solver.tao.getIterationNumber()
        reason = solver.tao.getConvergedReason()
        return f"{its} (r={reason})"
    except Exception as e:
        return f"FAIL: {e.__class__.__name__}"


STRETCHES = (0.0, 3.0, 4.0, 5.0)


print(f"2D Poisson control, N=32, alpha={ALPHA}, gatol={GATOL:.0e}")
print(f"{'stretch':>7}  {'h_ratio':>7}  {'LMVM (hist=100)':>18}  "
      f"{'NLS':>14}")
for stretch in STRETCHES:
    mesh = graded_square(stretch=stretch)
    # Fresh tape per solver so neither inherits the other's iterate.
    print(f"{stretch:7.1f}  {h_ratio(mesh):7.1f}  running...", flush=True)
    lmvm_iters = run(build_jhat(mesh), 'lmvm')
    nls_iters = run(build_jhat(mesh), 'nls')
    print(f"{stretch:7.1f}  {h_ratio(mesh):7.1f}  "
          f"{lmvm_iters:>18}  {nls_iters:>14}", flush=True)
