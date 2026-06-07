"""Per Josh's other suggestion: try using the H^1 Riesz map on the
Control instead of the (default) L^2 one. This is an *alternative* to
the LMVM-history-size workaround - the question is whether changing
the inner product on the control space alone is enough to restore
mesh-independence without touching the LMVM internals at all.

Mathematically: with riesz_map="L2", pyadjoint installs M^{-1} as
LMVM's initial Hessian. With riesz_map="H1", it installs (M+K)^{-1}
instead (mass + stiffness). H^1 is a "stronger" inner product, so the
resulting Hessian approximation should be better conditioned on graded
meshes.

This script runs the SAME Poisson distributed-control problem as
minimal_reproducer.py, but with riesz_map="H1" and the *default* LMVM
history of 5 - i.e. NO history-size hack. If Josh's suggestion works,
iteration counts should already be flat across grading levels.
"""
import numpy as np
from firedrake import *
from firedrake.adjoint import *
from pyadjoint.optimization.tao_solver import TAOSolver

ALPHA = 1.0e-4


def graded_square(n=32, stretch=3.0):
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


def run_tao_h1(mesh, gatol=1.0e-7, gttol=0.0):
    """Same problem as minimal_reproducer.py, but Control uses riesz_map='H1'
    and we DON'T set tao_lmvm_mat_lmvm_hist_size - so LMVM uses its default
    history of 5."""
    x, y = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 1)
    u, v = Function(V), TestFunction(V)
    m = Function(V)
    a = inner(grad(u), grad(v)) * dx
    L = m * v * dx
    F = (a - L == 0)
    bc = DirichletBC(V, 0, "on_boundary")
    d = Function(V).interpolate(sin(pi * x) * sin(pi * y))
    fwd_params = {'ksp_type': 'preonly', 'pc_type': 'lu'}

    m.zero()
    continue_annotation()
    with set_working_tape() as tape:
        solve(F, u, bcs=bc, solver_parameters=fwd_params)
        J = assemble(0.5 * (u - d) ** 2 * dx + 0.5 * ALPHA * m ** 2 * dx)
        # >>> the only meaningful change from minimal_reproducer.py: H1 <<<
        Jhat = ReducedFunctional(J, Control(m, riesz_map="H1"), tape=tape)
    pause_annotation()

    tao_params = {'tao_type': 'lmvm',
                  'tao_gatol': gatol,
                  'tao_grtol': 0.0,
                  'tao_gttol': gttol,
                  'tao_max_it': 2000,
                  'tao_max_funcs': 2000,     # fail fast rather than grind forever
                  'tao_ls_type': 'unit'}
    solver = TAOSolver(MinimizationProblem(Jhat), parameters=tao_params)
    try:
        solver.solve()
        its = solver.tao.getIterationNumber()
        reason = solver.tao.getConvergedReason()
        return f"{its} (reason={reason})"
    except Exception as e:
        return f"failed: {e.__class__.__name__}"


# gatol=1e-7 with H1 is much stricter than gatol=1e-7 with L2, because
# ||g||_H1 >= ||g||_L2. Empirically on the uniform mesh, ||g||_H1 at the
# L2-converged optimum is somewhere in (1e-5, 1e-3). gatol=1e-3 is the
# loosest value that lets H1 converge at all; gatol=1e-4 gives a tighter
# but still fair stopping point.

for gatol in (1.0e-3, 1.0e-4):
    print(f"\n=== H1 Riesz, gatol={gatol:.0e}, default LMVM history (5) ===")
    print(f"{'stretch':>7}  {'h_ratio':>9}  {'iters (H1, hist=5)':>22}")
    for stretch in (0.0, 3.0, 4.0, 5.0, 10.0, 12.0):
        mesh = graded_square(stretch=stretch)
        print(f"{stretch:7.1f}  {h_ratio(mesh):9.1f}  "
              f"{run_tao_h1(mesh, gatol=gatol):>22}", flush=True)
