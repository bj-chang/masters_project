"""Per Lawrence Mitchell's question: monitor BOTH the L^2 and l^2 norms of
the gradient at every TAO iteration, to see whether they are both converging
or only one of them.

Uses TAO's setMonitor callback to independently compute, at each outer
iteration:
  - L^2 norm of the gradient: sqrt(int |g|^2 dx) via the L^2 Riesz
    representation. This is the function-space norm (what TAO already
    reports as "Residual" in its built-in monitor);
  - l^2 norm of the dual coefficient vector: sqrt(g^T g), the raw
    Euclidean coefficient norm.
"""
import numpy as np

from firedrake import *
from firedrake.adjoint import *
from pyadjoint import MinimizationProblem
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


def run_with_norm_monitor(mesh, label):
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
        Jhat = ReducedFunctional(J, Control(m), tape=tape)
    pause_annotation()

    tao_params = {'tao_type': 'lmvm',
                  'tao_gatol': 1.0e-5,
                  'tao_grtol': 0.0,
                  'tao_gttol': 1.0e-7,
                  'tao_max_it': 5000,
                  'tao_ls_type': 'unit'}
    solver = TAOSolver(MinimizationProblem(Jhat), parameters=tao_params)

    # Custom monitor: at every outer iteration, pull the current iterate
    # out of TAO, recompute the gradient via the tape, and print both norms.
    m_buf = Function(V)

    def monitor(tao):
        it = tao.getIterationNumber()
        x_vec = tao.getSolution()
        m_buf.dat.data[:] = x_vec.getArray(readonly=True)
        Jhat(m_buf)
        g_dual = Jhat.derivative()
        g_primal = g_dual.riesz_representation(
            riesz_map="L2",
            solver_options={'ksp_type': 'preonly', 'pc_type': 'lu'},
        )
        L2 = float(assemble(inner(g_primal, g_primal) * dx)) ** 0.5
        l2 = float(np.linalg.norm(np.asarray(g_dual.dat.data_ro)))
        print(f"    iter {it:3d}    L2 = {L2:.6e}    l2 = {l2:.6e}",
              flush=True)

    solver.tao.setMonitor(monitor)

    print(f"\n=== {label} ===")
    solver.solve()
    print(f"    -> {solver.tao.getIterationNumber()} outer iters, "
          f"reason = {solver.tao.getConvergedReason()}")


for stretch in (0.0, 3.0, 5.0):
    mesh = graded_square(stretch=stretch)
    label = f"stretch={stretch}  (h_ratio ~ "
    DG0 = FunctionSpace(mesh, "DG", 0)
    h = Function(DG0).interpolate(CellDiameter(mesh)).dat.data_ro
    label += f"{h.max() / h.min():.1f})"
    run_with_norm_monitor(mesh, label)
