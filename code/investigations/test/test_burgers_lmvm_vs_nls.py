"""Burgers control: do LMVM and NLS both stay mesh-independent under grading?

The dissertation diagnosis (sec. 10.8) is that the LMVM blow-up
observed on Poisson does NOT transfer to Burgers - the backward-Euler
time stepping preconditions the high-frequency modes of the discrete
Hessian. This script tests both solvers on the same 1D Burgers control
problem across a grading sweep and prints a side-by-side iteration
table.

LMVM uses the default history (5). NLS uses the default line search.
Both target ``gatol = 1e-7``. The hypothesis is that on Burgers, both
solvers converge in roughly constant iteration count across stretches,
unlike Poisson where LMVM-with-default-history grows from ~130 to
several hundred iters as grading increases.

Run from ~/masters_project/.
"""
import numpy as np

from firedrake import *
from firedrake.adjoint import *
from pyadjoint.optimization.tao_solver import TAOSolver

ALPHA = 1.0e-4
NU = 1.0e-3
T_FINAL = 0.5
N_STEPS = 25
DT = T_FINAL / N_STEPS
N_MESH = 32
GATOL = 1.0e-7
MAX_IT = 500


def graded_interval(n=N_MESH, stretch=0.0):
    """Stretch the unit interval to cluster cells near x=0."""
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


def build_jhat(mesh):
    """Tape the forward Burgers solve and return the reduced functional."""
    x = SpatialCoordinate(mesh)[0]
    V = FunctionSpace(mesh, "CG", 1)

    u_old = Function(V)
    u_new = Function(V)
    m = Function(V)
    v = TestFunction(V)

    u0 = Function(V).interpolate(sin(pi * x))
    d = Function(V).interpolate(0.5 * sin(pi * x))
    bc = DirichletBC(V, 0, "on_boundary")

    dt_c = Constant(DT)
    F = ((u_new - u_old) / dt_c) * v * dx \
        + u_new * u_new.dx(0) * v * dx \
        + NU * u_new.dx(0) * v.dx(0) * dx \
        - m * v * dx

    fwd_params = {'snes_type': 'newtonls',
                  'ksp_type': 'preonly', 'pc_type': 'lu'}

    m.zero()
    continue_annotation()
    with set_working_tape() as tape:
        u_old.assign(u0)
        J_val = 0.0
        for _ in range(N_STEPS):
            solve(F == 0, u_new, bcs=bc, solver_parameters=fwd_params)
            J_val = J_val + DT * assemble(0.5 * (u_new - d) ** 2 * dx)
            u_old.assign(u_new)
        J_val = J_val + assemble(0.5 * ALPHA * m ** 2 * dx)
        Jhat = ReducedFunctional(J_val, Control(m), tape=tape)
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


STRETCHES = (0.0, 2.0, 3.0, 4.0)


print(f"1D Burgers control, N={N_MESH}, T={T_FINAL}, dt={DT}, "
      f"nu={NU}, alpha={ALPHA}, gatol={GATOL:.0e}")
print(f"{'stretch':>7}  {'h_ratio':>7}  {'LMVM (hist=5)':>18}  "
      f"{'NLS (default LS)':>20}")
for stretch in STRETCHES:
    mesh = graded_interval(stretch=stretch)
    # Build a fresh tape per solver to keep the comparison clean.
    lmvm_iters = run(build_jhat(mesh), 'lmvm')
    nls_iters = run(build_jhat(mesh), 'nls')
    print(f"{stretch:7.1f}  {h_ratio(mesh):7.1f}  "
          f"{lmvm_iters:>18}  {nls_iters:>20}", flush=True)
