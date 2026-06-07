"""Burgers control test: does NLS misbehave on a nonlinear problem
(where the line search is actually exercised)?

Per Josh Hope-Collins's observation: the Poisson distributed-control problem
is quadratic in m, so an exact-Newton method (NLS) lands the minimum in one
step and the line search is never exercised. Switching to Burgers (nonlinear
in u, hence non-quadratic in m) gives the line search something to do.

Forward problem: 1D viscous Burgers on (0, 1),
    u_t + u u_x - nu u_xx = m,    u(t, 0) = u(t, 1) = 0,
    u(0, x) = sin(pi x).
Control: a spatial source m, constant in time.
Objective: J(m) = (1/2) int_0^T ||u(t) - d||^2 dt + (alpha/2) ||m||^2.

For each grading level we run NLS twice - with the default line search and
with tao_ls_type='unit' (no line search) - and print iteration counts and
TAO convergence reasons side-by-side.

Run from ~/masters_project/.
"""
import numpy as np
from firedrake import *
from firedrake.adjoint import *
from pyadjoint.optimization.tao_solver import TAOSolver

ALPHA = 1.0e-4
NU = 1.0e-3                  # stronger convection (Re ~ 1000)
T_FINAL = 0.5                # longer horizon -> nonlinearity has time to bite
N_STEPS = 25
DT = T_FINAL / N_STEPS
N_MESH = 32


def graded_interval(n=N_MESH, stretch=0.0):
    """1D analogue of graded_square: stretch the unit interval to cluster
    cells near x=0."""
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


def run_burgers_nls(mesh, ls_type=None):
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

    forward_params = {'snes_type': 'newtonls',
                      'ksp_type': 'preonly', 'pc_type': 'lu'}

    m.zero()
    continue_annotation()
    with set_working_tape() as tape:
        u_old.assign(u0)
        J_val = 0.0
        for _ in range(N_STEPS):
            solve(F == 0, u_new, bcs=bc, solver_parameters=forward_params)
            J_val = J_val + DT * assemble(0.5 * (u_new - d) ** 2 * dx)
            u_old.assign(u_new)
        J_val = J_val + assemble(0.5 * ALPHA * m ** 2 * dx)
        Jhat = ReducedFunctional(J_val, Control(m), tape=tape)
    pause_annotation()

    tao_params = {'tao_type': 'nls',
                  'tao_gatol': 1.0e-6,
                  'tao_grtol': 0.0, 'tao_gttol': 0.0,
                  'tao_max_it': 200}
    if ls_type:
        tao_params['tao_ls_type'] = ls_type

    tao = TAOSolver(MinimizationProblem(Jhat), parameters=tao_params)
    try:
        tao.solve()
        its = tao.tao.getIterationNumber()
        reason = tao.tao.getConvergedReason()
        return f"{its} (reason={reason})"
    except Exception as e:
        return f"failed: {e.__class__.__name__}"


print(f"1D Burgers control, N={N_MESH}, T={T_FINAL}, dt={DT}, "
      f"nu={NU}, alpha={ALPHA}, gatol=1e-6")
print(f"{'stretch':>8}  {'h_ratio':>8}  "
      f"{'NLS, default LS':>20}  {'NLS, unit LS':>20}")
for stretch in (0.0, 2.0, 3.0):
    mesh = graded_interval(stretch=stretch)
    hr = h_ratio(mesh)
    default = run_burgers_nls(mesh)
    unit = run_burgers_nls(mesh, ls_type='unit')
    print(f"{stretch:8.1f}  {hr:8.1f}  {default:>20}  {unit:>20}", flush=True)
