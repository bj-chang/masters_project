import numpy as np
from firedrake import *
from firedrake.adjoint import *
from pyadjoint.optimization.tao_solver import TAOSolver

alpha = 1.0e-4


def graded_square(n=32, stretch=3.0):
    mesh = UnitSquareMesh(n, n)
    if stretch == 0.0:
        return mesh
    new = Function(mesh.coordinates.function_space())
    xy = mesh.coordinates.dat.data_ro
    new.dat.data[:] = (np.exp(stretch * xy) - 1.0) / (np.exp(stretch) - 1.0)
    return Mesh(new)


def run_nls(mesh):
    x, y = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 1)
    u, v = Function(V), TestFunction(V)
    m = Function(V)
    a = inner(grad(u), grad(v)) * dx
    L = m * v * dx
    F = (a - L == 0)
    bc = DirichletBC(V, 0, "on_boundary")
    d = Function(V).interpolate(sin(pi * x) * sin(pi * y))
    params = {'ksp_type': 'preonly', 'pc_type': 'lu'}
    m.zero()
    continue_annotation()
    with set_working_tape() as tape:
        solve(F, u, bcs=bc, solver_parameters=params)
        J = assemble(0.5 * (u - d)**2 * dx + 0.5 * alpha * m**2 * dx)
        Jhat = ReducedFunctional(J, Control(m), tape=tape)
    pause_annotation()
    # tao_params = {'tao_type': 'nls',
    #               'tao_gatol': 1.0e-7, 'tao_grtol': 0.0, 'tao_gttol': 0.0,
    #               'tao_max_it': 500, 'tao_ls_type': 'unit'}
    tao_params = {'tao_type': 'nls',
                  'tao_gatol': 1.0e-7, 'tao_grtol': 0.0, 'tao_gttol': 0.0,
                  'tao_max_it': 500}
    tao = TAOSolver(MinimizationProblem(Jhat), parameters=tao_params)
    try:
        tao.solve()
        its = tao.tao.getIterationNumber()
        reason = tao.tao.getConvergedReason()
        return f"{its} (reason={reason})"
    except Exception as e:
        return f"failed: {e.__class__.__name__}"


print(f"{'stretch':>8}  {'NLS, unit LS':>22}")
for s in (0.0, 3.0, 4.0, 5.0):
    res = run_nls(graded_square(stretch=s))
    print(f"{s:8.1f}  {res:>22}", flush=True)
