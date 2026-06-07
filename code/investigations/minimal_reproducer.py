import numpy as np
from firedrake import *
from firedrake.adjoint import *
from pyadjoint.optimization.tao_solver import TAOSolver


# To increase max storage
from firedrake.petsc import PETSc
PETSc.Options().setValue('-tao_lmvm_mat_lmvm_hist_size', 100)

alpha = 1.0e-4


# Helper functions
def graded_square(n=32, stretch=3.0):
    """Unit square mesh with triangles bunched up toward (0, 0).

    ``n`` is the number of triangles per side of the uniform base mesh.
    ``stretch=0`` returns the uniform mesh; bigger values give stronger
    grading (``3`` gives h_max/h_min ~ 20, ``5`` ~ 130).
    """
    mesh = UnitSquareMesh(n, n)
    if stretch == 0.0:
        return mesh
    new = Function(mesh.coordinates.function_space())
    xy = mesh.coordinates.dat.data_ro
    # Smooth map of [0,1] back onto itself that squashes points near 0
    # and stretches them near 1, applied elementwise to every coord
    new.dat.data[:] = (np.exp(stretch * xy) - 1.0) / (np.exp(stretch) - 1.0)
    return Mesh(new)


def h_ratio(mesh):
    """The cell size ratio h_max / h_min of a mesh."""
    DG0 = FunctionSpace(mesh, "DG", 0)
    h = Function(DG0).interpolate(CellDiameter(mesh)).dat.data_ro
    return h.max() / h.min()


# TAO solver
def run_tao(mesh):
    """Solve the L^2 Poisson distributed-control problem on ``mesh`` with TAO/LMVM.

    Returns the number of TAO/LMVM iterations to convergence.
    """
    # Function space, variational form, target field and boundary conds
    x, y = SpatialCoordinate(mesh)

    V = FunctionSpace(mesh, "CG", 1)
    u, v = Function(V), TestFunction(V)
    m = Function(V)

    a = inner(grad(u), grad(v)) * dx
    L = m * v * dx
    F = (a - L == 0)
    bc = DirichletBC(V, 0, "on_boundary")

    d = Function(V).interpolate(sin(pi * x) * sin(pi * y))

    # Use LU here so the forward solve is basically exact. With the default
    # solver the gradient is noisy around 1e-5, so we wouldnt reach the 
    # 1e-7 tolerance we want
    params = {'ksp_type': 'preonly', 'pc_type': 'lu'}

    # Tape the forward solve and build the reduced functional
    m.zero()
    continue_annotation()
    with set_working_tape() as tape:
        solve(F, u, bcs=bc, solver_parameters=params)
        J = assemble(0.5 * (u - d)**2 * dx + 0.5 * alpha * m**2 * dx)
        Jhat = ReducedFunctional(J, Control(m), tape=tape)   # note Control(m) uses the L2 Riesz map by default
    pause_annotation()

    # Run LMVM with an absolute gradient tolerance. tao_gttol should stop
    # the optimiser before the slowdown shows up.

    # =========== CHANGED =====================
    # tao_params = {'tao_type': 'lmvm',
    #               'tao_gatol': 1.0e-7, 'tao_grtol': 0.0, 'tao_gttol': 0.0,
    #               'tao_max_it': 5000}

    # # first suggestion - unit LS:
    # tao_params = {'tao_type': 'lmvm',
    #               'tao_gatol': 1.0e-7, 'tao_grtol': 0.0, 'tao_gttol': 0.0,
    #               'tao_max_it': 5000,
    #               'tao_ls_type': 'unit'}

    # # second suggestion: looser gatol, relative gttol active,
    # # monitor on so we can see the per-iteration gradient trajectory.
    # tao_params = {'tao_type': 'lmvm',
    #             #   'tao_gatol': 1.0e-7, 'tao_grtol': 0.0, 'tao_gttol': 0.0,
    #               'tao_gatol': 1.0e-5, 'tao_grtol': 0.0, 'tao_gttol': 1.0e-7,
    #               'tao_max_it': 5000,
    #               'tao_ls_type': 'unit',
    #               'tao_monitor': None
    #               }
    
    tao_params = {'tao_type': 'lmvm',
              'tao_gatol': 1.0e-7, 'tao_grtol': 0.0, 'tao_gttol': 0.0,
            #   'tao_gatol': 1.0e-5, 'tao_grtol': 0.0, 'tao_gttol': 1.0e-7,
              'tao_max_it': 5000,
              'tao_ls_type': 'unit',
            #   'tao_monitor': None,
              'tao_lmvm_mat_lmvm_hist_size': 100, 
            #   'tao_view': None,          
              }
    
    # # (keep the existing tao_params dict, but drop the hist_size entry -
    # #  we're setting it the other way)
    # tao_params = {'tao_type': 'lmvm',
    #             'tao_gatol': 1.0e-7, 'tao_grtol': 0.0, 'tao_gttol': 0.0,
    #             'tao_max_it': 5000,
    #             'tao_ls_type': 'unit',
    #             # 'tao_monitor': None,
    #             'tao_view': None
    #             }

    # 
    # PETSc.Options().setValue('-tao_lmvm_mat_lmvm_hist_size', 100)

    
    # tao = TAOSolver(MinimizationProblem(Jhat), parameters=tao_params)
    tao = TAOSolver(MinimizationProblem(Jhat), parameters=tao_params,
                options_prefix="")
    tao.solve()
    return tao.tao.getIterationNumber()


# Print over a few grading levels

print(f"{'stretch':>7}  {'h_ratio':>7}  {'TAO iters':>9}")
# for stretch in (0.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 20.0, 25.0):
for stretch in (0.0, 3.0, 4.0, 5.0):
    mesh = graded_square(stretch=stretch)
    print(f"{stretch:7.1f}  {h_ratio(mesh):7.1f}  {run_tao(mesh):9d}", flush=True)
