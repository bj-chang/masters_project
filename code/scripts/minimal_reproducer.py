import numpy as np
from firedrake import *
from firedrake.adjoint import *
from pyadjoint.optimization.tao_solver import TAOSolver

alpha = 1.0e-4


# Helper functions
def graded_square(n=32, stretch=3.0):
    """A unit square mesh with triangles bunched up toward (0, 0).

    :param n: The number of triangles along each side of the uniform base mesh.
    :param stretch: How aggressively to bunch the triangles. ``stretch=0``
        skips the remap and returns the uniform mesh; bigger values give
        stronger grading (``3`` gives h_max/h_min ~ 20, ``5`` ~ 130).
    :returns: a :class:`~firedrake.Mesh` whose triangles are bunched up
        toward (0, 0).
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
    """The cell size ratio h_max / h_min of a mesh.

    :param mesh: The :class:`~firedrake.Mesh` to measure.
    :returns: a ``float`` equal to the largest cell diameter divided by
        the smallest.
    """
    DG0 = FunctionSpace(mesh, "DG", 0)
    h = Function(DG0).interpolate(CellDiameter(mesh)).dat.data_ro
    return h.max() / h.min()


# TAO solver
def run_tao(mesh):
    """Solve the L^2 Poisson distributed-control problem on a mesh with TAO/LMVM.

    :param mesh: The :class:`~firedrake.Mesh` to solve on.
    :returns: an ``int`` giving the number of TAO/LMVM iterations to
        convergence.
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
    tao_params = {'tao_type': 'lmvm',
                  'tao_gatol': 1.0e-7, 'tao_grtol': 0.0, 'tao_gttol': 0.0,
                  'tao_max_it': 5000}
    tao = TAOSolver(MinimizationProblem(Jhat), parameters=tao_params)
    tao.solve()
    return tao.tao.getIterationNumber()


# Print over a few grading levels

print(f"{'stretch':>7}  {'h_ratio':>7}  {'TAO iters':>9}")
for stretch in (0.0, 3.0, 4.0, 5.0):
    mesh = graded_square(stretch=stretch)
    print(f"{stretch:7.1f}  {h_ratio(mesh):7.1f}  {run_tao(mesh):9d}", flush=True)
