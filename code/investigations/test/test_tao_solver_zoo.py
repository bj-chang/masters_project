"""TAO solver zoo on graded Poisson"""
import numpy as np

from firedrake import *
from firedrake.adjoint import *
from firedrake.petsc import PETSc
from pyadjoint.optimization.tao_solver import TAOSolver

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
    fwd_params = {'ksp_type': 'preonly', 'pc_type': 'lu'}

    m.zero()
    continue_annotation()
    with set_working_tape() as tape:
        solve(F == 0, u, bcs=bc, solver_parameters=fwd_params)
        J = assemble(0.5 * (u - d) ** 2 * dx + 0.5 * ALPHA * m ** 2 * dx)
        Jhat = ReducedFunctional(J, Control(m), tape=tape)
    pause_annotation()
    return Jhat


def run(Jhat, tao_type, extra_opts=None):
    """Run ``tao_type`` on ``Jhat`` and return the iteration count string.

    Any PETSc options in ``extra_opts`` are set before the solver is
    built and cleared after the solve, so options never leak between
    rows of the zoo table.
    """
    opts = PETSc.Options()
    if extra_opts:
        for k, v in extra_opts.items():
            opts.setValue(k, v)

    params = {'tao_type': tao_type,
              'tao_gatol': GATOL,
              'tao_grtol': 0.0, 'tao_gttol': 0.0,
              'tao_max_it': MAX_IT}
    try:
        solver = TAOSolver(MinimizationProblem(Jhat), parameters=params)
        solver.solve()
        its = solver.tao.getIterationNumber()
        reason = solver.tao.getConvergedReason()
        result = f"{its} (r={reason})"
    except Exception as e:
        result = f"FAIL: {e.__class__.__name__}"
    finally:
        if extra_opts:
            for k in extra_opts:
                try:
                    opts.delValue(k)
                except Exception:
                    pass
    return result



SOLVERS = [
    ("nls",   "nls",   None),
    ("bnls",  "bnls",  None),
    ("bntr",  "bntr",  None),
    ("bntl",  "bntl",  None),
    ("bnk",   "bnk",   None),
    
    ("lmvm",  "lmvm",  {'-tao_lmvm_mat_lmvm_hist_size': 100}),
    ("bqnls", "bqnls", None),
    
    ("cg",    "cg",    None),
    ("owlqn", "owlqn", None),
]

STRETCHES = (0.0, 3.0, 4.0, 5.0)


# Build meshes
meshes = [(s, graded_square(stretch=s)) for s in STRETCHES]
ratios = [h_ratio(m) for _, m in meshes]


col_labels = [f"h~{r:.0f}" for r in ratios]
print(f"2D Poisson control, N=32, alpha={ALPHA}, gatol={GATOL:.0e}, "
      f"max_it={MAX_IT}")
print(f"stretches: {STRETCHES}")
print(f"realised h_max/h_min: " +
      "  ".join(f"{r:.1f}" for r in ratios))
print()

header = f"{'solver':>13}  " + "  ".join(f"{c:>16}" for c in col_labels)
print(header)
print("-" * len(header))

for label, tao_type, extra_opts in SOLVERS:
    cells = []
    for s, mesh in meshes:
        print(f"  running {label} at stretch={s}...", flush=True)
        result = run(build_jhat(mesh), tao_type, extra_opts=extra_opts)
        cells.append(result)
    print(f"{label:>13}  " + "  ".join(f"{c:>16}" for c in cells),
          flush=True)
