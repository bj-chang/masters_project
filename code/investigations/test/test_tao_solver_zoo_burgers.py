"""TAO solver zoo on 1D Burgers control"""
import numpy as np

from firedrake import *
from firedrake.adjoint import *
from firedrake.petsc import PETSc
from pyadjoint.optimization.tao_solver import TAOSolver

ALPHA = 1.0e-4
NU = 1.0e-3
T_FINAL = 0.5
N_STEPS = 25
DT = T_FINAL / N_STEPS
N_MESH = 32
GATOL = 1.0e-7
MAX_IT = 1000


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

STRETCHES = (0.0, 2.0, 3.0, 4.0)


meshes = [(s, graded_interval(stretch=s)) for s in STRETCHES]
ratios = [h_ratio(m) for _, m in meshes]


col_labels = [f"h~{r:.0f}" for r in ratios]
print(f"1D Burgers control, N={N_MESH}, T={T_FINAL}, dt={DT}, "
      f"nu={NU}, alpha={ALPHA}, gatol={GATOL:.0e}, max_it={MAX_IT}")
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
