"""every tao solver that accepts this problem, on the graded meshes. zoo table in 10.10"""
import os
import sys


_PROJECT_CODE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
if _PROJECT_CODE not in sys.path:
    sys.path.insert(0, _PROJECT_CODE)

from firedrake import *
from firedrake.adjoint import *
from firedrake.petsc import PETSc
from pyadjoint.optimization.tao_solver import TAOSolver

from meshdep.meshes import graded_unit_square_from_file

ALPHA = 1.0e-4
GATOL = 1.0e-7
MAX_IT = 2000
MESH_DIR = "code/mesh_generation"


def build_jhat(mesh):
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
    ("lmvm",  "lmvm",  {'-tao_lmvm_mat_lmvm_hist_size': 100}),
    ("bqnls", "bqnls", None),
    ("bntl",  "bntl",  None),
]


TARGET_RATIOS = (4, 16, 64, 128)


print("Loading Netgen meshes from .msh files...", flush=True)
meshes = []
for r in TARGET_RATIOS:
    mesh, realised = graded_unit_square_from_file(
        f"{MESH_DIR}/graded_R{r}.msh"
    )
    print(f"  target r={r:3d}, realised h_max/h_min = {realised:6.2f}, "
          f"{mesh.num_cells()} cells", flush=True)
    meshes.append((r, mesh, realised))


col_labels = [f"r={r}\n(h~{realised:.0f})" for r, _, realised in meshes]

col_labels = [f"r={r}/{realised:.0f}" for r, _, realised in meshes]

print()
print(f"2D Poisson control on Netgen meshes, alpha={ALPHA}, "
      f"gatol={GATOL:.0e}, max_it={MAX_IT}")
print(f"target h_max/h_min ratios: {TARGET_RATIOS}")
print(f"realised h_max/h_min: " +
      "  ".join(f"{realised:.1f}" for _, _, realised in meshes))
print()

header = f"{'solver':>13}  " + "  ".join(f"{c:>16}" for c in col_labels)
print(header)
print("-" * len(header))


all_rows = []
for label, tao_type, extra_opts in SOLVERS:
    cells = []
    for r, mesh, realised in meshes:
        print(f"  running {label} at r={r}...", flush=True)
        result = run(build_jhat(mesh), tao_type, extra_opts=extra_opts)
        cells.append(result)
    row = f"{label:>13}  " + "  ".join(f"{c:>16}" for c in cells)
    print(row, flush=True)
    all_rows.append(row)


print()
print("=" * len(header))
print("FINAL TABLE")
print("=" * len(header))
print(f"2D Poisson control on Netgen meshes, alpha={ALPHA}, "
      f"gatol={GATOL:.0e}, max_it={MAX_IT}")
print(f"target h_max/h_min ratios: {TARGET_RATIOS}")
print(f"realised h_max/h_min: " +
      "  ".join(f"{realised:.1f}" for _, _, realised in meshes))
print()
print(header)
print("-" * len(header))
for row in all_rows:
    print(row)
print("=" * len(header), flush=True)
