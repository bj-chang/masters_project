"""panel (b): h1 regularised objective, hilbert lbfgs with the h1 riesz map on the graded meshes. gives the flat 4s"""
import sys
sys.path.insert(0, "/home/bjcwsl/masters_project/code")
from meshdep.meshes import graded_unit_square_from_file
from meshdep.optimisers import (
    solve_with_hilbert_lbfgs, solve_with_scipy_external_check,
)

MESH_DIR = "/home/bjcwsl/masters_project/code/mesh_generation"
RATIOS = [4, 8, 16, 32, 64, 128]
ALPHA = 1.0e-4


def build_jhat(mesh, riesz_map):
    from firedrake import (
        DirichletBC, Function, FunctionSpace, SpatialCoordinate,
        TestFunction, assemble, dx, grad, inner, pi, sin, solve,
    )
    from firedrake.adjoint import (
        Control, ReducedFunctional, continue_annotation,
        pause_annotation, set_working_tape,
    )
    V = FunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)
    d = Function(V).interpolate(sin(pi * x) * sin(pi * y))
    bc = DirichletBC(V, 0.0, "on_boundary")
    m = Function(V)
    continue_annotation()
    with set_working_tape() as tape:
        u = Function(V)
        v = TestFunction(V)
        F = inner(grad(u), grad(v)) * dx - m * v * dx
        solve(F == 0, u, bcs=bc,
              solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
        J = assemble(0.5 * (u - d)**2 * dx
                     + 0.5 * ALPHA * (m**2 + inner(grad(m), grad(m))) * dx)
        Jhat = ReducedFunctional(J, Control(m, riesz_map=riesz_map), tape=tape)
    pause_annotation()
    return Jhat


l2_row, h1_row, ratios, cells = [], [], [], []
for r in RATIOS:
    mesh, realised = graded_unit_square_from_file(f"{MESH_DIR}/graded_R{r}.msh")
    ratios.append(realised)
    cells.append(mesh.num_cells())
    print(f"r={r} realised={realised:.2f} cells={mesh.num_cells()}", flush=True)

    Jhat = build_jhat(mesh, "l2")
    res = solve_with_scipy_external_check(Jhat, eps=1.0e-7,
                                          test_riesz_map="H1", maxiter=10000)
    l2_row.append(res["iterations"])
    print(f"  ell2 (SciPy, H1-check): {res['iterations']}  J={res['final_J']:.4e}",
          flush=True)

    Jhat = build_jhat(mesh, "H1")
    res = solve_with_hilbert_lbfgs(Jhat, eps=1.0e-7, max_iter=500, history=5)
    h1_row.append(res["iterations"])
    print(f"  H1   (Hilbert L-BFGS):  {res['iterations']}  J={res['final_J']:.4e}",
          flush=True)

print("\n===== PANEL (b) FINAL =====")
print("ratios :", [f"{x:.2f}" for x in ratios])
print("cells  :", cells)
print("ell2   :", l2_row)
print("H1     :", h1_row)
