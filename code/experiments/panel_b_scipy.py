"""panel (b) scipy row: l2 lbfgs-b on the h1 regularised problem, convergence checked externally in the h1 norm"""
from meshdep.meshes import graded_unit_square_from_file
from meshdep.optimisers import solve_with_scipy_external_check


def build_h1reg_problem(mesh, alpha, metric):
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
                     + 0.5 * alpha * (m**2 + inner(grad(m), grad(m))) * dx)

        Jhat = ReducedFunctional(J, Control(m, riesz_map=metric), tape=tape)
    pause_annotation()
    return Jhat


def main():
    alpha = 1.0e-4
    ratios = [4, 8, 16, 32, 64, 128]
    print(f"{'ratio':>6} {'realised':>9} {'ell2 SciPy (H1-thresh)':>22}",
          flush=True)
    print("-" * 40, flush=True)
    for r in ratios:
        mesh, realised = graded_unit_square_from_file(
            f"mesh_generation/graded_R{r}.msh")
        Jhat = build_h1reg_problem(mesh, alpha, metric="l2")
        res = solve_with_scipy_external_check(
            Jhat, eps=1.0e-7, test_riesz_map="H1", maxiter=5000)
        print(f"{r:>6} {realised:>9.2f} {res['iterations']:>22}  "
              f"(J={res['final_J']:.6e})", flush=True)


if __name__ == "__main__":
    main()
