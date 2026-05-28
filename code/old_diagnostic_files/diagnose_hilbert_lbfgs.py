"""Test the Hilbert L-BFGS implementation against the Netgen R4 mesh.

If the implementation is correct, the L^2 row should converge in a
flat ~20-ish iterations regardless of mesh ratio (Schwedes' Moola
gives 22, 20, 22, 23, 23, 27).
"""

from firedrake import (
    DirichletBC, Function, FunctionSpace, SpatialCoordinate,
    TestFunction, assemble, dx, grad, inner, pi, sin, solve,
)
from firedrake.adjoint import (
    Control, ReducedFunctional, continue_annotation, pause_annotation,
    set_working_tape,
)

from pdeopt.meshes import graded_unit_square_from_file
from pdeopt.optimisers import solve_with_hilbert_lbfgs


def build_problem(mesh, alpha=1.0e-4, riesz_map="L2"):
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
        J = assemble(0.5 * (u - d)**2 * dx + 0.5 * alpha * m**2 * dx)
        Jhat = ReducedFunctional(J, Control(m, riesz_map=riesz_map),
                                 tape=tape)
    pause_annotation()
    return Jhat


def main():
    for ratio in [4, 8, 16, 32, 64, 128]:
        path = f"mesh_generation/graded_R{ratio}.msh"
        from firedrake import Mesh
        mesh = Mesh(path)
        print(f"\n##### Netgen R{ratio} mesh "
              f"({mesh.num_cells()} cells) #####")
        Jhat = build_problem(mesh, riesz_map="L2")
        result = solve_with_hilbert_lbfgs(
            Jhat, eps=1.0e-7, max_iter=200, history=5, verbose=True,
        )
        print(f"  -> iters={result['iterations']}, "
              f"converged={result['converged']}, "
              f"J={result['final_J']:.6e}, "
              f"||g||_H={result['final_grad_norm_H']:.3e}")


if __name__ == "__main__":
    main()
