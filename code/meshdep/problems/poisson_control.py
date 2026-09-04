"""the poisson control problem used by all the mesh dependence runs"""
from argparse import ArgumentParser

from firedrake import *
from firedrake.adjoint import (
    Control,
    ReducedFunctional,
    continue_annotation,
    pause_annotation,
    set_working_tape,
)

from meshdep.optimisers import solve_with_tao


def solve_poisson_control(mesh, alpha=1.0e-4, riesz_map="L2",
                          tao_gatol=1.0e-7, verbose=False):

    """taped forward solve + reduced functional for the control problem"""
    V = FunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)
    d = Function(V, name="d").interpolate(sin(pi * x) * sin(pi * y))
    bc = DirichletBC(V, 0.0, "on_boundary")
    m = Function(V, name="m")

    continue_annotation()
    with set_working_tape() as tape:
        u = Function(V, name="u")
        v = TestFunction(V)
        F = inner(grad(u), grad(v)) * dx - m * v * dx


        solve(F == 0, u, bcs=bc,
              solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})

        J = assemble(0.5 * (u - d)**2 * dx + 0.5 * alpha * m**2 * dx)
        Jhat = ReducedFunctional(J, Control(m, riesz_map=riesz_map),
                                 tape=tape)
    pause_annotation()

    return solve_with_tao(Jhat, tao_gatol=tao_gatol, verbose=verbose)


def main():
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("resolution", type=int,
                        help="Mesh resolution (n by n).")
    parser.add_argument("--alpha", type=float, default=1.0e-4,
                        help="Tikhonov parameter (default: 1e-4).")
    parser.add_argument("--riesz-map", choices=["l2", "L2", "H1"],
                        default="L2",
                        help="Riesz map on the control (default: L2).")
    parser.add_argument("--verbose", action="store_true",
                        help="Print TAO's per-iteration monitor.")
    args = parser.parse_args()

    mesh = UnitSquareMesh(args.resolution, args.resolution)
    result = solve_poisson_control(
        mesh, alpha=args.alpha, riesz_map=args.riesz_map,
        verbose=args.verbose,
    )

    print(f"resolution    = {args.resolution}")
    print(f"riesz_map     = {args.riesz_map}")
    print(f"iterations    = {result['iterations']}")
    print(f"final J       = {result['final_J']:.6e}")
    print(f"converged     = {result['converged']}")


if __name__ == "__main__":
    main()
