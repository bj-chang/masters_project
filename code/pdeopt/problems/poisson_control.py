"""Poisson optimal control problem.

Minimise ``J(u, m) = (1/2)|u-d|^2 + (alpha/2)|m|^2`` subject to
``-Delta u = m`` on the unit square with ``u = 0`` on the boundary.
"""

from argparse import ArgumentParser

from firedrake import (
    DirichletBC,
    Function,
    FunctionSpace,
    SpatialCoordinate,
    TestFunction,
    UnitSquareMesh,
    assemble,
    dx,
    grad,
    inner,
    pi,
    sin,
    solve,
)
from firedrake.adjoint import (
    Control,
    ReducedFunctional,
    continue_annotation,
    pause_annotation,
    set_working_tape,
)

from pdeopt.optimisers import solve_with_tao


def solve_poisson_control(mesh, alpha=1.0e-4, riesz_map="L2",
                          tao_gatol=1.0e-7, verbose=False,
                          convergence_riesz_map=None):
    """Tape the forward solve, build the reduced functional, run TAO.

    ``riesz_map`` is the inner product on the control space:
    ``'l2'``, ``'L2'`` or ``'H1'``.
    ``convergence_riesz_map`` (optional) is the H used by TAO's
    stopping test; defaults to the Control's own ``riesz_map``.
    """

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
        # LU: default CG+hypre has rtol=1e-5, which puts a noise floor
        # on the discrete gradient above the eps=1e-7 target.
        solve(F == 0, u, bcs=bc,
              solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})

        J = assemble(0.5 * (u - d)**2 * dx + 0.5 * alpha * m**2 * dx)
        Jhat = ReducedFunctional(J, Control(m, riesz_map=riesz_map),
                                 tape=tape)
    pause_annotation()

    return solve_with_tao(
        Jhat, tao_gatol=tao_gatol, verbose=verbose,
        convergence_riesz_map=convergence_riesz_map,
    )


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
