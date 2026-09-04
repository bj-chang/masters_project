"""firedrake forward poisson with the manufactured solution"""
from argparse import ArgumentParser

import numpy as np

from firedrake import *

def solve_forward_poisson_mms(resolution, degree=1):

    """solves with the manufactured source, returns the l2 error"""
    mesh = UnitSquareMesh(resolution, resolution)
    V = FunctionSpace(mesh, "CG", degree)
    u = TrialFunction(V)
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)
    u_exact = Function(V, name="u_exact").interpolate(sin(pi * x) * sin(pi * y))
    m = Function(V, name="m").interpolate(2.0 * pi**2 * sin(pi * x) * sin(pi * y))

    a = inner(grad(u), grad(v)) * dx
    L = m * v * dx
    bc = DirichletBC(V, 0.0, "on_boundary")

    u_h = Function(V, name="u")
    solve(a == L, u_h, bcs=bc)
    return errornorm(u_exact, u_h, norm_type="L2")


def convergence_test(resolutions=(8, 16, 32, 64), degree=1):

    rows = []
    previous_error = None
    previous_h = None
    for n in resolutions:
        h = 1.0 / n
        error = solve_forward_poisson_mms(n, degree=degree)
        if previous_error is None:
            rate = None
        else:
            rate = np.log(previous_error / error) / np.log(previous_h / h)
        rows.append((n, h, error, rate))
        previous_error = error
        previous_h = h
    return rows


def print_table(rows):

    print(f"{'N':>4}  {'h':>10}  {'L2 error':>14}  {'rate':>6}")
    print("-" * 40)
    for n, h, error, rate in rows:
        rate_str = "  -- " if rate is None else f"{rate:6.4f}"
        print(f"{n:>4}  {h:>10.4e}  {error:>14.6e}  {rate_str}")


def main():
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--degree", type=int, default=1,
        help="Polynomial degree of the Lagrange basis (default: 1).",
    )
    parser.add_argument(
        "--resolutions", type=int, nargs="+", default=[8, 16, 32, 64],
        help="Mesh resolutions to test (default: 8 16 32 64).",
    )
    args = parser.parse_args()

    rows = convergence_test(args.resolutions, degree=args.degree)
    print_table(rows)


if __name__ == "__main__":
    main()
