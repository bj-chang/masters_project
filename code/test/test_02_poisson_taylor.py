"""Taylor test for the Poisson optimal-control reduced functional.

A correct adjoint-based gradient must satisfy

    |J_tilde(m + h dm) - J_tilde(m) - h dJ_tilde(m; dm)| = O(h^2)

as h -> 0. The pyadjoint helper taylor_test returns the minimum
observed rate across a sequence of perturbations; we assert that it
is close to 2.
"""

import sys
import pytest

pytest.importorskip("firedrake")


def test_taylor_rate_is_quadratic():
    from firedrake import (
        DirichletBC, Function, FunctionSpace, SpatialCoordinate,
        TestFunction, UnitSquareMesh, assemble, dx, grad, inner, pi,
        sin, solve,
    )
    from firedrake.adjoint import (
        Control, ReducedFunctional, continue_annotation,
        pause_annotation, set_working_tape,
    )
    from pyadjoint import taylor_test

    mesh = UnitSquareMesh(16, 16)
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
        solve(F == 0, u, bcs=bc)
        J = assemble(0.5 * (u - d)**2 * dx + 0.5 * 1.0e-4 * m**2 * dx)
        Jhat = ReducedFunctional(J, Control(m, riesz_map="L2"), tape=tape)
    pause_annotation()

    direction = Function(V).interpolate(sin(pi * x) * sin(pi * y))
    rate = taylor_test(Jhat, m, direction)
    print(f"Achieved Taylor rate: {rate}")
    assert rate > 1.9


if __name__ == "__main__":
    pytest.main(sys.argv)
