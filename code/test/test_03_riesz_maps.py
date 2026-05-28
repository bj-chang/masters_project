"""Smoke test that the riesz_map keyword on Control changes the gradient.

If pyadjoint were silently ignoring the keyword, the entire mesh-
dependence argument would have no realisation in code. The test
confirms that selecting "l2" and "L2" produces different primal
gradients (they differ by the mass matrix).
"""

import sys
import pytest

pytest.importorskip("firedrake")


def _build_jhat(riesz_map):
    from firedrake import (
        DirichletBC, Function, FunctionSpace, SpatialCoordinate,
        TestFunction, UnitSquareMesh, assemble, dx, grad, inner, pi,
        sin, solve,
    )
    from firedrake.adjoint import (
        Control, ReducedFunctional, continue_annotation,
        pause_annotation, set_working_tape,
    )

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
        Jhat = ReducedFunctional(J, Control(m, riesz_map=riesz_map),
                                 tape=tape)
    pause_annotation()
    return Jhat


def test_l2_and_ell2_gradients_differ():
    from firedrake import errornorm, norm

    Jhat_ell2 = _build_jhat("l2")
    grad_ell2 = Jhat_ell2.derivative(apply_riesz=True)

    Jhat_L2 = _build_jhat("L2")
    grad_L2 = Jhat_L2.derivative(apply_riesz=True)

    diff = errornorm(grad_ell2, grad_L2)
    scale = max(norm(grad_ell2), norm(grad_L2), 1.0)
    assert diff > 1.0e-10 * scale


if __name__ == "__main__":
    pytest.main(sys.argv)
