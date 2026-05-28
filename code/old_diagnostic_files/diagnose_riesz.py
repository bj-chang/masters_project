"""Diagnostic: are the three Riesz maps actually wired up?

Findings from reading the pyadjoint 2025.10.1 source:

* ``Jhat.derivative()`` returns the *dual-space* derivative (a Cofunction
  in the Firedrake case), regardless of the ``riesz_map`` attached to
  the Control. The Riesz map is only applied when the caller passes
  ``apply_riesz=True`` (see ``pyadjoint/control.py`` ``Control.get_derivative``
  and ``pyadjoint/reduced_functional.py`` ``ReducedFunctional.derivative``).

* ``TAOSolver`` does *not* call ``derivative(apply_riesz=True)``. Instead,
  it installs the Riesz map as PETSc TAO's gradient-norm metric (via
  ``tao.setGradientNorm``) and as the initial Hessian approximation B_0
  in LMVM (via ``setLMVMH0`` plus a custom preconditioner). So the Riesz
  map is consumed *inside* PETSc TAO, not exposed in the gradient
  returned to the user.

This script confirms both:

1. ``Jhat.derivative()`` with no flag produces the same Cofunction for
   every value of ``riesz_map`` (the dual derivative does not depend on
   the inner product on the primal space).

2. ``Jhat.derivative(apply_riesz=True)`` produces three different
   *primal* gradients, one per Riesz map, which is what the dissertation
   needs.

3. The norm ``||g||_H`` that TAO checks against ``gttol`` is the dual
   norm with respect to the Riesz map ``H``. We compute it here for
   each Riesz map so we can compare with TAO's internal residuals.
"""

import numpy as np

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


def build_reduced_functional(mesh, riesz_map, alpha=1.0e-4):
    """Return a fresh ReducedFunctional for the Poisson control problem.

    The control ``m`` is the zero function on construction, so subsequent
    calls to ``Jhat.derivative`` evaluate the gradient at m=0.
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
        solve(F == 0, u, bcs=bc,
              solver_parameters={"ksp_type": "cg", "pc_type": "hypre"})
        J = assemble(0.5 * (u - d)**2 * dx + 0.5 * alpha * m**2 * dx)
        Jhat = ReducedFunctional(
            J, Control(m, riesz_map=riesz_map), tape=tape,
        )
    pause_annotation()
    return Jhat, V


def L2_norm_of_coeffs(coeffs, V):
    """L^2(Omega) norm of the field with the given coefficient vector."""
    f = Function(V)
    f.dat.data[:] = coeffs
    return float(np.sqrt(assemble(inner(f, f) * dx)))


def H1_norm_of_coeffs(coeffs, V):
    """H^1(Omega) norm of the field with the given coefficient vector."""
    f = Function(V)
    f.dat.data[:] = coeffs
    return float(np.sqrt(assemble((inner(f, f) + inner(grad(f), grad(f))) * dx)))


def print_block(title):
    bar = "=" * len(title)
    print(f"\n{bar}\n{title}\n{bar}")


def main():
    N = 16
    alpha = 1.0e-4
    print(f"Poisson optimal control, uniform {N}x{N} mesh, alpha = {alpha}")
    print("Gradient evaluated at m = 0.\n")

    mesh = UnitSquareMesh(N, N)
    riesz_maps = ["l2", "L2", "H1"]

    # ---------------------------------------------------------------
    # 1. Default Jhat.derivative() -- expected to be Riesz-independent
    # ---------------------------------------------------------------
    print_block("(1) Jhat.derivative()   [no apply_riesz flag]")
    print("Each row should be identical: derivative() returns the dual,")
    print("which does not depend on the Riesz map on the primal space.\n")

    dual_coeffs = {}
    for rm in riesz_maps:
        Jhat, V = build_reduced_functional(mesh, rm, alpha=alpha)
        g = Jhat.derivative()
        c = np.array(g.dat.data_ro, copy=True)
        dual_coeffs[rm] = (c, V)
        print(f"riesz_map = {rm!r:>5}    type = {type(g).__name__:<11}    "
              f"||coeffs||_2 = {np.linalg.norm(c):.6e}    "
              f"first 3 coeffs = {c[:3]}")

    keys = list(dual_coeffs.keys())
    print()
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            ca, _ = dual_coeffs[a]
            cb, _ = dual_coeffs[b]
            print(f"  ||g[{a}] - g[{b}]||_2 = "
                  f"{np.linalg.norm(ca - cb):.6e}    "
                  f"(should be 0)")

    # ---------------------------------------------------------------
    # 2. Jhat.derivative(apply_riesz=True) -- should be Riesz-DEPENDENT
    # ---------------------------------------------------------------
    print_block("(2) Jhat.derivative(apply_riesz=True)   [primal gradient]")
    print("Each row should be DIFFERENT: pyadjoint applies the Riesz map")
    print("of the control to convert the dual to a primal gradient.\n")

    primal_coeffs = {}
    for rm in riesz_maps:
        Jhat, V = build_reduced_functional(mesh, rm, alpha=alpha)
        g = Jhat.derivative(apply_riesz=True)
        c = np.array(g.dat.data_ro, copy=True)
        primal_coeffs[rm] = (c, V)
        l2_norm = L2_norm_of_coeffs(c, V)
        h1_norm = H1_norm_of_coeffs(c, V)
        print(f"riesz_map = {rm!r:>5}    type = {type(g).__name__:<11}    "
              f"||coeffs||_2 = {np.linalg.norm(c):.6e}")
        print(f"               ||g||_{{L^2}}     = {l2_norm:.6e}    "
              f"||g||_{{H^1}}     = {h1_norm:.6e}")
        print(f"               first 3 coeffs = {c[:3]}")

    print()
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            ca, V = primal_coeffs[a]
            cb, _ = primal_coeffs[b]
            l2 = L2_norm_of_coeffs(ca - cb, V)
            l2_rel = np.linalg.norm(ca - cb) / np.linalg.norm(ca)
            print(f"  ||g[{a}] - g[{b}]||_{{L^2}} = {l2:.6e}    "
                  f"relative coeff diff = {l2_rel:.6e}")

    # ---------------------------------------------------------------
    # 3. The dual norm ||J'||_{H^*} that TAO's gttol actually sees
    # ---------------------------------------------------------------
    #
    # PETSc TAO checks convergence on the H-norm of the gradient,
    # where H = setGradientNorm(M_inv). The H-norm of the dual residual
    # equals sqrt(<r, M_inv r>) = sqrt(<r, g_primal>) when g_primal is
    # the Riesz representer of r (i.e. M_inv applied to r).
    print_block("(3) Dual norm seen by TAO's gttol convergence check")
    print("This is sqrt(<dual r, primal g>), which equals ||g||_H,")
    print("the function-space norm of the primal gradient. The l2 row")
    print("uses identity for M_inv so this equals ||coeffs||_2 of the dual.\n")

    for rm in riesz_maps:
        r_dual, _ = dual_coeffs[rm]
        g_primal, V = primal_coeffs[rm]
        dual_norm = np.sqrt(np.dot(r_dual, g_primal))
        l2_dual = np.linalg.norm(r_dual)
        print(f"riesz_map = {rm!r:>5}    "
              f"||r||_2 (dual coeffs) = {l2_dual:.6e}    "
              f"||g||_H (TAO sees this) = {dual_norm:.6e}")


if __name__ == "__main__":
    main()
