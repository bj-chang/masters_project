"""Verify the convergence_riesz_map override is doing what we think.

For a ratio-4 graded mesh, run TAO/LMVM with riesz_map='l2'
(algorithm uses identity B_0, ill-conditioned LMVM) and:

  * once with the default pyadjoint convergence metric (which is the
    Control's riesz_map -- here l2, so TAO checks ||r||_l2 of the
    raw dual coefficients);
  * once with convergence_riesz_map='L2' override (TAO should now
    check ||R_L2(J')||_L2 instead).

After each, print:

  * the first few iterations' gradient norms (TAO monitor),
  * the *externally computed* ||g||_L2 and ||r||_l2 at the final
    iterate, so we can see which norm the override actually drove
    to 1e-7.

Run from the project root with the Firedrake venv.
"""

import numpy as np

from firedrake import (
    DirichletBC, Function, FunctionSpace, SpatialCoordinate,
    TestFunction, assemble, dx, grad, inner, pi, sin, solve,
)
from firedrake.adjoint import (
    Control, ReducedFunctional, continue_annotation, pause_annotation,
    set_working_tape,
)

from pdeopt.meshes import graded_unit_square_tensor
from pdeopt.optimisers import solve_with_tao


def build_problem(mesh, alpha=1.0e-4, riesz_map="l2"):
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
              solver_parameters={"ksp_type": "cg", "pc_type": "hypre"})
        J = assemble(0.5 * (u - d)**2 * dx + 0.5 * alpha * m**2 * dx)
        Jhat = ReducedFunctional(J, Control(m, riesz_map=riesz_map),
                                 tape=tape)
    pause_annotation()
    return Jhat, V, m


def L2_norm(coeffs, V):
    f = Function(V)
    f.dat.data[:] = coeffs
    return float(np.sqrt(assemble(inner(f, f) * dx)))


def main():
    print("Building graded ratio-4 mesh, n=16")
    mesh, realised = graded_unit_square_tensor(h_ratio=4.0, n=16)
    print(f"  realised h_max/h_min = {realised:.2f}\n")

    # Reference: at m=0 the dual gradient r has these norms.
    Jhat_l2, V, m = build_problem(mesh, riesz_map="l2")
    r0 = Jhat_l2.derivative()
    r0_l2 = float(np.linalg.norm(r0.dat.data_ro))
    g0_L2 = Jhat_l2.derivative(apply_riesz=False)  # still dual
    # primal L2 representer
    Jhat_L2, _, _ = build_problem(mesh, riesz_map="L2")
    g0_L2_primal = Jhat_L2.derivative(apply_riesz=True)
    g0_L2_norm = float(np.sqrt(assemble(inner(g0_L2_primal, g0_L2_primal)*dx)))
    print(f"At m=0 (graded ratio 4, n=16):")
    print(f"  ||r||_l2 (dual coeffs)     = {r0_l2:.6e}")
    print(f"  ||R_L2(J')||_L2            = {g0_L2_norm:.6e}\n")

    for override in [None, "L2"]:
        label = override if override is not None else "default (= riesz_map)"
        print(f"=== TAO/LMVM, riesz_map='l2', convergence={label} ===")
        Jhat, _, _ = build_problem(mesh, riesz_map="l2")
        result = solve_with_tao(
            Jhat, tao_gatol=1e-7, tao_max_it=200, tao_max_funcs=400,
            history=5, verbose=False, convergence_riesz_map=override,
        )
        # Externally evaluate ||r||_l2 and ||g||_L2 at the final iterate.
        m_opt = result["m_opt"]
        # rebuild adjoint at this m
        Jhat_eval_l2, V_e, _ = build_problem(mesh, riesz_map="l2")
        Jhat_eval_l2(m_opt)
        r_final = Jhat_eval_l2.derivative()
        r_final_l2 = float(np.linalg.norm(r_final.dat.data_ro))

        Jhat_eval_L2, _, _ = build_problem(mesh, riesz_map="L2")
        Jhat_eval_L2(m_opt)
        g_final_L2 = Jhat_eval_L2.derivative(apply_riesz=True)
        g_final_L2_norm = float(
            np.sqrt(assemble(inner(g_final_L2, g_final_L2) * dx))
        )

        print(f"  iterations  = {result['iterations']}")
        print(f"  converged   = {result['converged']}")
        print(f"  final J     = {result['final_J']:.6e}")
        print(f"  ||r||_l2 at exit  = {r_final_l2:.6e}")
        print(f"  ||g||_L2 at exit  = {g_final_L2_norm:.6e}")
        print()


if __name__ == "__main__":
    main()
