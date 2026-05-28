"""Second-level probe: is solver.tao.setGradientNorm() being silently no-oped?

We do three things to isolate the bug:

  (1) Inspect TAO's reported converged reason and final gradient norm
      *as reported by TAO itself* (so we know which norm it's using).
  (2) After overriding, manually call TaoComputeGradientNorm by reading
      back the gradient and applying the matrix we installed, to verify
      tao->gradient_norm is what we set.
  (3) Try a "wild" override (scale the L^2 inverse by 1e6); if TAO is
      really using our metric, this should change the convergence
      behaviour dramatically.
"""

import numpy as np
from petsc4py import PETSc

from firedrake import (
    DirichletBC, Function, FunctionSpace, SpatialCoordinate,
    TestFunction, TrialFunction, assemble, dx, grad, inner, pi, sin, solve,
)
from firedrake.adjoint import (
    Control, ReducedFunctional, continue_annotation, pause_annotation,
    set_working_tape,
)
from pyadjoint import MinimizationProblem, Control as PAControl
from pyadjoint.optimization.tao_solver import TAOSolver, RieszMapMat

from pdeopt.meshes import graded_unit_square_tensor


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
    return Jhat, V


def run_case(label, mesh, override, scale=1.0, max_it=20):
    print(f"--- {label} ---")
    Jhat, V = build_problem(mesh, riesz_map="l2")
    parameters = {
        "tao_type": "lmvm",
        "tao_gatol": 1e-7,
        "tao_grtol": 0.0,
        "tao_gttol": 0.0,
        "tao_max_it": max_it,
        "tao_max_funcs": 4 * max_it,
        "tao_lmvm_num_vecs": 5,
        "tao_monitor": None,
    }
    solver = TAOSolver(MinimizationProblem(Jhat), parameters=parameters)
    if override is not None:
        inner_control = Jhat.controls[0]
        test_control = PAControl(inner_control.control, riesz_map=override)
        test_minv = RieszMapMat([test_control], comm=solver.tao.getComm())
        if scale != 1.0:
            # Wrap test_minv in a scaled python Mat so we can amplify.
            class ScaledCtx:
                def __init__(self, M, s):
                    self.M = M; self.s = s
                def mult(self, mat, x, y):
                    self.M.mult(x, y)
                    y.scale(self.s)
            n, N = test_minv.getLocalSize()[0], test_minv.getSize()[0]
            scaled = PETSc.Mat().createPython(((n, N), (n, N)),
                                              ScaledCtx(test_minv, scale),
                                              comm=solver.tao.getComm())
            scaled.setOption(PETSc.Mat.Option.SYMMETRIC, True)
            scaled.setUp(); scaled.assemble()
            solver.tao.setGradientNorm(scaled)
            solver._scaled = scaled; solver._test_minv = test_minv
        else:
            solver.tao.setGradientNorm(test_minv)
            solver._test_minv = test_minv
        solver._test_control = test_control

    try:
        solver.solve()
    except Exception as e:
        print(f"  solver raised: {type(e).__name__}")

    tao = solver.tao
    reason = tao.getConvergedReason()
    its = tao.getIterationNumber()
    status = tao.getSolutionStatus()
    # petsc4py returns (its, fval, gnorm, cnorm, xdiff, reason)
    gnorm = status[2]
    print(f"  converged reason = {reason}")
    print(f"  iterations       = {its}")
    print(f"  TAO's gnorm      = {gnorm:.6e}")
    print()


def main():
    print("Building ratio-4 mesh, n=16")
    mesh, ratio = graded_unit_square_tensor(h_ratio=4.0, n=16)
    print(f"  realised ratio = {ratio:.2f}\n")

    run_case("no override (= l2 metric)", mesh, override=None, max_it=20)
    run_case("override = L2", mesh, override="L2", max_it=20)
    run_case("override = L2 scaled x1e6", mesh, override="L2",
             scale=1e6, max_it=20)


if __name__ == "__main__":
    main()
