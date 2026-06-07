"""Minimal library-only TAO/LMVM case for the Poisson control problem.

Canonical pyadjoint recipe (per David Ham's notes), nothing custom:
  - build the reduced functional with a single Control(m); the default
    Riesz map is L2 (verified: riesz_map=None gives the same primal
    representation as riesz_map="L2"),
  - hand it to TAOSolver and run.

We do NOT call setGradientNorm, we do NOT build a second control, and
we do NOT use a hand-rolled L-BFGS. pyadjoint's TAOSolver itself
installs the L2 gradient norm (setGradientNorm) and the initial Hessian
H0 = M^{-1} (setLMVMH0) from the Control's Riesz map.

Question: does the library optimiser converge mesh-independently if the
inner product is set correctly?

  - Uniform refinement (h -> 0 uniformly): YES, the count is flat. So
    "does L2 work normally?" -- yes, on uniform meshes.
  - Graded (Netgen) meshes: NO, the count jumps to the hundreds. This
    is the minimal case where the library setup fails. The cause is the
    fixed H0 = M^{-1}: TAO uses it as-is and does not apply the standard
    L-BFGS rescaling gamma_k = <s,y>/<y,y>, which on a graded mesh has
    the wrong overall scale.

Run from ~/masters_project/.
"""

from firedrake import *
from firedrake.adjoint import *
from pyadjoint import MinimizationProblem
from pyadjoint.optimization.tao_solver import TAOSolver

ALPHA = 1.0e-4
MESH_DIR = "code/mesh_generation"
LU = {"ksp_type": "preonly", "pc_type": "lu"}

# Minimal LMVM options: absolute gradient tolerance only (Schwedes
# eps = 1e-7), defaults for everything else.
TAO_PARAMS = {
    "tao_type": "lmvm",
    "tao_gatol": 1.0e-7,
    "tao_grtol": 0.0,
    "tao_gttol": 0.0,
}


def solve_tao(mesh):
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
        solve(F == 0, u, bcs=bc, solver_parameters=LU)
        J = assemble(0.5 * (u - d)**2 * dx + 0.5 * ALPHA * m**2 * dx)
        Jhat = ReducedFunctional(J, Control(m), tape=tape)   # default = L2
    pause_annotation()

    solver = TAOSolver(MinimizationProblem(Jhat), parameters=TAO_PARAMS)
    solver.solve()
    return solver.tao.getIterationNumber()


def main():
    uniform = [(f"uniform N={n}", UnitSquareMesh(n, n))
               for n in (16, 32, 64, 128)]
    graded = [(f"graded R{r}", Mesh(f"{MESH_DIR}/graded_R{r}.msh"))
              for r in (4, 16)]

    print("\n=== canonical TAO/LMVM, default L2 Riesz map ===")
    print(f"{'mesh':>14}  {'TAO iterations':>14}")
    for label, mesh in uniform + graded:
        print(f"{label:>14}  {solve_tao(mesh):>14}", flush=True)


if __name__ == "__main__":
    main()
