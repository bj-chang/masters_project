"""Demonstrates that pyadjoint's TAOSolver disables L-BFGS initial-Hessian
rescaling when it installs a fixed J0 = M^{-1} via setLMVMH0(...).

NOTE on status (added retrospectively). The rescaling behaviour
demonstrated here is real: the three tests below show conclusively
that pyadjoint's setLMVMH0 call switches off PETSc's automatic
gamma_k = <s,y>/<y,y> rescaling. We originally suspected this was
the cause of the mesh-dependence observed in
original_failure_reproducer.py. It is not. The actual cause is the
LMVM history size (default 5) being too small to capture the
spectrum of the discrete Hessian on strongly graded meshes; with
history >= 25 the iteration reaches machine precision on every
mesh tested. See test_nls.py for the noise-floor probe that
established this, and the dissertation, sec. 10.8 ("TAO/LMVM
revisited") for the full diagnostic chain.

This file is kept because the underlying observation (J0 disables
auto-scaling) is correct and useful background when reading PETSc
TAO behaviour, even though it is not the root cause we initially
took it for.

Standard L-BFGS rescales its initial Hessian guess every step by
gamma_k = <s,y>/<y,y>. This rescaling is what makes the method
mesh-independent on non-uniform meshes. PETSc/TAO has it built in (the
LMVM "scalar"/"diagonal" scale types, on by default).

pyadjoint's TAOSolver, for an LMVM solver, installs the Riesz map as a
*fixed* initial Hessian H0 = M^{-1} via tao.setLMVMH0(...) (see
tao_solver.py, the `if tao.getType() in {LMVM, BLMVM}` block). PETSc
disables the automatic scaling whenever a user J0 is supplied this way,
so the rescaling is silently switched off.

Three experiments, all on the Poisson distributed-control problem with
the canonical setup - Control(m) (default L2 Riesz map) + TAOSolver,
tao_type=lmvm:

  Test 1 (symptom). Canonical TAO on uniform meshes. The count is flat
     (mesh-independent), so nothing looks wrong here.

  Test 2 (the obvious knob does nothing). Graded mesh, sweeping
     tao_lmvm_mat_lmvm_scale_type over none/scalar/diagonal. TAO's
     "scalar" scaling IS the standard rescaling, but it has no effect
     here: pyadjoint has already set the fixed H0, which disables it.

  Test 3 (decisive). Graded mesh, multiply that fixed H0 by a constant c
     and sweep c. A rescaling optimiser is invariant to a constant scale
     on H0 (it cancels out). The count instead swings from ~670 to ~6,
     proving the scale is fixed - i.e. NOT rescaled.

Run from ~/masters_project/.
"""

from firedrake import *
from firedrake.adjoint import *
from firedrake.petsc import PETSc
from pyadjoint import MinimizationProblem
from pyadjoint.optimization.tao_solver import TAOSolver, RieszMapMat

ALPHA = 1.0e-4
MESH_DIR = "code/mesh_generation"
LU = {"ksp_type": "preonly", "pc_type": "lu"}
PARAMS = {
    "tao_type": "lmvm",
    "tao_gatol": 1.0e-7, "tao_grtol": 0.0, "tao_gttol": 0.0,
    "tao_max_it": 2000, "tao_max_funcs": 20000,
}


def build(mesh):
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
        Jhat = ReducedFunctional(J, Control(m), tape=tape)
    pause_annotation()
    return Jhat


def run_canonical(Jhat, extra=None):
    """Canonical pyadjoint TAO/LMVM, optionally with extra PETSc options."""
    params = dict(PARAMS)
    if extra:
        params.update(extra)
    solver = TAOSolver(MinimizationProblem(Jhat), parameters=params)
    solver.solve()
    return solver.tao.getIterationNumber()


class _Empty:
    pass


class _ScaledMinvPC:
    """Apply c * M^{-1} as the initial-Hessian preconditioner."""
    def __init__(self, minv, c):
        self.minv, self.c = minv, c

    def apply(self, pc, x, y):
        self.minv.mult(x, y)
        y.scale(self.c)


def run_scaled(Jhat, c):
    """Canonical pyadjoint TAO/LMVM, but with H0 = c * M^{-1}."""
    solver = TAOSolver(MinimizationProblem(Jhat), parameters=PARAMS)
    tao = solver.tao
    comm = tao.getComm()
    minv = RieszMapMat(Jhat.controls, comm=comm)
    (n, N), _ = minv.getSizes()
    H0 = PETSc.Mat().createPython(((n, N), (n, N)), _Empty(), comm=comm)
    H0.setOption(PETSc.Mat.Option.SYMMETRIC, True)
    H0.setUp()
    pc = PETSc.PC().createPython(_ScaledMinvPC(minv, c), comm=comm)
    pc.setOperators(H0)
    pc.setUp()
    tao.setLMVMH0(H0)
    ksp = tao.getLMVMH0KSP()
    ksp.setType(PETSc.KSP.Type.PREONLY)
    ksp.setTolerances(rtol=0.0, atol=0.0, max_it=1)
    ksp.setPC(pc)
    solver._keep = (minv, H0, pc)
    solver.solve()
    return tao.getIterationNumber()


def main():
    graded = f"{MESH_DIR}/graded_R16.msh"

    print("=== Test 1: canonical TAO on uniform meshes (looks fine) ===")
    print(f"{'mesh':>14}  {'iterations':>10}")
    for n in (32, 64, 128):
        its = run_canonical(build(UnitSquareMesh(n, n)))
        print(f"{'uniform N=' + str(n):>14}  {its:>10}", flush=True)

    print("\n=== Test 2: graded R16, sweep tao_lmvm_mat_lmvm_scale_type ===")
    print(f"{'scale_type':>14}  {'iterations':>10}")
    for st in ("none", "scalar", "diagonal"):
        its = run_canonical(build(Mesh(graded)),
                            extra={"tao_lmvm_mat_lmvm_scale_type": st})
        print(f"{st:>14}  {its:>10}", flush=True)

    print("\n=== Test 3: graded R16, scale H0 = c * M^-1, sweep c ===")
    print(f"{'c':>14}  {'iterations':>10}")
    print(f"{'none (c=1)':>14}  {run_canonical(build(Mesh(graded))):>10}",
          flush=True)
    for c in (1.0, 1e1, 1e2, 1e3, 1e4):
        print(f"{c:>14.0e}  {run_scaled(build(Mesh(graded)), c):>10}",
              flush=True)

    print("\nTest 1: nothing wrong on uniform meshes.")
    print("Test 2: the standard scaling knob has no effect -> it is disabled.")
    print("Test 3: count swings with a constant scale -> H0 is fixed, not"
          " rescaled.")


if __name__ == "__main__":
    main()
