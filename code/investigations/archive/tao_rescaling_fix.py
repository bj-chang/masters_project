"""Get *standard* library TAO/LMVM to rescale, via a change of variables.

The problem: pyadjoint's TAOSolver hands TAO a fixed initial Hessian
H0 = M^{-1}, which switches off TAO's built-in scalar rescaling (proved
in tao_rescaling_proof.py). We do not want to hand-pick a scale or write
our own optimiser.

The fix: solve in a new variable x = D m, where D = diag(sqrt(lumped
mass)). In this variable the ordinary Euclidean (l2) inner product
equals the L2 inner product on m:

    <x, x'>_l2 = sum_i ML_i m_i m'_i = <m, m'>_L2 (lumped).

So we run *plain* TAO/LMVM on x in l2 -- no setLMVMH0, no
setGradientNorm. Plain TAO/LMVM rescales its initial Hessian by
gamma_k = <s,y>/<y,y> by default, so the rescaling happens
automatically, and because l2-on-x = L2-on-m it happens in the L2
metric. TAO's default l2 gradient-norm stopping test also equals the
L2 gradient norm in this variable, so gatol = 1e-7 still means the
L2 gradient.

The change of variables is applied purely at the vector level inside
the callbacks; the tape is the ordinary J(m). The optimiser is
unmodified library TAO.

Run from ~/masters_project/.
"""

import numpy as np

from firedrake import *
from firedrake.adjoint import *
from firedrake.petsc import PETSc

ALPHA = 1.0e-4
MESH_DIR = "code/mesh_generation"
LU = {"ksp_type": "preonly", "pc_type": "lu"}


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
    return Jhat, V


def solve_tao_changevar(Jhat, V, gatol=1.0e-7, history=5, max_it=2000):
    mfun = Function(V)
    ndof = mfun.dat.data.shape[0]

    # lumped mass = row sums of the mass matrix = integral of each basis fn
    ML = assemble(TestFunction(V) * dx).dat.data_ro.copy()
    dvec = np.sqrt(ML)                       # D = diag(dvec),  m = x / dvec

    def objective(tao, x):
        mfun.dat.data[:] = x.getArray(readonly=True) / dvec
        return float(Jhat(mfun))

    def gradient(tao, x, g):
        mfun.dat.data[:] = x.getArray(readonly=True) / dvec
        Jhat(mfun)
        dJdm = Jhat.derivative().dat.data_ro      # dual gradient wrt m
        g.setArray(dJdm / dvec)                   # chain rule: dJ/dx = D^-1 dJ/dm

    tao = PETSc.TAO().create(comm=PETSc.COMM_SELF)
    tao.setType(PETSc.TAO.Type.LMVM)

    xvec = PETSc.Vec().createSeq(ndof, comm=PETSc.COMM_SELF)
    xvec.set(0.0)                                 # m0 = 0  ->  x0 = 0
    gvec = xvec.duplicate()
    tao.setSolution(xvec)
    tao.setObjective(objective)
    tao.setGradient(gradient, gvec)

    opts = PETSc.Options()
    for k, val in {"tao_type": "lmvm", "tao_gatol": gatol,
                   "tao_grtol": 0.0, "tao_gttol": 0.0,
                   "tao_lmvm_num_vecs": history, "tao_max_it": max_it}.items():
        opts[k] = val
    tao.setFromOptions()
    tao.solve()
    its = tao.getIterationNumber()
    tao.destroy()
    return its


def main():
    uniform = [(f"uniform N={n}", UnitSquareMesh(n, n)) for n in (32, 64, 128)]
    graded = [(f"graded R{r}", Mesh(f"{MESH_DIR}/graded_R{r}.msh"))
              for r in (4, 16, 32)]

    print("\n=== standard TAO/LMVM in changed variable x = D m (l2 = L2) ===")
    print(f"{'mesh':>14}  {'TAO iterations':>14}")
    for label, mesh in uniform + graded:
        Jhat, V = build(mesh)
        print(f"{label:>14}  {solve_tao_changevar(Jhat, V):>14}", flush=True)


if __name__ == "__main__":
    main()
