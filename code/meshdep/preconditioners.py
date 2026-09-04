"""riesz map pcs for the inner ksp. the AuxiliaryOperatorPC route silently fails on pyadjoints python mat so RieszMapPCContext is the one actually used"""
from firedrake import (
    AuxiliaryOperatorPC,
    TestFunction,
    TrialFunction,
    assemble,
    dx,
    grad,
    inner,
)
from firedrake.petsc import PETSc


class RieszMapL2(AuxiliaryOperatorPC):

    """auxiliary operator pc returning the mass form. doesnt attach to pyadjoints mat, kept for the record"""
    _prefix = "riesz_l2_"

    def form(self, pc, v, u):
        a = inner(u, v) * dx
        bcs = None
        return (a, bcs)


class RieszMapH1(AuxiliaryOperatorPC):

    """same but mass + stiffness"""
    _prefix = "riesz_h1_"

    def form(self, pc, v, u):
        a = (inner(u, v) + inner(grad(u), grad(v))) * dx
        bcs = None
        return (a, bcs)


class RieszMapPCContext:

    """plain petsc python pc. assembles Mh (or Mh+Kh) on V and applies the inverse by lu"""
    def __init__(self, V, riesz_map="L2"):
        self.V = V
        if riesz_map not in ("L2", "H1"):
            raise ValueError(f"riesz_map must be 'L2' or 'H1', got {riesz_map}")
        self.riesz_map = riesz_map
        self._ksp = None

    def setUp(self, pc):


        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        if self.riesz_map == "L2":
            a = inner(u, v) * dx
        else:
            a = (inner(u, v) + inner(grad(u), grad(v))) * dx


        M = assemble(a, mat_type='aij')
        ksp = PETSc.KSP().create(comm=pc.getComm())

        ksp.setOperators(M.M.handle)
        ksp.setType('preonly')
        ksp.getPC().setType('lu')
        ksp.setUp()

        self._M = M
        self._ksp = ksp

    def apply(self, pc, x, y):
        self._ksp.solve(x, y)

    def applyTranspose(self, pc, x, y):

        self._ksp.solve(x, y)
