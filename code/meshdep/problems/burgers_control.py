"""Scalar viscous Burgers optimal control problem (Firedrake + pyadjoint).

Minimise
    J(u, m) = (1/2) integral_0^T ||u(t) - d||^2 dt + (alpha/2) ||m||^2
subject to
    u_t + u u_x - nu u_xx = m   on (0, T) x (0, 1),
    u = 0                        on the boundary,
    u(0, x) = u_0(x).

The control ``m`` is constant in time. The forward problem is solved
with backward Euler in time and continuous P1 elements in space.
Sibling of the plain-Python ``burgers.py``, which builds the same
problem from scratch.
"""

from argparse import ArgumentParser

from firedrake import *
from firedrake.adjoint import (
    Control,
    ReducedFunctional,
    continue_annotation,
    pause_annotation,
    set_working_tape,
)

from meshdep.optimisers import solve_with_tao


def solve_burgers_control(mesh, alpha=1.0e-4, nu=1.0e-3,
                          T=0.5, n_steps=25,
                          riesz_map="L2", tao_gatol=1.0e-7,
                          tao_max_funcs=50000, verbose=False):
    """Tape the forward Burgers solve, build the reduced functional, run TAO.

    ``riesz_map`` is the inner product on the control space
    (``'l2'``, ``'L2'`` or ``'H1'``). ``TAOSolver`` uses it for both
    the LMVM initial Hessian and the convergence norm.
    ``tao_max_funcs`` caps the number of forward solves, since Burgers
    is much more expensive per iteration than Poisson.
    """
    x = SpatialCoordinate(mesh)[0]
    V = FunctionSpace(mesh, "CG", 1)
    bc = DirichletBC(V, 0.0, "on_boundary")

    u_old = Function(V, name="u_old")
    u_new = Function(V, name="u")
    m = Function(V, name="m")
    v = TestFunction(V)

    u0 = Function(V).interpolate(sin(pi * x))
    d = Function(V).interpolate(0.5 * sin(pi * x))

    dt = T / n_steps
    dt_c = Constant(dt)

    # Backward-Euler Burgers residual at time t_{n+1}.
    F = ((u_new - u_old) / dt_c) * v * dx \
        + u_new * u_new.dx(0) * v * dx \
        + nu * u_new.dx(0) * v.dx(0) * dx \
        - m * v * dx

    fwd_params = {
        'snes_type': 'newtonls',
        'ksp_type': 'preonly',
        'pc_type': 'lu',
    }

    m.zero()
    continue_annotation()
    with set_working_tape() as tape:
        u_old.assign(u0)
        J_val = 0.0
        for _ in range(n_steps):
            solve(F == 0, u_new, bcs=bc, solver_parameters=fwd_params)
            J_val = J_val + dt * assemble(0.5 * (u_new - d) ** 2 * dx)
            u_old.assign(u_new)
        J_val = J_val + assemble(0.5 * alpha * m ** 2 * dx)
        Jhat = ReducedFunctional(
            J_val, Control(m, riesz_map=riesz_map), tape=tape,
        )
    pause_annotation()

    return solve_with_tao(
        Jhat, tao_gatol=tao_gatol, tao_max_funcs=tao_max_funcs,
        verbose=verbose,
    )


def main():
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("resolution", type=int,
                        help="Mesh resolution (number of intervals).")
    parser.add_argument("--alpha", type=float, default=1.0e-4)
    parser.add_argument("--nu", type=float, default=1.0e-3)
    parser.add_argument("--T", type=float, default=0.5)
    parser.add_argument("--n-steps", type=int, default=25)
    parser.add_argument("--riesz-map", choices=["l2", "L2", "H1"],
                        default="L2",
                        help="Riesz map on the control (default: L2).")
    parser.add_argument("--tao-gatol", type=float, default=1.0e-7)
    parser.add_argument("--tao-max-funcs", type=int, default=50000)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    mesh = UnitIntervalMesh(args.resolution)
    result = solve_burgers_control(
        mesh, alpha=args.alpha, nu=args.nu,
        T=args.T, n_steps=args.n_steps,
        riesz_map=args.riesz_map,
        tao_gatol=args.tao_gatol,
        tao_max_funcs=args.tao_max_funcs,
        verbose=args.verbose,
    )

    print(f"resolution    = {args.resolution}")
    print(f"riesz_map     = {args.riesz_map}")
    print(f"iterations    = {result['iterations']}")
    print(f"final J       = {result['final_J']:.6e}")
    print(f"converged     = {result['converged']}")


if __name__ == "__main__":
    main()
