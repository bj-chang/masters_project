"""Optimisation wrappers used by the experiment scripts.

``solve_with_tao`` and ``solve_with_scipy`` are wrappers around the
PETSc TAO and SciPy L-BFGS-B optimisers from pyadjoint.
``solve_with_scipy_external_check`` is the same SciPy run but with the
convergence test done in a chosen Hilbert-space norm rather than the
ambient l^2 norm. ``solve_with_hilbert_lbfgs`` is a small L-BFGS
implementation in which every dot product uses the inner product
chosen on the Control.
"""

from pyadjoint import MinimizationProblem, minimize
from pyadjoint.optimization.tao_solver import (
    TAOConvergenceError,
    TAOSolver,
)


class _ConvergedExternally(Exception):
    """Raised by a SciPy callback once our convergence test is met."""
    pass


def solve_with_tao(Jhat, tao_gatol=1.0e-7, tao_gttol=0.0, tao_grtol=0.0,
                   tao_max_it=50000, tao_max_funcs=50000, history=5,
                   verbose=False):
    """Run TAO/LMVM on the reduced functional.

    The defaults put TAO on the absolute test ``|g|_H <= gatol`` only,
    to match Schwedes' ``eps = 1e-7``. The H-norm used by the
    convergence test (and by the LMVM initial Hessian) is the one
    ``TAOSolver`` installs from the Control's ``riesz_map``; we do not
    override it.

    Returns a dict with keys ``m_opt``, ``iterations``, ``final_J``
    and ``converged``.
    """

    parameters = {
        "tao_type": "lmvm",
        "tao_gatol": tao_gatol,
        "tao_grtol": tao_grtol,
        "tao_gttol": tao_gttol,
        "tao_max_it": tao_max_it,
        "tao_max_funcs": tao_max_funcs,
        "tao_lmvm_num_vecs": history,
    }
    if verbose:
        parameters["tao_monitor"] = None

    solver = TAOSolver(MinimizationProblem(Jhat), parameters=parameters)

    try:
        m_opt = solver.solve()
        converged = True
    except TAOConvergenceError:
        m_opt = Jhat.controls[0].tape_value()
        converged = False

    iterations = solver.tao.getIterationNumber()
    final_J = float(Jhat(m_opt))

    return {
        "m_opt": m_opt,
        "iterations": iterations,
        "final_J": final_J,
        "converged": converged,
    }


def solve_with_scipy(Jhat, gtol=1.0e-7, maxiter=5000, method="L-BFGS-B"):
    """Run SciPy's L-BFGS-B. Returns a dict like ``solve_with_tao``."""

    iteration_counter = [0]

    def callback(_):
        iteration_counter[0] += 1

    options = {"gtol": gtol, "maxiter": maxiter}
    m_opt = minimize(Jhat, method=method, options=options,
                     callback=callback)
    final_J = float(Jhat(m_opt))
    converged = iteration_counter[0] < maxiter

    return {
        "m_opt": m_opt,
        "iterations": iteration_counter[0],
        "final_J": final_J,
        "converged": converged,
    }


def solve_with_hilbert_lbfgs(Jhat, eps=1.0e-7, max_iter=500, history=5,
                             verbose=False):
    """L-BFGS using the H-inner product on the Control throughout.

    The Control's ``riesz_map`` must be ``'L2'`` or ``'H1'``. Every dot
    product in the two-loop recursion uses ``<.,.>_H`` and convergence
    is declared once ``|g|_H <= eps`` with ``g`` the H-primal Riesz
    representer of ``dJ``. The Riesz solve uses LU so the gradient is
    computed to ~machine precision (Firedrake's default RieszMap uses
    CG with ``rtol=1e-5``, which would otherwise put a noise floor on
    ``|g|_H`` of order 1e-5).
    """

    from firedrake import (
        Cofunction, Function, LinearVariationalProblem,
        LinearVariationalSolver, TestFunction, TrialFunction,
        assemble, dx, grad, inner,
    )

    control = Jhat.controls[0]
    riesz_map = control.riesz_map
    V = control.control.function_space()

    if riesz_map == "L2":
        def H_inner(a, b):
            return float(assemble(inner(a, b) * dx))
    elif riesz_map == "H1":
        def H_inner(a, b):
            return float(assemble(
                (inner(a, b) + inner(grad(a), grad(b))) * dx
            ))
    else:
        raise ValueError(
            f"riesz_map must be 'L2' or 'H1', got {riesz_map!r}"
        )

    def H_norm(f):
        return H_inner(f, f) ** 0.5

    # Riesz solver: (M or M+K) * primal = dual.
    u_tr, v_te = TrialFunction(V), TestFunction(V)
    if riesz_map == "L2":
        riesz_form = inner(u_tr, v_te) * dx
    else:
        riesz_form = (
            inner(u_tr, v_te) + inner(grad(u_tr), grad(v_te))
        ) * dx
    _rhs = Cofunction(V.dual())
    _primal = Function(V)
    _riesz_solver = LinearVariationalSolver(
        LinearVariationalProblem(
            riesz_form, _rhs, _primal,
            constant_jacobian=True,
        ),
        solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
    )

    def primal_gradient():
        dual = Jhat.derivative()
        _rhs.assign(dual)
        _riesz_solver.solve()
        return Function(V).assign(_primal)

    m = Function(V).assign(control.tape_value())
    J_curr = float(Jhat(m))
    g_curr = primal_gradient()
    g_norm = H_norm(g_curr)

    if verbose:
        print(f"  iter   0: J={J_curr:.6e}  ||g||_H={g_norm:.6e}")

    s_history, y_history, rho_history = [], [], []

    for k in range(max_iter):
        if g_norm <= eps:
            return {
                "m_opt": m, "iterations": k, "final_J": J_curr,
                "converged": True, "final_grad_norm_H": g_norm,
            }

        # Two-loop recursion in the H-metric.
        q = Function(V).assign(g_curr)
        alphas = []
        for s_i, y_i, rho_i in zip(reversed(s_history),
                                    reversed(y_history),
                                    reversed(rho_history)):
            a = rho_i * H_inner(s_i, q)
            alphas.append(a)
            q.assign(q - a * y_i)
        # H_0 = gamma_k * I with gamma_k = <s,y>_H / <y,y>_H. Without
        # this rescaling the algorithm plateaus well above eps.
        if s_history:
            yy = H_inner(y_history[-1], y_history[-1])
            sy = 1.0 / rho_history[-1]
            gamma = sy / yy if yy > 0 else 1.0
        else:
            gamma = 1.0
        z = Function(V).assign(gamma * q)
        for s_i, y_i, rho_i, a in zip(s_history, y_history,
                                       rho_history, reversed(alphas)):
            beta = rho_i * H_inner(y_i, z)
            z.assign(z + (a - beta) * s_i)
        p = Function(V).assign(-1.0 * z)

        # Steepest descent fallback if p is not a descent direction.
        dphi0 = H_inner(g_curr, p)
        if dphi0 >= 0.0:
            p.assign(-1.0 * g_curr)
            dphi0 = -g_norm * g_norm

        # Backtracking Armijo.
        alpha = 1.0
        c1 = 1.0e-4
        m_trial = Function(V)
        line_search_ok = False
        for _ in range(50):
            m_trial.assign(m + alpha * p)
            J_trial = float(Jhat(m_trial))
            if J_trial <= J_curr + c1 * alpha * dphi0:
                line_search_ok = True
                break
            alpha *= 0.5
        if not line_search_ok:
            return {
                "m_opt": m, "iterations": k + 1, "final_J": J_curr,
                "converged": False, "final_grad_norm_H": g_norm,
            }

        g_new = primal_gradient()
        s_new = Function(V).assign(m_trial - m)
        y_new = Function(V).assign(g_new - g_curr)
        sy = H_inner(s_new, y_new)
        if sy > 1.0e-20:
            if len(s_history) >= history:
                s_history.pop(0)
                y_history.pop(0)
                rho_history.pop(0)
            s_history.append(s_new)
            y_history.append(y_new)
            rho_history.append(1.0 / sy)

        m.assign(m_trial)
        J_curr = J_trial
        g_curr.assign(g_new)
        g_norm = H_norm(g_curr)

        if verbose:
            print(f"  iter {k+1:>3d}: J={J_curr:.6e}  "
                  f"||g||_H={g_norm:.6e}  alpha={alpha:.3e}")

    return {
        "m_opt": m, "iterations": max_iter, "final_J": J_curr,
        "converged": False, "final_grad_norm_H": g_norm,
    }


def solve_with_scipy_external_check(Jhat, eps=1.0e-7,
                                    test_riesz_map="L2", maxiter=5000):
    """SciPy L-BFGS-B with the convergence test done in an H-norm.

    SciPy's own ``gtol`` and ``ftol`` are pinned to 1e-30 so they
    never trigger. A callback recomputes ``|g|_H`` at each iterate
    (with ``H = test_riesz_map``, either ``'L2'`` or ``'H1'``) and
    raises ``_ConvergedExternally`` when ``|g|_H <= eps``.
    """

    from firedrake import Function, assemble, dx, grad, inner

    inner_control = Jhat.controls[0]
    m_func = inner_control.control
    V = m_func.function_space()

    if test_riesz_map == "L2":
        def primal_norm_sq(f):
            return assemble(inner(f, f) * dx)
    elif test_riesz_map == "H1":
        def primal_norm_sq(f):
            return assemble(
                (inner(f, f) + inner(grad(f), grad(f))) * dx
            )
    else:
        raise ValueError(
            f"test_riesz_map must be 'L2' or 'H1', got {test_riesz_map!r}"
        )

    iteration_counter = [0]
    last_grad_norm = [None]

    def callback(xk):
        iteration_counter[0] += 1
        # Re-evaluate at xk so the gradient we read is at the current
        # iterate, not the previous one.
        m_xk = Function(V)
        m_xk.dat.data[:] = xk
        Jhat(m_xk)
        Jhat.derivative()
        primal = m_func._ad_convert_riesz(
            inner_control.block_variable.adj_value,
            riesz_map=test_riesz_map,
        )
        norm = float(primal_norm_sq(primal)) ** 0.5
        last_grad_norm[0] = norm
        if norm <= eps:
            raise _ConvergedExternally()

    options = {
        "gtol": 1.0e-30,
        "ftol": 1.0e-30,
        "maxiter": maxiter,
    }

    converged = False
    try:
        m_opt = minimize(Jhat, method="L-BFGS-B", options=options,
                         callback=callback)
        converged = (last_grad_norm[0] is not None
                     and last_grad_norm[0] <= eps)
    except _ConvergedExternally:
        m_opt = inner_control.tape_value()
        converged = True

    final_J = float(Jhat(m_opt))
    return {
        "m_opt": m_opt,
        "iterations": iteration_counter[0],
        "final_J": final_J,
        "converged": converged,
        "final_grad_norm_H": last_grad_norm[0],
    }
