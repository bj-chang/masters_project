"""optimiser wrappers: tao, scipy, scipy with the external convergence check, and the hilbert space lbfgs"""
from pyadjoint import MinimizationProblem, minimize
from pyadjoint.optimization.tao_solver import (
    TAOConvergenceError,
    TAOSolver,
)


class _ConvergedExternally(Exception):
    pass


def solve_with_tao(Jhat, tao_gatol=1.0e-7, tao_gttol=0.0, tao_grtol=0.0,
                   tao_max_it=50000, tao_max_funcs=50000, history=5,
                   verbose=False):

    """runs pyadjoints TAOSolver, returns the iteration count"""
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

    """plain scipy minimize on the l2 coefficients"""
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

    """two loop lbfgs with every inner product taken in H, armijo backtracking, riesz solve by lu"""
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


        q = Function(V).assign(g_curr)
        alphas = []
        for s_i, y_i, rho_i in zip(reversed(s_history),
                                    reversed(y_history),
                                    reversed(rho_history)):
            a = rho_i * H_inner(s_i, q)
            alphas.append(a)
            q.assign(q - a * y_i)


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


        dphi0 = H_inner(g_curr, p)
        if dphi0 >= 0.0:
            p.assign(-1.0 * g_curr)
            dphi0 = -g_norm * g_norm


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
                                    test_riesz_map="L2", maxiter=5000,
                                    maxcor=5):

    """scipy lbfgs-b with its own tolerances disabled, halted by an external check on the chosen gradient norm. maxcor=5 to match the hilbert lbfgs"""
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
        "maxcor": maxcor,
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
