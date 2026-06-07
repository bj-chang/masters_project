"""Scalar viscous Burgers optimal control problem in 1D.

Forward equation:

    u_t + u u_x - nu u_xx = m   in (0, T) x (0, 1),
    u = 0                        on the boundary,
    u(0, x) = u_0(x).

Plain Python / NumPy / SciPy implementation: P1 in space, backward
Euler in time, Newton's method for the nonlinear term, with a
discrete adjoint and a Taylor test for the optimisation pipeline.

Subcommands:
  ``forward``     - single forward solve at default settings
  ``convergence`` - MMS convergence study 
  ``taylor``      - Taylor test for the gradient 
"""

from argparse import ArgumentParser
import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# Mesh, mass and stiffness matrices, and quadrature

# Two-point Gauss quadrature on the reference interval [-1, 1].
GAUSS_POINTS = np.array([-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)])
GAUSS_WEIGHTS = np.array([1.0, 1.0])


def make_uniform_mesh_1d(num_elements, a=0.0, b=1.0):
    """Build a uniform 1D mesh on [a, b].

    Returns the node coordinates and an array of element connectivity,
    where each row contains the two node indices of one interval element.
    """

    x = np.linspace(a, b, num_elements + 1)
    elements = np.column_stack([
        np.arange(num_elements),
        np.arange(1, num_elements + 1),
    ])
    return x, elements


def assemble_p1_mass_stiffness_1d(x, elements):
    """Assemble the global P1 mass and stiffness matrices on a 1D mesh.

    For each element [x_i, x_{i+1}] we use the local matrices

        M_e = (h/6) [[2, 1],          K_e = (1/h) [[ 1, -1],
                     [1, 2]],                      [-1,  1]],

    and accumulate them into sparse global matrices.
    """

    n_nodes = len(x)
    mass_rows, mass_cols, mass_vals = [], [], []
    stiff_rows, stiff_cols, stiff_vals = [], [], []

    mass_ref = np.array([[2.0, 1.0], [1.0, 2.0]]) / 6.0
    stiff_ref = np.array([[1.0, -1.0], [-1.0, 1.0]])

    for elem in elements:
        i, j = elem
        h = x[j] - x[i]
        M_loc = h * mass_ref
        K_loc = (1.0 / h) * stiff_ref
        for a in range(2):
            for b in range(2):
                mass_rows.append(elem[a])
                mass_cols.append(elem[b])
                mass_vals.append(M_loc[a, b])
                stiff_rows.append(elem[a])
                stiff_cols.append(elem[b])
                stiff_vals.append(K_loc[a, b])

    M = sp.csr_matrix((mass_vals, (mass_rows, mass_cols)),
                      shape=(n_nodes, n_nodes))
    K = sp.csr_matrix((stiff_vals, (stiff_rows, stiff_cols)),
                      shape=(n_nodes, n_nodes))
    return M, K


def assemble_load_vector(x, elements, source_fn, t):
    """Assemble the load vector for a source m(x, t) by Gauss quadrature."""

    F = np.zeros(len(x))
    for elem in elements:
        i, j = elem
        x_left, x_right = x[i], x[j]
        h = x_right - x_left
        local = np.zeros(2)
        for xi, w in zip(GAUSS_POINTS, GAUSS_WEIGHTS):
            x_q = 0.5 * (x_left + x_right) + 0.5 * h * xi
            phi = np.array([(1.0 - xi) / 2.0, (1.0 + xi) / 2.0])
            local += w * float(source_fn(x_q, t)) * phi * (h / 2.0)
        F[elem] += local
    return F


def assemble_convection_residual_and_jacobian(x, elements, U):
    """Assemble the nonlinear convection residual c(U; v) and its Jacobian.

    The Burgers convection term is treated in the form

        c(U; v) = integral of u_h (u_h)_x v_h dx,

    so that on a P1 element with constant (u_h)_x and linear u_h the
    local contribution is straightforward to write down by hand.
    """

    n_nodes = len(x)
    R_conv = np.zeros(n_nodes)
    jac_rows, jac_cols, jac_vals = [], [], []
    mass_ref = np.array([[2.0, 1.0], [1.0, 2.0]]) / 6.0

    # Local derivative pattern: if u_loc = [u_i, u_{i+1}], then
    # u_x = (-u_i + u_{i+1}) / h.
    derivative_vector = np.array([-1.0, 1.0])

    for elem in elements:
        i, j = elem
        h = x[j] - x[i]
        u_loc = U[elem]
        M_loc = h * mass_ref
        q = M_loc @ u_loc
        grad_u = (derivative_vector @ u_loc) / h

        R_loc = grad_u * q
        J_loc = np.outer(q, derivative_vector) / h + grad_u * M_loc

        R_conv[elem] += R_loc
        for a in range(2):
            for b in range(2):
                jac_rows.append(elem[a])
                jac_cols.append(elem[b])
                jac_vals.append(J_loc[a, b])

    J_conv = sp.csr_matrix((jac_vals, (jac_rows, jac_cols)),
                           shape=(n_nodes, n_nodes))
    return R_conv, J_conv


# Forward solve with a callable source term

def solve_forward_burgers_1d(
    num_elements=80,
    T=0.1,
    num_steps=400,
    nu=0.05,
    u0_fn=lambda x: np.sin(np.pi * x),
    source_fn=lambda x, t: 0.0,
    newton_tol=1e-10,
    max_newton_iters=25,
):
    """Solve the forward scalar viscous Burgers equation in 1D.

    P1 in space, backward Euler in time, Newton at each implicit step.
    The source ``source_fn(x, t)`` is supplied as a callable.

    Returns a dict with the mesh, time grid, full state history, mass
    and stiffness matrices, and per-step Newton iteration counts.
    """

    x, elements = make_uniform_mesh_1d(num_elements)
    M, K = assemble_p1_mass_stiffness_1d(x, elements)

    dt = T / num_steps
    times = np.linspace(0.0, T, num_steps + 1)

    U_prev = np.array([u0_fn(x_i) for x_i in x], dtype=float)
    U_prev[0] = 0.0
    U_prev[-1] = 0.0

    states = [U_prev.copy()]
    newton_iterations = []

    # The linear part of the Jacobian is fixed across Newton iterations.
    linear_part = M / dt + nu * K

    for n in range(num_steps):
        t_next = times[n + 1]
        F = assemble_load_vector(x, elements, source_fn, t_next)
        U = U_prev.copy()

        for k in range(max_newton_iters):
            C, J_C = assemble_convection_residual_and_jacobian(x, elements, U)
            R = M @ ((U - U_prev) / dt) + C + nu * (K @ U) - F
            J = linear_part + J_C

            R_int = R[1:-1]
            J_int = J[1:-1, 1:-1]
            delta_int = spla.spsolve(J_int, -R_int)
            U[1:-1] += delta_int

            if np.linalg.norm(delta_int, ord=np.inf) < newton_tol:
                newton_iterations.append(k + 1)
                break
        else:
            raise RuntimeError(f"Newton did not converge at time step {n + 1}.")

        U[0] = 0.0
        U[-1] = 0.0
        states.append(U.copy())
        U_prev = U

    return {
        "x": x,
        "elements": elements,
        "times": times,
        "states": np.array(states),
        "M": M,
        "K": K,
        "dt": dt,
        "nu": nu,
        "newton_iterations": np.array(newton_iterations),
    }


# Manufactured solution and convergence study

def manufactured_exact_solution(t, x):
    """Smooth exact solution u_exact(t, x) = exp(-t) sin(pi x)."""

    return np.exp(-t) * np.sin(np.pi * x)


def manufactured_source(x, t, nu=0.05):
    """Source term so that the manufactured exact solution satisfies Burgers."""

    u    =  np.exp(-t) * np.sin(np.pi * x)
    u_x  =  np.exp(-t) * np.pi * np.cos(np.pi * x)
    u_xx = -np.exp(-t) * (np.pi**2) * np.sin(np.pi * x)
    u_t  = -np.exp(-t) * np.sin(np.pi * x)
    return u_t + u * u_x - nu * u_xx


def compute_l2_error(x, elements, uh, exact_fn, t):
    """Compute ||u_h - u_exact||_{L2} on the mesh by Gauss quadrature."""

    error_sq = 0.0
    for elem in elements:
        i, j = elem
        x_left, x_right = x[i], x[j]
        h = x_right - x_left
        u_loc = uh[elem]
        for xi, w in zip(GAUSS_POINTS, GAUSS_WEIGHTS):
            x_q = 0.5 * (x_left + x_right) + 0.5 * h * xi
            phi = np.array([(1.0 - xi) / 2.0, (1.0 + xi) / 2.0])
            u_h_q = u_loc @ phi
            diff = u_h_q - exact_fn(t, x_q)
            error_sq += w * diff**2 * (h / 2.0)
    return math.sqrt(error_sq)


def compute_discrete_space_time_l2_error(result, exact_fn):
    """Compute (dt sum_n ||u_h^n - u_exact(t_n)||^2)^(1/2)."""

    dt = result["dt"]
    x = result["x"]
    elements = result["elements"]
    states = result["states"]
    times = result["times"]

    error_sq = 0.0
    for U_n, t_n in zip(states[1:], times[1:]):
        slice_error = compute_l2_error(x, elements, U_n, exact_fn, t_n)
        error_sq += dt * slice_error**2
    return math.sqrt(error_sq)


def convergence_test(mesh_sizes=(10, 20, 40, 80), T=0.1, nu=0.05,
                     steps_per_element=8):
    """Run the manufactured-solution convergence study.

    Refines space and time together by setting
    ``num_steps = steps_per_element * num_elements`` so that dt is
    proportional to h. Returns a list of dictionaries with the errors
    and observed rates for each refinement level, plus the average
    number of Newton iterations per time step.
    """

    rows = []
    previous_final = None
    previous_spacetime = None

    for N in mesh_sizes:
        num_steps = steps_per_element * N

        result = solve_forward_burgers_1d(
            num_elements=N,
            T=T,
            num_steps=num_steps,
            nu=nu,
            u0_fn=lambda x: manufactured_exact_solution(0.0, x),
            source_fn=lambda x, t: manufactured_source(x, t, nu=nu),
        )

        final_error = compute_l2_error(
            result["x"], result["elements"], result["states"][-1],
            manufactured_exact_solution, T,
        )
        spacetime_error = compute_discrete_space_time_l2_error(
            result, manufactured_exact_solution,
        )

        final_rate = None if previous_final is None else \
            math.log(previous_final / final_error, 2.0)
        spacetime_rate = None if previous_spacetime is None else \
            math.log(previous_spacetime / spacetime_error, 2.0)

        rows.append({
            "num_elements": N,
            "num_steps": num_steps,
            "h": 1.0 / N,
            "final_time_L2_error": final_error,
            "final_time_rate": final_rate,
            "discrete_space_time_L2_error": spacetime_error,
            "space_time_rate": spacetime_rate,
            "avg_newton_iters": float(result["newton_iterations"].mean()),
        })

        previous_final = final_error
        previous_spacetime = spacetime_error

    return rows


def print_convergence_table(rows):
    """Pretty-print the output of ``convergence_test``."""

    header = (
        f"{'N':>4}  {'Nt':>5}  {'h':>8}  "
        f"{'final L2':>14}  {'rate':>6}  "
        f"{'space-time L2':>14}  {'rate':>6}  {'Newton':>6}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        fr = "  --  " if row["final_time_rate"] is None \
             else f"{row['final_time_rate']:6.4f}"
        sr = "  --  " if row["space_time_rate"] is None \
             else f"{row['space_time_rate']:6.4f}"
        print(
            f"{row['num_elements']:>4}  {row['num_steps']:>5}  "
            f"{row['h']:>8.4f}  "
            f"{row['final_time_L2_error']:>14.6e}  {fr}  "
            f"{row['discrete_space_time_L2_error']:>14.6e}  {sr}  "
            f"{row['avg_newton_iters']:>6.2f}"
        )


# Forward solve with a control history (one nodal vector per step)

def build_control_history_from_callable(x, times, control_fn):
    """Sample a callable control m(x, t) at the mesh nodes for each implicit step.

    Returns an array of shape ``(num_steps, num_nodes)`` where row n
    contains the nodal values of the control used in the step from
    t_n to t_{n+1}.
    """

    control_history = np.zeros((len(times) - 1, len(x)))
    for n, t in enumerate(times[1:]):
        control_history[n] = np.array([control_fn(x_i, t) for x_i in x],
                                      dtype=float)
    return control_history


def solve_forward_burgers_with_control(
    num_elements=80,
    T=0.2,
    num_steps=200,
    nu=0.05,
    u0_fn=lambda x: 0.0,
    control_history=None,
    control_fn=None,
    newton_tol=1e-10,
    max_newton_iters=25,
):
    """Forward Burgers solve with a control supplied as a time-indexed array.

    Same as ``solve_forward_burgers_1d`` but the source is a P1
    control field per time step, so the load is ``M @ m^n``.
    """

    x, elements = make_uniform_mesh_1d(num_elements)
    M, K = assemble_p1_mass_stiffness_1d(x, elements)

    dt = T / num_steps
    times = np.linspace(0.0, T, num_steps + 1)

    if control_history is None:
        if control_fn is None:
            control_fn = lambda x, t: 0.0
        control_history = build_control_history_from_callable(
            x, times, control_fn,
        )
    else:
        control_history = np.asarray(control_history, dtype=float)
        expected_shape = (num_steps, len(x))
        if control_history.shape != expected_shape:
            raise ValueError(
                f"control_history should have shape {expected_shape}, "
                f"got {control_history.shape}."
            )

    U_prev = np.array([u0_fn(x_i) for x_i in x], dtype=float)
    U_prev[0] = 0.0
    U_prev[-1] = 0.0

    states = [U_prev.copy()]
    newton_iterations = []
    linear_part = M / dt + nu * K

    for n in range(num_steps):
        F = M @ control_history[n]
        U = U_prev.copy()
        for k in range(max_newton_iters):
            C, J_C = assemble_convection_residual_and_jacobian(x, elements, U)
            R = M @ ((U - U_prev) / dt) + C + nu * (K @ U) - F
            J = linear_part + J_C
            R_int = R[1:-1]
            J_int = J[1:-1, 1:-1]
            delta_int = spla.spsolve(J_int, -R_int)
            U[1:-1] += delta_int
            if np.linalg.norm(delta_int, ord=np.inf) < newton_tol:
                newton_iterations.append(k + 1)
                break
        else:
            raise RuntimeError(f"Newton did not converge at time step {n + 1}.")
        U[0] = 0.0
        U[-1] = 0.0
        states.append(U.copy())
        U_prev = U

    return {
        "x": x,
        "elements": elements,
        "M": M,
        "K": K,
        "dt": dt,
        "times": times,
        "nu": nu,
        "states": np.array(states),
        "control_history": control_history,
        "newton_iterations": np.array(newton_iterations),
    }



# Tracking objective, discrete adjoint, and L2-gradient

def evaluate_tracking_objective(result, target_history, alpha):
    """Evaluate the fully discrete tracking objective

        J_h = (dt/2) sum_n ||U^n - D^n||_M^2
            + (alpha dt/2) sum_n ||m^n||_M^2,

    where ||v||_M^2 = v^T M v is the mass-matrix L2 norm.
    """

    M = result["M"]
    dt = result["dt"]
    states = result["states"][1:]
    control_history = result["control_history"]
    target_history = np.asarray(target_history, dtype=float)

    if target_history.shape != states.shape:
        raise ValueError(
            f"target_history should have shape {states.shape}, "
            f"got {target_history.shape}."
        )

    total = 0.0
    for U_n, D_n, m_n in zip(states, target_history, control_history):
        state_error = U_n - D_n
        total += 0.5 * dt * (state_error @ (M @ state_error))
        total += 0.5 * alpha * dt * (m_n @ (M @ m_n))
    return float(total)


def solve_discrete_adjoint(result, target_history):
    """Solve the discrete adjoint equation backward in time.

    Derived from the fully discrete forward scheme directly, not from
    discretising the continuous adjoint PDE.
    """

    x = result["x"]
    elements = result["elements"]
    M = result["M"]
    K = result["K"]
    dt = result["dt"]
    nu = result["nu"]
    states = result["states"]

    num_steps = len(result["times"]) - 1
    target_history = np.asarray(target_history, dtype=float)

    adjoints = np.zeros((num_steps, len(x)))
    p_next = np.zeros(len(x))

    for n in reversed(range(num_steps)):
        U_n = states[n + 1]
        _, J_conv = assemble_convection_residual_and_jacobian(
            x, elements, U_n,
        )
        A_n = M / dt + nu * K + J_conv

        rhs = dt * (M @ (U_n - target_history[n]))
        if n < num_steps - 1:
            rhs += (M @ p_next) / dt

        p_int = spla.spsolve(A_n[1:-1, 1:-1].T, rhs[1:-1])
        adjoints[n, 1:-1] = p_int
        p_next = adjoints[n].copy()

    return adjoints


def compute_discrete_l2_gradient(result, target_history, alpha):
    """Compute the discrete L2-gradient grad^n = alpha m^n + p^n / dt.

    This is the discrete analogue of the continuous formula
    ``grad J(m) = lambda + alpha m`` from the dissertation.
    """

    adjoints = solve_discrete_adjoint(result, target_history)
    gradient = alpha * result["control_history"] + adjoints / result["dt"]
    return gradient, adjoints


def control_inner_product(control_a, control_b, M, dt):
    """Discrete L2-like inner product on control histories.

        <a, b> = dt sum_n a_n^T M b_n.
    """

    total = 0.0
    for a_n, b_n in zip(control_a, control_b):
        total += dt * (a_n @ (M @ b_n))
    return float(total)


# Taylor test


def reduced_objective_only(control_history, num_elements, T, num_steps,
                           nu, u0_fn, target_history, alpha):
    """Solve the forward problem for a given control and return J_h(m)."""

    result = solve_forward_burgers_with_control(
        num_elements=num_elements,
        T=T,
        num_steps=num_steps,
        nu=nu,
        u0_fn=u0_fn,
        control_history=control_history,
    )
    return evaluate_tracking_objective(result, target_history, alpha)


def run_taylor_test(base_control_history, direction_history, target_history,
                    num_elements, T, num_steps, nu, u0_fn, alpha,
                    h_values=(1e-1, 5e-2, 2.5e-2, 1.25e-2, 6.25e-3)):
    """Run a Taylor-style gradient test for the reduced objective.

    Returns ``(rows, directional_derivative)``, where ``rows`` is a
    list of dicts with the first- and second-order remainders and
    their observed convergence rates.
    """

    base_result = solve_forward_burgers_with_control(
        num_elements=num_elements,
        T=T,
        num_steps=num_steps,
        nu=nu,
        u0_fn=u0_fn,
        control_history=base_control_history,
    )

    J0 = evaluate_tracking_objective(base_result, target_history, alpha)
    gradient, _ = compute_discrete_l2_gradient(
        base_result, target_history, alpha,
    )

    direction_norm = math.sqrt(control_inner_product(
        direction_history, direction_history,
        base_result["M"], base_result["dt"],
    ))
    direction_unit = direction_history / direction_norm

    directional_derivative = control_inner_product(
        gradient, direction_unit,
        base_result["M"], base_result["dt"],
    )

    rows = []
    previous_first = None
    previous_second = None
    for h in h_values:
        Jh = reduced_objective_only(
            base_control_history + h * direction_unit,
            num_elements=num_elements, T=T, num_steps=num_steps, nu=nu,
            u0_fn=u0_fn, target_history=target_history, alpha=alpha,
        )
        first_remainder = abs(Jh - J0)
        second_remainder = abs(Jh - J0 - h * directional_derivative)

        first_rate = None if previous_first is None else \
            math.log(previous_first / first_remainder) / math.log(2.0)
        second_rate = None if previous_second is None else \
            math.log(previous_second / second_remainder) / math.log(2.0)

        rows.append({
            "h": h,
            "first_remainder": first_remainder,
            "first_rate": first_rate,
            "second_remainder": second_remainder,
            "second_rate": second_rate,
        })
        previous_first = first_remainder
        previous_second = second_remainder

    return rows, directional_derivative


def print_taylor_table(rows):
    """Pretty-print the output of ``run_taylor_test``."""

    header = (f"{'h':>10}  {'|R1|':>14}  {'rate':>6}  "
              f"{'|R2|':>14}  {'rate':>6}")
    print(header)
    print("-" * len(header))
    for row in rows:
        r1 = "  --  " if row["first_rate"] is None \
             else f"{row['first_rate']:6.4f}"
        r2 = "  --  " if row["second_rate"] is None \
             else f"{row['second_rate']:6.4f}"
        print(
            f"{row['h']:>10.4e}  "
            f"{row['first_remainder']:>14.6e}  {r1}  "
            f"{row['second_remainder']:>14.6e}  {r2}"
        )


# CLI

def main():
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_fwd = sub.add_parser("forward",
                           help="Run one forward Burgers solve.")
    p_fwd.add_argument("--num-elements", type=int, default=80)
    p_fwd.add_argument("--num-steps", type=int, default=400)
    p_fwd.add_argument("--T", type=float, default=0.1)
    p_fwd.add_argument("--nu", type=float, default=0.05)

    p_conv = sub.add_parser("convergence",
                            help="Run the MMS convergence study.")
    p_conv.add_argument("--mesh-sizes", type=int, nargs="+",
                        default=[10, 20, 40, 80])
    p_conv.add_argument("--T", type=float, default=0.1)
    p_conv.add_argument("--nu", type=float, default=0.05)

    p_tay = sub.add_parser("taylor",
                           help="Run the Taylor-test for the gradient.")
    p_tay.add_argument("--num-elements", type=int, default=20)
    p_tay.add_argument("--num-steps", type=int, default=60)
    p_tay.add_argument("--T", type=float, default=0.2)
    p_tay.add_argument("--nu", type=float, default=0.05)
    p_tay.add_argument("--alpha", type=float, default=1e-3)

    args = parser.parse_args()

    if args.command == "forward":
        result = solve_forward_burgers_1d(
            num_elements=args.num_elements,
            T=args.T,
            num_steps=args.num_steps,
            nu=args.nu,
            u0_fn=lambda x: math.sin(math.pi * x),
        )
        print(f"final time = {result['times'][-1]:.4f}")
        print(f"max |u(T)| = {np.max(np.abs(result['states'][-1])):.6e}")
        print(f"avg Newton iterations = "
              f"{result['newton_iterations'].mean():.2f}")

    elif args.command == "convergence":
        rows = convergence_test(
            mesh_sizes=tuple(args.mesh_sizes),
            T=args.T,
            nu=args.nu,
        )
        print_convergence_table(rows)

    elif args.command == "taylor":
        # Build a target trajectory from a smooth reference control.
        x, elements = make_uniform_mesh_1d(args.num_elements)
        times = np.linspace(0.0, args.T, args.num_steps + 1)
        u0_fn = lambda x: 0.0

        reference_control = build_control_history_from_callable(
            x, times,
            lambda x, t: math.sin(math.pi * x) * math.cos(2.0 * t),
        )
        ref = solve_forward_burgers_with_control(
            num_elements=args.num_elements, T=args.T,
            num_steps=args.num_steps, nu=args.nu, u0_fn=u0_fn,
            control_history=reference_control,
        )
        target_history = ref["states"][1:]

        trial_control = build_control_history_from_callable(
            x, times, lambda x, t: 0.5 * math.sin(math.pi * x),
        )
        direction_history = build_control_history_from_callable(
            x, times,
            lambda x, t: (math.sin(math.pi * x)
                          + 0.3 * math.sin(2.0 * math.pi * x)),
        )

        rows, dJ = run_taylor_test(
            base_control_history=trial_control,
            direction_history=direction_history,
            target_history=target_history,
            num_elements=args.num_elements, T=args.T,
            num_steps=args.num_steps, nu=args.nu, u0_fn=u0_fn,
            alpha=args.alpha,
        )
        print(f"directional derivative = {dJ:.6e}")
        print_taylor_table(rows)


if __name__ == "__main__":
    main()
