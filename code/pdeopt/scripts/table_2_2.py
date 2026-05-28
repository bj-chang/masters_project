"""Reproduce Table 2.2 of Schwedes et al. (2017).

For each ``(mesh, optimiser, riesz_map)`` row of a panel, prints the
iteration count at convergence ``eps = 1e-7`` on
``||R_H(J')||_H``.
"""

from argparse import ArgumentParser

from pdeopt.meshes import (
    graded_unit_square_from_file,
    graded_unit_square_tensor,
)
from pdeopt.optimisers import (
    solve_with_hilbert_lbfgs,
    solve_with_scipy,
    solve_with_scipy_external_check,
)
from pdeopt.problems.poisson_control import solve_poisson_control


# The l^2 rows use SciPy (its native metric); the function-space row
# uses the Hilbert-space L-BFGS in pdeopt.optimisers. TAO LMVM is kept
# available but stalls on this problem and is not used by default.
PANEL_A_ROWS = [
    ("scipy_lbfgs_ext", "l2"),
    ("hilbert_lbfgs", "L2"),
]
PANEL_B_ROWS = [
    ("scipy_lbfgs_ext", "l2"),
    ("hilbert_lbfgs", "H1"),
]
# Convergence metric per panel: L^2 for (a), H^1 for (b).
PANEL_CONVERGENCE_RIESZ = {"a": "L2", "b": "H1"}


def _build_poisson_problem(mesh, alpha, riesz_map="l2"):
    """Build the reduced functional with the chosen Control riesz_map."""

    from firedrake import (
        DirichletBC, Function, FunctionSpace, SpatialCoordinate,
        TestFunction, assemble, dx, grad, inner, pi, sin, solve,
    )
    from firedrake.adjoint import (
        Control, ReducedFunctional, continue_annotation,
        pause_annotation, set_working_tape,
    )

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
        # LU: see note in pdeopt.problems.poisson_control.
        solve(F == 0, u, bcs=bc,
              solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
        J = assemble(0.5 * (u - d)**2 * dx + 0.5 * alpha * m**2 * dx)
        Jhat = ReducedFunctional(J, Control(m, riesz_map=riesz_map),
                                 tape=tape)
    pause_annotation()

    return Jhat


def run_panel(meshes, rows, alpha=1.0e-4, convergence_riesz_map="L2"):
    """Run every ``(mesh, optimiser, riesz_map)`` combination for a panel.

    Returns ``{(mesh_label, optimiser_name, riesz_map): iteration_count}``.
    """

    results = {}
    for label, mesh in meshes:
        for optimiser_name, riesz_map in rows:
            print(f"  {label}, {optimiser_name}, {riesz_map}...", flush=True)

            if optimiser_name == "tao_lmvm":
                result = solve_poisson_control(
                    mesh, alpha=alpha, riesz_map=riesz_map,
                    convergence_riesz_map=convergence_riesz_map,
                )
            elif optimiser_name == "hilbert_lbfgs":
                Jhat = _build_poisson_problem(
                    mesh, alpha, riesz_map=riesz_map,
                )
                result = solve_with_hilbert_lbfgs(
                    Jhat, eps=1.0e-7, max_iter=500, history=5,
                )
            elif optimiser_name == "scipy_lbfgs_ext":
                Jhat = _build_poisson_problem(
                    mesh, alpha, riesz_map="l2",
                )
                result = solve_with_scipy_external_check(
                    Jhat, eps=1.0e-7,
                    test_riesz_map=convergence_riesz_map,
                    maxiter=5000,
                )
            else:
                Jhat = _build_poisson_problem(mesh, alpha, riesz_map="l2")
                result = solve_with_scipy(Jhat)

            results[(label, optimiser_name, riesz_map)] = result["iterations"]
            print(
                f"    iters={result['iterations']:5d}  "
                f"J={result['final_J']:.3e}",
                flush=True,
            )
    return results


def print_console_table(panel_label, meshes, rows, results):
    """Print one panel as a fixed-width text table."""

    mesh_labels = [label for label, _ in meshes]
    print(f"\n{panel_label}")
    print("-" * (32 + 7 * len(mesh_labels)))
    print(f"{'inner product':>14}  {'implementation':>16}  "
          + "  ".join(f"{l:>5}" for l in mesh_labels))
    print("-" * (32 + 7 * len(mesh_labels)))
    for optimiser_name, riesz_map in rows:
        cells = []
        for label in mesh_labels:
            value = results.get((label, optimiser_name, riesz_map))
            cells.append(f"{value:>5}" if value is not None else "  -- ")
        print(f"{riesz_map:>14}  {optimiser_name:>16}  "
              + "  ".join(cells))


def main():
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ratios", type=int, nargs="+",
        default=[4, 8, 16, 32, 64, 128],
        help="Target h_max / h_min ratios.",
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0e-4,
        help="Tikhonov parameter (default: 1e-4).",
    )
    parser.add_argument(
        "--n", type=int, default=32,
        help="Base mesh resolution for the tensor fallback "
             "(default: 32). Ignored when --mesh-source=netgen.",
    )
    parser.add_argument(
        "--mesh-source", choices=["netgen", "tensor"], default="netgen",
        help="netgen: load mesh_generation/graded_R<ratio>.msh; "
             "tensor: use the Netgen-free fallback. Default: netgen.",
    )
    parser.add_argument(
        "--mesh-dir", default="mesh_generation",
        help="Directory holding the .msh files (default: mesh_generation).",
    )
    parser.add_argument(
        "--panel", choices=["a", "b", "both"], default="both",
        help="Which panel(s) to run (default: both).",
    )
    args = parser.parse_args()

    meshes = []
    for ratio in args.ratios:
        if args.mesh_source == "netgen":
            path = f"{args.mesh_dir}/graded_R{ratio}.msh"
            print(f"Loading Netgen mesh {path}...", flush=True)
            mesh, realised = graded_unit_square_from_file(path)
        else:
            print(f"Building tensor mesh with target h_max/h_min = "
                  f"{ratio}...", flush=True)
            mesh, realised = graded_unit_square_tensor(
                h_ratio=float(ratio), n=args.n,
            )
        meshes.append((str(ratio), mesh))
        print(f"  {mesh.num_cells()} cells, realised ratio = {realised:.2f}")

    if args.panel in ("a", "both"):
        print("\nPanel (a): H = L^2(Omega)")
        results_a = run_panel(
            meshes, PANEL_A_ROWS, alpha=args.alpha,
            convergence_riesz_map=PANEL_CONVERGENCE_RIESZ["a"],
        )
        print_console_table("Panel (a): H = L^2", meshes,
                            PANEL_A_ROWS, results_a)

    if args.panel in ("b", "both"):
        print("\nPanel (b): H = H^1(Omega)")
        results_b = run_panel(
            meshes, PANEL_B_ROWS, alpha=args.alpha,
            convergence_riesz_map=PANEL_CONVERGENCE_RIESZ["b"],
        )
        print_console_table("Panel (b): H = H^1", meshes,
                            PANEL_B_ROWS, results_b)


if __name__ == "__main__":
    main()
