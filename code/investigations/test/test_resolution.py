"""Per Josh: plot the optimised solution at various grading levels to
check whether the FE discretisation is still properly resolving the
true solution at high grid ratios, or whether the high-stretch meshes
have effectively turned the problem into a *different* one.

For each chosen stretch:
  - build the graded mesh,
  - run the L^2 Poisson distributed-control optimisation,
  - re-do a forward solve at the optimal m to get u_opt,
  - plot the mesh, u_opt, the target d, and |u_opt - d| (the misfit).

A poorly resolved high-stretch case will show:
  - u_opt visibly different from d (compare panels 2 and 3),
  - large concentrated misfit in the |u_opt - d| panel,
  - sometimes the mesh panel will already look obviously broken.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from firedrake import *
from firedrake.adjoint import *
from firedrake.pyplot import tripcolor, triplot
from firedrake.petsc import PETSc
from pyadjoint.optimization.tao_solver import TAOSolver

# Same history-size fix as the reproducer.
PETSc.Options().setValue('-tao_lmvm_mat_lmvm_hist_size', 100)

ALPHA = 1.0e-4
OUTPUT_DIR = "code/investigations/plots/discretisation_breakdown"


def graded_square(n=32, stretch=3.0):
    mesh = UnitSquareMesh(n, n)
    if stretch == 0.0:
        return mesh
    new = Function(mesh.coordinates.function_space())
    xy = mesh.coordinates.dat.data_ro
    new.dat.data[:] = (np.exp(stretch * xy) - 1.0) / (np.exp(stretch) - 1.0)
    return Mesh(new)


def h_ratio(mesh):
    DG0 = FunctionSpace(mesh, "DG", 0)
    h = Function(DG0).interpolate(CellDiameter(mesh)).dat.data_ro
    return h.max() / h.min()


def error_estimator(mesh, V):
    """Per-cell linear-interpolation error estimate: |grad(d_exact)| * h.

    Per Josh's suggestion - this is a quantitative version of the
    "is the mesh actually resolving the smooth target?" question.

    Important: we take the gradient of the SYMBOLIC expression for d,
    not of d after it's been interpolated into V. The interpolant of d
    is piecewise linear and its gradient is piecewise constant, which
    would tell us nothing about the underlying error. The exact grad(d)
    is what the local FE error scales with.

    Returns a Function in V holding |grad(d)| * CellDiameter, so larger
    values flag cells where the mesh struggles to resolve d.
    """
    x, y = SpatialCoordinate(mesh)
    d_expr = sin(pi * x) * sin(pi * y)          # SYMBOLIC, not interpolated
    grad_norm = sqrt(inner(grad(d_expr), grad(d_expr)))
    return Function(V).interpolate(grad_norm * CellDiameter(mesh))


def solve_problem(mesh):
    """Run the Poisson control optimisation and return u_opt, m_opt, d, iters."""
    x, y = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 1)
    u, v = Function(V), TestFunction(V)
    m = Function(V)
    a = inner(grad(u), grad(v)) * dx
    L = m * v * dx
    F = (a - L == 0)
    bc = DirichletBC(V, 0, "on_boundary")
    d = Function(V).interpolate(sin(pi * x) * sin(pi * y))
    fwd_params = {'ksp_type': 'preonly', 'pc_type': 'lu'}

    m.zero()
    continue_annotation()
    with set_working_tape() as tape:
        solve(F, u, bcs=bc, solver_parameters=fwd_params)
        J = assemble(0.5 * (u - d) ** 2 * dx + 0.5 * ALPHA * m ** 2 * dx)
        Jhat = ReducedFunctional(J, Control(m), tape=tape)
    pause_annotation()

    tao_params = {'tao_type': 'lmvm',
                  'tao_gatol': 1.0e-7, 'tao_grtol': 0.0, 'tao_gttol': 0.0,
                  'tao_max_it': 5000, 'tao_ls_type': 'unit'}
    solver = TAOSolver(MinimizationProblem(Jhat), parameters=tao_params)
    m_opt = solver.solve()

    # Forward solve at the optimal m (outside the tape) to get u_opt.
    u_opt = Function(V)
    solve(inner(grad(u_opt), grad(v)) * dx - m_opt * v * dx == 0,
          u_opt, bcs=bc, solver_parameters=fwd_params)

    iters = solver.tao.getIterationNumber()
    return u_opt, m_opt, d, iters


def plot_one(stretch):
    mesh = graded_square(stretch=stretch)
    ratio = h_ratio(mesh)
    print(f"  stretch={stretch:5.1f}  h_ratio={ratio:.3e}  optimising...",
          flush=True)
    u_opt, m_opt, d, iters = solve_problem(mesh)
    V = u_opt.function_space()

    # Misfit u_opt - d on the SAME function space.
    misfit = Function(V).assign(u_opt - d)
    misfit_L2 = float(assemble(misfit ** 2 * dx)) ** 0.5

    # Josh's error estimator: |grad(d_exact)| * h per cell.
    err_est = error_estimator(mesh, V)
    err_est_max = float(err_est.dat.data_ro.max())

    # Save VTK for ParaView, one file per stretch with all fields.
    vtk_dir = "code/investigations/output/discretisation_breakdown"
    os.makedirs(vtk_dir, exist_ok=True)
    u_opt.rename("u_opt")
    m_opt.rename("m_opt")
    d.rename("target_d")
    misfit.rename("misfit")
    err_est.rename("err_estimator")
    vtk_name = f"{vtk_dir}/stretch_{stretch:04.1f}.pvd"
    VTKFile(vtk_name).write(u_opt, m_opt, d, misfit, err_est)
    print(f"    VTK -> {vtk_name}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (0,0) mesh
    triplot(mesh, axes=axes[0, 0])
    axes[0, 0].set_title(
        f"Mesh\nstretch={stretch}, h_ratio={ratio:.2e}")
    axes[0, 0].set_aspect('equal')

    # (0,1) optimal state u
    cs = tripcolor(u_opt, axes=axes[0, 1], cmap='viridis')
    plt.colorbar(cs, ax=axes[0, 1])
    axes[0, 1].set_title(f"Optimal u  (TAO iters = {iters})")
    axes[0, 1].set_aspect('equal')

    # (0,2) target d
    cs = tripcolor(d, axes=axes[0, 2], cmap='viridis')
    plt.colorbar(cs, ax=axes[0, 2])
    axes[0, 2].set_title("Target  d = sin(pi x) sin(pi y)")
    axes[0, 2].set_aspect('equal')

    # (1,0) misfit |u_opt - d|  -  the ACTUAL local error
    misfit_abs = Function(V)
    misfit_abs.dat.data[:] = np.abs(misfit.dat.data_ro)
    cs = tripcolor(misfit_abs, axes=axes[1, 0], cmap='magma')
    plt.colorbar(cs, ax=axes[1, 0])
    axes[1, 0].set_title(
        f"|u_opt - d|   (||u-d||_L^2 = {misfit_L2:.3e})")
    axes[1, 0].set_aspect('equal')

    # (1,1) Josh's error estimator |grad(d)| * h  -  the PREDICTED error
    cs = tripcolor(err_est, axes=axes[1, 1], cmap='magma')
    plt.colorbar(cs, ax=axes[1, 1])
    axes[1, 1].set_title(
        f"|grad(d)| * h   (max = {err_est_max:.3e})")
    axes[1, 1].set_aspect('equal')

    # (1,2) hide - nothing to plot here.
    axes[1, 2].axis('off')

    fig.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fname = (f"{OUTPUT_DIR}/stretch_{stretch:04.1f}"
             f"_ratio_{ratio:.0e}.png")
    fig.savefig(fname, dpi=100)
    plt.close(fig)
    print(f"    iters={iters}, ||u-d||_L^2={misfit_L2:.3e}, "
          f"max(|grad(d)|*h)={err_est_max:.3e}, saved {fname}",
          flush=True)


if __name__ == "__main__":
    print("Plotting optimised solutions across grading levels...")
    for stretch in (0.0, 5.0, 10.0, 12.0, 15.0, 20.0, 25.0):
        plot_one(stretch)
    print(f"\nDone. PNGs in {OUTPUT_DIR}/")
