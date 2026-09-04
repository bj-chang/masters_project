"""makes burgers_evolution, poisson_fields and random_meshes pdfs"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

sys.path.insert(0, os.path.abspath("code"))
sys.path.insert(0, os.path.abspath("code/mesh_generation"))

FIG = "dissertation/figures/"
plt.style.use("seaborn-v0_8-whitegrid")


from random_refine import (
    initial_uniform_arrays, bisect_refine, check_conforming,
    DEFAULT_P_REFINE, DEFAULT_N_INITIAL,
)


def interior_random_arrays(level, region=(0.3, 0.7, 0.3, 0.7), seed=42,
                           p_refine=DEFAULT_P_REFINE,
                           n_initial=DEFAULT_N_INITIAL):
    x_min, x_max, y_min, y_max = region
    verts, cells = initial_uniform_arrays(n_initial)
    rng = np.random.default_rng(seed)
    for _ in range(level):
        centroids = verts[cells].mean(axis=1)
        in_region = (
            (centroids[:, 0] > x_min) & (centroids[:, 0] < x_max)
            & (centroids[:, 1] > y_min) & (centroids[:, 1] < y_max)
        )
        marked = (rng.random(len(cells)) < p_refine) & in_region
        verts, cells = bisect_refine(verts, cells, marked)
        check_conforming(verts, cells)
    return verts, cells


def mesh_panel(ax, verts, cells, title):
    tri = Triangulation(verts[:, 0], verts[:, 1], cells)
    ax.triplot(tri, color="#1f4e9e", lw=0.35)
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.grid(False)


fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3))
for ax, lvl in zip(axes, [4, 8, 12]):
    verts, cells = interior_random_arrays(lvl)
    mesh_panel(ax, verts, cells, f"Level {lvl} ({len(cells)} Cells)")
fig.tight_layout()
fig.savefig(FIG + "random_meshes.pdf")
plt.close(fig)
print("saved", FIG + "random_meshes.pdf")


from meshdep.problems.burgers import solve_forward_burgers_1d

T, NU = 1.0, 0.05
res = solve_forward_burgers_1d(num_elements=200, T=T, num_steps=800, nu=NU)
x, times, U = res["x"], res["times"], res["states"]

fig, ax = plt.subplots(figsize=(7.0, 4.3))
snaps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
colors = plt.cm.viridis(np.linspace(0, 0.92, len(snaps)))
for t_snap, c in zip(snaps, colors):
    k = int(np.argmin(np.abs(times - t_snap)))
    ax.plot(x, U[k], color=c, lw=2, label=f"$t = {t_snap:.2f}$")
ax.set_xlabel("$x$")
ax.set_ylabel("$u(x, t)$")
ax.set_title(rf"Forward Burgers: Nonlinear Steepening ($\nu = {NU}$)")
ax.set_ylim(-0.04, 1.30)
ax.legend(loc="upper left", ncol=2, framealpha=1.0,
          facecolor="white", edgecolor="0.7")
fig.tight_layout()
fig.savefig(FIG + "burgers_evolution.pdf")
plt.close(fig)
print("saved", FIG + "burgers_evolution.pdf")


from firedrake import (
    UnitSquareMesh, FunctionSpace, Function, SpatialCoordinate,
    TrialFunctions, TestFunctions, DirichletBC, solve, dx, grad, inner,
    sin, pi,
)

ALPHA = 1.0e-4
mesh = UnitSquareMesh(48, 48)
V = FunctionSpace(mesh, "CG", 1)
W = V * V
xc, yc = SpatialCoordinate(mesh)
d = Function(V).interpolate(sin(pi * xc) * sin(pi * yc))


u, lam = TrialFunctions(W)
v, w = TestFunctions(W)
a = (inner(grad(u), grad(v)) + (1.0 / ALPHA) * lam * v) * dx \
    + (inner(grad(lam), grad(w)) - u * w) * dx
L = -d * w * dx
bcs = [DirichletBC(W.sub(0), 0.0, "on_boundary"),
       DirichletBC(W.sub(1), 0.0, "on_boundary")]
sol = Function(W)
solve(a == L, sol, bcs=bcs,
      solver_parameters={"ksp_type": "preonly", "pc_type": "lu",
                         "pc_factor_mat_solver_type": "mumps"})
u_s, lam_s = sol.subfunctions
m_s = Function(V).interpolate(-lam_s / ALPHA)

coords = mesh.coordinates.dat.data_ro
cell_nodes = V.cell_node_list
tri = Triangulation(coords[:, 0], coords[:, 1], cell_nodes)

panels = [(d, r"Desired State $d$"),
          (u_s, r"Recovered State $u^{*}$"),
          (m_s, r"Recovered Control $m^{*}$"),
          (lam_s, r"Adjoint $\lambda$")]

fig, axes = plt.subplots(1, 4, figsize=(16.0, 3.6))
for ax, (fn, title) in zip(axes, panels):
    vals = fn.dat.data_ro
    cf = ax.tricontourf(tri, vals, levels=40, cmap="viridis")
    fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.grid(False)
fig.tight_layout()
fig.savefig(FIG + "poisson_fields.pdf")
plt.close(fig)
print("saved", FIG + "poisson_fields.pdf")
