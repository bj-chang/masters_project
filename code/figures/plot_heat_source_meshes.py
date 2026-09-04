"""radiator and window example for 10.2. manufactured solution so the errors are exact, uniform vs graded mesh"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

sys.path.insert(0, "/home/bjcwsl/masters_project/code")

import firedrake as fd
from mesh_generation.random_refine import (
    bisect_refine, h_ratio, initial_uniform_arrays, write_gmsh22,
)

FIG = "dissertation/figures/"
BLUE = "#1f4e9e"
plt.style.use("seaborn-v0_8-whitegrid")


FEATURES = [((0.06, 0.35), 1.0, 0.040),
            ((1.00, 0.68), -0.8, 0.055)]
BASE_N = 16
ROUNDS = 10
R0 = 0.30


def exact_and_source(mesh):
    x, y = fd.SpatialCoordinate(mesh)
    u = 0
    f = 0
    for (cx, cy), amp, sig in FEATURES:
        r2 = (x - cx) ** 2 + (y - cy) ** 2
        g = amp * fd.exp(-r2 / (2 * sig ** 2))
        u = u + g
        f = f + (2 / sig ** 2 - r2 / sig ** 4) * g
    return u, f


def l2_error(mesh):
    V = fd.FunctionSpace(mesh, "CG", 1)
    u_ex, f = exact_and_source(mesh)
    w, v = fd.TrialFunction(V), fd.TestFunction(V)
    a = fd.inner(fd.grad(w), fd.grad(v)) * fd.dx
    L = f * v * fd.dx
    bc = fd.DirichletBC(V, u_ex, "on_boundary")
    uh = fd.Function(V)
    fd.solve(a == L, uh, bcs=bc,
             solver_parameters={"ksp_type": "cg", "pc_type": "hypre",
                                "ksp_rtol": 1e-12})
    return float(fd.sqrt(fd.assemble((uh - u_ex) ** 2 * fd.dx))), uh


def graded_arrays(n_initial, n_rounds, r0):
    verts, cells = initial_uniform_arrays(n_initial)
    for k in range(n_rounds):
        radius = r0 * 0.5 ** (k // 2)
        c = verts[cells].mean(axis=1)
        marked = np.zeros(len(cells), dtype=bool)
        for (cx, cy), _, _ in FEATURES:
            marked |= np.hypot(c[:, 0] - cx, c[:, 1] - cy) <= radius
        verts, cells = bisect_refine(verts, cells, marked)
    return verts, cells


uniform = {}
for n in (16, 32, 64, 128, 256):
    m = fd.UnitSquareMesh(n, n)
    err, _ = l2_error(m)
    uniform[n] = (m, m.num_cells(), err)


verts, cells = graded_arrays(BASE_N, ROUNDS, R0)
write_gmsh22("/tmp/heat_graded.msh", verts, cells)
m_graded = fd.Mesh("/tmp/heat_graded.msh")
g_cells = m_graded.num_cells()
g_ratio = h_ratio(verts, cells)
g_err, _ = l2_error(m_graded)

print(f"{'mesh':<26}{'cells':>9}{'ratio':>8}{'L2 error':>13}")
for n, (_, nc, e) in uniform.items():
    print(f"{'uniform ' + str(n) + 'x' + str(n):<26}{nc:>9}{1:>8}{e:>13.3e}")
print(f"{'graded to both features':<26}{g_cells:>9}{g_ratio:>8.0f}{g_err:>13.3e}")

match = min((n for n, (_, _, e) in uniform.items() if e <= g_err), default=None)
if match is not None:
    mc = uniform[match][1]
    print(f"\ncheapest uniform mesh at least as accurate as the graded one: "
          f"{match}x{match} = {mc} cells, "
          f"{mc / g_cells:.0f}x the graded cell count")


def tri_of(mesh):
    c = mesh.coordinates.dat.data_ro
    t = mesh.coordinates.cell_node_map().values
    return Triangulation(c[:, 0], c[:, 1], t)


fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3))


gx, gy = np.meshgrid(np.linspace(0, 1, 800), np.linspace(0, 1, 800))
U = np.zeros_like(gx)
for (cx, cy), amp, sig in FEATURES:
    U += amp * np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * sig ** 2))
lim = np.abs(U).max()
im = axes[0].imshow(U, origin="lower", extent=(0, 1, 0, 1), cmap="RdBu_r",
                    vmin=-lim, vmax=lim, interpolation="bilinear")
axes[0].set_title("Temperature In A Heated Room\nRadiator (Left) "
                  "And Window (Right)", fontsize=10.5)
fig.colorbar(im, ax=axes[0], fraction=0.046)

panels = [(uniform[BASE_N][0], "Uniform Mesh", uniform[BASE_N][1],
           1.0, uniform[BASE_N][2]),
          (m_graded, "Mesh Graded Towards Both Features", g_cells,
           g_ratio, g_err)]
for ax, (mesh, name, nc, ratio, err) in zip(axes[1:], panels):
    ax.triplot(tri_of(mesh), color=BLUE, lw=0.25)
    ax.set_title(f"{name}\n{nc} Cells, "
                 f"$h_{{\\max}}/h_{{\\min}} = {ratio:.0f}$, "
                 f"$L^2$ Error {err:.1e}", fontsize=10.5)

for ax in axes:
    ax.set_aspect("equal")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.grid(False)

fig.tight_layout()
fig.savefig(FIG + "heat_source_meshes.pdf")
plt.close(fig)
print("\nsaved", FIG + "heat_source_meshes.pdf")
