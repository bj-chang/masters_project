"""makes poisson_meshes, graded_meshes and precond_matching pdfs"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

FIG = "dissertation/figures/"
BLUE, ORANGE, GREY = "#1f4e9e", "#d95f02", "#5d6d7e"
plt.style.use("seaborn-v0_8-whitegrid")


cells = np.array([314, 772, 1970, 5450, 16020, 46998], dtype=float)


l2_base = np.array([65, 108, 341, 614, 1105, 2115], dtype=float)
l2_l2pc = np.array([14, 13, 13, 14, 13, 14], dtype=float)
l2_h1pc = np.array([131, 205, 323, 417, 522, 526], dtype=float)


h1_l2pc = np.array([370, 747, 1600, 3449, 7434, 10000], dtype=float)
h1_h1pc = np.array([10, 10, 8, 8, 10, 9], dtype=float)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)

ax = axes[0]
ax.loglog(cells, l2_base, "^-", color=GREY, lw=2, ms=7, label="no PC")
ax.loglog(cells, l2_l2pc, "s-", color=BLUE, lw=2, ms=7, label="$L^2$ Riesz PC")
ax.loglog(cells, l2_h1pc, "o-", color=ORANGE, lw=2, ms=7, label="$H^1$ Riesz PC")
ax.set_title("$L^2$-Regularised Problem")
ax.set_xlabel("number of cells")
ax.set_ylabel("total inner-CG iterations")
ax.set_ylim(4, 4.0e4)
ax.legend(loc="upper left", framealpha=1.0, facecolor="white", edgecolor="0.7")
ax.annotate("matched:\nflat", xy=(3000, 13), xytext=(1500, 25),
            color=BLUE, fontweight="bold", fontsize=9)

ax = axes[1]
ax.loglog(cells, h1_l2pc, "s-", color=BLUE, lw=2, ms=7, label="$L^2$ Riesz PC")
ax.loglog(cells, h1_h1pc, "o-", color=ORANGE, lw=2, ms=7, label="$H^1$ Riesz PC")
ax.set_title("$H^1$-Regularised Problem")
ax.set_xlabel("number of cells")
ax.legend(loc="upper left", framealpha=1.0, facecolor="white", edgecolor="0.7")
ax.annotate("matched:\nflat", xy=(3000, 9), xytext=(1500, 16),
            color=ORANGE, fontweight="bold", fontsize=9)

fig.tight_layout()
fig.savefig(FIG + "precond_matching.pdf")
plt.close(fig)
print("saved", FIG + "precond_matching.pdf")


def read_gmsh22(path):
    with open(path) as fh:
        lines = [ln.strip() for ln in fh]
    i = lines.index("$Nodes")
    nnodes = int(lines[i + 1])
    coords = {}
    for ln in lines[i + 2: i + 2 + nnodes]:
        p = ln.split()
        coords[int(p[0])] = (float(p[1]), float(p[2]))
    j = lines.index("$Elements")
    nel = int(lines[j + 1])
    tris = []
    for ln in lines[j + 2: j + 2 + nel]:
        p = ln.split()
        if int(p[1]) == 2:
            ntags = int(p[2])
            tris.append([int(v) for v in p[3 + ntags: 6 + ntags]])
    ids = sorted(coords)
    index = {nid: k for k, nid in enumerate(ids)}
    x = np.array([coords[n][0] for n in ids])
    y = np.array([coords[n][1] for n in ids])
    t = np.array([[index[v] for v in tri] for tri in tris])
    return x, y, t


def mesh_panel(ax, x, y, t, title):
    ax.triplot(Triangulation(x, y, t), color="#1f4e9e", lw=0.35)
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.grid(False)


fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3))
for ax, r in zip(axes, [4, 16, 64]):
    x, y, t = read_gmsh22(f"code/mesh_generation/graded_R{r}.msh")
    mesh_panel(ax, x, y, t, f"Ratio {r} ({len(t)} Cells)")
fig.tight_layout()
fig.savefig(FIG + "graded_meshes.pdf")
plt.close(fig)
print("saved", FIG + "graded_meshes.pdf")


def structured_unit_square(nx):
    xs = np.linspace(0, 1, nx + 1)
    X, Y = np.meshgrid(xs, xs)
    x, y = X.ravel(), Y.ravel()
    tris = []
    for j in range(nx):
        for i in range(nx):
            a = j * (nx + 1) + i
            b, c, d = a + 1, a + nx + 1, a + nx + 2
            tris += [[a, b, d], [a, d, c]]
    return x, y, np.array(tris)


fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3))
for ax, nx in zip(axes, [4, 8, 16]):
    x, y, t = structured_unit_square(nx)
    mesh_panel(ax, x, y, t, f"$n_x = {nx}$")
fig.tight_layout()
fig.savefig(FIG + "poisson_meshes.pdf")
plt.close(fig)
print("saved", FIG + "poisson_meshes.pdf")
