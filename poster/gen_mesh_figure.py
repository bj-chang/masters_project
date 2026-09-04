"""Generate random-refined mesh figures (Schwedes Fig 2.2 style) for the poster."""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CODE_DIR = os.path.join(PROJECT_ROOT, "code")
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")

if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import matplotlib.pyplot as plt
import numpy as np

from firedrake import Mesh
from firedrake.pyplot import triplot
from investigations.random_refine import (
    bisect_refine,
    check_conforming,
    initial_uniform_arrays,
    write_gmsh22,
    h_ratio,
)

os.makedirs(FIG_DIR, exist_ok=True)

TMP_DIR = "/tmp/random_refined_meshes"
os.makedirs(TMP_DIR, exist_ok=True)

P_REFINE = 0.35
SEED = 42
N_INITIAL = 8
LEVELS_TO_SHOW = (4, 6, 8, 10)

verts, cells = initial_uniform_arrays(N_INITIAL)
rng = np.random.default_rng(SEED)

for target_level in range(max(LEVELS_TO_SHOW) + 1):
    if target_level in LEVELS_TO_SHOW:
        msh_path = f"{TMP_DIR}/poster_level_{target_level}.msh"
        write_gmsh22(msh_path, verts, cells)
        mesh = Mesh(msh_path)
        realised = h_ratio(verts, cells)

        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        triplot(mesh, axes=ax,
                interior_kw={"linewidths": 0.5, "edgecolors": "black"},
                boundary_kw={"linewidths": 1.0, "colors": "black"})
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        out = os.path.join(FIG_DIR, f"random_mesh_level{target_level}.pdf")
        plt.savefig(out, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        print(f"Saved {out}, level={target_level}, "
              f"cells={mesh.num_cells()}, realised h_max/h_min={realised:.2f}")

    if target_level < max(LEVELS_TO_SHOW):
        marked = rng.random(len(cells)) < P_REFINE
        verts, cells = bisect_refine(verts, cells, marked)
        check_conforming(verts, cells)
