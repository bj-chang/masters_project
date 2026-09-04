"""writes graded_R{4..128}.msh. 8x8 base, rivara bisection inside [0.4,0.6]^2, realised ratio an exact power of two"""
import sys

sys.path.insert(0, "/home/bjcwsl/masters_project/code/mesh_generation")

import numpy as np

from random_refine import (
    bisect_refine, h_ratio, initial_uniform_arrays, write_gmsh22,
)

RATIOS = [4, 8, 16, 32, 64, 128]
N_INITIAL = 8
REGION = (0.4, 0.6, 0.4, 0.6)
OUT_DIR = "/home/bjcwsl/masters_project/code/mesh_generation"


def deterministic_interior_mesh(n_initial, n_rounds, region):

    verts, cells = initial_uniform_arrays(n_initial)
    x0, x1, y0, y1 = region
    for _ in range(n_rounds):
        c = verts[cells].mean(axis=1)
        marked = ((c[:, 0] >= x0) & (c[:, 0] <= x1) &
                  (c[:, 1] >= y0) & (c[:, 1] <= y1))
        verts, cells = bisect_refine(verts, cells, marked)
    return verts, cells


for r in RATIOS:
    rounds = 2 * int(round(np.log2(r)))
    verts, cells = deterministic_interior_mesh(N_INITIAL, rounds, REGION)
    ratio = h_ratio(verts, cells)
    out = f"{OUT_DIR}/graded_R{r}.msh"
    write_gmsh22(out, verts, cells)
    print(f"wrote graded_R{r}.msh: {len(cells)} cells, realised ratio {ratio:.2f}")

print("done")
