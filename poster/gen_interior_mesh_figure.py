"""Generate visualisations of interior random-refined meshes for the poster.

Uses the interior sub-square [0.3, 0.7]^2 (does not touch any Dirichlet
boundary edge). The mesh construction is shared with the experiment
scripts via ``investigations.random_refine.interior_random_refined_mesh``.
"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CODE_DIR = os.path.join(PROJECT_ROOT, "code")
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")

if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import matplotlib.pyplot as plt

from firedrake.pyplot import triplot
from investigations.random_refine import interior_random_refined_mesh

os.makedirs(FIG_DIR, exist_ok=True)

REGION = (0.3, 0.7, 0.3, 0.7)
LEVELS_TO_SHOW = (0, 2, 4, 8, 12, 14)

for level in LEVELS_TO_SHOW:
    mesh, realised, n_cells = interior_random_refined_mesh(level, region=REGION)

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

    out = os.path.join(FIG_DIR, f"interior_level{level}.pdf")
    plt.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved {out}, level={level}, "
          f"cells={n_cells}, realised h_max/h_min={realised:.2f}")
