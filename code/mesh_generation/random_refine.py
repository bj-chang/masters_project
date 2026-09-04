"""rivara longest edge bisection with random marking (schwedes 2.3.2 style) and a minimal gmsh 2.2 writer"""
import os
import sys

import numpy as np

from firedrake import Mesh, UnitSquareMesh

DEFAULT_P_REFINE = 0.35
DEFAULT_N_INITIAL = 8


def _edge_key(a, b):
    return (a, b) if a < b else (b, a)


def initial_uniform_arrays(n=DEFAULT_N_INITIAL):
    """n by n crossed triangle base mesh as plain arrays"""
    m = UnitSquareMesh(n, n)
    verts = np.asarray(m.coordinates.dat.data_ro)
    cells = np.asarray(m.coordinates.function_space().cell_node_map().values)
    return verts.copy(), cells.copy()


def bisect_refine(vertices, cells, marked):
    """rivara longest edge bisection of the marked cells, propagated so the mesh stays conforming"""
    verts_list = vertices.tolist()
    cells_list = [list(c) for c in cells.tolist()]
    alive = [True] * len(cells_list)
    edge_cells = {}
    edge_mid = {}

    def add_cell(idx):
        a, b, c = cells_list[idx]
        for e in [_edge_key(a, b), _edge_key(b, c), _edge_key(c, a)]:
            edge_cells.setdefault(e, set()).add(idx)

    def remove_cell(idx):
        a, b, c = cells_list[idx]
        for e in [_edge_key(a, b), _edge_key(b, c), _edge_key(c, a)]:
            edge_cells.get(e, set()).discard(idx)

    for i in range(len(cells_list)):
        add_cell(i)

    def edge_len_sq(a, b):
        pa = np.asarray(verts_list[a])
        pb = np.asarray(verts_list[b])
        d = pa - pb
        return float(d @ d)

    def longest_edge_key(idx):
        a, b, c = cells_list[idx]
        lens = [
            (edge_len_sq(a, b), _edge_key(a, b)),
            (edge_len_sq(b, c), _edge_key(b, c)),
            (edge_len_sq(c, a), _edge_key(c, a)),
        ]
        lens.sort(reverse=True)
        return lens[0][1]

    def get_mid(a, b):
        key = _edge_key(a, b)
        if key not in edge_mid:
            edge_mid[key] = len(verts_list)
            pa = np.asarray(verts_list[a])
            pb = np.asarray(verts_list[b])
            verts_list.append(((pa + pb) / 2).tolist())
        return edge_mid[key]

    def bisect_at(idx, edge):
        e0, e1 = edge
        a, b, c = cells_list[idx]
        if {a, b} == {e0, e1}:
            v0, v1, opp = a, b, c
        elif {b, c} == {e0, e1}:
            v0, v1, opp = b, c, a
        else:
            v0, v1, opp = c, a, b
        m = get_mid(v0, v1)
        remove_cell(idx)
        alive[idx] = False
        cells_list.append([v0, m, opp])
        cells_list.append([m, v1, opp])
        alive.extend([True, True])
        add_cell(len(cells_list) - 2)
        add_cell(len(cells_list) - 1)

    def refine_cell(idx):
        while alive[idx]:
            ledge = longest_edge_key(idx)
            others = [n for n in edge_cells.get(ledge, set())
                      if n != idx and alive[n]]
            if not others:
                bisect_at(idx, ledge)
                return
            n = next(iter(others))
            if longest_edge_key(n) == ledge:
                bisect_at(idx, ledge)
                bisect_at(n, ledge)
                return
            refine_cell(n)

    for i in range(len(cells_list)):
        if i < len(marked) and bool(marked[i]) and alive[i]:
            refine_cell(i)

    final_cells = [cells_list[i] for i in range(len(cells_list)) if alive[i]]
    return np.asarray(verts_list), np.asarray(final_cells, dtype=np.int64)


def check_conforming(vertices, cells):
    """asserts there are no hanging nodes"""
    edge_count = {}
    for cell in cells:
        a, b, c = cell
        for e in [_edge_key(a, b), _edge_key(b, c), _edge_key(c, a)]:
            edge_count[e] = edge_count.get(e, 0) + 1
    bad = [(e, n) for e, n in edge_count.items() if n not in (1, 2)]
    if bad:
        raise RuntimeError(
            f"Non-conforming mesh: {len(bad)} edges with bad incidence: "
            f"{bad[:5]}..."
        )


def write_gmsh22(path, vertices, cells):
    """minimal gmsh 2.2 ascii writer"""
    with open(path, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write(f"$Nodes\n{len(vertices)}\n")
        for i, v in enumerate(vertices):
            f.write(f"{i + 1} {v[0]:.15g} {v[1]:.15g} 0\n")
        f.write("$EndNodes\n")
        f.write(f"$Elements\n{len(cells)}\n")
        for i, c in enumerate(cells):
            f.write(
                f"{i + 1} 2 2 0 0 "
                f"{c[0] + 1} {c[1] + 1} {c[2] + 1}\n"
            )
        f.write("$EndElements\n")


def h_ratio(vertices, cells):
    """realised hmax/hmin from the cell diameters"""
    h = np.zeros(len(cells))
    for i, cell in enumerate(cells):
        v0, v1, v2 = vertices[cell[0]], vertices[cell[1]], vertices[cell[2]]
        d01 = float(np.linalg.norm(v0 - v1))
        d12 = float(np.linalg.norm(v1 - v2))
        d20 = float(np.linalg.norm(v2 - v0))
        h[i] = max(d01, d12, d20)
    return float(h.max() / h.min())


def random_refined_mesh(
    level,
    seed=42,
    p_refine=DEFAULT_P_REFINE,
    n_initial=DEFAULT_N_INITIAL,
    tmp_dir="/tmp/random_refined_meshes",
):
    """whole domain random marking, ratio saturates around 6"""
    os.makedirs(tmp_dir, exist_ok=True)
    verts, cells = initial_uniform_arrays(n_initial)
    rng = np.random.default_rng(seed)
    for _ in range(level):
        marked = rng.random(len(cells)) < p_refine
        verts, cells = bisect_refine(verts, cells, marked)
        check_conforming(verts, cells)
    realised = h_ratio(verts, cells)
    n_cells = len(cells)
    msh_path = f"{tmp_dir}/level_{level}_n{n_initial}_p{p_refine}.msh"
    write_gmsh22(msh_path, verts, cells)
    mesh = Mesh(msh_path)
    return mesh, realised, n_cells


def interior_random_refined_mesh(
    level,
    region=(0.3, 0.7, 0.3, 0.7),
    seed=42,
    p_refine=DEFAULT_P_REFINE,
    n_initial=DEFAULT_N_INITIAL,
    tmp_dir="/tmp/interior_random_refined_meshes",
):
    """random marking restricted to the interior box, fixed seed so the meshes are reproducible"""
    os.makedirs(tmp_dir, exist_ok=True)
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
    realised = h_ratio(verts, cells)
    n_cells = len(cells)
    region_tag = f"{x_min}-{x_max}-{y_min}-{y_max}"
    msh_path = (f"{tmp_dir}/level_{level}_n{n_initial}"
                f"_p{p_refine}_r{region_tag}.msh")
    write_gmsh22(msh_path, verts, cells)
    mesh = Mesh(msh_path)
    return mesh, realised, n_cells
