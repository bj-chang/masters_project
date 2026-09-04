"""precondition with (Mh + w Kh)^-1 and sweep w. every positive w still grows with the mesh, only w=0 is flat"""
import json
import sys

sys.path.insert(0, "/home/bjcwsl/masters_project/code")

from firedrake import (
    DirichletBC, Function, FunctionSpace, RieszMap, SpatialCoordinate,
    TestFunction, TrialFunction, assemble, dx, grad, inner, pi, sin, solve,
)
from firedrake.adjoint import (
    Control, ReducedFunctional,
    continue_annotation, pause_annotation, set_working_tape,
)
from pyadjoint import MinimizationProblem
from pyadjoint.optimization.tao_solver import TAOSolver
from mesh_generation.random_refine import interior_random_refined_mesh

ALPHA = 1.0e-4
REGION = (0.3, 0.7, 0.3, 0.7)
TMP_DIR = "/tmp/interior_random_h1_sweep"
LU = {"ksp_type": "preonly", "pc_type": "lu"}
WEIGHTS = [1.0, 1e-1, 1e-2, 1e-3, 1e-4]
LEVELS = [4, 6, 8, 10, 12, 14]
OUT = ("/tmp/claude-1000/-home-bjcwsl-masters-project/"
       "8d1f5f6b-c14f-4430-b09e-dedad66f1077/scratchpad/h1_weight_data.json")
NEW_PC = "pyadjoint.optimization.tao_solver.RieszMapPC"


def weighted_h1_map(V, w):
    u, v = TrialFunction(V), TestFunction(V)
    form = inner(u, v) * dx + w * inner(grad(u), grad(v)) * dx
    return RieszMap(form, solver_parameters=LU)


def build_jhat(mesh, riesz_map):
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
        solve(F == 0, u, bcs=bc, solver_parameters=LU)
        J = assemble(0.5 * (u - d) ** 2 * dx + 0.5 * ALPHA * m ** 2 * dx)
        Jhat = ReducedFunctional(J, Control(m, riesz_map=riesz_map), tape=tape)
    pause_annotation()
    return Jhat, V


def run(mesh, riesz_map, use_new_pc):
    Jhat, V = build_jhat(mesh, riesz_map)
    params = {
        "tao_type": "nls", "tao_gatol": 1e-10, "tao_grtol": 1e-7,
        "tao_gttol": 0.0, "tao_max_it": 50, "tao_nls_ksp_rtol": 1e-4,
    }
    if use_new_pc:
        params["tao_nls_pc_type"] = "python"
        params["tao_nls_pc_python_type"] = NEW_PC
    solver = TAOSolver(MinimizationProblem(Jhat), parameters=params)
    ksp = solver.tao.getKSP()
    ksp.setConvergenceHistory(reset=False)
    solver.solve()
    return solver.tao.getIterationNumber(), len(ksp.getConvergenceHistory())


rows = []
for lvl in LEVELS:
    mesh, realised, ncells = interior_random_refined_mesh(
        lvl, region=REGION, seed=42, p_refine=0.35, n_initial=8,
        tmp_dir=TMP_DIR)
    V = FunctionSpace(mesh, "CG", 1)

    o_l2, i_l2 = run(mesh, "L2", use_new_pc=True)
    row = {"level": lvl, "ratio": realised, "cells": ncells,
           "l2_out": o_l2, "l2_in": i_l2, "weights": {}}
    msg = (f"L={lvl:>2} ratio={realised:6.1f} cells={ncells:>6} | "
           f"L2 out={o_l2} in={i_l2:>3}")

    for w in WEIGHTS:
        rmap = weighted_h1_map(V, w)
        o, i = run(mesh, rmap, use_new_pc=True)
        row["weights"][str(w)] = {"out": o, "in": i}
        msg += f" | w={w:g} out={o} in={i:>4}"

    rows.append(row)
    print(msg, flush=True)
    with open(OUT, "w") as f:
        json.dump(rows, f, indent=2)

print("DONE")
