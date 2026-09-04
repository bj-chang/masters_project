"""p1 poisson assembled by hand, algorithm 1 in the diss"""
from argparse import ArgumentParser
import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


REFERENCE_GRADS = np.array([
    [-1.0, -1.0],
    [ 1.0,  0.0],
    [ 0.0,  1.0],
])


QUAD_POINTS = np.array([
    [1.0 / 6.0, 1.0 / 6.0],
    [2.0 / 3.0, 1.0 / 6.0],
    [1.0 / 6.0, 2.0 / 3.0],
])

QUAD_WEIGHTS = np.array([
    1.0 / 6.0,
    1.0 / 6.0,
    1.0 / 6.0,
])


def make_unit_square_tri_mesh(nx, ny=None):
    if ny is None:
        ny = nx

    xs = np.linspace(0.0, 1.0, nx + 1)
    ys = np.linspace(0.0, 1.0, ny + 1)


    vertices = np.array([(x, y) for y in ys for x in xs], dtype=float)

    def vertex_id(i, j):
        return j * (nx + 1) + i

    triangles = []
    for j in range(ny):
        for i in range(nx):
            v00 = vertex_id(i,     j)
            v10 = vertex_id(i + 1, j)
            v01 = vertex_id(i,     j + 1)
            v11 = vertex_id(i + 1, j + 1)
            triangles.append([v00, v10, v11])
            triangles.append([v00, v11, v01])

    return vertices, np.array(triangles, dtype=int)


def p1_basis_values(xi, eta):

    return np.array([
        1.0 - xi - eta,
        xi,
        eta,
    ])


def boundary_nodes(vertices, tol=1.0e-12):

    x = vertices[:, 0]
    y = vertices[:, 1]
    on_boundary = (
        (x < tol) | (x > 1.0 - tol) | (y < tol) | (y > 1.0 - tol)
    )
    return np.flatnonzero(on_boundary)


def assemble_poisson_p1(nx, rhs_function):

    """element loop: map to the reference triangle, quadrature for the load, dirichlet rows overwritten"""
    vertices, triangles = make_unit_square_tri_mesh(nx)
    n_vertices = len(vertices)

    A = sp.lil_matrix((n_vertices, n_vertices))
    b = np.zeros(n_vertices)

    for tri in triangles:
        coords = vertices[tri]


        x0 = coords[0]
        J = np.column_stack((coords[1] - coords[0], coords[2] - coords[0]))
        detJ = np.linalg.det(J)
        abs_detJ = abs(detJ)
        area = 0.5 * abs_detJ


        invJ = np.linalg.inv(J)
        grad_phys = REFERENCE_GRADS @ invJ


        local_A = area * (grad_phys @ grad_phys.T)


        local_b = np.zeros(3)
        for (xi, eta), w in zip(QUAD_POINTS, QUAD_WEIGHTS):
            phi = p1_basis_values(xi, eta)
            xq = x0 + J @ np.array([xi, eta])
            local_b += w * abs_detJ * rhs_function(xq[0], xq[1]) * phi


        for a, Arow in enumerate(tri):
            b[Arow] += local_b[a]
            for bcol, Acol in enumerate(tri):
                A[Arow, Acol] += local_A[a, bcol]


    bdry = boundary_nodes(vertices)
    b[bdry] = 0.0
    for i in bdry:
        A.rows[i] = [i]
        A.data[i] = [1.0]

    return vertices, triangles, A.tocsr(), b


def solve_poisson_p1(nx):

    """assembles and solves, returns the error against the manufactured solution"""
    u_exact = lambda x, y: math.sin(math.pi * x) * math.sin(math.pi * y)
    rhs = lambda x, y: 2.0 * math.pi**2 * u_exact(x, y)

    vertices, triangles, A, b = assemble_poisson_p1(nx, rhs)
    uh = spla.spsolve(A, b)


    error_sq = 0.0
    for tri in triangles:
        coords = vertices[tri]
        x0 = coords[0]
        J = np.column_stack((coords[1] - coords[0], coords[2] - coords[0]))
        abs_detJ = abs(np.linalg.det(J))
        uh_local = uh[tri]
        for (xi, eta), w in zip(QUAD_POINTS, QUAD_WEIGHTS):
            phi = p1_basis_values(xi, eta)
            xq = x0 + J @ np.array([xi, eta])
            uh_q = np.dot(uh_local, phi)
            ue_q = u_exact(xq[0], xq[1])
            error_sq += w * abs_detJ * (uh_q - ue_q)**2

    return {
        "vertices": vertices,
        "triangles": triangles,
        "uh": uh,
        "l2_error": math.sqrt(error_sq),
        "h": 1.0 / nx,
    }


def convergence_test(mesh_sizes=(4, 8, 16, 32)):

    rows = []
    previous_error = None
    for nx in mesh_sizes:
        result = solve_poisson_p1(nx)
        error = result["l2_error"]
        if previous_error is None:
            rate = None
        else:
            rate = math.log(previous_error / error, 2.0)
        rows.append((nx, result["h"], error, rate))
        previous_error = error
    return rows


def print_table(rows):

    print(f"{'nx':>4}  {'h':>10}  {'L2 error':>16}  {'rate':>6}")
    print("-" * 44)
    for nx, h, error, rate in rows:
        rate_str = "  --  " if rate is None else f"{rate:6.4f}"
        print(f"{nx:>4}  {h:>10.6f}  {error:>16.10e}  {rate_str}")


def main():
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--resolutions", type=int, nargs="+", default=[4, 8, 16, 32],
        help="Mesh resolutions to test (default: 4 8 16 32).",
    )
    args = parser.parse_args()
    rows = convergence_test(args.resolutions)
    print_table(rows)


if __name__ == "__main__":
    main()
