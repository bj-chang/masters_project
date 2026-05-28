"""Forward Poisson solve, by hand in plain Python with P1 elements.

Solves ``-Delta u = m`` on ``(0,1)^2`` with ``u = 0`` on the boundary,
with the manufactured solution ``u_exact = sin(pi x) sin(pi y)`` and
source ``m = 2 pi^2 u_exact``. Reproduces the "by hand" half of the
Section 6 convergence table.
"""

from argparse import ArgumentParser
import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# Reference triangle basis functions:
#   phi_1 = 1 - xi - eta
#   phi_2 = xi
#   phi_3 = eta
#
# Their gradients with respect to the reference coordinates (xi, eta)
# are constant on the reference triangle.
REFERENCE_GRADS = np.array([
    [-1.0, -1.0],
    [ 1.0,  0.0],
    [ 0.0,  1.0],
])


# A simple degree-2 quadrature rule on the reference triangle.
# The reference triangle has vertices (0,0), (1,0), (0,1) and area 1/2.
# This three-point rule is exact for polynomials up to degree 2.
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
    """Build a structured triangular mesh of the unit square.

    Parameters
    ----------
    nx, ny : int
        Number of rectangular subdivisions in each direction. Each
        rectangle is split into two triangles, giving a mesh with
        ``2 * nx * ny`` triangles and ``(nx + 1) * (ny + 1)`` vertices.

    Returns
    -------
    vertices : ndarray, shape (N, 2)
        Coordinates of all mesh vertices.
    triangles : ndarray, shape (T, 3)
        Vertex indices for each triangle.
    """

    if ny is None:
        ny = nx

    xs = np.linspace(0.0, 1.0, nx + 1)
    ys = np.linspace(0.0, 1.0, ny + 1)

    # Vertex ordering: loop over rows in y, then within each row over x.
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
    """Evaluate the three P1 basis functions on the reference triangle."""

    return np.array([
        1.0 - xi - eta,
        xi,
        eta,
    ])


def boundary_nodes(vertices, tol=1.0e-12):
    """Return the indices of vertices lying on the boundary of the unit square."""

    x = vertices[:, 0]
    y = vertices[:, 1]
    on_boundary = (
        (x < tol) | (x > 1.0 - tol) | (y < tol) | (y > 1.0 - tol)
    )
    return np.flatnonzero(on_boundary)


def assemble_poisson_p1(nx, rhs_function):
    """Assemble the global system A u = b for the Poisson problem.

    Uses P1 elements on a structured triangular mesh, with homogeneous
    Dirichlet boundary conditions. Returns the mesh together with the
    assembled (sparse) stiffness matrix and load vector.
    """

    vertices, triangles = make_unit_square_tri_mesh(nx)
    n_vertices = len(vertices)

    A = sp.lil_matrix((n_vertices, n_vertices))
    b = np.zeros(n_vertices)

    for tri in triangles:
        coords = vertices[tri]

        # Affine map from reference triangle to physical triangle.
        x0 = coords[0]
        J = np.column_stack((coords[1] - coords[0], coords[2] - coords[0]))
        detJ = np.linalg.det(J)
        abs_detJ = abs(detJ)
        area = 0.5 * abs_detJ

        # For affine P1 elements, the physical gradients are constant
        # on each cell and obtained by the chain rule.
        invJ = np.linalg.inv(J)
        grad_phys = REFERENCE_GRADS @ invJ

        # Local stiffness matrix.
        local_A = area * (grad_phys @ grad_phys.T)

        # Local load vector by quadrature.
        local_b = np.zeros(3)
        for (xi, eta), w in zip(QUAD_POINTS, QUAD_WEIGHTS):
            phi = p1_basis_values(xi, eta)
            xq = x0 + J @ np.array([xi, eta])
            local_b += w * abs_detJ * rhs_function(xq[0], xq[1]) * phi

        # Local-to-global assembly.
        for a, Arow in enumerate(tri):
            b[Arow] += local_b[a]
            for bcol, Acol in enumerate(tri):
                A[Arow, Acol] += local_A[a, bcol]

    # Impose homogeneous Dirichlet boundary conditions by overwriting
    # the rows corresponding to boundary nodes.
    bdry = boundary_nodes(vertices)
    b[bdry] = 0.0
    for i in bdry:
        A.rows[i] = [i]
        A.data[i] = [1.0]

    return vertices, triangles, A.tocsr(), b


def solve_poisson_p1(nx):
    """Solve the manufactured Poisson problem with P1 finite elements.

    The exact solution is u_exact = sin(pi x) sin(pi y), so the source
    term is m = 2 pi^2 sin(pi x) sin(pi y). Returns a dictionary
    containing the mesh, the coefficient vector and the L2 error.
    """

    u_exact = lambda x, y: math.sin(math.pi * x) * math.sin(math.pi * y)
    rhs = lambda x, y: 2.0 * math.pi**2 * u_exact(x, y)

    vertices, triangles, A, b = assemble_poisson_p1(nx, rhs)
    uh = spla.spsolve(A, b)

    # Compute an L2 error by quadrature over each triangle.
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
    """Run a convergence study for the plain Python P1 solver.

    Returns a list of ``(nx, h, l2_error, rate)`` tuples, with ``rate``
    set to ``None`` for the coarsest mesh.
    """

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
    """Pretty-print the rows returned by ``convergence_test``."""

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
