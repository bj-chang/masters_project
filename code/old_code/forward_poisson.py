# forward_poisson.py
#
# Solve the forward problem:
#   -Delta u = m   in Omega
#    u = 0         on the boundary of Omega
#
# Weak form:
#   Find u in H_0^1(Omega) such that
#       integral over Omega of grad(u) dot grad(v) dx
#       =
#       integral over Omega of m * v dx
#   for all v in H_0^1(Omega).

from firedrake import *

def solve_forward_poisson(
    n: int = 32,
    degree: int = 1,
    alpha_scale_m: float = 1.0,
    output_pvd: str = "forward_solution.pvd",
):
    """
    Parameters
    ----------
    n : int
        Mesh resolution n x n
    degree : int
        Polynomial degree for FE space
    alpha_scale_m : float
        Scalar multiple applied to the source term m
    output_pvd : str
        Output name for the pvd file
    """

    # Mesh and function space 
    mesh = UnitSquareMesh(n, n)
    V = FunctionSpace(mesh, "CG", degree)  # CG = continuous Galerkin, H^1 space

    # Trial and test functions 
    u = TrialFunction(V) # represents unknown u symbolically
    v = TestFunction(V) # represents arbitrary test function v

    # Define a source term m(x) 
    x, y = SpatialCoordinate(mesh)

    # Example source: smooth bump function in the interior
    m_expr = exp(-50.0 * ((x - 0.5)**2 + (y - 0.5)**2))

    # Firedrake function in space V to store source term values on the mesh
    m = Function(V, name="m") # name="m" is just a label
    # Fills the function m vy evaluating the expression on the mesh and storing it in the finite element representation
    m.interpolate(alpha_scale_m * m_expr) 

    # Variational problem 
    a = inner(grad(u), grad(v)) * dx # bilinear form a(u,v) = integral of grad(u) dot grad(v)

    # Linear form: L(v) = integral of m * v
    L = m * v * dx

    # Homogeneous Dirichlet boundary condition u = 0 
    bc = DirichletBC(V, 0.0, "on_boundary")

    # Solve the linear system 
    u_sol = Function(V, name="u") # allocates a func to hold the soln
    solve(a == L, u_sol, bcs=bc)

    # Sanity checks:
    # For -Delta u = m with zero boundary conditions and m >= 0,
    # the solution u should be non-negative inside the domain.
    u_min = u_sol.dat.data_ro.min()
    u_max = u_sol.dat.data_ro.max()
    print(f"u range: min={u_min:.6e}, max={u_max:.6e}")

    # pvd file output for visualisation
    File(output_pvd).write(u_sol, m)

    return mesh, V, m, u_sol


if __name__ == "__main__":
    solve_forward_poisson(
        n=48,            
        degree=1,        
        alpha_scale_m=1.0,
        output_pvd="forward_solution.pvd",
    )
