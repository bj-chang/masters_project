import numpy as np
from firedrake import *

def solve_forward_poisson_mms(
    n: int = 32,
    degree: int = 1,
    output_pvd: str = None,
):
    """
    Solves the Poisson problem using a manufactured solution to compute the L2 error.
    """
    # Mesh and function space 
    mesh = UnitSquareMesh(n, n)
    V = FunctionSpace(mesh, "CG", degree) 

    # Trial and test functions 
    u = TrialFunction(V)
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)

    # Choose an exact manufactured solution that satisfies u=0 on the boundary
    u_exact_expr = sin(pi * x) * sin(pi * y)
    u_exact = Function(V, name="u_exact")
    u_exact.interpolate(u_exact_expr)

    # Derive the corresponding source term m = -Delta(u_exact)
    # Delta(sin(pi*x)*sin(pi*y)) = -2 * pi^2 * sin(pi*x)*sin(pi*y)
    m_expr = 2.0 * pi**2 * sin(pi * x) * sin(pi * y)
    m = Function(V, name="m")
    m.interpolate(m_expr) 

    # Variational problem 
    a = inner(grad(u), grad(v)) * dx
    L = m * v * dx

    # Homogeneous Dirichlet boundary condition u = 0 
    bc = DirichletBC(V, 0.0, "on_boundary")

    # Solve the linear system 
    u_sol = Function(V, name="u")
    solve(a == L, u_sol, bcs=bc)

    # Output to PVD if requested
    if output_pvd:
        File(output_pvd).write(u_sol, m, u_exact)

    # Compute the L2 error between the numerical solution and the exact solution
    error_l2 = errornorm(u_exact, u_sol, norm_type="L2")
    
    return error_l2

def run_convergence_test():
    # Sequence of increasing mesh resolutions
    resolutions = [8, 16, 32, 64]
    errors = []
    
    print("Running MMS Convergence Test...")
    for n in resolutions:
        err = solve_forward_poisson_mms(n=n, degree=1)
        errors.append(err)
        print(f"Resolution {n:2d}x{n:2d} | L2 Error: {err:.6e}")
        
    print("\nCalculating Convergence Rates...")
    # Calculate the convergence rate q for each refinement step
    for i in range(1, len(errors)):
        # h is inversely proportional to n, so h1/h2 = n2/n1
        h_ratio = resolutions[i] / resolutions[i-1] 
        error_ratio = errors[i-1] / errors[i]
        
        # q = ln(E_1 / E_2) / ln(h_1 / h_2)
        rate = np.log(error_ratio) / np.log(h_ratio)
        print(f"Refinement {resolutions[i-1]:2d} -> {resolutions[i]:2d} | Rate (q) = {rate:.4f}")

if __name__ == "__main__":
    run_convergence_test()