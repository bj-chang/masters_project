"""mesh builders and loaders"""
import numpy as np

from firedrake import *


def uniform_unit_square(n):

    """n by n unit square split into triangles"""
    return UnitSquareMesh(n, n)


def graded_unit_square_from_file(path):

    """loads a gmsh mesh, returns (mesh, realised hmax/hmin)"""
    mesh = Mesh(path)
    DG0 = FunctionSpace(mesh, "DG", 0)
    h = Function(DG0).interpolate(CellDiameter(mesh))
    h_values = h.dat.data_ro
    realised_ratio = float(h_values.max()) / float(h_values.min())
    return mesh, realised_ratio


def graded_unit_square_tensor(h_ratio, n=32):

    if h_ratio < 1.0:
        raise ValueError("h_ratio must be >= 1")

    alpha = float(np.arccosh(np.sqrt(float(h_ratio)))) if h_ratio > 1.0 else 0.0

    mesh = UnitSquareMesh(n, n)

    if alpha > 0.0:
        coords = mesh.coordinates.dat.data
        tanh_a = np.tanh(alpha)
        warped = 0.5 * (1.0 + np.tanh(alpha * (2.0 * coords - 1.0)) / tanh_a)
        coords[:] = warped

    DG0 = FunctionSpace(mesh, "DG", 0)
    h = Function(DG0).interpolate(CellDiameter(mesh))
    h_values = h.dat.data_ro
    realised_ratio = float(h_values.max()) / float(h_values.min())

    return mesh, realised_ratio


def graded_unit_square(h_ratio, h_max=0.2):


    from netgen.geom2d import SplineGeometry

    geo = SplineGeometry()
    geo.AddRectangle(p1=(0.0, 0.0), p2=(1.0, 1.0),
                     bc="outer", leftdomain=1, rightdomain=0)
    geo.AddRectangle(p1=(0.0, 0.0), p2=(0.4, 0.4),
                     bc="inner", leftdomain=2, rightdomain=1)
    geo.SetMaterial(1, "coarse")
    geo.SetMaterial(2, "fine")
    geo.SetDomainMaxH(2, h_max / h_ratio)

    ngmesh = geo.GenerateMesh(maxh=h_max)
    mesh = Mesh(ngmesh)

    DG0 = FunctionSpace(mesh, "DG", 0)
    h = Function(DG0).interpolate(CellDiameter(mesh))
    h_values = h.dat.data_ro
    realised_ratio = float(h_values.max()) / float(h_values.min())

    return mesh, realised_ratio
