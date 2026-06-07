"""Mesh constructors for the experiments. Can make a uniform mesh, use a mesh
from a file, manually generate a mesh, or use Netgen to generate a mesh."""

import numpy as np

from firedrake import *


def uniform_unit_square(n):
    """Uniform n-by-n triangulation of the unit square."""

    return UnitSquareMesh(n, n)


def graded_unit_square_from_file(path):
    """Load a pre-generated graded mesh from a Gmsh .msh file.

    Returns ``(mesh, realised h_max/h_min)``.
    """

    mesh = Mesh(path)
    DG0 = FunctionSpace(mesh, "DG", 0)
    h = Function(DG0).interpolate(CellDiameter(mesh))
    h_values = h.dat.data_ro
    realised_ratio = float(h_values.max()) / float(h_values.min())
    return mesh, realised_ratio


def graded_unit_square_tensor(h_ratio, n=32):
    """Tensor-product graded mesh of the unit square that does not use Netgen.

    Warps each coord of ``UnitSquareMesh(n, n)`` using
    ``f(s) = 1/2 (1 + tanh(alpha*(2s-1))/tanh(alpha))`` with
    ``alpha = arccosh(sqrt(h_ratio))``. Used as a fallback if Netgen
    isn't available. 
    
    Returns ``(mesh, realised h_max/h_min)``.
    """

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
    """Graded unit-square mesh made using Netgen.

    Coarse outer rectangle with target cell size ``h_max`` sits around a finer
    sub-square ``(0, 0.4)^2`` with target ``h_max / h_ratio``.
    
    Returns ``(mesh, realised h_max/h_min)``.
    """

    # Only import here, so that other parts can work without Netgen
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
