"""Generate the six graded meshes for Table 2.2.

Recipe mirrors meshdep/meshes.py:graded_unit_square: a unit square with
a coarse outer region (h <= 0.2) and a fine inner 0.4 x 0.4 sub-square
in the lower-left corner with h <= 0.2 / ratio. Each mesh is exported
to a Gmsh v2 .msh file that Firedrake can load directly.
"""

from netgen.geom2d import SplineGeometry

RATIOS = [4, 8, 16, 32, 64, 128]
H_MAX = 0.2


def build_geometry():
    geo = SplineGeometry()

    # Shared points so the inner sub-square's left/bottom edges coincide
    # exactly with the outer boundary, instead of being duplicated splines.
    p_00   = geo.AppendPoint(0.0, 0.0)
    p_a0   = geo.AppendPoint(0.4, 0.0)
    p_10   = geo.AppendPoint(1.0, 0.0)
    p_11   = geo.AppendPoint(1.0, 1.0)
    p_01   = geo.AppendPoint(0.0, 1.0)
    p_0a   = geo.AppendPoint(0.0, 0.4)
    p_aa   = geo.AppendPoint(0.4, 0.4)

    # Inner sub-square boundary on the outer domain edge (domain 2 vs outside).
    geo.Append(["line", p_00, p_a0], leftdomain=2, rightdomain=0, bc="outer")
    geo.Append(["line", p_0a, p_00], leftdomain=2, rightdomain=0, bc="outer")

    # Outer-only boundary segments (domain 1 vs outside).
    geo.Append(["line", p_a0, p_10], leftdomain=1, rightdomain=0, bc="outer")
    geo.Append(["line", p_10, p_11], leftdomain=1, rightdomain=0, bc="outer")
    geo.Append(["line", p_11, p_01], leftdomain=1, rightdomain=0, bc="outer")
    geo.Append(["line", p_01, p_0a], leftdomain=1, rightdomain=0, bc="outer")

    # Interface between fine inner and coarse outer (domain 2 vs domain 1).
    geo.Append(["line", p_a0, p_aa], leftdomain=2, rightdomain=1, bc="inner")
    geo.Append(["line", p_aa, p_0a], leftdomain=2, rightdomain=1, bc="inner")

    geo.SetMaterial(1, "coarse")
    geo.SetMaterial(2, "fine")
    return geo


for ratio in RATIOS:
    geo = build_geometry()
    geo.SetDomainMaxH(2, H_MAX / float(ratio))

    ngmesh = geo.GenerateMesh(maxh=H_MAX)
    out = f"graded_R{ratio}.msh"
    ngmesh.Export(out, "Gmsh2 Format")
    print(f"wrote {out}  ({ngmesh.Elements2D().NumPy().shape[0]} elements)")

print("done")
