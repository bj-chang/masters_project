"""Plot iteration counts vs realised h_max/h_min for both mesh constructions.

If the underlying phenomenon is the same, points from both mesh types
should trace out the same curve.
"""
import os
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---------- Netgen graded meshes (from preliminary_report.tex) ----------
# r = 4, 8, 16, 32, 64, 128
netgen_ratio     = [5.9, 11.0, 23.5,   46.0, 115.0, 194.0]
netgen_scipy     = [46,  None, 195,    None, 786,   None]
netgen_hilbert   = [8,   None,   7,    None,   8,   9]
netgen_nls_base  = [56,  None, 147,    None, 432,   807]
netgen_nls_pc    = [15,  None,  15,    None,  15,    15]

# ---------- Random-refined meshes (measured today) ----------
# levels 0, 2, 4, 6, 8, 10
random_ratio     = [1.00, 2.00, 2.83, 4.00, 5.66, 5.66]
random_scipy     = [13,   31,   35,   47,   47,   55]
random_hilbert   = [6,    8,    6,    6,    5,    4]
random_nls_base  = [24,   39,   45,   50,   61,   73]
random_nls_pc    = [10,   14,   14,   13,   13,   12]


def _clean(xs, ys):
    return zip(*[(x, y) for x, y in zip(xs, ys) if y is not None])


NETGEN_KW  = dict(marker='o', markersize=8, linestyle='-',  linewidth=1.5)
RANDOM_KW  = dict(marker='s', markersize=7, linestyle='--', linewidth=1.5)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# ----- Left: outer L-BFGS -----
xs, ys = _clean(netgen_ratio, netgen_scipy)
ax1.plot(list(xs), list(ys), color='#c8102e', label='SciPy L-BFGS-B ($\\ell^2$), Netgen graded', **NETGEN_KW)
xs, ys = _clean(random_ratio, random_scipy)
ax1.plot(list(xs), list(ys), color='#c8102e', label='SciPy L-BFGS-B ($\\ell^2$), random-refined', **RANDOM_KW)
xs, ys = _clean(netgen_ratio, netgen_hilbert)
ax1.plot(list(xs), list(ys), color='#0000cd', label='Hilbert L-BFGS ($L^2$), Netgen graded', **NETGEN_KW)
xs, ys = _clean(random_ratio, random_hilbert)
ax1.plot(list(xs), list(ys), color='#0000cd', label='Hilbert L-BFGS ($L^2$), random-refined', **RANDOM_KW)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel(r'Realised $h_{\max} / h_{\min}$')
ax1.set_ylabel('Outer L-BFGS iterations')
ax1.set_title('Panel (a): outer L-BFGS')
ax1.grid(True, which='both', linestyle=':', alpha=0.5)
ax1.legend(fontsize=9, loc='upper left')

# ----- Right: inner CG -----
xs, ys = _clean(netgen_ratio, netgen_nls_base)
ax2.plot(list(xs), list(ys), color='#c8102e', label='TAO/NLS inner CG, no PC, Netgen graded', **NETGEN_KW)
xs, ys = _clean(random_ratio, random_nls_base)
ax2.plot(list(xs), list(ys), color='#c8102e', label='TAO/NLS inner CG, no PC, random-refined', **RANDOM_KW)
xs, ys = _clean(netgen_ratio, netgen_nls_pc)
ax2.plot(list(xs), list(ys), color='#0000cd', label='TAO/NLS inner CG + $M_h^{-1}$, Netgen graded', **NETGEN_KW)
xs, ys = _clean(random_ratio, random_nls_pc)
ax2.plot(list(xs), list(ys), color='#0000cd', label='TAO/NLS inner CG + $M_h^{-1}$, random-refined', **RANDOM_KW)

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel(r'Realised $h_{\max} / h_{\min}$')
ax2.set_ylabel('Total inner-CG iterations')
ax2.set_title('Inner CG of TAO/NLS')
ax2.grid(True, which='both', linestyle=':', alpha=0.5)
ax2.legend(fontsize=9, loc='upper left')

fig.suptitle('Iteration counts vs realised mesh ratio: '
             'Netgen graded (circles, solid) vs random-refined (squares, dashed)',
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])

out_pdf = os.path.join(FIG_DIR, 'comparison_netgen_vs_random.pdf')
out_png = os.path.join(FIG_DIR, 'comparison_netgen_vs_random.png')
fig.savefig(out_pdf, bbox_inches='tight')
fig.savefig(out_png, bbox_inches='tight', dpi=150)
plt.close(fig)
print(f"Saved {out_pdf}")
print(f"Saved {out_png}")


# ---------- Standalone: outer L-BFGS ----------
fig, ax = plt.subplots(figsize=(7, 5.5))
xs, ys = _clean(netgen_ratio, netgen_scipy)
ax.plot(list(xs), list(ys), color='#c8102e',
        label='SciPy L-BFGS-B ($\\ell^2$), Netgen graded', **NETGEN_KW)
xs, ys = _clean(random_ratio, random_scipy)
ax.plot(list(xs), list(ys), color='#c8102e',
        label='SciPy L-BFGS-B ($\\ell^2$), random-refined', **RANDOM_KW)
xs, ys = _clean(netgen_ratio, netgen_hilbert)
ax.plot(list(xs), list(ys), color='#0000cd',
        label='Hilbert L-BFGS ($L^2$), Netgen graded', **NETGEN_KW)
xs, ys = _clean(random_ratio, random_hilbert)
ax.plot(list(xs), list(ys), color='#0000cd',
        label='Hilbert L-BFGS ($L^2$), random-refined', **RANDOM_KW)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'Realised $h_{\max} / h_{\min}$', fontsize=12)
ax.set_ylabel('Outer L-BFGS iterations', fontsize=12)
ax.grid(True, which='both', linestyle=':', alpha=0.5)
ax.legend(fontsize=10, loc='upper left')
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'comparison_outer_lbfgs.pdf'),
            bbox_inches='tight')
plt.close(fig)
print("Saved comparison_outer_lbfgs.pdf")

# ---------- Standalone: inner CG ----------
fig, ax = plt.subplots(figsize=(7, 5.5))
xs, ys = _clean(netgen_ratio, netgen_nls_base)
ax.plot(list(xs), list(ys), color='#c8102e',
        label='TAO/NLS inner CG, no PC, Netgen graded', **NETGEN_KW)
xs, ys = _clean(random_ratio, random_nls_base)
ax.plot(list(xs), list(ys), color='#c8102e',
        label='TAO/NLS inner CG, no PC, random-refined', **RANDOM_KW)
xs, ys = _clean(netgen_ratio, netgen_nls_pc)
ax.plot(list(xs), list(ys), color='#0000cd',
        label='TAO/NLS inner CG + $M_h^{-1}$, Netgen graded', **NETGEN_KW)
xs, ys = _clean(random_ratio, random_nls_pc)
ax.plot(list(xs), list(ys), color='#0000cd',
        label='TAO/NLS inner CG + $M_h^{-1}$, random-refined', **RANDOM_KW)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'Realised $h_{\max} / h_{\min}$', fontsize=12)
ax.set_ylabel('Total inner-CG iterations', fontsize=12)
ax.grid(True, which='both', linestyle=':', alpha=0.5)
ax.legend(fontsize=10, loc='upper left')
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'comparison_inner_cg.pdf'),
            bbox_inches='tight')
plt.close(fig)
print("Saved comparison_inner_cg.pdf")
