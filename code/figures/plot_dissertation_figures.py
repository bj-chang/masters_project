"""makes poisson_convergence, poisson_convergence_loglog, burgers_convergence, panel_a and precond_inner_cg pdfs"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "dissertation/figures/"
BLUE, RED, ORANGE, GREY = "#1f4e9e", "#c0392b", "#d95f02", "#5d6d7e"

plt.style.use("seaborn-v0_8-whitegrid")


def finish(fig, ax, out, legend_loc, ncol=1):
    leg = ax.legend(loc=legend_loc, ncol=ncol, framealpha=1.0,
                    facecolor="white", edgecolor="0.7")
    leg.get_frame().set_linewidth(0.8)
    fig.tight_layout()
    fig.savefig(FIG + out)
    plt.close(fig)
    print("saved", FIG + out)


h = 1.0 / np.array([4, 8, 16, 32, 64, 128, 256, 512], dtype=float)
byhand = np.array([7.5963e-02, 2.0409e-02, 5.2009e-03, 1.3066e-03,
                   3.2704e-04, 8.1786e-05, 2.0448e-05, 5.1121e-06])
firedrake = np.array([6.2103e-02, 1.8332e-02, 4.7854e-03, 1.2095e-03,
                      3.0321e-04, 7.5855e-05, 1.8967e-05, 4.7420e-06])


fig, ax = plt.subplots(figsize=(7.0, 4.3))
ax.loglog(h, byhand, "o-", color=BLUE, lw=2, ms=7, label="by-hand $P_1$")
ax.loglog(h, firedrake, "s-", color=RED, lw=2, ms=7, label="Firedrake")
C = firedrake[-1] / h[-1] ** 2
ax.loglog(h, 0.25 * C * h ** 2, "k--", lw=1.5, label="slope 2")
ax.set_xlabel("mesh size $h$")
ax.set_ylabel(r"$\|u - u_h\|_{L^2}$")
ax.set_title("Poisson: Manufactured-Solution Convergence")
ax.set_ylim(top=ax.get_ylim()[1] * 3)
finish(fig, ax, "poisson_convergence_loglog.pdf", "upper left")


fig, ax = plt.subplots(figsize=(7.0, 4.3))
ax.semilogx(h, byhand / h ** 2, "o-", color=BLUE, lw=2, ms=7,
            label="by-hand $P_1$")
ax.semilogx(h, firedrake / h ** 2, "s-", color=RED, lw=2, ms=7,
            label="Firedrake")
ax.set_xlabel("mesh size $h$")
ax.set_ylabel(r"$\|u - u_h\|_{L^2} / h^2$")
ax.set_title("Poisson: $L^2$ Error Rescaled by $h^2$")
lo, hi = ax.get_ylim()
ax.set_ylim(lo - 0.35 * (hi - lo), hi)
finish(fig, ax, "poisson_convergence.pdf", "lower left")


hb = np.array([0.1, 0.05, 0.025, 0.0125])
e_final = np.array([5.840e-3, 1.452e-3, 3.577e-4, 8.679e-5])
e_st = np.array([1.838e-3, 4.590e-4, 1.140e-4, 2.810e-5])

fig, ax = plt.subplots(figsize=(7.0, 4.3))
ax.semilogx(hb, e_final / hb ** 2, "o-", color=BLUE, lw=2, ms=7,
            label="final-time $L^2$ error")
ax.semilogx(hb, e_st / hb ** 2, "s-", color=RED, lw=2, ms=7,
            label="space-time $L^2$ error")
ax.set_xlabel("mesh size $h$")
ax.set_ylabel(r"$\|u - u_h\|_{L^2} / h^2$")
ax.set_title("Burgers: $L^2$ Error Rescaled by $h^2$")
lo, hi = ax.get_ylim()
ax.set_ylim(lo, hi + 0.55 * (hi - lo))
finish(fig, ax, "burgers_convergence.pdf", "upper left", ncol=2)


r = np.array([4, 8, 16, 32, 64])
scipy_it = np.array([81, 180, 357, 675, 1490])
hilbert_it = np.array([6, 6, 6, 6, 6])

fig, ax = plt.subplots(figsize=(7.0, 4.3))
ax.loglog(r, scipy_it, "o-", color=ORANGE, lw=2.2, ms=8,
          label=r"SciPy L-BFGS-B ($\ell^2$)")
ax.loglog(r, hilbert_it, "s-", color=BLUE, lw=2.2, ms=8,
          label="Hilbert-space L-BFGS ($L^2$)")
ax.set_xscale("log", base=2)
ax.set_xticks(r)
ax.set_xticklabels([str(v) for v in r])
ax.set_xlabel(r"target $h_{\max}/h_{\min}$")
ax.set_ylabel("outer iterations to converge")
ax.set_title("Panel (a): Outer Iteration Count Against Mesh Ratio")
ax.set_ylim(3, 6000)
ax.annotate(r"$\propto$ mesh ratio", xy=(16, 357), xytext=(20, 120),
            color=ORANGE, fontweight="bold", fontsize=10)
ax.annotate("constant at 6", xy=(16, 6), xytext=(11, 8.2),
            color=BLUE, fontweight="bold", fontsize=10)
finish(fig, ax, "panel_a.pdf", "upper left")


rc = np.array([4, 16, 64, 128])
base = np.array([61, 179, 573, 1052])
riesz = np.array([12, 12, 12, 12])

fig, ax = plt.subplots(figsize=(7.0, 4.3))
ax.loglog(rc, base, "o-", color=ORANGE, lw=2.2, ms=8,
          label=r"no preconditioner ($\ell^2$)")
ax.loglog(rc, riesz, "s-", color=BLUE, lw=2.2, ms=8,
          label=r"with $L^2$ Riesz PC ($M_h^{-1}$)")
ax.set_xscale("log", base=2)
ax.set_xticks(rc)
ax.set_xticklabels([str(v) for v in rc])
ax.set_xlabel(r"realised $h_{\max}/h_{\min}$")
ax.set_ylabel("total inner-CG iterations")
ax.set_title("Inner-CG Iteration Count With and Without the Riesz Map")
ax.set_ylim(6, 6000)
ax.annotate(r"$88\times$ at $r = 128$", xy=(128, 1052), xytext=(30, 2100),
            color=ORANGE, fontweight="bold", fontsize=10)
ax.annotate("flat at 12", xy=(64, 12), xytext=(40, 16),
            color=BLUE, fontweight="bold", fontsize=10)
finish(fig, ax, "precond_inner_cg.pdf", "upper left")
