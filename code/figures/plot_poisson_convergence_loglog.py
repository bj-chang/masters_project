"""loglog poisson convergence with the offset slope 2 reference line"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


h = 1.0 / np.array([4, 8, 16, 32, 64, 128, 256, 512], dtype=float)
byhand = np.array([7.5963e-02, 2.0409e-02, 5.2009e-03, 1.3066e-03,
                   3.2704e-04, 8.1786e-05, 2.0448e-05, 5.1121e-06])
firedrake = np.array([6.2103e-02, 1.8332e-02, 4.7854e-03, 1.2095e-03,
                      3.0321e-04, 7.5855e-05, 1.8967e-05, 4.7420e-06])

plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(7.0, 4.3))

ax.loglog(h, byhand, "o-", color="#1f4e9e", lw=2, ms=7, label="by-hand $P_1$")
ax.loglog(h, firedrake, "s-", color="#c0392b", lw=2, ms=7, label="Firedrake")


C = firedrake[-1] / h[-1] ** 2
offset = 0.25
ax.loglog(h, offset * C * h ** 2, "k--", lw=1.5, label="slope 2")

ax.set_xlabel("mesh size $h$")
ax.set_ylabel(r"$\|u - u_h\|_{L^2}$")
ax.set_title("Poisson: manufactured-solution convergence")
ax.legend(loc="upper left")
fig.tight_layout()

out = "dissertation/figures/poisson_convergence_loglog.pdf"
fig.savefig(out)
print("saved", out)
