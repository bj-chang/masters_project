"""Test that the Burgers forward solver converges at the expected rate.

This mirrors test_01_forward_poisson_convergence.py for the
nonlinear time-dependent problem. With backward Euler in time and
P1 in space, refined together so that dt is proportional to h, we
expect both the final-time L2 error and the discrete space-time L2
error to decay at rate close to 2.
"""

import sys
import pytest

from meshdep.problems.burgers import convergence_test


def test_burgers_convergence():
    rows = convergence_test(mesh_sizes=(10, 20, 40), T=0.1, nu=0.05)

    final_rates = [r["final_time_rate"] for r in rows[1:]]
    spacetime_rates = [r["space_time_rate"] for r in rows[1:]]

    print(f"final-time rates: {final_rates}")
    print(f"space-time rates: {spacetime_rates}")
    print("expected: close to 2")

    for rate in final_rates + spacetime_rates:
        assert rate > 1.8


if __name__ == "__main__":
    pytest.main(sys.argv)
