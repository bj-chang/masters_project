"""Test that the forward Poisson solver converges at the expected rate.

This mirrors test_12_poisson_convergence.py in the course package: we
expect P1 elements to give a rate close to 2 in the L2 norm.
"""

import sys
import numpy as np
import pytest

pytest.importorskip("firedrake")

from pdeopt.problems.forward_poisson import solve_forward_poisson_mms


def test_convergence():
    resolutions = [8, 16, 32, 64]
    errors = [solve_forward_poisson_mms(n, degree=1) for n in resolutions]
    rates = np.array([
        np.log(errors[i] / errors[i+1])
        / np.log(resolutions[i+1] / resolutions[i])
        for i in range(len(resolutions) - 1)
    ])
    print(f"Achieved convergence rates: {rates}")
    print("Expected: close to 2")
    assert (rates > 1.8).all()


if __name__ == "__main__":
    pytest.main(sys.argv)
