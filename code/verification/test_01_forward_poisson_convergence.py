"""poisson mms convergence for both implementations, tables in section 6"""
import sys
import numpy as np
import pytest

pytest.importorskip("firedrake")

from meshdep.problems.forward_poisson import solve_forward_poisson_mms


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
