"""byhand poisson assembler check"""
import math
import sys

import numpy as np
import pytest

from meshdep.problems.forward_poisson_byhand import solve_poisson_p1


def test_convergence():
    resolutions = [4, 8, 16, 32]
    errors = [solve_poisson_p1(n)["l2_error"] for n in resolutions]
    rates = np.array([
        math.log(errors[i] / errors[i+1], 2.0)
        for i in range(len(errors) - 1)
    ])
    print(f"Achieved convergence rates: {rates}")
    print("Expected: close to 2")
    assert (rates > 1.8).all()


if __name__ == "__main__":
    pytest.main(sys.argv)
