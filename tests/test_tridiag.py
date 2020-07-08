import numpy as np
from numpy.testing import assert_allclose
from openamundsen.tridiag import solve_tridiag
from scipy.sparse import diags


def test_solve_tridiag():
    N = 100

    a = np.append([0], np.random.rand(N - 1))
    b = np.random.rand(N)
    c = np.append(np.random.rand(N - 1), [0])
    x = np.random.rand(N)

    A = diags([a[1:], b, c[:-1]], [-1, 0, 1])
    d = A @ x

    x_solved = solve_tridiag(a, b, c, d)
    assert_allclose(x_solved, x)
