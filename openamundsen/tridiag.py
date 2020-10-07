from numba import njit


@njit(cache=True)
def solve_tridiag(a, b, c, d, overwrite_bd=False):
    """
    Solve a tridiagonal equation system using the Thomas algorithm.

    The equivalent using scipy.linalg.solve_banded would be:

        ab = np.zeros((3, len(a)))
        ab[0, 1:] = c[:-1]
        ab[1, :] = b
        ab[2, :-1] = a[1:]
        x = scipy.linalg.solve_banded((1, 1), ab, d)

    Parameters
    ----------
    a : ndarray
        Lower diagonal of the tridiagonal matrix as a length n array with
        elements (0, a_2, a_3, ..., a_n).

    b : ndarray
        Main diagonal of the tridiagonal matrix as a length n array with
        elements (b_1, ..., b_n).

    c : ndarray
        Upper diagonal of the tridiagonal matrix as a length n array with
        elements (c_1, ..., c_n-1, 0).

    d : ndarray
        Right hand side of the matrix system as a length n array.

    Returns
    -------
    x : ndarray
        Solution vector as a length n array.
    """
    n = len(d)  # number of equations

    if not overwrite_bd:
        b = b.copy()
        d = d.copy()

    for k in range(1, n):
        m = a[k] / b[k - 1]
        b[k] = b[k] - m * c[k - 1]
        d[k] = d[k] - m * d[k - 1]

    x = b
    x[-1] = d[-1] / b[-1]

    for k in range(n - 2, 0 - 1, -1):
        x[k] = (d[k] - c[k] * x[k + 1]) / b[k]

    return x


def solve_tridiag_array(a, b, c, d, overwrite_bd=False):
    """
    Solve multiple tridiagonal equation systems using the Thomas algorithm.
    Parameters are same as for solve_tridiag(), however are in this case arrays
    with dimensions (n, num_equations).
    """
    n = d.shape[0]  # number of equations

    if not overwrite_bd:
        b = b.copy()
        d = d.copy()

    for k in range(1, n):
        m = a[k, :] / b[k - 1, :]
        b[k, :] = b[k, :] - m * c[k - 1, :]
        d[k, :] = d[k, :] - m * d[k - 1, :]

    x = b
    x[-1, :] = d[-1, :] / b[-1, :]

    for k in range(n - 2, 0 - 1, -1):
        x[k, :] = (d[k, :] - c[k, :] * x[k + 1, :]) / b[k, :]

    return x
