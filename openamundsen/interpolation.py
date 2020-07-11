import numba
import numpy as np


@numba.njit(
    numba.double[:](
        numba.double[:],
        numba.double[:],
        numba.double[:],
        numba.double[:],
        numba.double[:],
        numba.double,
        numba.double,
    ),
    parallel=True,
    cache=True,
)
def _idw(x_points, y_points, z_points, x_targets, y_targets, power, smoothing):
    """
    Interpolate a set or irregularly distributed points using inverse distance
    weighting.
    This function performs the actual interpolation; for the parameters see the
    documentation of `idw` (which is only a wrapper for this function).
    """
    num_points = len(x_points)
    num_targets = len(x_targets)
    data = np.zeros(num_targets)

    for target_num in numba.prange(num_targets):
        x = x_targets[target_num]
        y = y_targets[target_num]
        w = 0.0
        total = 0.0
        dist_is_0 = False

        for k in range(num_points):
            dist = np.sqrt((x - x_points[k]) ** 2 + (y - y_points[k]) ** 2 + smoothing ** 2)

            if dist == 0.0:
                data[target_num] = z_points[k]
                dist_is_0 = True
                break

            w += 1.0 / dist ** power

        if not dist_is_0:
            for k in range(num_points):
                dist = np.sqrt((x - x_points[k]) ** 2 + (y - y_points[k]) ** 2 + smoothing ** 2)
                total += z_points[k] / dist ** power

            data[target_num] = total / w

    return data


def idw(x, y, z, x_targets, y_targets, power=2, smoothing=0):
    """
    Interpolate a set or irregularly distributed points using inverse distance
    weighting.

    Parameters
    ----------
    x, y : ndarray
        x and y coordinates of the known points.

    z : ndarray
        Values to be interpolated.

    x_targets, y_targets : ndarray
        x and y coordinates of the interpolation targets.

    power : float, default 2
        Weighting power. The default value of 2 corresponds to the classic
        inverse distance squared weighting.

    smoothing : float, default 0
        Smoothing parameter. Increasing this value will produce smoother
        results, but means that the interpolation is no longer exact.

    Returns
    -------
    data : ndarray
        Interpolated values for the target locations.
    """
    assert len(x) == len(y) == len(z)

    return _idw(
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        np.asarray(z, dtype=float),
        np.asarray(x_targets, dtype=float),
        np.asarray(y_targets, dtype=float),
        float(power),
        float(smoothing),
    )
