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


def idw(x, y, z, x_targets, y_targets, power=2, smoothing=0, ignore_nan=True):
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

    ignore_nan : bool, default True
        Ignore NaN values in the interpolation points.

    Returns
    -------
    data : ndarray
        Interpolated values for the target locations.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    assert len(x) == len(y) == len(z)

    if ignore_nan:
        pos = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        x = x[pos]
        y = y[pos]
        z = z[pos]

    # If no input points are available return an all-nan array
    if len(x) == 0:
        return np.full(x_targets.shape, np.nan)

    return _idw(
        x,
        y,
        z,
        np.asarray(x_targets, dtype=float),
        np.asarray(y_targets, dtype=float),
        float(power),
        float(smoothing),
    )
