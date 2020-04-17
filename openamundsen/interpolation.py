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
)
def _idw(x_points, y_points, z_points, x_targets, y_targets, power, smoothing):
    num_points = len(x_points)
    num_targets = len(x_targets)
    data = np.zeros(num_targets)

    for target_num in range(num_targets):
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
    Wrapper function for _idw to ensure correct data types.
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
