import numpy as np


def slope_aspect(dem, res):
    """
    Calculate terrain slope and aspect.

    Parameters
    ----------
    dem : ndarray
        Terrain elevation (m).

    res : float
        DEM resolution (m).

    Returns
    -------
    slope, aspect, unit_vec : ndarrays
        Terrain slope (degrees), aspect (degrees) and unit vector perpendicular
        to the slope.
    """
    right = np.roll(dem, 1, axis=1)
    up = np.roll(dem, 1, axis=0)
    diag = np.roll(np.roll(dem, 1, axis=1), 1, axis=0)

    normal_vec = np.full((dem.shape[0], dem.shape[1], 3), np.nan)
    normal_vec[:, :, 0] = 0.5 * res * (dem - right + up - diag)
    normal_vec[:, :, 1] = 0.5 * res * (dem + right - up - diag)
    normal_vec[:, :, 2] = res**2

    # fill last column and row
    normal_vec[-1, :, :] = normal_vec[-2, :, :]
    normal_vec[:, -1, :] = normal_vec[:, -2, :]

    normal_vec_len = np.sqrt(  # length of the normal vector
        normal_vec[:, :, 0]**2 +
        normal_vec[:, :, 1]**2 +
        normal_vec[:, :, 2]**2
    )

    unit_vec = np.zeros(normal_vec.shape)
    for i in range(3):
        unit_vec[:, :, i] = normal_vec[:, :, i] / normal_vec_len

    slope = np.rad2deg(np.arccos(unit_vec[:, :, 2]))
    aspect = np.rad2deg(np.arctan2(-unit_vec[:, :, 1], unit_vec[:, :, 0])) + 90
    aspect[aspect < 0] += 360

    return slope, aspect, unit_vec
