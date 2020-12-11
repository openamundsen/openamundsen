from numba import njit, prange
import numpy as np


@njit(cache=True, parallel=True)
def shadows(dem, res, sun_vec, num_sweeps=1):
    """
    Calculate terrain shadowing after Corripio (2003).

    Parameters
    ----------
    dem : ndarray
        Terrain elevation (m).

    res : float
        DEM resolution (m).

    sun_vec : ndarray
        Vector describing the position of the sun.

    num_sweeps : int, default 1
        Number of sweeps to perform in each direction. E.g. if num_sweeps = 3,
        sun paths are calculated for all elements in the first 3 rows and
        columns of the DEM in the direction of the sun.
        num_sweeps can be set to -1 to consider all pixels of the DEM.

    Returns
    -------
    illum : ndarray
        Contains 1 for illuminated and 0 for shadowed pixels.

    References
    ----------
    .. [1] Corripio, J. G. (2003). Vectorial algebra algorithms for calculating
       terrain parameters from DEMs and solar radiation modelling in mountainous
       terrain. International Journal of Geographical Information Science, 17(1),
       1–23. https://doi.org/10.1080/13658810210157796
    """
    inv_sun_vec = -sun_vec / np.max(np.abs(sun_vec[:2]))

    normal_sun_vec = np.zeros(3)
    normal_sun_vec[2] = np.sqrt(sun_vec[0]**2 + sun_vec[1]**2)
    normal_sun_vec[0] = -sun_vec[0] * sun_vec[2] / normal_sun_vec[2]
    normal_sun_vec[1] = -sun_vec[1] * sun_vec[2] / normal_sun_vec[2]

    # Determine origin of scanning lines in the direction of the sun
    if num_sweeps == -1:
        num_sweeps = max(dem.shape)
    pos = np.zeros(dem.shape, dtype=np.uint8)
    if sun_vec[0] < 0 and sun_vec[1] < 0:  # sun is in the Northwest
        pos[:, :num_sweeps] = 1
        pos[:num_sweeps, :] = 1
    elif sun_vec[0] < 0 and sun_vec[1] >= 0:  # sun is in the Southwest
        pos[-num_sweeps:, :] = 1
        pos[:, :num_sweeps] = 1
    elif sun_vec[0] >= 0 and sun_vec[1] < 0:  # sun is in the Northeast
        pos[:num_sweeps, :] = 1
        pos[:, -num_sweeps:] = 1
    elif sun_vec[0] >= 0 and sun_vec[1] >= 0:  # sun is in the Northwest
        pos[-num_sweeps:, :] = 1
        pos[:, -num_sweeps:] = 1
    i_vals, j_vals = np.nonzero(pos)

    illum = np.full(dem.shape, 1, dtype=np.uint8)

    for idx_num in prange(len(i_vals)):
        i = i_vals[idx_num]
        j = j_vals[idx_num]
        n = 0
        max_z_proj = -np.inf

        while True:
            dx = inv_sun_vec[0] * n
            dy = inv_sun_vec[1] * n
            jdx = int(np.round(j + dx))
            idy = int(np.round(i + dy))
            n += 1

            if jdx < 0 or jdx >= dem.shape[1] or idy < 0 or idy >= dem.shape[0]:
                break

            vec_to_orig = np.array([
                dx * res,
                dy * res,
                dem[idy, jdx],
            ])
            z_proj = np.dot(vec_to_orig, normal_sun_vec)

            if z_proj < max_z_proj:
                illum[idy, jdx] = 0
            else:
                max_z_proj = z_proj

    return illum
