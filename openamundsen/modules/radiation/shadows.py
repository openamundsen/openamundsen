import numba
import numpy as np


@numba.njit(
    numba.uint8[:, :](
        numba.double[:, :],
        numba.double,
        numba.double[:],
    ),
    cache=True,
)
def shadows(dem, res, sun_vec):
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
    if sun_vec[0] < 0:  # sun is in the West
        j_end = 0
    else:
        j_end = dem.shape[1] - 1  # sun is in the East

    if sun_vec[1] < 0:  # sun is in the North
        i_end = 0
    else:
        i_end = dem.shape[0] - 1  # sun is in the South

    illum = np.full(dem.shape, 1, dtype=np.uint8)

    for direction in ('x', 'y'):
        if direction == 'x':
            i_vals = np.array([i_end])
            j_vals = np.arange(dem.shape[1])
        elif direction == 'y':
            i_vals = np.arange(dem.shape[0])
            j_vals = np.array([j_end])

        for i in i_vals:
            for j in j_vals:
                n = 0
                z_cmp = -1e30

                while True:
                    dx = inv_sun_vec[0] * n
                    dy = inv_sun_vec[1] * n
                    jdx = int(np.round(j + dx))
                    idy = int(np.round(i + dy))

                    if jdx < 0 or jdx >= dem.shape[1] or idy < 0 or idy >= dem.shape[0]:
                        break

                    vec_to_orig = np.array([
                        dx * res,
                        dy * res,
                        dem[idy, jdx],
                    ])
                    z_proj = np.dot(vec_to_orig, normal_sun_vec)

                    if z_proj < z_cmp:
                        illum[idy, jdx] = 0
                    else:
                        z_cmp = z_proj

                    n += 1

    return illum
