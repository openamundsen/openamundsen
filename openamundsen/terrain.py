import numpy as np
import openamundsen.modules.radiation


def slope_aspect(dem, res):
    """
    Calculate terrain slope and aspect using Horn's formula.

    Parameters
    ----------
    dem : ndarray
        Terrain elevation (m).

    res : float
        DEM resolution (m).

    Returns
    -------
    slope : ndarray
        Slope (degrees).

    aspect : ndarray
        Aspect (degrees) in azimuth direction, i.e., 0° means North-facing, 90°
        East-facing, 180° South-facing, and 270° West-facing.
    """
    # 3x3 window around cell e is labeled as follows:
    # [ a b c ]
    # [ d e f ]
    # [ g h i ]

    za = np.roll(dem, [1, 1], axis=[0, 1])
    zb = np.roll(dem, 1, axis=0)
    zc = np.roll(dem, [1, -1], axis=[0, 1])
    zd = np.roll(dem, 1, axis=1)
    zf = np.roll(dem, -1, axis=1)
    zg = np.roll(dem, [-1, 1], axis=[0, 1])
    zh = np.roll(dem, -1, axis=0)
    zi = np.roll(dem, [-1, -1], axis=[0, 1])

    za[:, 0] = za[:, 1]
    za[0, :] = za[1, :]
    zb[0, :] = zb[1, :]
    zc[:, -1] = zc[:, -2]
    zc[0, :] = zc[1, :]
    zd[:, 0] = zd[:, 1]
    zf[:, -1] = zf[:, -2]
    zg[:, 0] = zg[:, 1]
    zg[-1, :] = zg[-2, :]
    zh[-1, :] = zh[-2, :]
    zi[-1, :] = zi[-2, :]
    zi[:, -1] = zi[:, -2]

    dzdx = ((zc + 2 * zf + zi) - (za + 2 * zd + zg)) / (8 * res)
    dzdy = ((zg + 2 * zh + zi) - (za + 2 * zb + zc)) / (8 * res)
    slope = np.rad2deg(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))

    aspect = np.rad2deg(np.arctan2(dzdy, -dzdx))

    pos1 = aspect < 0
    pos2 = aspect > 90
    pos3 = (aspect >= 0) & (aspect <= 90)
    aspect[pos1] = 90 - aspect[pos1]
    aspect[pos2] = 360 - aspect[pos2] + 90
    aspect[pos3] = 90 - aspect[pos3]

    return slope, aspect


def normal_vector(dem, res):
    """
    Calculate a vector normal to the surface after Corripio (2003).

    Parameters
    ----------
    dem : ndarray
        Terrain elevation (m).

    res : float
        DEM resolution (m).

    Returns
    -------
    normal_vec : ndarray with dimensions (3, rows, cols)
        Unit vector perpendicular to the surface.

    References
    ----------
    .. [1] Corripio, J. G. (2003). Vectorial algebra algorithms for calculating
       terrain parameters from DEMs and solar radiation modelling in mountainous
       terrain. International Journal of Geographical Information Science, 17(1),
       1–23. https://doi.org/10.1080/13658810210157796
    """
    right = np.roll(dem, -1, axis=1)
    up = np.roll(dem, 1, axis=0)
    diag = np.roll(dem, (1, -1), axis=(0, 1))

    normal_vec = np.full((3, dem.shape[0], dem.shape[1]), np.nan)
    normal_vec[0, :, :] = 0.5 * res * (dem - right + up - diag)
    normal_vec[1, :, :] = 0.5 * res * (dem + right - up - diag)
    normal_vec[2, :, :] = res**2

    # fill first row and last column
    normal_vec[:, 0, :] = normal_vec[:, 1, :]
    normal_vec[:, :, -1] = normal_vec[:, :, -2]

    normal_vec_len = np.sqrt(  # length of the normal vector
        normal_vec[0, :, :]**2 +
        normal_vec[1, :, :]**2 +
        normal_vec[2, :, :]**2
    )
    normal_vec /= normal_vec_len  # make it a unit vector

    return normal_vec


def sky_view_factor(dem, res, azim_step=10, elev_step=1, logger=None):
    """
    Calculate the sky view factor for a DEM after Corripio (2003).
    The sky view factor is the hemispherical fraction of unobstructed sky
    visible from any point. Calculation is performed using a hillshading
    algorithm at regular azimuth and elevation angle intervals.

    Parameters
    ----------
    dem : ndarray
        Terrain elevation (m).

    res : float
        DEM resolution (m).

    azim_step : int, default 10
        Azimuth angle interval.

    elev_step : int, default 1
        Elevation angle interval.

    logger : Logger, optional
        Logger instance for printing status messages.

    Returns
    -------
    svf : ndarray
        Sky view factor (values between 0 and 1).

    References
    ----------
    .. [1] Corripio, J. G. (2003). Vectorial algebra algorithms for calculating
       terrain parameters from DEMs and solar radiation modelling in mountainous
       terrain. International Journal of Geographical Information Science, 17(1),
       1–23. https://doi.org/10.1080/13658810210157796
    """
    slope, _ = slope_aspect(dem, res)

    min_azim_angle = 0
    max_azim_angle = 360
    azim_angles = np.arange(min_azim_angle, max_azim_angle, azim_step)

    min_elev_angle = 1
    max_elev_angle = int(np.ceil(np.nanmax(slope)))
    elev_angles = np.arange(min_elev_angle, max_elev_angle)[::-1]

    svf = np.zeros(dem.shape)

    for azim_angle in azim_angles:
        if logger is not None:
            logger.debug(f'Calculating sky view factor: azimuth={azim_angle}')

        azim_rad = np.deg2rad(azim_angle)
        svf_cur = np.full(dem.shape, max_elev_angle)

        for elev_angle in elev_angles:
            elev_rad = np.deg2rad(elev_angle)
            sun_vec = np.array([
                np.sin(azim_rad) * np.cos(elev_rad),
                -np.cos(azim_rad) * np.cos(elev_rad),
                np.sin(elev_rad),
            ])

            illum = openamundsen.modules.radiation.shadows(dem, res, sun_vec)
            svf_cur[illum == 1] = elev_angle

        svf += np.cos(np.deg2rad(svf_cur))**2

    svf /= len(azim_angles)
    return svf
