from numba import njit, prange
import numpy as np
from openamundsen.modules.radiation import shadows


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
    # TODO check/improve calculation for small DEMs
    if any(np.array(dem.shape) < (3, 3)):
        slope = np.full(dem.shape, 0)
        aspect = np.full(dem.shape, np.nan)
        return slope, aspect

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
    normal_vec = np.full((3, dem.shape[0], dem.shape[1]), np.nan)

    # TODO check/improve calculation for small DEMs
    # Here it is assumed that the DEM is flat
    if any(np.array(dem.shape) < (3, 3)):
        normal_vec[0, :, :] = 0
        normal_vec[1, :, :] = 0
        normal_vec[2, :, :] = 1
        return normal_vec

    right = np.roll(dem, -1, axis=1)
    up = np.roll(dem, 1, axis=0)
    diag = np.roll(dem, (1, -1), axis=(0, 1))

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


def sky_view_factor(
        dem,
        res,
        min_azim=0,
        max_azim=360,
        azim_step=10,
        elev_step=1,
        num_sweeps=1,
        logger=None,
):
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

    min_azim : int, default 0
        Minimum azimuth angle (degrees).

    max_azim : int, default 360
        Maximum azimuth angle (degrees).

    azim_step : int, default 10
        Azimuth angle interval (degrees).

    elev_step : int, default 1
        Elevation angle interval (degrees).

    num_sweeps : int, default 1
        Number of sweeps in each direction when calculating shadows.

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
    # Set SVF to 1 for 1x1 grids
    if dem.shape == (1, 1):
        return np.full(dem.shape, 1.)

    dem = dem.astype(float, copy=False)  # convert to float64 if necessary
    slope, _ = slope_aspect(dem, res)
    svf = np.zeros(dem.shape)

    azim_angles = np.arange(min_azim, max_azim, azim_step)
    min_elev_angle = 1
    max_elev_angle = max(
        int(np.ceil(np.nanmax(slope))),
        min_elev_angle,
    )
    elev_angles = np.arange(min_elev_angle, max_elev_angle + 1)[::-1]

    for azim_num, azim_angle in enumerate(azim_angles):
        if logger is not None:
            logger.info(f'Calculating sky view factor for azimuth={azim_angle}° '
                        f'({azim_num + 1}/{len(azim_angles)})')

        azim_rad = np.deg2rad(azim_angle)
        svf_cur = np.full(dem.shape, max_elev_angle)

        for elev_angle in elev_angles:
            elev_rad = np.deg2rad(elev_angle)
            sun_vec = np.array([
                np.sin(azim_rad) * np.cos(elev_rad),
                -np.cos(azim_rad) * np.cos(elev_rad),
                np.sin(elev_rad),
            ])

            illum = shadows(dem, res, sun_vec, num_sweeps=num_sweeps)
            svf_cur[illum == 1] = elev_angle

        svf += np.cos(np.deg2rad(svf_cur))**2

    svf /= len(azim_angles)
    return svf


def curvature(dem, res, kind, L=None):
    """
    Calculate curvature for a digital terrain model.

    Parameters
    ----------
    dem : ndarray
        Terrain elevation (m).

    res : float
        DEM resolution (m).

    kind : str
        Curvature calculation method. Allowed values are:
            - 'liston': calculate topographic curvature according to [1].

    L : float, default None
        Length scale required for kind='liston' (m).

    Returns
    -------
    curv : ndarray
        Curvature array.

    References
    ----------
    .. [1] Liston, G. E., Haehnel, R. B., Sturm, M., Hiemstra, C. A.,
       Berezovskaya, S., & Tabler, R.  D. (2007). Simulating complex snow
       distributions in windy environments using SnowTran-3D. Journal of
       Glaciology, 53(181), 241–256.
    """
    if kind == 'liston':
        if L is None:
            L = res

        L_px = int(np.ceil(L / res))

        z = dem
        z_n = _shift_arr(dem, 0, L_px, mode='edge')
        z_ne = _shift_arr(dem, 1, L_px, mode='edge')
        z_e = _shift_arr(dem, 2, L_px, mode='edge')
        z_se = _shift_arr(dem, 3, L_px, mode='edge')
        z_s = _shift_arr(dem, 4, L_px, mode='edge')
        z_sw = _shift_arr(dem, 5, L_px, mode='edge')
        z_w = _shift_arr(dem, 6, L_px, mode='edge')
        z_nw = _shift_arr(dem, 7, L_px, mode='edge')

        curv = (
            (z - (z_w + z_e) / 2.) / (2 * L)
            + (z - (z_s + z_n) / 2.) / (2 * L)
            + (z - (z_sw + z_ne) / 2.) / (2 * np.sqrt(2) * L)
            + (z - (z_nw + z_se) / 2.) / (2 * np.sqrt(2) * L)
        ) / 4.
    else:
        raise NotImplementedError

    return curv


def openness(dem, res, L, negative=False, mean=True):
    """
    Calculate topographic openness for a DEM following [1].

    Parameters
    ----------
    dem : ndarray
        Terrain elevation (m).

    res : float
        DEM resolution (m).

    L : float
        Radial distance to consider for calculating openness for each pixel (m).

    negative : bool, default False
        Calculate negative instead of positive openness.

    mean : bool, default True
        Return openness averaged over all eight compass directions.

    Returns
    -------
    opn : ndarray
        Openness (radians).

    References
    ----------
    .. [1] Yokoyama, R., Shirasawa, M., & Pike, R. J. (2002). Visualizing
       topography by openness: A new application of image processing to digital
       elevation models. Photogrammetric Engineering and Remote Sensing, 68(3),
       257–266.
    """
    dirs = np.arange(8)
    opn = np.full((len(dirs), dem.shape[0], dem.shape[1]), np.inf)

    if negative:
        dem = -dem

    for dir in dirs:
        opn[dir, :, :] = _openness_dir(dem, res, L, dir)

    if mean:
        opn = opn.mean(axis=0)

    return opn


@njit(cache=True, parallel=True)
def _openness_dir(dem, res, L, dir):
    """
    Calculate topographic openness for a DEM and a single compass direction.

    Parameters
    ----------
    dem : ndarray
        Terrain elevation (m).

    res : float
        DEM resolution (m).

    L : float
        Radial distance to consider for calculating openness for each pixel (m).

    dir : int
        Direction for which to calculate openness, ranging from 0 (northwest) to
        7 (west).

    Returns
    -------
    opn_dir : ndarray
        Openness (radians).
    """
    opn_dir = np.full(dem.shape, np.inf)

    for i in prange(int(np.ceil(L / res))):
        dist = res * (i + 1) * [1, np.sqrt(2)][dir % 2]
        Z_shift = _shift_arr_retain(dem, dir, i)
        angle = np.pi / 2 - np.arctan2(Z_shift - dem, dist)

        idxs = np.flatnonzero(angle < opn_dir)
        opn_dir.ravel()[idxs] = angle.ravel()[idxs]

    return opn_dir


def _shift_arr(M, dir, n, mode='retain'):
    """
    Shift an array along one of the eight (inter)cardinal directions.

    Parameters
    ----------
    M : ndarray
        Input array.

    dir : int
        Direction along to shift the array, ranging from 0 (north) to 7
        (northwest).

    n : int
        Number of pixels to be shifted.

    mode : str, default 'retain'
        If 'retain', pixels are padded with the values from the original array.
        All other values (e.g., 'edge' for padding with the edge values) are
        passed to np.pad().

    Returns
    -------
    S : ndarray
        Shifted array.
    """
    if mode == 'retain':
        return _shift_arr_retain(M, dir, n)
    else:
        if dir == 0:  # north
            return np.pad(M, ((0, n), (0, 0)), mode=mode)[n:, :]
        elif dir == 1:  # northeast
            return np.pad(M, ((0, n), (n, 0)), mode=mode)[n:, :-n]
        elif dir == 2:  # east
            return np.pad(M, ((0, 0), (n, 0)), mode=mode)[:, :-n]
        elif dir == 3:  # southeast
            return np.pad(M, ((n, 0), (n, 0)), mode=mode)[:-n, :-n]
        elif dir == 4:  # south
            return np.pad(M, ((n, 0), (0, 0)), mode=mode)[:-n, :]
        elif dir == 5:  # southwest
            return np.pad(M, ((n, 0), (0, n)), mode=mode)[:-n, n:]
        elif dir == 6:  # west
            return np.pad(M, ((0, 0), (0, n)), mode=mode)[:, n:]
        elif dir == 7:  # northwest
            return np.pad(M, ((0, n), (0, n)), mode=mode)[n:, n:]


@njit(cache=True)
def _shift_arr_retain(M, dir, n):
    """
    Shift an array along one of the eight (inter)cardinal directions.
    Pixels padded to the edges of the axes retain the value from the
    original array.

    Parameters
    ----------
    M : ndarray
        Input array.

    dir : int
        Direction along to shift the array, ranging from 0 (north) to 7
        (northwest).

    n : int
        Number of pixels to be shifted.

    Returns
    -------
    S : ndarray
        Shifted array.
    """
    S = M.copy()

    if dir == 0:  # north
        S[:-n - 1, :] = M[1 + n:, :]
    elif dir == 1:  # northeast
        S[:-n - 1, 1 + n:] = M[1 + n:, :-n - 1]
    elif dir == 2:  # east
        S[:, 1 + n:] = M[:, :-n - 1]
    elif dir == 3:  # southeast
        S[1 + n:, 1 + n:] = M[:-n - 1, :-n - 1]
    elif dir == 4:  # south
        S[1 + n:, :] = M[:-n - 1, :]
    elif dir == 5:  # southwest
        S[1 + n:, :-n - 1] = M[:-n - 1, 1 + n:]
    elif dir == 6:  # west
        S[:, :-n - 1] = M[:, 1 + n:]
    elif dir == 7:  # northwest
        S[:-n - 1, :-n - 1] = M[1 + n:, 1 + n:]

    return S
