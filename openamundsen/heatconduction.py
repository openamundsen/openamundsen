from numba import njit
import numpy as np
from openamundsen import tridiag


@njit(cache=True)
def temp_change(
    dx,
    dt,
    T,
    therm_cond,
    T_bottom,
    dx_bottom,
    therm_cond_bottom,
    top_heat_flux,
    heat_cap,
    replace_nan=True,
):
    """
    Calculate the change in temperature over time by implicitly solving the
    one-dimensional heat equation following [1].

    Parameters
    ----------
    dx : ndarray
        Layer thicknesses (m).

    dt : float
        Timestep (s).

    T : ndarray
        Current temperatures (K).

    therm_cond : ndarray
        Thermal conductivities (W m-1 K-1).

    T_bottom : float
        Temperature of the bottom layer (K).

    dx_bottom : float
        Thickness of the bottom layer (m).

    top_heat_flux : float
        Heat flux from the top (W m-2).

    heat_cap : ndarray
        Areal heat capacities (J K-1 m-2).

    replace_nan : bool, default True
        Replace nan values in the calculated temperature changes with 0.

    Returns
    -------
    dT : ndarray
        Change in temperature over the timestep.

    References
    ----------
    .. [1] Essery, R. (2015). A factorial snowpack model (FSM 1.0).
       Geoscientific Model Development, 8(12), 3867–3876.
       https://doi.org/10.5194/gmd-8-3867-2015
    """
    N = len(T)  # number of layers

    # Calculate thermal conductivity between layers
    therm_cond_between = np.full(N, np.nan)
    for k in range(N - 1):
        therm_cond_between[k] = 2 / (dx[k] / therm_cond[k] + dx[k + 1] / therm_cond[k + 1])
    therm_cond_between[N - 1] = 2 / (dx[N - 1] / therm_cond[N - 1] + dx_bottom / therm_cond_bottom)

    if N == 1:
        temp_change = np.array(
            [
                (top_heat_flux + therm_cond_between[0] * (T_bottom - T[0]))
                * dt
                / (heat_cap[0] + therm_cond_between[0] * dt)
            ]
        )
    else:
        a = np.zeros(N)  # below-diagonal matrix elements
        b = np.zeros(N)  # diagonal matrix elements
        c = np.zeros(N)  # above-diagonal matrix elements
        d = np.zeros(N)  # right hand side of the matrix equation

        a[0] = 0
        b[0] = heat_cap[0] + therm_cond_between[0] * dt
        c[0] = -therm_cond_between[0] * dt
        d[0] = (top_heat_flux - therm_cond_between[0] * (T[0] - T[1])) * dt

        for k in range(1, N - 1):
            a[k] = c[k - 1]
            b[k] = heat_cap[k] + (therm_cond_between[k - 1] + therm_cond_between[k]) * dt
            c[k] = -therm_cond_between[k] * dt
            d[k] = (
                therm_cond_between[k - 1] * (T[k - 1] - T[k]) * dt
                + therm_cond_between[k] * (T[k + 1] - T[k]) * dt
            )

        k = N - 1
        a[k] = c[k - 1]
        b[k] = heat_cap[k] + (therm_cond_between[k - 1] + therm_cond_between[k]) * dt
        c[k] = 0
        d[k] = (
            therm_cond_between[k - 1] * (T[k - 1] - T[k]) * dt
            + therm_cond_between[k] * (T_bottom - T[k]) * dt
        )

        temp_change = tridiag.solve_tridiag(a, b, c, d)

    if replace_nan:
        temp_change[np.isnan(temp_change)] = 0.

    return temp_change


def temp_change_array(
    dx,
    dt,
    T,
    therm_cond,
    T_bottom,
    dx_bottom,
    therm_cond_bottom,
    top_heat_flux,
    heat_cap,
    replace_nan=True,
):
    """
    Array version of temp_change().
    Can be applied if the number of layers is the same for all pixels.

    Parameters
    ----------
    dx : ndarray
        Layer thicknesses (m).

    dt : float
        Timestep (s).

    T : ndarray(ndim=2)
        Current temperatures (K).

    therm_cond : ndarray(ndim=2)
        Thermal conductivities (W m-1 K-1).

    T_bottom : ndarray(ndim=1)
        Temperature of the bottom layer (K).

    dx_bottom : ndarray(ndim=1)
        Thickness of the bottom layer (m).

    top_heat_flux : ndarray(ndim=1)
        Heat flux from the top (W m-2).

    heat_cap : ndarray(ndim=2)
        Areal heat capacities (J K-1 m-2).

    replace_nan : bool, default True
        Replace nan values in the calculated temperature changes with 0.

    Returns
    -------
    dT : ndarray(ndim=2)
        Change in temperature over the timestep.
    """
    N = T.shape[0]  # number of layers

    # Calculate thermal conductivity between layers
    therm_cond_between = np.full(T.shape, np.nan)
    for k in range(N - 1):
        therm_cond_between[k, :] = 2 / (dx[k, :] / therm_cond[k, :] + dx[k + 1, :] / therm_cond[k + 1, :])
    therm_cond_between[N - 1, :] = 2 / (dx[N - 1, :] / therm_cond[N - 1, :] + dx_bottom / therm_cond_bottom)

    if N == 1:
        temp_change = np.atleast_2d(
            (top_heat_flux + therm_cond_between[0, :] * (T_bottom - T[0, :]))
            * dt / (heat_cap[0, :] + therm_cond_between[0, :] * dt)
        )
    else:
        a = np.zeros(T.shape)  # below-diagonal matrix elements
        b = np.zeros(T.shape)  # diagonal matrix elements
        c = np.zeros(T.shape)  # above-diagonal matrix elements
        d = np.zeros(T.shape)  # right hand side of the matrix equation

        a[0, :] = 0
        b[0, :] = heat_cap[0, :] + therm_cond_between[0, :] * dt
        c[0, :] = -therm_cond_between[0, :] * dt
        d[0, :] = (top_heat_flux - therm_cond_between[0, :] * (T[0, :] - T[1, :])) * dt

        for k in range(1, N - 1):
            a[k, :] = c[k - 1, :]
            b[k, :] = heat_cap[k, :] + (therm_cond_between[k - 1, :] + therm_cond_between[k, :]) * dt
            c[k, :] = -therm_cond_between[k, :] * dt
            d[k, :] = (
                therm_cond_between[k - 1, :] * (T[k - 1, :] - T[k, :]) * dt
                + therm_cond_between[k, :] * (T[k + 1, :] - T[k, :]) * dt
            )

        k = N - 1
        a[k, :] = c[k - 1, :]
        b[k, :] = heat_cap[k, :] + (therm_cond_between[k - 1, :] + therm_cond_between[k, :]) * dt
        c[k, :] = 0
        d[k, :] = (
            therm_cond_between[k - 1, :] * (T[k - 1, :] - T[k, :]) * dt
            + therm_cond_between[k, :] * (T_bottom - T[k, :]) * dt
        )

        temp_change = tridiag.solve_tridiag_array(a, b, c, d)

    if replace_nan:
        temp_change[np.isnan(temp_change)] = 0.

    return temp_change
