from openamundsen import constants as c
import numpy as np


# Magnus formula coefficients a (Pa), b, and c (°C)
VAPOR_PRESSURE_COEFFS_ICE = (610.71, 22.44, 272.44)
VAPOR_PRESSURE_COEFFS_WATER = (610.78, 17.08, 234.18)


def atmospheric_pressure(elev):
    """
    Calculate atmospheric pressure.

    Parameters
    ----------
    elev : numeric
        Height above sea level (m).

    Returns
    -------
    pressure : numeric
        Atmospheric pressure (Pa).
    """
    return c.STANDARD_ATMOSPHERE * (
        1 - c.GRAVITATIONAL_ACCELERATION * np.asarray(elev)
        / (c.SPEC_HEAT_CAP_DRY_AIR * c.STANDARD_SEA_LEVEL_TEMPERATURE)
    )**(c.SPEC_HEAT_CAP_DRY_AIR * c.MOLAR_MASS_DRY_AIR / c.UNIVERSAL_GAS_CONSTANT)


def dry_air_density(temp, pressure):
    """
    Calculate the density of dry air.

    Parameters
    ----------
    temp : numeric
        Air temperature (K).

    pressure : numeric
        Atmospheric pressure (Pa).

    Returns
    -------
    density : numeric
        Dry air density (kg m-3).
    """
    return pressure * (c.GAS_CONSTANT_DRY_AIR * temp)


def latent_heat_of_vaporization(temp):
    """
    Calculate the latent heat of vaporization.

    Parameters
    ----------
    temp : numeric
        Air temperature (K).

    Returns
    -------
    lat_heat_vap : numeric
        Latent heat of vaporization (J kg-1).
    """
    return c.LATENT_HEAT_OF_VAPORIZATION - 2.361e3 * (np.asarray(temp) - c.T0)


def saturation_vapor_pressure(temp, over=None):
    """
    Calculate saturation vapor pressure for a given temperature.

    Parameters
    ----------
    temp : numeric
        Air temperature (K).

    over : str, default None
        Can be 'water' (to calculate vapor pressure over water), 'ice' (to
        calculate vapor pressure over ice), or None (to decide depending on
        temperature).

    Returns
    -------
    sat_vapor_pressure : numeric
        Saturation vapor pressure (Pa).
    """
    temp_c = np.asarray(temp) - c.T0

    if over is None:
        ix = (temp_c >= 0.).astype(int)  # contains 1 for indexes with positive temperatures, and 0 otherwise
    elif over == 'water':
        ix = np.full(temp_c.shape, 1)
    elif over == 'ice':
        ix = np.full(temp_c.shape, 0)
    else:
        raise NotImplementedError

    ca = np.array([VAPOR_PRESSURE_COEFFS_ICE[0], VAPOR_PRESSURE_COEFFS_WATER[0]])
    cb = np.array([VAPOR_PRESSURE_COEFFS_ICE[1], VAPOR_PRESSURE_COEFFS_WATER[1]])
    cc = np.array([VAPOR_PRESSURE_COEFFS_ICE[2], VAPOR_PRESSURE_COEFFS_WATER[2]])

    return ca[ix] * np.exp(cb[ix] * temp_c / (cc[ix] + temp_c))


def vapor_pressure(temp, rel_hum, over=None):
    """
    Calculate vapor pressure.

    Parameters
    ----------
    temp : numeric
        Air temperature (K).

    rel_hum : numeric
        Relative humidity (%).

    over : str, default None
        Can be 'water' (to calculate vapor pressure over water), 'ice' (to
        calculate vapor pressure over ice), or None (to decide depending on
        temperature).

    Returns
    -------
    vapor_pressure : numeric
        Vapor pressure (Pa).
    """
    return saturation_vapor_pressure(temp, over=over) * np.asarray(rel_hum) / 100


def specific_humidity(atmospheric_pressure, vapor_pressure):
    """
    Calculate specific humidity.

    Parameters
    ----------
    atmospheric_pressure : numeric
        Atmospheric pressure (Pa).

    vapor_pressure : numeric
        Vapor pressure (Pa).

    Returns
    -------
    spec_hum : numeric
        Specific humidity (kg kg-1).
    """
    atmospheric_pressure = np.asarray(atmospheric_pressure)
    vapor_pressure = np.asarray(vapor_pressure)
    return 0.622 * vapor_pressure / (atmospheric_pressure - 0.378 * vapor_pressure)


def absolute_humidity(temp, vapor_pressure):
    """
    Calculate absolute humidity.

    Parameters
    ----------
    temp : numeric
        Air temperature (K).

    vapor_pressure : numeric
        Vapor pressure (Pa).

    Returns
    -------
    abs_hum : numeric
        Absolute humidity (kg m-3).
    """
    return np.asarray(vapor_pressure) / (c.SPEC_GAS_CONSTANT_WATER_VAPOR * np.asarray(temp))


def relative_humidity(temp, abs_hum):
    """
    Calculate relative humidity from absolute humidity.

    Parameters
    ----------
    temp : numeric
        Air temperature (K).

    abs_hum : numeric
        Absolute humidity (kg m-3).

    Returns
    -------
    rel_hum : numeric
        Relative humidity (%).
    """
    temp = np.asarray(temp)
    abs_hum = np.asarray(abs_hum)
    vap_press = c.SPEC_GAS_CONSTANT_WATER_VAPOR * temp * abs_hum
    sat_vap_press = saturation_vapor_pressure(temp)
    return 100 * vap_press / sat_vap_press


def specific_heat_capacity_moist_air(spec_hum):
    """
    Calculate the specific heat capacity of moist air.

    Parameters
    ----------
    spec_hum : numeric
        Specific humidity (kg kg-1).

    Returns
    -------
    spec_heat_cap_moist_air : numeric
        Specific heat capacity of moist air (J kg-1 K-1).
    """
    return c.SPEC_HEAT_CAP_DRY_AIR * (1 + 0.84 * np.asarray(spec_hum))


def psychrometric_constant(atmos_press, spec_heat_cap, lat_heat_vap):
    """
    Calculate the psychrometric constant.

    Parameters
    ----------
    atmos_press : numeric
        Atmospheric pressure (Pa).

    spec_heat_cap : numeric
        Specific heat capacity (J kg-1 K-1).

    lat_heat_vap : numeric
        Latent heat of vaporization (J kg-1).

    Returns
    -------
    psychrometric_constant : numeric
        Psychrometric constant (Pa K-1).
    """
    return np.asarray(spec_heat_cap) * np.asarray(atmos_press) / (0.622 * np.asarray(lat_heat_vap))


def _water_vapor_pressure_difference(temp, wet_bulb_temp, vap_press, psych_const):
    """
    Evaluate the psychrometric formula
        e_l - (e_w - gamma * (T_a - T_w)).

    Parameters
    ----------
    temp : numeric
        Air temperature (K).

    wet_bulb_temp : numeric
        Wet-bulb temperature (K).

    vap_press : numeric
        Vapor pressure (Pa).

    psych_const : numeric
        Psychrometric constant (Pa K-1).

    Returns
    -------
    wat_vap_press_diff : numeric
        Water vapor pressure difference (Pa).
    """
    sat_vap_press_wet_bulb = saturation_vapor_pressure(wet_bulb_temp)
    return vap_press - (sat_vap_press_wet_bulb - psych_const * (temp - wet_bulb_temp))


def wet_bulb_temperature(temp, rel_hum, vap_press, psych_const):
    """
    Calculate wet-bulb temperature for given ambient conditions.
    Wet-bulb temperature is calculated iteratively using a secant method.

    Parameters
    ----------
    temp : numeric
        Air temperature (K).

    rel_hum : numeric
        Relative humidity (%).

    vap_press : numeric
        Vapor pressure (Pa).

    psych_const : numeric
        Psychrometric constant (Pa K-1).

    Returns
    -------
    wet_bulb_temp : numeric
        Wet-bulb temperature (K).
    """
    temp = np.asarray(temp)
    rel_hum = np.asarray(rel_hum)
    vap_press = np.asarray(vap_press)
    psych_const = np.asarray(psych_const)

    tol = 1e-2  # stopping criterion (iteration continues until abs(xk_1 - xk) < tol)

    xk_1 = temp - 10  # first start value (x_(k-1))
    xk = temp.copy()  # second start value (x_k)
    yk_1 = _water_vapor_pressure_difference(temp, xk_1, vap_press, psych_const)
    yk = _water_vapor_pressure_difference(temp, xk, vap_press, psych_const)

    while True:
        d = (xk - xk_1) / (yk - yk_1) * yk  # secant method
        iter_pos = np.abs(d) > tol

        if iter_pos.sum() == 0:
            break

        xk_1[iter_pos] = xk[iter_pos]
        yk_1[iter_pos] = yk[iter_pos]
        xk[iter_pos] -= d[iter_pos]
        yk[iter_pos] = _water_vapor_pressure_difference(
            temp[iter_pos],
            xk[iter_pos],
            vap_press[iter_pos],
            psych_const[iter_pos],
        )

    return xk


def dew_point_temperature(temp, rel_hum):
    """
    Calculate dew point temperature.

    Parameters
    ----------
    temp : numeric
        Air temperature (K).

    rel_hum : numeric
        Relative humidity (%).

    Returns
    -------
    dew_point_temp : numeric
        Dew point temperature (K).
    """
    ca, cb, cc = VAPOR_PRESSURE_COEFFS_WATER
    vap_press_water = vapor_pressure(temp, rel_hum, 'water')
    td_c = cc * np.log(vap_press_water / ca) / (cb - np.log(vap_press_water / ca))
    return td_c + c.T0


def precipitable_water(temp, vap_press):
    """
    Calculate precipitable water after Prata (1996).

    Parameters
    ----------
    temp : numeric
        Air temperature (K).

    vap_press : numeric
        Vapor pressure (Pa).

    Returns
    -------
    precipitable_water : numeric
        Precipitable water (kg m-2).

    References
    ----------
    .. [1] Prata, A. J. (1996). A new long-wave formula for estimating downward
       clear-sky radiation at the surface. Quarterly Journal of the Royal
       Meteorological Society, 122(533), 1127–1151. doi:10.1002/qj.49712253306
    """
    vap_press_hpa = vap_press / 100
    u = 46.5 * vap_press_hpa / temp  # in g cm-2
    return u * (1e-3 * 1e4)  # in kg m-2


def cloud_fraction_from_humidity(temp, rel_hum, elev, temp_lapse_rate, dew_point_temp_lapse_rate):
    """
    Calculate cloud fraction from relative humidity at the 700 hPa level
    following Liston and Elder (2006).

    Parameters
    ----------
    temp : numeric
        Air temperature (K).

    rel_hum : numeric
        Relative humidity (%).

    elev : numeric
        Elevation (m).

    temp_lapse_rate : numeric
        Temperature lapse rate (K m-1).

    dew_point_temp_lapse_rate : numeric
        Dew point temperature lapse rate (K m-1).

    Returns
    -------
    cloud_fraction : numeric
        Cloud fraction (0-1).

    References
    ----------
    .. [1] Liston, G. E., & Elder, K. (2006). A Meteorological Distribution
       System for High-Resolution Terrestrial Modeling (MicroMet). Journal of
       Hydrometeorology, 7(2), 217–234. https://doi.org/10.1175/JHM486.1
    """
    temp = np.asarray(temp)
    rel_hum = np.asarray(rel_hum)
    elev = np.asarray(elev)

    td = dew_point_temperature(temp, rel_hum)
    elev_diff = elev - 3000  # 700 hPa = 3000 m in a standard atmosphere

    temp700 = temp - (temp_lapse_rate * elev_diff)  # T in 700 hPa
    td700 = td - (dew_point_temp_lapse_rate * elev_diff)  # Td in 700 hPa
    sat_vap_press700 = saturation_vapor_pressure(temp700)
    vap_press700 = saturation_vapor_pressure(td700)
    rel_hum700 = 100 * vap_press700 / sat_vap_press700  # RH in 700 hPa

    cloud_frac = 0.832 * np.exp((rel_hum700 - 100) / 41.6)
    return cloud_frac.clip(0, 1)
