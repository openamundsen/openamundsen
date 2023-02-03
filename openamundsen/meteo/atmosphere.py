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


def pressure_to_altitude(pressure):
    """
    Calculate altitude from atmospheric pressure.

    This is the inverse function for atmospheric_pressure().

    Parameters
    ----------
    pressure : numeric
        Atmospheric pressure (Pa).

    Returns
    -------
    elev : numeric
        Height above sea level (m).
    """
    exp = c.SPEC_HEAT_CAP_DRY_AIR * c.MOLAR_MASS_DRY_AIR / c.UNIVERSAL_GAS_CONSTANT
    return (
        (1 - (np.asarray(pressure) / c.STANDARD_ATMOSPHERE)**(1 / exp))
        * c.SPEC_HEAT_CAP_DRY_AIR * c.STANDARD_SEA_LEVEL_TEMPERATURE
        / c.GRAVITATIONAL_ACCELERATION
    )


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
        pos = temp_c >= 0
        ca = np.where(pos, VAPOR_PRESSURE_COEFFS_WATER[0], VAPOR_PRESSURE_COEFFS_ICE[0])
        cb = np.where(pos, VAPOR_PRESSURE_COEFFS_WATER[1], VAPOR_PRESSURE_COEFFS_ICE[1])
        cc = np.where(pos, VAPOR_PRESSURE_COEFFS_WATER[2], VAPOR_PRESSURE_COEFFS_ICE[2])
    elif over == 'water':
        ca = VAPOR_PRESSURE_COEFFS_WATER[0]
        cb = VAPOR_PRESSURE_COEFFS_WATER[1]
        cc = VAPOR_PRESSURE_COEFFS_WATER[2]
    elif over == 'ice':
        ca = VAPOR_PRESSURE_COEFFS_ICE[0]
        cb = VAPOR_PRESSURE_COEFFS_ICE[1]
        cc = VAPOR_PRESSURE_COEFFS_ICE[2]
    else:
        raise NotImplementedError

    return ca * np.exp(cb * temp_c / (cc + temp_c))


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

    tol = 1e-2  # stopping criterion (iteration continues until abs(x0 - x1) < tol)

    x0 = temp - 10  # first start value (x_(k-1))
    x1 = temp.copy()  # second start value (x_k)
    y0 = _water_vapor_pressure_difference(temp, x0, vap_press, psych_const)
    y1 = _water_vapor_pressure_difference(temp, x1, vap_press, psych_const)

    while True:
        d = (x1 - x0) / (y1 - y0) * y1  # secant method
        iter_pos = np.abs(d) > tol

        if iter_pos.sum() == 0:
            break

        x0[iter_pos] = x1[iter_pos]
        y0[iter_pos] = y1[iter_pos]
        x1[iter_pos] -= d[iter_pos]
        y1[iter_pos] = _water_vapor_pressure_difference(
            temp[iter_pos],
            x1[iter_pos],
            vap_press[iter_pos],
            psych_const[iter_pos],
        )

    return x1


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


def cloud_fraction_from_humidity(
    temp,
    rel_hum,
    elev,
    temp_lapse_rate,
    dew_point_temp_lapse_rate,
    pressure_level=70000,
    saturation_cloud_fraction=0.832,
    e_folding_humidity=58.4,
):
    """
    Calculate cloud fraction from relative humidity at the 700 hPa level
    following Walcek (1994) and Liston and Elder (2006).

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

    pressure_level : float, default 70000
        Pressure level (Pa) on which the calculation should be performed.
        Default value of 700 hPa is from [2].

    saturation_cloud_fraction : float, default 0.832
        Cloud fraction at 100% relative humidity. Default value from [2].

    e_folding_humidity : float, default 58.4
        "e-folding relative humidity", i.e., the relative humidity depression
        below 100% where cloud fraction decreases to 37% (exp(-1)) of its value
        at 100% humidity. Default value from [2].

    Returns
    -------
    cloud_fraction : numeric
        Cloud fraction (0-1).

    References
    ----------
    .. [1] Walcek, C. J. (1994). Cloud Cover and Its Relationship to Relative
       Humidity during a Springtime Midlatitude Cyclone. Monthly Weather Review,
       122(6), 1021–1035.
       https://doi.org/10.1175/1520-0493(1994)122<1021:CCAIRT>2.0.CO;2

    .. [2] Liston, G. E., & Elder, K. (2006). A Meteorological Distribution
       System for High-Resolution Terrestrial Modeling (MicroMet). Journal of
       Hydrometeorology, 7(2), 217–234. https://doi.org/10.1175/JHM486.1
    """
    temp = np.asarray(temp)
    rel_hum = np.asarray(rel_hum)
    elev = np.asarray(elev)

    pressure_level_elev = pressure_to_altitude(pressure_level)
    elev_diff = elev - pressure_level_elev

    td = dew_point_temperature(temp, rel_hum)
    temp700 = temp - (temp_lapse_rate * elev_diff)  # T in 700 hPa
    td700 = td - (dew_point_temp_lapse_rate * elev_diff)  # Td in 700 hPa
    sat_vap_press700 = saturation_vapor_pressure(temp700)
    vap_press700 = saturation_vapor_pressure(td700)
    rel_hum700 = 100 * vap_press700 / sat_vap_press700  # RH in 700 hPa

    cloud_frac = (  # eq. (1) in [1], eq. (20) in [2]
        saturation_cloud_fraction
        * np.exp((rel_hum700 - 100) / (100 - e_folding_humidity))
    )
    return cloud_frac.clip(0, 1)


def cloud_factor_from_cloud_fraction(cloud_fraction):
    """
    Calculate the cloud factor (i.e., the ratio of the actual global radiation
    and the clear-sky global radiation) from the cloud fraction after Greuell
    et al. (1997).

    Parameters
    ----------
    cloud_fraction : numeric
        Cloud fraction (0-1).

    Returns
    -------
    cloud_factor : numeric
        Cloud factor (0-1).

    References
    ----------
    .. [1] Greuell, W., Knap, W. H., & Smeets, P. C. (1997). Elevational
       changes in meteorological variables along a midlatitude glacier during
       summer. Journal of Geophysical Research, 102(D22), 25941–25954.
       https://doi.org/10.1029/97JD02083
    """
    cloud_fraction = np.asarray(cloud_fraction)
    cloud_factor = 1 - 0.233 * cloud_fraction - 0.415 * cloud_fraction**2
    return cloud_factor.clip(0, 1)


def cloud_fraction_from_cloud_factor(cloud_factor):
    """
    Calculate cloud fraction from cloud factor after Greuell et al. (1997),
    inverted fit function.

    Parameters
    ----------
    cloud_factor : numeric
        Cloud factor (0-1).

    Returns
    -------
    cloud_fraction : numeric
        Cloud fraction (0-1).

    References
    ----------
    .. [1] Greuell, W., Knap, W. H., & Smeets, P. C. (1997). Elevational
       changes in meteorological variables along a midlatitude glacier during
       summer. Journal of Geophysical Research, 102(D22), 25941–25954.
       https://doi.org/10.1029/97JD02083
    """
    cloud_factor = np.asarray(cloud_factor)
    cloud_fraction = -1.4059 * cloud_factor**2 + 0.4473 * cloud_factor + 0.997
    return cloud_fraction.clip(0, 1)


def clear_sky_emissivity(prec_wat):
    """
    Calculate emissivity of the atmosphere under clear-sky conditions after
    Prata (1996).

    Parameters
    ----------
    prec_wat : numeric
        Precipitable water (kg m-2).

    Returns
    -------
    emissivity : numeric
        Emissivity (0-1).

    References
    ----------
    .. [1] Prata, A. J. (1996). A new long-wave formula for estimating downward
       clear-sky radiation at the surface. Quarterly Journal of the Royal
       Meteorological Society, 122(533), 1127–1151. doi:10.1002/qj.49712253306
    """
    prec_wat_cm = np.asarray(prec_wat) / 10  # kg m-2 (= mm) to g cm-2 (= cm)
    return 1 - (1 + prec_wat_cm) * np.exp(-np.sqrt(1.2 + 3 * prec_wat_cm))


def precipitation_phase(temp, threshold_temp=c.T0, temp_range=0., method='linear'):
    """
    Calculate precipitation phase.

    Parameters
    ----------
    temp : numeric
        Temperature (in K) on which to perform the partitioning (e.g., air
        temperature or wet-bulb temperature).

    threshold_temp : float, default 273.15
        Threshold temperature (in K) at which 50% of precipitation falls as
        snow.

    temp_range : float, default 0
        Temperature range in which mixed precipitation can occur. If 0,
        precipitation can only be either rain or snow.

    method : str, default 'linear'
        Method for interpolating precipitation phase within the
        (threshold_temp - temp_range/2, threshold_temp + temp_range/2) range.

    Returns
    -------
    snowfall_fraction : numeric
        Fraction of precipitation falling as snow (0-1).
    """

    if method != 'linear':
        raise NotImplementedError

    temp = np.asarray(temp)

    if temp_range < 0:
        raise ValueError('temp_range must be positive')
    elif temp_range == 0:
        snowfall_frac = (temp < threshold_temp) * 1.
    else:
        t1 = threshold_temp - temp_range / 2.
        t2 = threshold_temp + temp_range / 2.
        snowfall_frac = (1 - (temp - t1) / (t2 - t1)).clip(0, 1)

    return snowfall_frac


def log_wind_profile(ref_wind_speed, ref_height, height, roughness_length):
    """
    Calculate wind speed at a different height based on a logarithmic wind
    profile.

    Parameters
    ----------
    ref_wind_speed : numeric
        Reference wind speed (m s-1).

    ref_height : float
        Height of the reference wind speed (m).

    height : float
        New height.

    roughness_length : float
        Surface roughness length (m).

    Returns
    -------
    wind_speed : numeric
        Wind speed at the new height (m s-1).
    """
    return ref_wind_speed * (
        np.log(height / roughness_length)
        / np.log(ref_height / roughness_length)
    )


def wind_to_uv(ws, wd):
    """
    Convert wind speed and direction to u and v components.

    Parameters
    ----------
    ws : numeric
        Wind speed (m s-1).

    wd : numeric
        Wind direction (meteorological degrees, i.e., 0 = North, 90 = East).

    Returns
    -------
    u : numeric
        Zonal component (m s-1).

    v : numeric
        Meridional component (m s-1).
    """
    u = -ws * np.sin(np.deg2rad(wd))
    v = -ws * np.cos(np.deg2rad(wd))
    return u, v


def wind_from_uv(u, v):
    """
    Convert u and v components to wind speed and direction.

    Parameters
    ----------
    u : numeric
        Zonal component (m s-1).

    v : numeric
        Meridional component (m s-1).

    Returns
    -------
    ws : numeric
        Wind speed (m s-1).

    wd : numeric
        Wind direction (meteorological degrees, i.e., 0 = North, 90 = East).
    """
    ws = np.sqrt(u**2 + v**2)
    wd = (270 - np.rad2deg(np.arctan2(v, u))) % 360
    return ws, wd
