import numpy as np
import openamundsen.constants as c
import pandas as pd


def day_angle(doy):
    """
    Return the day of the year in angular form.

    Parameters
    ----------
    doy : int
        Day of the year (Jan 1 = 1).

    Returns
    -------
    day_angle : float
        Day angle in radians.
    """
    # TODO
    #   Use 2*pi/365 for normal years and 2*pi/366 for leap years
    #   instead of 365.25?
    return (2. * np.pi / c.DAYS_PER_YEAR) * (doy - 1)


def equation_of_time(doy, method='spencer'):
    """
    Calculate the equation of time, i.e., the difference in time between solar
    noon at 0 degrees longitude and 12:00 UTC.

    Parameters
    ----------
    doy : int
        Day of the year (Jan 1 = 1).

    Returns
    -------
    eot_m : float
        Equation of time in minutes for the given day. The value is within
        +/- 16 minutes throughout the entire year.

    References
    ----------
    .. [1] J. W. Spencer, "Fourier series representation of the position of the
       sun" in Search 2 (5), p. 172 (1971)
    """
    # TODO
    #   Implement Reda method (https://www.nrel.gov/docs/fy08osti/34302.pdf),
    #   possibly PVCDROM (see https://github.com/pvlib/pvlib-python/blob/master/pvlib/solarposition.py)

    da = day_angle(doy)

    if method == 'spencer':
        eot = (
            0.0000075 +
            0.001868 * np.cos(da) - 0.032077 * np.sin(da) -
            0.014615 * np.cos(2 * da) - 0.040849 * np.sin(2 * da)
        )  # in radians
    else:
        raise NotImplementedError(f'Unsupported method: {method}')

    eot_m = c.HOURS_PER_DAY * c.MINUTES_PER_HOUR / (2 * np.pi) * eot  # in minutes
    return eot_m


def declination_angle(doy):
    """
    Calculate the solar declination angle after Bourges (1985).

    Parameters
    ----------
    doy : int
        Day of the year (Jan 1 = 1).

    Returns
    -------
    declination : float
        Solar declination angle in degrees.

    References
    ----------
    .. [1] Bernard Bourges. Improvement in solar declination computation. Solar
       Energy, 1985, 35(4), pp.367-369.
    """
    pass
    
    day_number = np.deg2rad((360 / c.DAYS_PER_YEAR) * (doy - 79.346))
    declination = (
        0.3723 + 23.2567 * np.sin(day_number) - 0.7580 * np.cos(day_number)
        + 0.1149 * np.sin(2 * day_number) + 0.3656 * np.cos(2 * day_number)
        - 0.1712 * np.sin(3 * day_number) + 0.0201 * np.cos(3 * day_number)
    )
    return declination


def hour_angle(date, timezone, lon, eot):
    """
    Calculate the hour angle, i.e., the angular displacement of the sun east or
    west of the local meridian due to rotation of the earth on its axis at
    15° per hour.

    Parameters
    ----------
    date : datetime-like
        Local date and time.

    timezone : int
        Timezone, e.g. 1 for CET.

    lon : float
        Longitude.

    eot : float
        Equation of time in minutes.

    Returns
    -------
    hour_angle : float
        Hour angle in degrees.
    """
    date = pd.to_datetime(date)
    hour = (date - date.normalize()).total_seconds() / c.SECONDS_PER_HOUR  # fractional hour of the day

    lstm = c.STANDARD_TIMEZONE_WIDTH * timezone  # local standard time meridian
    tc = c.MINUTES_PER_DEGREE_OF_EARTH_ROTATION * (lon - lstm) + eot  # time correction (minutes)
    lst = hour + tc / c.MINUTES_PER_HOUR  # local solar time
    ha = c.SUN_DEGREES_PER_HOUR * (lst - 12)

    return ha


def sun_vector(lat, ha, dec):
    """
    Calculate the vector defining the position of the sun after Corripio
    (2003).

    Parameters
    ----------
    lat : float
        Latitude (degrees).

    ha : float
        Hour angle (degrees).

    dec : float
        Declination angle (degrees).

    Returns
    -------
    vec : ndarray
        Solar vector.

    References
    ----------
    .. [1] Corripio, J. G. (2003). Vectorial algebra algorithms for calculating
       terrain parameters from DEMs and solar radiation modelling in mountainous
       terrain. International Journal of Geographical Information Science, 17(1),
       1–23. https://doi.org/10.1080/13658810210157796
    """
    lat_rad = np.deg2rad(lat)
    ha_rad = np.deg2rad(ha)
    dec_rad = np.deg2rad(dec)

    return np.array([
        -np.sin(ha_rad) * np.cos(dec_rad),
        np.sin(lat_rad) * np.cos(ha_rad) * np.cos(dec_rad) - np.cos(lat_rad) * np.sin(dec_rad),
        np.cos(lat_rad) * np.cos(ha_rad) * np.cos(dec_rad) + np.sin(lat_rad) * np.sin(dec_rad),
    ])


def sun_parameters(date, lon, lat, timezone):
    """
    Calculate sun related parameters for a specified date and position.

    Parameters
    ----------
    date : datetime-like
        Local date and time.

    lon : float
        Longitude (degrees).

    lat : float
        Latitude (degrees).

    timezone : int
        Timezone, e.g. 1 for CET.

    Returns
    -------
    d : dict
        Dictionary containing the following keys:
        - 'day_angle': day angle (radians)
        - 'hour_angle': hour angle (degrees)
        - 'declination_angle': declination angle (degrees)
        - 'equation_of_time': equation of time (minutes)
        - 'sun_vector': vector describing the position of the sun
        - 'zenith_angle': zenith angle (degrees)
        - 'sun_over_horizon': True if the sun is over the horizon
    """
    date = pd.to_datetime(date)
    eot = equation_of_time(date.dayofyear)
    da = day_angle(date.dayofyear)
    ha = hour_angle(date, timezone, lon, eot)
    dec = declination_angle(date.dayofyear)
    sv = sun_vector(lat, ha, dec)
    zenith_angle = np.rad2deg(np.arccos(sv[2]))

    return {
        'day_angle': da,
        'hour_angle': ha,
        'declination_angle': dec,
        'equation_of_time': eot,
        'sun_vector': sv,
        'zenith_angle': zenith_angle,
        'sun_over_horizon': zenith_angle < 90,
    }
