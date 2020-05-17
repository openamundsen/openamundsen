import numpy as np
from numpy.testing import assert_allclose
import openamundsen.modules.radiation as rad
import pandas as pd
from pvlib import solarposition


def test_equation_of_time():
    doys = np.arange(366) + 1
    eot_oa = rad.equation_of_time(doys)
    eot_pv = solarposition.equation_of_time_spencer71(doys)
    assert_allclose(eot_pv, eot_oa, atol=0.15)  # error is due to 365 vs 365.25 used in calculating day angle


def test_declination_angle():
    doys = np.arange(366) + 1
    dec_oa = rad.declination_angle(doys)
    dec_pv = np.rad2deg(solarposition.declination_spencer71(doys))
    assert_allclose(dec_pv, dec_oa, atol=0.35)


def test_hour_angle():
    lon = 11.400375
    timezone = 1

    dates = pd.DatetimeIndex([
        '2020-01-02 08:43:37',
        '2020-07-15 16:55:31',
        '2020-09-01 12:23:11',
        '2020-11-15 23:40:00',
        '2020-12-24 15:34:11',
    ]).tz_localize(f'Etc/GMT-{timezone}')
    # (Etc/GMT-1 corresponds to GMT+1 in the traditional sense
    # (https://stackoverflow.com/questions/4008960/pytz-and-etc-gmt-5))

    eots = rad.equation_of_time(np.array([date.dayofyear for date in dates]))
    has_oa = rad.hour_angle(dates, timezone, lon, eots)
    has_pv = solarposition.hour_angle(dates, lon, eots)
    assert_allclose(has_pv, has_oa, atol=0.01)
