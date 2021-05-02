import numpy as np
from numpy.testing import assert_allclose
from openamundsen.fileio.fieldoutput import _freq_write_dates
import pandas as pd
import pytest


def test_freq_write_dates():
    dates = pd.date_range(start='2021-01-01 00:00', end='2021-12-31 23:00', freq='H')
    assert dates.equals(_freq_write_dates(dates, 'H', False))
    wd = _freq_write_dates(dates, '3H', False)
    assert np.all(wd.hour.isin([0, 3, 6, 9, 12, 15, 18, 21]))
    wd = _freq_write_dates(dates, 'D', False)
    assert np.all(wd.hour == 0)
    assert dates.normalize().unique().equals(wd.normalize())
    wd = _freq_write_dates(dates, 'D', True)
    assert np.all(wd.hour == 23)
    wd = _freq_write_dates(dates, 'M', True)
    assert np.all(wd.day.isin([28, 30, 31]))
    assert np.all(wd.hour == 23)

    dates = pd.date_range(start='2021-01-01 06:00', end='2021-12-31 21:00', freq='3H')
    wd = _freq_write_dates(dates, 'D', False)
    assert np.all(wd.hour == 6)
    wd = _freq_write_dates(dates, 'D', True)
    assert np.all(wd.hour == 3)
    with pytest.raises(ValueError):
        wd = _freq_write_dates(dates, 'H', False)
    with pytest.raises(ValueError):
        wd = _freq_write_dates(dates, 'H', True)

    dates = pd.date_range(start='2021-01-01 02:00', end='2021-12-31 04:00', freq='3H')
    wd = _freq_write_dates(dates, 'D', False)
    assert np.all(wd.hour == 2)

    dates = pd.date_range(start='2021-01-01 02:00', end='2021-01-01 07:00', freq='H')
    assert len(_freq_write_dates(dates, 'D', False)) == 1
    assert len(_freq_write_dates(dates, 'D', True)) == 0

    dates = pd.date_range(start='2021-01-15 00:00', end='2021-03-15 21:00', freq='3H')
    wd = _freq_write_dates(dates, 'M', True)
    assert len(wd) == 2
