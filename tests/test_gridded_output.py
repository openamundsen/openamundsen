import numpy as np
from numpy.testing import assert_allclose
import openamundsen as oa
from openamundsen.fileio.griddedoutput import _freq_write_dates
import pandas as pd
import pytest
import rasterio
import xarray as xr


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


@pytest.mark.parametrize('fmt', ['netcdf', 'ascii', 'geotiff'])
def test_formats(fmt, base_config, tmp_path):
    config = base_config.copy()
    config.end_date = '2020-01-16'
    config.results_dir = tmp_path
    grid_cfg = config.output_data.grids
    grid_cfg.format = fmt
    grid_cfg.variables = [
        {'var': 'meteo.temp', 'freq': 'D'},
        {'var': 'meteo.precip', 'freq': 'D', 'agg': 'sum'},
        {'var': 'meteo.sw_in', 'dates': ['2020-01-15 12:00'], 'name': 'myvar'},
    ]

    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run()

    if fmt == 'netcdf':
        ds = xr.open_dataset(tmp_path / 'output_grids.nc')
        assert 'temp' in ds.variables
        assert ds.time1.to_index().equals(pd.DatetimeIndex([
            '2020-01-15 00:00',
            '2020-01-16 00:00',
        ]))
        assert 'precip' in ds.variables
        assert ds.time2.to_index().equals(pd.DatetimeIndex([
            '2020-01-15 21:00',
            '2020-01-16 21:00',
        ]))
        assert pd.DatetimeIndex(ds.time2_bounds.values.ravel()).equals(pd.DatetimeIndex([
            '2020-01-15 00:00:00',
            '2020-01-15 21:00:00',
            '2020-01-15 21:00:00',
            '2020-01-16 21:00:00',
        ]))
        assert 'myvar' in ds.variables
        assert ds.time3.to_index().equals(pd.DatetimeIndex(['2020-01-15 12:00']))
    else:
        if fmt == 'ascii':
            ext = 'asc'
        elif fmt == 'geotiff':
            ext = 'tif'

        fn = tmp_path / f'temp_2020-01-16T0000.{ext}'
        with rasterio.open(fn) as ds:
            ds.read(1)

        fn = tmp_path / f'precip_2020-01-16T0000_2020-01-16T2100.{ext}'
        with rasterio.open(fn) as ds:
            ds.read(1)

        fn = tmp_path / f'myvar_2020-01-15T1200.{ext}'
        with rasterio.open(fn) as ds:
            ds.read(1)


def test_values(base_config, tmp_path):
    config = base_config.copy()
    config.start_date = '2020-01-17'
    config.end_date = '2020-01-19'
    config.results_dir = tmp_path
    grid_cfg = config.output_data.grids
    grid_cfg.variables = [
        {'var': 'meteo.temp', 'freq': 'D', 'agg': 'mean'},
        {'var': 'meteo.precip', 'freq': 'D', 'agg': 'sum'},
        {'var': 'meteo.sw_in', 'dates': ['2020-01-17 12:00']},
        {'var': 'snow.swe', 'freq': 'D'},
    ]

    model = oa.OpenAmundsen(config)
    model.initialize()
    data_vals = {
        'temp': {},
        'precip': {},
        'sw_in': {},
        'swe': {},
    }

    for date in model.dates:
        model.run_single()
        date = date.strftime('%Y-%m-%d %H:%M')
        data_vals['temp'][date] = model.state.meteo.temp.copy()
        data_vals['precip'][date] = model.state.meteo.precip.copy()
        data_vals['sw_in'][date] = model.state.meteo.sw_in.copy()
        data_vals['swe'][date] = model.state.snow.swe.copy()

    ds = xr.load_dataset(tmp_path / 'output_grids.nc')
    assert_allclose(
        data_vals['sw_in']['2020-01-17 12:00'],
        ds.sw_in.loc['2020-01-17 12:00', :, :].values,
    )

    assert_allclose(
        data_vals['swe']['2020-01-19 00:00'],
        ds.swe.loc['2020-01-19 00:00', :, :].values,
    )

    mean_temp = (
        data_vals['temp']['2020-01-18 00:00']
        + data_vals['temp']['2020-01-18 03:00']
        + data_vals['temp']['2020-01-18 06:00']
        + data_vals['temp']['2020-01-18 09:00']
        + data_vals['temp']['2020-01-18 12:00']
        + data_vals['temp']['2020-01-18 15:00']
        + data_vals['temp']['2020-01-18 18:00']
        + data_vals['temp']['2020-01-18 21:00']
    ) / 8
    assert_allclose(
        mean_temp,
        ds.temp.loc['2020-01-18 21:00', :, :].values,
    )

    precip_sum = (
        data_vals['precip']['2020-01-18 00:00']
        + data_vals['precip']['2020-01-18 03:00']
        + data_vals['precip']['2020-01-18 06:00']
        + data_vals['precip']['2020-01-18 09:00']
        + data_vals['precip']['2020-01-18 12:00']
        + data_vals['precip']['2020-01-18 15:00']
        + data_vals['precip']['2020-01-18 18:00']
        + data_vals['precip']['2020-01-18 21:00']
    )
    assert_allclose(
        precip_sum,
        ds.precip.loc['2020-01-18 21:00', :, :].values,
    )


def test_data_type(base_config, tmp_path):
    config = base_config.copy()
    config.start_date = '2020-01-17'
    config.end_date = '2020-01-17'
    config.results_dir = tmp_path
    grid_cfg = config.output_data.grids
    grid_cfg.variables = [
        {'var': 'meteo.temp', 'freq': 'D', 'agg': 'mean'},
        {'var': 'snow.num_layers', 'freq': 'D'},
        {'var': 'snow.num_layers', 'freq': 'D', 'agg': 'sum', 'name': 'num_layers_sum'},
        {'var': 'snow.num_layers', 'freq': 'D', 'agg': 'mean', 'name': 'num_layers_mean'},
    ]

    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run()

    ds = xr.open_dataset(tmp_path / 'output_grids.nc')
    assert np.issubdtype(ds.temp.dtype, np.float32)
    assert np.issubdtype(ds.num_layers.dtype, np.integer)
    assert np.issubdtype(ds.num_layers_sum.dtype, np.integer)
    assert np.issubdtype(ds.num_layers_mean.dtype, np.float32)


def test_nothing_to_write(base_config, tmp_path):
    config = base_config.copy()
    config.start_date = '2020-01-01'
    config.end_date = '2020-01-01 03:00'
    config.results_dir = tmp_path
    grid_cfg = config.output_data.grids
    grid_cfg.variables = [
        {'var': 'snow.swe', 'freq': 'D', 'agg': 'sum'},
        {'var': 'meteo.temp', 'dates': ['2020-01-02 12:00']},
    ]
    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run()
    assert not (tmp_path / 'output_grids.nc').exists()

    grid_cfg.variables = [
        {'var': 'snow.swe', 'freq': 'D', 'agg': 'sum'},
        {'var': 'meteo.temp', 'dates': ['2020-01-01 02:00']},
    ]
    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run()
    ds = xr.open_dataset(tmp_path / 'output_grids.nc')
    assert list(ds.data_vars.keys()) == ['temp']