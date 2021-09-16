from .conftest import base_config
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import openamundsen as oa
import openamundsen.errors as errors
import pandas as pd
import pytest
import xarray as xr


@pytest.mark.parametrize('fmt', ['netcdf', 'csv', 'memory'])
def test_formats(fmt, tmp_path):
    config = base_config()
    config.end_date = '2020-01-16'
    config.results_dir = tmp_path
    config.output_data.timeseries.format = fmt
    config.output_data.timeseries.variables = [{'var': 'snow.num_layers'}]

    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run()

    point_ids = ['bellavista', 'latschbloder', 'proviantdepot']

    if fmt in ('netcdf', 'memory'):
        if fmt == 'netcdf':
            ds = xr.open_dataset(tmp_path / 'output_timeseries.nc')
        elif fmt == 'memory':
            ds = model.point_output.data

        assert ds.time.to_index().equals(model.dates)
        assert_array_equal(ds.point, point_ids)
        assert_array_equal(
            list(ds.coords.keys()),
            ['time', 'point', 'lon', 'lat', 'alt', 'x', 'y', 'soil_layer', 'snow_layer'],
        )
        assert ds.temp.dims == ('time', 'point')
        assert ds.snow_thickness.dims == ('time', 'snow_layer', 'point')
        assert ds.soil_temp.dims == ('time', 'soil_layer', 'point')
        assert ds.temp.dtype == np.float32
        assert np.issubdtype(ds.num_layers.dtype, np.integer)
        assert np.all(ds.temp > 250.)
    elif fmt == 'csv':
        for point_id in point_ids:
            assert (tmp_path / f'point_{point_id}.csv').exists()

        df = pd.read_csv(tmp_path / 'point_bellavista.csv', index_col='time', parse_dates=True)
        assert df.index.equals(model.dates)
        assert df.temp.dtype == np.float64
        assert np.issubdtype(df.num_layers.dtype, np.integer)
        assert 'snow_thickness0' in df
        assert np.all(df.temp > 250.)


def test_values():
    config = base_config()
    config.end_date = '2020-01-15'

    model = oa.OpenAmundsen(config)
    model.initialize()

    point = 'proviantdepot'
    row = int(model.meteo.sel(station=point).row)
    col = int(model.meteo.sel(station=point).col)

    data_temp = pd.Series(index=model.dates, dtype=float)
    data_soil_temp1 = pd.Series(index=model.dates, dtype=float)

    for date in model.dates:
        model.run_single()
        data_temp[date] = model.state.meteo.temp[row, col]
        data_soil_temp1[date] = model.state.soil.temp[1, row, col]

    ds = model.point_output.data.sel(point=point)
    assert_allclose(ds.temp.values, data_temp)
    assert_allclose(ds.soil_temp.isel(soil_layer=1).values, data_soil_temp1)


@pytest.mark.parametrize('write_freq', ['M', '7H', '3H', '10min'])
def test_write_freq(write_freq, tmp_path):
    config = base_config()
    config.end_date = '2020-01-15'
    config.results_dir = tmp_path
    config.output_data.timeseries.format = 'netcdf'
    config.output_data.timeseries.write_freq = write_freq

    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run()

    ds = xr.open_dataset(tmp_path / 'output_timeseries.nc')
    assert ds.time.to_index().equals(model.dates)


def test_points():
    bc = base_config()
    bc.end_date = '2020-01-15 00:00'

    config = bc.copy()
    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run()
    ds = model.point_output.data
    assert_array_equal(ds.point, ['bellavista', 'latschbloder', 'proviantdepot'])

    config = bc.copy()
    config.output_data.timeseries.add_default_points = False
    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run()
    ds = model.point_output.data
    assert ds.point.size == 0

    config = bc.copy()
    config.output_data.timeseries.add_default_points = False
    config.output_data.timeseries.points.append({
        'x': 640367,
        'y': 5182896,
    })
    config.output_data.timeseries.points.append({
        'x': 645378,
        'y': 5190907,
        'name': 'mypoint',
    })
    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run()
    ds = model.point_output.data
    assert_array_equal(ds.point, ['point1', 'mypoint'])
    assert_allclose(ds.alt, [3181.89, 1948.97])

    # Duplicate point name
    config = bc.copy()
    config.output_data.timeseries.points.append({
        'x': 640367,
        'y': 5182896,
        'name': 'bellavista',
    })
    model = oa.OpenAmundsen(config)
    with pytest.raises(errors.ConfigurationError):
        model.initialize()

    # Duplicate point name
    config = bc.copy()
    config.output_data.timeseries.points.append({
        'x': 640367,
        'y': 5182896,
        'name': 'bellavista',
    })
    model = oa.OpenAmundsen(config)
    with pytest.raises(errors.ConfigurationError):
        model.initialize()

    # Point not within grid
    config = bc.copy()
    config.output_data.timeseries.points.append({
        'x': 637152,
        'y': 5196427,
    })
    model = oa.OpenAmundsen(config)
    with pytest.raises(errors.ConfigurationError):
        model.initialize()


def test_variables():
    bc = base_config()
    bc.end_date = '2020-01-15 00:00'
    bc.output_data.timeseries.variables = []

    config = bc.copy()
    config.output_data.timeseries.add_default_variables = False
    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run()
    ds = model.point_output.data
    assert len(ds.data_vars) == 0

    config = bc.copy()
    config.output_data.timeseries.add_default_variables = False
    config.output_data.timeseries.variables.append({'var': 'meteo.spec_heat_cap_moist_air'})
    config.output_data.timeseries.variables.append({
        'var': 'surface.conductance',
        'name': 'myvar',
    })
    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run()
    ds = model.point_output.data
    assert_array_equal(ds.data_vars, ['spec_heat_cap_moist_air', 'myvar'])

    # Invalid variable name
    config = bc.copy()
    config.output_data.timeseries.variables.append({'var': 'meteo.asdf'})
    model = oa.OpenAmundsen(config)
    with pytest.raises(errors.ConfigurationError):
        model.initialize()

    # Output name already in use
    config = bc.copy()
    config.output_data.timeseries.variables.append({'var': 'meteo.temp'})
    model = oa.OpenAmundsen(config)
    with pytest.raises(errors.ConfigurationError):
        model.initialize()
