from .conftest import base_config
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import openamundsen as oa
import openamundsen.errors as errors
import pandas as pd
from pathlib import Path
import pytest
import xarray as xr


def compare_tsp(tsp, **kwargs):
    d = dict(
        first_of_run=False,
        strict_first_of_year=False,
        strict_first_of_month=False,
        strict_first_of_day=False,
        first_of_year=False,
        first_of_month=False,
        first_of_day=False,
        last_of_run=False,
        strict_last_of_year=False,
        strict_last_of_month=False,
        strict_last_of_day=False,
        last_of_year=False,
        last_of_month=False,
        last_of_day=False,
    )
    d.update(kwargs)

    for key in d.keys():
        assert getattr(tsp, key) == d[key], f'{key} should be {d[key]}'


def test_timestep_properties():
    config = base_config()
    config.start_date = '2015-07-28 00:00'
    config.end_date = '2015-12-31'
    config.timestep = 'H'

    model = oa.OpenAmundsen(config)
    model.initialize()

    model.run_single()
    compare_tsp(
        model.timestep_props,
        first_of_run=True,
        first_of_year=True,
        first_of_month=True,
        first_of_day=True,
        strict_first_of_day=True,
    )

    model.run_single()
    compare_tsp(model.timestep_props)

    while model.date < pd.Timestamp('2015-07-29 00:00'):
        model.run_single()
    compare_tsp(
        model.timestep_props,
        first_of_day=True,
        strict_first_of_day=True,
    )

    while model.date < pd.Timestamp('2015-08-01 00:00'):
        model.run_single()
    compare_tsp(
        model.timestep_props,
        first_of_month=True,
        first_of_day=True,
        strict_first_of_month=True,
        strict_first_of_day=True,
    )

    config.start_date = '2015-07-28 23:00'
    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run_single()
    compare_tsp(
        model.timestep_props,
        first_of_run=True,
        first_of_year=True,
        first_of_month=True,
        first_of_day=True,
        last_of_day=True,
        strict_last_of_day=True,
    )

    config.start_date = '2015-08-01 00:00'
    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run_single()
    compare_tsp(
        model.timestep_props,
        first_of_run=True,
        first_of_year=True,
        first_of_month=True,
        first_of_day=True,
        strict_first_of_month=True,
        strict_first_of_day=True,
    )

    config.start_date = '2014-12-31 23:00'
    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run_single()
    compare_tsp(
        model.timestep_props,
        first_of_run=True,
        first_of_year=True,
        first_of_month=True,
        first_of_day=True,
        last_of_year=True,
        last_of_month=True,
        last_of_day=True,
        strict_last_of_year=True,
        strict_last_of_month=True,
        strict_last_of_day=True,
    )

    model.run_single()
    compare_tsp(
        model.timestep_props,
        first_of_year=True,
        first_of_month=True,
        first_of_day=True,
        strict_first_of_year=True,
        strict_first_of_month=True,
        strict_first_of_day=True,
    )

    config.end_date = '2015-12-15 12:00'
    config.start_date = config.end_date
    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run_single()
    compare_tsp(
        model.timestep_props,
        first_of_run=True,
        last_of_run=True,
        first_of_year=True,
        first_of_month=True,
        first_of_day=True,
        last_of_year=True,
        last_of_month=True,
        last_of_day=True,
    )


@pytest.mark.slow
def test_state_variable_reset():
    config = base_config()
    config.start_date = '2019-10-01'
    config.end_date = '2020-05-31'
    config.timestep = '3H'
    config.evapotranspiration.enabled = True
    config.canopy.enabled = True

    model = oa.OpenAmundsen(config)
    model.initialize()

    for category in model.state.categories:
        for var_name in model.state[category]._meta.keys():
            full_var_name = f'{category}.{var_name}'
            config.output_data.grids.variables.append({
                'var': full_var_name,
                'name': full_var_name,
            })

    config.reset_state_variables = False
    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run()
    ds0 = model.gridded_output.data

    config.reset_state_variables = True
    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run()
    ds1 = model.gridded_output.data

    compare_vars = [v['name'] for v in config.output_data.grids.variables]
    success = True

    for v in compare_vars:
        arr0 = ds0[v].values
        arr1 = ds1[v].values

        if not np.allclose(arr0, arr1, equal_nan=True):
            print(f'Mismatch in variable {v}\n'
                  f'Without reset:\n{arr0}\n'
                  f'With reset:\n{arr1}')
            success = False

    assert success


def test_simulation_timezone(tmp_path):
    ds = xr.load_dataset(f'{pytest.DATA_DIR}/meteo/rofental/netcdf/proviantdepot.nc')
    ds.to_netcdf(tmp_path / 'proviantdepot.nc')

    config = base_config()
    config.timestep = 'H'
    config.input_data.meteo.dir = str(tmp_path)

    config.start_date = '2020-01-15 00:00'
    config.end_date = '2020-01-15 23:00'
    config.simulation_timezone = None
    model1 = oa.OpenAmundsen(config)
    model1.initialize()
    model1.run()

    ds.shift(time=-1).to_netcdf(tmp_path / 'proviantdepot.nc')

    config.start_date = '2020-01-14 23:00'
    config.end_date = '2020-01-15 22:00'
    config.simulation_timezone = 0
    model2 = oa.OpenAmundsen(config)
    model2.initialize()
    model2.run()

    ds1 = model1.point_output.data
    ds2 = model2.point_output.data
    ds2['time'] = ds2['time'] + pd.Timedelta(hours=1)
    xr.testing.assert_identical(ds1, ds2)


def test_roi(tmp_path):
    config = base_config()
    config.start_date = '2020-01-15 12:00'
    config.end_date = '2020-01-15 12:00'

    for p in Path(config.input_data.grids.dir).glob('*.asc'):
        if not p.name.startswith('roi_'):
            (tmp_path / p.name).symlink_to(p)

    config.input_data.grids.dir = str(tmp_path)

    # No ROI set
    model1 = oa.OpenAmundsen(config)
    model1.initialize()
    assert np.all(model1.grid.roi)
    model1.run()

    # ROI manually set
    roi = model1.grid.roi.copy()
    roi[:10, :10] = False
    oa.fileio.write_raster_file(
        oa.util.raster_filename('roi', config),
        roi.astype(np.uint8),
        model1.grid.transform,
        driver='AAIGrid',
        dtype='uint8',
    )
    model2 = oa.OpenAmundsen(config)
    model2.initialize()
    assert_array_equal(roi, model2.grid.roi)
    model2.run()

    # ROI should be False where DEM is NaN
    Path(oa.util.raster_filename('roi', config)).unlink()
    dem = model2.state.base.dem
    dem[-10:, -10:] = np.nan
    oa.fileio.write_raster_file(
        oa.util.raster_filename('dem', config),
        dem,
        model2.grid.transform,
        driver='AAIGrid',
    )
    model3 = oa.OpenAmundsen(config)
    model3.initialize()
    assert np.all(model3.grid.roi[np.isfinite(dem)])
    assert not np.any(model3.grid.roi[np.isnan(dem)])
    model3.run()

    for model in (model1, model2, model3):
        for var in ('meteo.temp', 'snow.swe'):
            var_data = model.state[var]
            assert np.all(np.isfinite(var_data[model.grid.roi]))
            assert np.all(np.isnan(var_data[~model.grid.roi]))


def test_extend_roi_with_stations(tmp_path):
    config = base_config()
    config.start_date = '2020-07-15 12:00'
    config.end_date = '2020-07-15 12:00'

    model = oa.OpenAmundsen(config)
    model.initialize()

    for p in Path(config.input_data.grids.dir).glob('*.asc'):
        if not p.name.startswith('roi_'):
            (tmp_path / p.name).symlink_to(p)

    config.input_data.grids.dir = str(tmp_path)

    station_id = 'proviantdepot'
    meteo_station = model.meteo.sel(station=station_id)
    row = meteo_station.row
    col = meteo_station.col

    roi = model.grid.roi
    roi[row, col] = False
    oa.fileio.write_raster_file(
        oa.util.raster_filename('roi', config),
        roi.astype(np.uint8),
        model.grid.transform,
        driver='AAIGrid',
        dtype='uint8',
    )

    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run()
    assert not model.meteo.within_roi.loc[station_id]
    assert np.isnan(model.state.meteo.temp[row, col])
    assert np.isnan(model.state.meteo.sw_in[row, col])

    config.extend_roi_with_stations = True
    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run()
    assert model.meteo.within_roi.loc[station_id]
    assert np.isfinite(model.state.meteo.temp[row, col])
    assert np.isfinite(model.state.meteo.sw_in[row, col])
