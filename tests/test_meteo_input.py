from .conftest import base_config
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import openamundsen as oa
from openamundsen import errors, meteo as oameteo
import pandas as pd
from pathlib import Path
import pytest
import xarray as xr


def meteo_to_netcdf(ds, out_dir):
    ds = ds.copy(deep=True)
    td = ds.time[1] - ds.time[0]
    timestep = td.values / np.timedelta64(1, 's')

    ds = ds[[
        'station_name',
        'lon',
        'lat',
        'alt',
        'time',
        'temp',
        'precip',
        'rel_hum',
        'sw_in',
        'wind_speed',
        'wind_dir',
    ]].rename_vars({
        'temp': 'tas',
        'precip': 'pr',
        'rel_hum': 'hurs',
        'sw_in': 'rsds',
        'wind_speed': 'wss',
        'wind_dir': 'wind_dir',
    })

    ds['pr'] /= timestep
    ds['pr'].attrs = {
        'standard_name': 'precipitation_flux',
        'units': 'kg m-2 s-1',
    }

    for station_num in range(len(ds.station)):
        ds_cur = ds.isel(station=station_num).copy()
        station_id = str(ds_cur.station.values)
        station_name = str(ds_cur.station_name.values)
        ds_cur.attrs = {
            'Conventions': 'CF-1.6',
            'station_name': station_name,
        }
        ds_cur = ds_cur.drop_vars(['station', 'station_name'])
        ds_cur.to_netcdf(f'{out_dir}/{station_id}.nc')


def test_formats():
    config = base_config()
    config.start_date = '2015-07-28'
    config.end_date = '2020-12-31'
    config.timestep = 'H'

    meta = pd.read_csv(f'{pytest.DATA_DIR}/meteo/rofental/csv/stations.csv', index_col=0)
    meta = meta.sort_index()

    meteo = {}
    for fmt in ('netcdf', 'csv'):
        config.input_data.meteo.format = fmt
        config.input_data.meteo.dir = f'{pytest.DATA_DIR}/meteo/rofental/{fmt}'
        model = oa.OpenAmundsen(config)
        model.initialize()
        ds = model.meteo.sortby('station')
        assert ds.time.to_index().equals(model.dates)
        assert_array_equal(meta['name'], ds.station_name)
        assert_array_equal(meta['alt'], ds.alt)
        meteo[fmt] = ds

    xr.testing.assert_allclose(meteo['netcdf'], meteo['csv'])


def test_format_memory(tmp_path):
    config = base_config()
    model = oa.OpenAmundsen(config)
    model.initialize()
    meteo1 = model.meteo

    config.input_data.meteo.format = 'memory'
    model = oa.OpenAmundsen(config)
    model.initialize(meteo=oa.forcing.strip_point_dataset(meteo1))
    meteo2 = model.meteo
    assert meteo1.identical(meteo2)

    config.start_date = meteo1.indexes['time'][1]
    model = oa.OpenAmundsen(config)
    with pytest.raises(errors.MeteoDataError):
        model.initialize(meteo=oa.forcing.strip_point_dataset(meteo1))


def test_format_callback():
    def callback(model):
        model.state.meteo.temp[:] = meteo['temp'][model.date_idx]
        model.state.meteo.precip[:] = meteo['precip'][model.date_idx]
        model.state.meteo.wind_speed[:] = meteo['wind_speed'][model.date_idx]
        model.state.meteo.rel_hum[:] = meteo['rel_hum'][model.date_idx]
        model.state.meteo.cloud_factor[:] = meteo['cloud_factor'][model.date_idx]
        model.state.meteo.cloud_fraction[:] = meteo['cloud_fraction'][model.date_idx]

    config = base_config()
    config.start_date = '2019-12-01'
    config.end_date = '2019-12-15'

    model = oa.OpenAmundsen(config)
    model.initialize()

    for category in model.state.categories:
        for var_name in model.state[category]._meta.keys():
            full_var_name = f'{category}.{var_name}'
            config.output_data.grids.variables.append({
                'var': full_var_name,
                'name': full_var_name,
            })

    meteo = {
        'temp': [],
        'precip': [],
        'wind_speed': [],
        'rel_hum': [],
        'cloud_factor': [],
        'cloud_fraction': [],
    }

    model = oa.OpenAmundsen(config)
    model.initialize()
    for date in model.dates:
        model.run_single()
        meteo['temp'].append(model.state.meteo.temp.copy())
        meteo['precip'].append(model.state.meteo.precip.copy())
        meteo['wind_speed'].append(model.state.meteo.wind_speed.copy())
        meteo['rel_hum'].append(model.state.meteo.rel_hum.copy())
        meteo['cloud_factor'].append(model.state.meteo.cloud_factor.copy())
        meteo['cloud_fraction'].append(model.state.meteo.cloud_fraction.copy())
    data_ref = model.gridded_output.data.copy()

    config.input_data.meteo = {'format': 'callback'}
    model = oa.OpenAmundsen(config)
    with pytest.raises(errors.MeteoDataError):
        model.initialize()
    model.initialize(meteo_callback=callback)
    model.run()
    data_cb = model.gridded_output.data.copy()

    assert data_ref.identical(data_cb)


def test_no_files_found(tmp_path):
    config = base_config()
    config.input_data.meteo.dir = str(tmp_path)

    for fmt in ('netcdf', 'csv'):
        config.input_data.meteo.format = fmt
        model = oa.OpenAmundsen(config)
        with pytest.raises(errors.MeteoDataError):
            model.initialize()


def test_missing_csv_metadata_columns(tmp_path):
    config = base_config()
    config.input_data.meteo.format = 'csv'
    config.input_data.meteo.dir = str(tmp_path)

    p_orig = Path(f'{pytest.DATA_DIR}/meteo/rofental/csv')
    meta = pd.read_csv(p_orig / 'stations.csv', index_col=0)
    meta[['name', 'x', 'y']].loc[['bellavista']].to_csv(tmp_path / 'stations.csv')
    (tmp_path / 'bellavista.csv').symlink_to(p_orig / 'bellavista.csv')

    model = oa.OpenAmundsen(config)
    with pytest.raises(errors.MeteoDataError):
        model.initialize()


def test_missing_records_inbetween(tmp_path):
    config = base_config()
    config.end_date = '2020-04-30'
    config.timestep = 'H'
    config.input_data.meteo.format = 'csv'
    config.input_data.meteo.dir = str(tmp_path)

    p_orig = Path(f'{pytest.DATA_DIR}/meteo/rofental/csv')

    meta = pd.read_csv(p_orig / 'stations.csv', index_col=0)
    meta.loc[['bellavista']].to_csv(tmp_path / 'stations.csv')

    df = pd.read_csv(p_orig / 'bellavista.csv', index_col=0, parse_dates=True)
    df = df.loc[config.start_date:config.end_date]
    df = df.drop(df.index[37:38])
    df.to_csv(tmp_path / 'bellavista.csv')

    model = oa.OpenAmundsen(config)
    with pytest.raises(errors.MeteoDataError):
        model.initialize()


def test_missing_records_start_end():
    config = base_config()
    config.start_date = '1900-01-01'
    config.end_date = '2100-12-31'

    model = oa.OpenAmundsen(config)
    with pytest.raises(errors.MeteoDataError):
        model.initialize()


def test_station_selection(tmp_path):
    config = base_config()
    model = oa.OpenAmundsen(config)
    model.initialize()

    dummy_station = model.meteo.isel(station=0).copy()
    dummy_station['station'] = 'dummy'
    dummy_station['station_name'] = 'Dummy station'
    dummy_station['lon'] = 11.7
    dummy_station['lat'] = 47.21
    ds = xr.concat([model.meteo, dummy_station], dim='station')
    meteo_to_netcdf(ds, tmp_path)

    config.input_data.meteo.dir = str(tmp_path)
    config.input_data.meteo.bounds = 'grid'
    model = oa.OpenAmundsen(config)
    model.initialize()
    assert_array_equal(
        sorted(model.meteo.station.values),
        ['bellavista', 'latschbloder', 'proviantdepot'],
    )

    config.input_data.meteo.bounds = 'global'
    model = oa.OpenAmundsen(config)
    model.initialize()
    assert_array_equal(
        sorted(model.meteo.station.values),
        ['bellavista', 'dummy', 'latschbloder', 'proviantdepot'],
    )

    config.input_data.meteo.bounds = [
        636800,
        5182550,
        636850,
        5182580,
    ]
    model = oa.OpenAmundsen(config)
    model.initialize()
    assert_array_equal(
        sorted(model.meteo.station.values),
        ['bellavista'],
    )

    config.input_data.meteo.bounds = [
        636800,
        5182550,
        636801,
        5182551,
    ]
    model = oa.OpenAmundsen(config)
    with pytest.raises(errors.MeteoDataError):
        model.initialize()

    config.input_data.meteo.bounds = 'global'
    config.input_data.meteo.exclude = ['bellavista', 'latschbloder']
    model = oa.OpenAmundsen(config)
    model.initialize()
    assert_array_equal(
        sorted(model.meteo.station.values),
        ['dummy', 'proviantdepot'],
    )

    config.input_data.meteo.bounds = 'grid'
    config.input_data.meteo.exclude = ['bellavista', 'latschbloder', 'proviantdepot']
    model = oa.OpenAmundsen(config)
    with pytest.raises(errors.MeteoDataError):
        model.initialize()

    config.input_data.meteo.bounds = 'grid'
    config.input_data.meteo.exclude = []
    config.input_data.meteo.include = ['dummy']
    model = oa.OpenAmundsen(config)
    model.initialize()
    assert_array_equal(
        sorted(model.meteo.station.values),
        ['bellavista', 'dummy', 'latschbloder', 'proviantdepot'],
    )

    config.input_data.meteo.include = []
    config.input_data.meteo.exclude = ['dummy', 'dummy2']
    model = oa.OpenAmundsen(config)
    model.initialize()

    config.input_data.meteo.include = ['dummy', 'dummy2']
    config.input_data.meteo.exclude = []
    model = oa.OpenAmundsen(config)
    with pytest.raises(errors.MeteoDataError):
        model.initialize()


def test_missing_variables(tmp_path):
    ds = xr.load_dataset(f'{pytest.DATA_DIR}/meteo/rofental/netcdf/proviantdepot.nc')
    ds = ds.drop_vars(['tas', 'hurs', 'rsds', 'wss', 'wind_dir'])
    ds.to_netcdf(tmp_path / 'proviantdepot.nc')

    config = base_config()
    config.input_data.meteo.dir = str(tmp_path)
    model = oa.OpenAmundsen(config)
    model.initialize()

    assert 'temp' in model.meteo
    assert 'precip' in model.meteo
    assert 'rel_hum' in model.meteo
    assert 'sw_in' in model.meteo
    assert 'wind_speed' in model.meteo
    assert 'wind_dir' not in model.meteo


def test_netcdf_precip_units(tmp_path):
    config = base_config()
    config.input_data.meteo.dir = str(tmp_path)

    ds = xr.load_dataset(f'{pytest.DATA_DIR}/meteo/rofental/netcdf/proviantdepot.nc')
    ds.pr.attrs['units'] = 'kg m-2 s-1'
    ds.to_netcdf(tmp_path / 'proviantdepot.nc')
    model = oa.OpenAmundsen(config)
    model.initialize()
    meteo1 = model.meteo.copy()

    ds['pr'] *= 3600
    ds.pr.attrs['units'] = 'kg m-2'
    ds.to_netcdf(tmp_path / 'proviantdepot.nc')
    model = oa.OpenAmundsen(config)
    model.initialize()
    meteo2 = model.meteo.copy()

    xr.testing.assert_allclose(meteo1.precip, meteo2.precip)

    ds.pr.attrs['units'] = 'mm h-1'
    ds.to_netcdf(tmp_path / 'proviantdepot.nc')
    model = oa.OpenAmundsen(config)
    with pytest.raises(errors.MeteoDataError):
        model.initialize()


def test_crs(tmp_path):
    config = base_config()

    model = oa.OpenAmundsen(config)
    model.initialize()
    ds1 = model.meteo[['lon', 'lat', 'x', 'y']].sortby('station')

    p_orig = Path(f'{pytest.DATA_DIR}/meteo/rofental/csv')
    meta = pd.read_csv(p_orig / 'stations.csv', index_col=0)
    x_new, y_new = oa.util.transform_coords(meta.x, meta.y, 'epsg:32632', 'epsg:3416')
    meta.x = x_new
    meta.y = y_new
    meta.to_csv(tmp_path / 'stations.csv')

    for station_id in meta.index:
        (tmp_path / f'{station_id}.csv').symlink_to(p_orig / f'{station_id}.csv')

    config.input_data.meteo.format = 'csv'
    config.input_data.meteo.dir = str(tmp_path)
    config.input_data.meteo.crs = 'epsg:3416'

    model = oa.OpenAmundsen(config)
    model.initialize()
    ds2 = model.meteo[['lon', 'lat', 'x', 'y']].sortby('station')

    xr.testing.assert_allclose(ds1, ds2)


def test_grid_cell_assignment(tmp_path):
    config = base_config()
    config.start_date = '2020-01-01'
    config.end_date = '2020-01-01'

    model = oa.OpenAmundsen(config)
    model.initialize()
    grid_xs = model.grid.xs
    grid_ys = model.grid.ys

    config.input_data.meteo.format = 'csv'
    config.input_data.meteo.dir = str(tmp_path)

    station_coords_colrows = [  # (x, y, col, row)
        (grid_xs[0], grid_ys[0], 0, 0),
        (grid_xs[0] + 1, grid_ys[0] - 1, 0, 0),
        (grid_xs[1] + 1, grid_ys[0] - 1, 1, 0),
        (grid_xs[1] + 1, grid_ys[3] - 1, 1, 3),
        (grid_xs[-1] + 1, grid_ys[-1] - 1, len(grid_xs) - 1, len(grid_ys) - 1),
        (grid_xs[0], grid_ys[-1], 0, len(grid_ys) - 1),
    ]

    meta = pd.DataFrame(
        index=[f'dummy{i}' for i in range(len(station_coords_colrows))],
        data={
            'name': None,
            'x': [s[0] for s in station_coords_colrows],
            'y': [s[1] for s in station_coords_colrows],
            'alt': 0.,
        },
    )
    meta.to_csv(tmp_path / 'stations.csv')

    p_orig = Path(f'{pytest.DATA_DIR}/meteo/rofental/csv')
    for station_id in meta.index:
        (tmp_path / f'{station_id}.csv').symlink_to(p_orig / 'proviantdepot.csv')

    model = oa.OpenAmundsen(config)
    model.initialize()
    ds = model.meteo

    for i, s in enumerate(station_coords_colrows):
        ds_s = ds.sel(station=f'dummy{i}')
        expected_col = s[2]
        expected_row = s[3]
        assert int(ds_s.col) == expected_col
        assert int(ds_s.row) == expected_row


@pytest.mark.parametrize('fmt', ['netcdf', 'csv'])
def test_slice_and_resample(fmt, tmp_path):
    def meteo_to_df(model):
        return (
            model.meteo
            .sel(station='bellavista')
            .to_dataframe()
            [params]
        ).astype(float)

    def matches_start_and_end_date(model):
        idx = model.meteo.indexes['time']
        return (
            idx[0] == model.config.start_date
            and idx[-1] == model.config.end_date
        )

    params = ['temp', 'precip', 'rel_hum', 'sw_in', 'wind_speed']

    agg_funcs_inst = {p: lambda s: s.iloc[-1] for p in params}
    agg_funcs_inst['precip'] = pd.Series.sum
    agg_funcs_res = {p: pd.Series.mean for p in params}
    agg_funcs_res['precip'] = pd.Series.sum

    config = base_config()
    config.start_date = '2015-11-20'
    config.end_date = '2015-11-30'
    config.timestep = 'H'
    config.input_data.meteo.format = fmt
    config.input_data.meteo.dir = f'{pytest.DATA_DIR}/meteo/rofental/{fmt}'
    model = oa.OpenAmundsen(config)
    model.initialize()
    df_h = meteo_to_df(model)

    config.timestep = '3H'
    model = oa.OpenAmundsen(config)
    model.initialize()
    assert matches_start_and_end_date(model)
    df_res = meteo_to_df(model)
    pd.testing.assert_series_equal(
        df_h.loc['2015-11-30 13:00':'2015-11-30 15:00'].agg(agg_funcs_inst, skipna=False),
        df_res.loc['2015-11-30 15:00'],
        check_exact=False,
        check_names=False,
    )
    assert np.isnan(df_res.loc['2015-11-26 15:00'].precip)
    assert_allclose(
        df_h.loc[:df_res.index[-1]].precip.sum(),
        df_res.precip.sum(),
    )

    config.input_data.meteo.aggregate_when_downsampling = True
    model = oa.OpenAmundsen(config)
    model.initialize()
    assert matches_start_and_end_date(model)
    df_res = meteo_to_df(model)
    pd.testing.assert_series_equal(
        df_h.loc['2015-11-30 13:00':'2015-11-30 15:00'].agg(agg_funcs_res, skipna=False),
        df_res.loc['2015-11-30 15:00'],
        check_exact=False,
        check_names=False,
    )
    assert np.isnan(df_res.loc['2015-11-26 15:00'].precip)

    config.start_date = '2015-11-25'
    config.end_date = '2015-11-29'
    config.timestep = 'D'
    config.input_data.meteo.aggregate_when_downsampling = True
    model = oa.OpenAmundsen(config)
    model.initialize()
    assert matches_start_and_end_date(model)
    df_res = meteo_to_df(model)
    pd.testing.assert_series_equal(
        df_h.loc['2015-11-25 01:00':'2015-11-26 00:00'].agg(agg_funcs_res, skipna=False),
        df_res.loc['2015-11-25'],
        check_exact=False,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        df_h.loc['2015-11-29 01:00':'2015-11-30 00:00'].agg(agg_funcs_res, skipna=False),
        df_res.loc['2015-11-29'],
        check_exact=False,
        check_names=False,
    )

    config.start_date = '2015-11-01'
    config.end_date = '2015-11-09'
    config.timestep = '2D'
    wf = config.output_data.timeseries.write_freq
    config.output_data.timeseries.write_freq = '10D'
    model = oa.OpenAmundsen(config)
    config.output_data.timeseries.write_freq = wf
    with pytest.raises(errors.MeteoDataError):
        model.initialize()

    config.start_date = '2015-11-01 01:00'
    config.end_date = '2015-11-30 22:00'
    config.timestep = '3H'
    config.input_data.meteo.aggregate_when_downsampling = False
    model = oa.OpenAmundsen(config)
    model.initialize()
    assert matches_start_and_end_date(model)
    df_res = meteo_to_df(model)
    pd.testing.assert_series_equal(
        df_h.loc['2015-11-30 14:00':'2015-11-30 16:00'].agg(agg_funcs_inst, skipna=False),
        df_res.loc['2015-11-30 16:00'],
        check_exact=False,
        check_names=False,
    )

    config.input_data.meteo.aggregate_when_downsampling = True
    model = oa.OpenAmundsen(config)
    model.initialize()
    assert matches_start_and_end_date(model)
    df_res = meteo_to_df(model)
    pd.testing.assert_series_equal(
        df_h.loc['2015-11-30 14:00':'2015-11-30 16:00'].agg(agg_funcs_res, skipna=False),
        df_res.loc['2015-11-30 16:00'],
        check_exact=False,
        check_names=False,
    )

    config.start_date = '2015-11-01 01:37'
    config.end_date = '2015-11-30 22:37'
    config.timestep = '3H'
    config.input_data.meteo.aggregate_when_downsampling = False
    model = oa.OpenAmundsen(config)
    with pytest.raises(errors.MeteoDataError):
        model.initialize()

    config.start_date = '2015-11-29'
    config.end_date = '2015-11-30'
    config.timestep = '10min'
    model = oa.OpenAmundsen(config)
    with pytest.raises(errors.MeteoDataError):
        model.initialize()

    # Wind direction
    config.start_date = '2015-12-20'
    config.end_date = '2015-12-30'
    config.timestep = 'H'
    config.input_data.meteo.aggregate_when_downsampling = False
    model = oa.OpenAmundsen(config)
    model.initialize()
    df_h = model.meteo.sel(station='bellavista').to_dataframe()[['wind_speed', 'wind_dir']]

    config.timestep = '3H'
    model = oa.OpenAmundsen(config)
    model.initialize()
    df_res = model.meteo.sel(station='bellavista').to_dataframe()[['wind_speed', 'wind_dir']]
    assert np.allclose(
        df_h['wind_dir'].loc['2015-12-20 15:00'],
        df_res['wind_dir'].loc['2015-12-20 15:00'],
    )

    config.input_data.meteo.aggregate_when_downsampling = True
    model = oa.OpenAmundsen(config)
    model.initialize()
    df_res = model.meteo.sel(station='bellavista').to_dataframe()[['wind_speed', 'wind_dir']]
    wind_us, wind_vs = oameteo.wind_to_uv(
        df_h['wind_speed'].loc['2015-12-20 13:00':'2015-12-20 15:00'],
        df_h['wind_dir'].loc['2015-12-20 13:00':'2015-12-20 15:00'],
    )
    wind_u_mean = wind_us.mean()
    wind_v_mean = wind_vs.mean()
    _, wind_dir = oameteo.wind_from_uv(wind_u_mean, wind_v_mean)
    assert np.allclose(
        df_res['wind_dir'].loc['2015-12-20 15:00'],
        wind_dir,
    )


def test_non_hourly_input(tmp_path):
    config = base_config()
    config.start_date = '2015-11-20'
    config.end_date = '2015-11-30'
    config.timestep = '3H'

    model = oa.OpenAmundsen(config)
    model.initialize()
    meteo_to_netcdf(model.meteo, tmp_path)
    ds1 = model.meteo

    config.input_data.meteo.dir = str(tmp_path)
    model = oa.OpenAmundsen(config)
    model.initialize()
    ds2 = model.meteo

    xr.testing.assert_allclose(ds1, ds2)


@pytest.mark.parametrize('aggregate', [False, True])
def test_resample_with_non_matching_start_date(aggregate, tmp_path):
    ds = xr.load_dataset(f'{pytest.DATA_DIR}/meteo/rofental/netcdf/proviantdepot.nc')
    ds = ds.sel(time=slice('2020-11-03 02:00', None))
    ds.to_netcdf(tmp_path / 'proviantdepot.nc')

    p_orig = Path(f'{pytest.DATA_DIR}/meteo/rofental/netcdf')
    for station_id in ('bellavista', 'latschbloder'):
        (tmp_path / f'{station_id}.nc').symlink_to(p_orig / f'{station_id}.nc')

    config = base_config()
    config.start_date = '2020-11-01'
    config.end_date = '2020-11-30'
    config.timestep = '3H'
    config.input_data.meteo.dir = str(tmp_path)
    config.input_data.meteo.aggregate_when_downsampling = aggregate

    model = oa.OpenAmundsen(config)
    model.initialize()
    assert model.meteo.indexes['time'].equals(model.dates)
