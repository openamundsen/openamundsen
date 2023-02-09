from pathlib import Path

import numpy as np
import openamundsen as oa
import openamundsen.errors as errors
import pytest
import rasterio
import rasterio.windows
import xarray as xr
from numpy.testing import assert_allclose, assert_equal

from .conftest import base_config


def base_cloudiness_config():
    config = base_config()
    config.start_date = '2019-10-05'
    config.end_date = '2019-10-10'
    config.input_data.meteo.bounds = 'global'
    return config


def test_clear_sky_fraction():
    nan_pos = slice('2019-10-06 00:00', '2019-10-07 00:00')

    config = base_cloudiness_config()
    config.meteo.interpolation.cloudiness.method = 'clear_sky_fraction'
    config.meteo.interpolation.cloudiness.allow_fallback = False
    model = oa.OpenAmundsen(config)
    model.initialize()

    model.meteo.sw_in.loc['proviantdepot', nan_pos] = np.nan
    model.run()
    ds = model.point_output.data
    assert np.all(ds.sw_in.notnull())

    model = oa.OpenAmundsen(config)
    model.initialize()
    model.meteo.sw_in.loc[:, nan_pos] = np.nan
    model.run()
    ds = model.point_output.data
    # When allow_fallback=False, values are nan during daytime
    assert np.all(ds.sw_in.sel(time=slice('2019-10-06 12:00', '2019-10-06 15:00')).isnull())
    assert np.all(ds.sw_in.sel(time=slice('2019-10-06 00:00', '2019-10-06 03:00')).notnull())

    config.meteo.interpolation.cloudiness.allow_fallback = True
    model = oa.OpenAmundsen(config)
    model.initialize()
    model.meteo.sw_in.loc[:, nan_pos] = np.nan
    model.run()
    ds = model.point_output.data
    assert np.all(ds.sw_in.notnull())
    # Daytime values should be calculated using the "humidity" method here (not constant)
    assert not np.array_equal(
        ds.cloud_factor.loc['2019-10-06 12:00'].values,
        ds.cloud_factor.loc['2019-10-06 15:00'].values,
    )

    config.meteo.interpolation.cloudiness.clear_sky_fraction_night_method = 'humidity'
    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run()
    ds = model.point_output.data
    # Nighttime values should not be constant here
    assert not np.array_equal(
        ds.cloud_factor.loc['2019-10-06 00:00'].values,
        ds.cloud_factor.loc['2019-10-06 03:00'].values,
    )

    config.meteo.interpolation.cloudiness.clear_sky_fraction_night_method = 'constant'
    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run()
    ds = model.point_output.data
    # Nighttime values should be constant here
    assert_equal(
        ds.cloud_factor.loc['2019-10-06 00:00'].values,
        ds.cloud_factor.loc['2019-10-06 03:00'].values,
    )


def test_humidity():
    nan_pos = slice('2019-10-06 00:00', '2019-10-07 00:00')

    config = base_cloudiness_config()
    config.meteo.interpolation.cloudiness.method = 'humidity'
    config.meteo.interpolation.cloudiness.allow_fallback = False
    model = oa.OpenAmundsen(config)
    model.initialize()

    model.meteo.rel_hum.loc['proviantdepot', nan_pos] = np.nan
    model.run()
    ds = model.point_output.data
    assert np.all(ds.sw_in.notnull())

    model = oa.OpenAmundsen(config)
    model.initialize()
    model.meteo.rel_hum.loc[:, nan_pos] = np.nan
    model.run()
    ds = model.point_output.data
    assert np.all(ds.sw_in.sel(time=nan_pos).isnull())
    assert np.all(ds.sw_in.drop_sel(time=ds.sel(time=nan_pos).time).notnull())


def test_prescribed(tmp_path):
    config = base_cloudiness_config()
    config.meteo.interpolation.cloudiness.method = 'clear_sky_fraction'
    config.meteo.interpolation.cloudiness.allow_fallback = False
    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run()
    ds_ref = model.point_output.data

    for station_id in ds_ref.indexes['point']:
        ds = xr.load_dataset(f'{config.input_data.meteo.dir}/{station_id}.nc')
        ds = ds.sel(time=ds_ref.time)
        ds['cloud_cover'] = xr.DataArray(
            ds_ref.cloud_factor.sel(point=station_id).values * 100,
            coords=ds.coords,
            dims=ds.dims,
        )
        ds = ds.drop_vars('sw_in', errors='ignore')
        ds.to_netcdf(f'{tmp_path}/{station_id}.nc')

    config.meteo.interpolation.cloudiness.method = 'prescribed'
    model = oa.OpenAmundsen(config)
    with pytest.raises(errors.MeteoDataError):
        model.initialize()

    config.input_data.meteo.dir = str(tmp_path)
    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run()
    ds = model.point_output.data
    assert_allclose(ds_ref.cloud_factor, ds.cloud_factor, atol=0.05)
    # (differences are (likely) due to interpolation from station positions to grid points)


def test_outside_grid_data(tmp_path):
    config = base_cloudiness_config()
    config.meteo.interpolation.cloudiness.allow_fallback = False
    config.output_data.timeseries.points.append({
        'x': 638483,
        'y': 5190972,
    })
    config.output_data.timeseries.add_default_points = False

    config.meteo.interpolation.cloudiness.method = 'clear_sky_fraction'
    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run()
    ds_ref_csf = model.point_output.data

    config.meteo.interpolation.cloudiness.method = 'humidity'
    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run()
    ds_ref_hum = model.point_output.data

    src_dir = Path(config.input_data.grids.dir)
    dst_dir = tmp_path
    window = rasterio.windows.Window(col_off=5, row_off=2, width=5, height=4)
    src_files = src_dir.glob('*.asc')
    for src_file in src_files:
        with rasterio.open(src_file) as src_ds:
            kwargs = src_ds.meta.copy()
            kwargs.update({
                'height': window.height,
                'width': window.width,
                'transform': rasterio.windows.transform(window, src_ds.transform),
            })

            with rasterio.open(dst_dir / src_file.name, 'w', **kwargs) as dst_ds:
                dst_ds.write(src_ds.read(window=window))

    config.input_data.grids.dir = str(dst_dir)

    config.meteo.interpolation.cloudiness.method = 'clear_sky_fraction'
    model = oa.OpenAmundsen(config)
    model.initialize()
    assert not np.any(model.meteo.within_grid_extent)
    model.run()
    ds = model.point_output.data
    # Daytime values must be nan
    assert np.all(ds.sw_in.sel(time=ds.indexes['time'].hour.isin([9, 12, 15])).isnull())

    config.meteo.interpolation.cloudiness.method = 'humidity'
    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run()
    ds = model.point_output.data
    assert np.all(ds.sw_in.notnull())
    assert_allclose(ds_ref_hum.sw_in, ds.sw_in, atol=25)

    # Now enable the extended grids
    grid_file_suffix = f'{config.domain}_{config.resolution}.asc'
    (tmp_path / f'extended-dem_{grid_file_suffix}').symlink_to(src_dir / f'dem_{grid_file_suffix}')
    (tmp_path / f'extended-svf_{grid_file_suffix}').symlink_to(src_dir / f'svf_{grid_file_suffix}')

    config.meteo.interpolation.cloudiness.method = 'clear_sky_fraction'
    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run()
    ds = model.point_output.data
    assert np.all(ds.sw_in.notnull())
    assert_allclose(ds_ref_csf.sw_in, ds.sw_in, rtol=0.03)

    config.meteo.interpolation.cloudiness.method = 'humidity'
    model = oa.OpenAmundsen(config)
    model.initialize()
    model.run()
    ds = model.point_output.data
    assert np.all(ds.sw_in.notnull())
    assert_allclose(ds_ref_hum.sw_in, ds.sw_in, rtol=0.04)
