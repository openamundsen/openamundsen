import numpy as np
import openamundsen as oa
from openamundsen import constants
from pathlib import Path
import xarray as xr


def read_netcdf_meteo_file(filename):
    """
    Read a meteo data file in NetCDF format and
    - check if the time, lon, lat, and alt variables are included
    - rename the variables according to NETCDF_VAR_MAPPINGS
    - check if the units are as expected (METEO_VAR_METADATA)
    - remove all unsupported variables
    - set the station id and name as attributes.
    """
    ds = xr.load_dataset(filename)

    # rename variables
    ds_vars = list(ds.variables.keys())
    rename_vars = set(constants.NETCDF_VAR_MAPPINGS.keys()) & set(ds_vars)
    rename_dict = {v: constants.NETCDF_VAR_MAPPINGS[v] for v in rename_vars}
    ds = ds.rename_vars(rename_dict)

    for var in ('time', 'lon', 'lat', 'alt'):
        if var not in ds:
            raise oa.errors.MeteoDataError(f'File is missing "{var}" variable: {filename}')

    for var, meta in constants.METEO_VAR_METADATA.items():
        units = meta['units']

        if var in ds and ds[var].units != units:
            raise oa.errors.MeteoDataError(
                f'{var} has wrong units in {filename}: expected {units}, got {ds[var].units}'
            )

    # remove all unsupported variables
    ds_vars = list(ds.variables.keys())
    allowed_vars = ['time', 'lon', 'lat', 'alt'] + list(constants.NETCDF_VAR_MAPPINGS.values())
    drop_vars = list(set(ds_vars) - set(allowed_vars))
    ds = ds.drop_vars(drop_vars)

    station_id = Path(filename).stem

    # set station name to station id (filename without extension) if not present
    if 'station_name' in ds.attrs:
        station_name = ds.attrs['station_name']
    else:
        station_name = station_id

    # remove all attributes except id and name
    ds.attrs = {
        'station_id': station_id,
        'station_name': station_name,
    }

    return ds


def combine_meteo_datasets(datasets):
    """
    Combine a list of meteo datasets as read by read_netcdf_meteo_file.
    The datasets are merged by adding an additional "station" (= station id)
    dimension.
    """
    datasets_processed = []

    for ds in datasets:
        ds = ds.copy()

        for var, meta in constants.METEO_VAR_METADATA.items():
            # add missing variables to the dataset
            if var not in ds:
                ds[var] = xr.DataArray(
                    data=np.full(ds.time.shape, np.nan),
                    coords=ds.coords,
                    dims=('time',),
                )

            ds[var].attrs = {
                'standard_name': meta['standard_name'],
                'units': meta['units'],
            }

        ds.coords['station'] = ds.attrs['station_id']
        ds['station_name'] = ds.attrs['station_name']
        ds.attrs = {}

        datasets_processed.append(ds)

    return xr.combine_nested(datasets_processed, concat_dim='station')
