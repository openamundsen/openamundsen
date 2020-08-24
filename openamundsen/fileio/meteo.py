import loguru
import numpy as np
import openamundsen as oa
from openamundsen import constants, errors, util
import pandas as pd
from pathlib import Path
import xarray as xr


def read_meteo_data_netcdf(meteo_data_dir, start_date, end_date, freq='H', logger=None):
    """
    Read all available stations in NetCDF format for a given period.

    Parameters
    ----------
    meteo_data_dir : path-like
        Location of the NetCDF files.

    start_date : datetime-like
        Start date.

    end_date : datetime-like
        End date.

    freq : str
        Pandas-compatible frequency string (e.g. '3H') to which the data should
        be resampled.

    logger : logger, default None
        Logger to use for status messages.

    Returns
    -------
    ds : Dataset
        Station data.
    """
    if logger is None:
        logger = loguru.logger

    meteo_data_dir = Path(meteo_data_dir)
    nc_files = sorted(list(meteo_data_dir.glob('*.nc')))

    if len(nc_files) == 0:
        raise errors.MeteoDataError('No meteo files found')

    datasets = []

    for nc_file in nc_files:
        logger.info(f'Reading meteo file: {nc_file}')

        ds = read_netcdf_meteo_file(nc_file)
        ds = ds.sel(time=slice(start_date, end_date))
        ds = _resample_dataset(ds, freq)

        if ds.dims['time'] == 0:
            logger.warning('File contains no meteo data for the specified period')
        else:
            datasets.append(ds)

    if len(datasets) == 0:
        raise errors.MeteoDataError('No meteo data available for the specified period')

    return combine_meteo_datasets(datasets)


def read_meteo_data_csv(meteo_data_dir, start_date, end_date, crs, freq='H', logger=None):
    """
    Read all available stations in CSV format for a given period.
    This function expects a file named `stations.csv` in the specified
    directory containing the metadata (IDs, names, x/y coordinates, altitudes)
    of the stations. Station files must be in the same directory and must be
    named `<station_id>.csv`

    Parameters
    ----------
    meteo_data_dir : path-like
        Location of the NetCDF files.

    start_date : datetime-like
        Start date.

    end_date : datetime-like
        End date.

    crs : str
        CRS of the station coordinates specified in the stations.csv file.

    freq : str
        Pandas-compatible frequency string (e.g. '3H') to which the data should
        be resampled.

    logger : logger, default None
        Logger to use for status messages.

    Returns
    -------
    ds : Dataset
        Station data.
    """
    if logger is None:
        logger = loguru.logger

    meta = pd.read_csv(f'{meteo_data_dir}/stations.csv', index_col=0)

    datasets = []

    for station_id, row in meta.iterrows():
        filename = f'{meteo_data_dir}/{station_id}.csv'
        logger.info(f'Reading meteo file: {filename}')
        ds = read_csv_meteo_file(
            filename,
            station_id,
            row['name'],
            row['x'],
            row['y'],
            row['alt'],
            crs,
        )

        ds = ds.sel(time=slice(start_date, end_date))
        ds = _resample_dataset(ds, freq)

        if ds.dims['time'] == 0:
            logger.warning('File contains no meteo data for the specified period')
        else:
            datasets.append(ds)

    if len(datasets) == 0:
        raise errors.MeteoDataError('No meteo data available for the specified period')

    return combine_meteo_datasets(datasets)


def read_netcdf_meteo_file(filename):
    """
    Read a meteo data file in NetCDF format and
    - check if the time, lon, lat, and alt variables are included
    - rename the variables according to NETCDF_VAR_MAPPINGS
    - check if the units are as expected (METEO_VAR_METADATA)
    - remove all unsupported variables
    - set the station id and name as attributes.

    Parameters
    ----------
    filename : path-like
        CSV filename.

    Returns
    -------
    ds : Dataset
        Station data.
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


def read_csv_meteo_file(filename, station_id, station_name, x, y, alt, crs):
    """
    Read a meteorological data file in CSV format and return it as a Dataset.
    Unlike in read_netcdf_meteo_file, here it is assumed that precipitation is
    specified as a sum over the time step (i.e., kg m-2) instead of a flux (kg
    m-2 s-1).

    Parameters
    ----------
    filename : path-like
        CSV filename.

    station_id : str
        Station ID (must be unique).

    station_name : str
        Station name.

    x : float
        Station x coordinate.

    y : float
        Station y coordinate.

    alt : float
        Station altitude.

    crs : str
        CRS of the x/y coordinates.

    Returns
    -------
    ds : Dataset
        Station data.
    """
    param_mappings = {
        'temp': 'temp',
        'precip': 'precip',
        'rel_hum': 'rel_hum',
        'sw_in': 'sw_in',
        'wind_speed': 'wind_speed',
    }

    lon, lat = util.transform_coords(x, y, crs, constants.CRS_WGS84)

    df = pd.read_csv(filename, parse_dates=True, index_col=0)
    df = df.rename(columns=param_mappings)

    if 'precip' in df.columns:
        df.precip /= constants.SECONDS_PER_HOUR  # convert to kg m-2 s-1
        # TODO check if the time step is really 3600 s and if precip is really a sum

    datadict = {param: (['time'], df[param].astype(np.float32)) for param in df.columns}
    datadict.update(
        time=(['time'], df.index),
        lon=([], lon),
        lat=([], lat),
        alt=([], alt),
    )
    ds = xr.Dataset(datadict)

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


def _resample_dataset(ds, freq):
    """
    Resample a dataset to a given frequency.

    For all parameters except precipitation, the instantaneous values at the
    target timestamps are taken, i.e.  no aggregation is performed. For
    precipitation flux, the mean over the respective time intervals is
    calculated.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset.

    freq : str
        Pandas-compatible frequency string (e.g. '3H'). Must be an exact subset
        of the original frequency of the data.

    Returns
    -------
    ds_res : Dataset
        Resampled dataset.
    """
    # return ds
    # ds.resample() is somehow extremely slow, so we resample using pandas
    df = ds.to_dataframe().drop(columns=['lon', 'lat', 'alt'])

    # Take the instantaneous values for all parameters except precipitation flux
    # (where we calculate the mean)
    df_res = df.asfreq(freq)

    if 'precip' in df:
        df_res['precip'] = df['precip'].resample(freq, label='right', closed='right').mean()

    # Check if the desired frequency is a subset of the original frequency of the
    # data (e.g., resampling hourly to 3-hourly data is ok, but not hourly to
    # 1.5-hourly)
    if not df.index.intersection(df_res.index).equals(df_res.index):
        raise errors.MeteoDataError(f'Resampling from freq "{df.index.inferred_freq}" '
                                    f'to "{freq}" not supported')

    # Create a new dataset with the resampled time series
    ds_res = ds[['lon', 'lat', 'alt', 'time']]
    ds_res['time'] = df_res.index
    ds_res.attrs = ds.attrs

    for param in df_res.columns:
        ds_res[param] = df_res[param]

    return ds_res
