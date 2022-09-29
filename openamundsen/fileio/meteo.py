import loguru
from openamundsen import constants, errors, forcing, meteo as oameteo, util
import pandas as pd
from pathlib import Path
import xarray as xr


def read_meteo_data(
    meteo_format,
    meteo_data_dir,
    start_date,
    end_date,
    meteo_crs=None,
    grid_crs=None,
    bounds=None,
    exclude=None,
    include=None,
    freq='H',
    aggregate=False,
    logger=None,
):
    """
    Read all available stations in NetCDF or CSV format for a given period.

    In case of NetCDF input, all available .nc files in `meteo_data_dir` are
    used.
    For CSV input a file named `stations.csv` must be provided in
    the specified directory containing the metadata (IDs, names, x/y
    coordinates, altitudes) of the stations. Station files must be in the same
    directory and must be named `<station_id>.csv`

    Parameters
    ----------
    meteo_format : str
        Data format (either 'netcdf' or 'csv').

    meteo_data_dir : path-like
        Location of the NetCDF files.

    start_date : datetime-like
        Start date.

    end_date : datetime-like
        End date.

    meteo_crs : str
        CRS of the station coordinates specified in the stations.csv file
        (required only when `meteo_format` is 'csv').

    grid_crs : str
        CRS of the model grid.

    bounds : list or None
        If specified, use only stations within the region specified as a list of
        (x_min, y_min, x_max, y_max) coordinates in the model grid CRS.

    exclude: list or None
        List of station IDs to exclude.

    include: list or None
        List of station IDs to include even if otherwise excluded via `bounds`
        or `exclude`.

    freq : str
        Pandas-compatible frequency string (e.g. '3H') to which the data should
        be resampled.

    aggregate : boolean, default False
        Aggregate data when downsampling to a lower frequency or take
        instantaneous values.

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

    meta = _read_meteo_metadata(meteo_format, meteo_data_dir, meteo_crs, grid_crs)
    if meta.empty:
        raise errors.MeteoDataError('No stations found')

    meta = _apply_station_rules(meta, bounds, exclude, include)
    if meta.empty:
        raise errors.MeteoDataError('No stations available after applying station rules')

    datasets = []
    for station_id in meta.index:
        if meteo_format == 'netcdf':
            filename = meteo_data_dir / f'{station_id}.nc'
            logger.info(f'Reading meteo file: {filename}')
            ds = read_netcdf_meteo_file(filename)
        elif meteo_format == 'csv':
            filename = meteo_data_dir / f'{station_id}.csv'
            logger.info(f'Reading meteo file: {filename}')
            ds = read_csv_meteo_file(
                filename,
                station_id,
                meta.loc[station_id, 'name'],
                meta.loc[station_id, 'x'],
                meta.loc[station_id, 'y'],
                meta.loc[station_id, 'alt'],
                grid_crs,
            )

        ds = _slice_and_resample_dataset(
            ds,
            start_date,
            end_date,
            freq,
            aggregate=aggregate,
        )

        if ds.dims['time'] == 0:
            logger.warning('File contains no meteo data for the specified period')
        else:
            datasets.append(ds)

    if len(datasets) == 0:
        raise errors.MeteoDataError('No meteo data available for the specified period')

    ds_combined = forcing.combine_point_datasets(datasets)
    dates_combined = ds_combined.time.to_index()
    if dates_combined[0] > start_date or dates_combined[-1] < end_date:
        raise errors.MeteoDataError('Insufficient meteo data available.\n'
                                    f'Requested period: {start_date}..{end_date}\n'
                                    f'Available period: {dates_combined[0]}..{dates_combined[-1]}')
    
    return ds_combined


def read_netcdf_meteo_file(filename):
    """
    Read a meteo data file in NetCDF format and
    - check if the time, lon, lat, and alt variables are included
    - rename the variables according to NETCDF_VAR_MAPPINGS
    - convert precipitation rates to sums if necessary
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

    ds = forcing.make_point_dataset(data=ds, point_id=_netcdf_station_id(filename))
    return ds


def read_csv_meteo_file(filename, station_id, station_name, x, y, alt, crs):
    """
    Read a meteorological data file in CSV format and return it as a Dataset.
    Unlike in read_netcdf_meteo_file, here it is assumed that precipitation is
    specified as a sum over the time step (i.e., kg m-2) instead of a rate (kg
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
        pass
        # TODO check if the time step is really 3600 s and if precip is really a sum

    ds = forcing.make_point_dataset(
        data=df,
        point_id=station_id,
        name=station_name,
        lon=lon,
        lat=lat,
        alt=alt,
    )

    return ds


def _slice_and_resample_dataset(ds, start_date, end_date, freq, aggregate=False):
    """
    Slice a dataset to a given date range and optionally resample to a given
    time frequency.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset.

    start_date : datetime-like
        Start date.

    end_date : datetime-like
        End date.

    freq : str
        Pandas-compatible frequency string (e.g. '3H'). Must be an exact subset
        of the original frequency of the data.

    aggregate : boolean, default False
        Aggregate data when downsampling to a lower frequency or take
        instantaneous values.

    Returns
    -------
    ds : Dataset
        Sliced and resampled dataset.
    """
    freq_td = util.offset_to_timedelta(freq)
    inferred_freq = ds.time.to_index().inferred_freq
    td_1d = pd.Timedelta('1d')

    if inferred_freq is None and ds.dims['time'] > 2:
        # ("> 2" because inferring the frequency requires at least 3 points, so for shorter
        # time series inferred_freq is always None)
        raise errors.MeteoDataError('File contains missing records or non-uniform timesteps')

    if (
        aggregate
        and freq_td == td_1d
        and util.offset_to_timedelta(inferred_freq) < td_1d
    ):
        end_date = (end_date + td_1d).normalize()

    ds = ds.sel(time=slice(start_date, end_date))

    if inferred_freq is not None and inferred_freq != freq:
        ds = _resample_dataset(ds, start_date, end_date, freq, aggregate=aggregate)

    return ds


def _resample_dataset(ds, start_date, end_date, freq, aggregate=False):
    """
    Resample a dataset to a given time frequency.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset.

    start_date : datetime-like
        Start date.

    end_date : datetime-like
        End date.

    freq : str
        Pandas-compatible frequency string (e.g. '3H'). Must be an exact subset
        of the original frequency of the data.

    aggregate : boolean, default False
        Aggregate data when downsampling to a lower frequency or take
        instantaneous values.

    Returns
    -------
    ds_res : Dataset
        Resampled dataset.
    """
    td = util.offset_to_timedelta(freq)
    td_1d = pd.Timedelta('1d')
    if td < td_1d:
        resample_kwargs = dict(label='right', closed='right', origin=pd.Timestamp(start_date))
    elif td == td_1d:
        resample_kwargs = dict(label='left', closed='right', origin='start')
    else:
        raise errors.MeteoDataError('Resampling to frequencies > 1 day is not supported')

    if ds.dims['time'] == 0:
        # Nothing to resample
        return ds

    # ds.resample() is extremely slow for some reason, so we resample using pandas
    df = ds.to_dataframe().drop(columns=['lon', 'lat', 'alt'])

    if aggregate:
        # Calculate averages
        df_res = df.resample(freq, **resample_kwargs).mean()

        # Wind direction must be aggregated separately
        if 'wind_dir' in df.columns:
            df_res['wind_dir'] = _aggregate_wind_dir(df, freq, resample_kwargs)

        # We might end up with an extra bin after resampling; remove it here
        df_res = df_res.loc[start_date:end_date]
    else:
        # Take the instantaneous values
        df_res = df.reindex(pd.date_range(
            start=start_date,
            end=end_date,
            freq=freq,
            name='time',
        )).loc[df.index[0]:df.index[-1]]

    # Precipitation is summed up regardless of the aggregation setting
    if 'precip' in df:
        df_res['precip'] = df['precip'].resample(
            freq,
            **resample_kwargs,
        ).agg(pd.Series.sum, skipna=False)

    # Check if the desired frequency is a subset of the original frequency of the
    # data (e.g., resampling hourly to 3-hourly data is ok, but not hourly to
    # 1.5-hourly, or upsampling in general)
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


def _netcdf_station_id(filename):
    """
    Return the station ID of a station in NetCDF format, i.e., the base name of
    the file without the .nc extension.
    """
    return Path(filename).stem


def _read_meteo_metadata(meteo_format, meteo_data_dir, meteo_crs, grid_crs):
    """
    Read the metadata of the available meteorological stations.

    Parameters
    ----------
    meteo_format : str
        Data format (either 'netcdf' or 'csv').

    meteo_data_dir : path-like
        Location of the NetCDF files.

    meteo_crs : str
        CRS of the station coordinates specified in the stations.csv file
        (required only when `meteo_format` is 'csv').

    grid_crs : str
        CRS of the model grid.

    Returns
    -------
    meta : DataFrame
        DataFrame containing the station IDs as index and the columns `name`,
        `x` (x coordinate in the grid CRS), `y` (y coordinate in the grid CRS),
        and `alt`.
    """
    if meteo_format == 'netcdf':
        nc_files = sorted(list(meteo_data_dir.glob('*.nc')))
        if len(nc_files) == 0:
            raise errors.MeteoDataError(f'No meteo data files found in {meteo_data_dir}')

        station_ids = []
        names = []
        lons = []
        lats = []
        alts = []

        for nc_file in nc_files:
            with xr.open_dataset(nc_file) as ds:
                station_ids.append(_netcdf_station_id(nc_file))
                names.append(None)  # name is not needed here, is set in read_netcdf_meteo_file
                lons.append(float(ds.lon.values))
                lats.append(float(ds.lat.values))
                alts.append(float(ds.alt.values))

        # Transform lon/lat coordinates to grid CRS
        xs_gridcrs, ys_gridcrs = util.transform_coords(
            lons,
            lats,
            constants.CRS_WGS84,
            grid_crs,
        )

        meta = pd.DataFrame(
            index=station_ids,
            data=dict(
                name=names,
                x=xs_gridcrs,
                y=ys_gridcrs,
                alt=alts,
            ),
        )
    elif meteo_format == 'csv':
        filename = f'{meteo_data_dir}/stations.csv'
        try:
            meta = pd.read_csv(filename, index_col=0)
        except FileNotFoundError:
            raise errors.MeteoDataError(f'Missing station metadata file ({filename})')

        required_cols = ['name', 'x', 'y', 'alt']
        if not all(pd.Index(required_cols).isin(meta.columns)):
            raise errors.MeteoDataError('Station metadata file does not contain all required '
                                        f'columns ({required_cols})')

        # Transform x/y coordinates from meteo CRS to grid CRS
        xs_gridcrs, ys_gridcrs = util.transform_coords(
            meta.x,
            meta.y,
            meteo_crs,
            grid_crs,
        )

        meta.x = xs_gridcrs
        meta.y = ys_gridcrs
    else:
        raise NotImplementedError

    return meta


def _apply_station_rules(meta, bounds, exclude, include):
    """
    Extend or reduce the list of stations to read in using a defined region or
    exclusion/inclusion lists.

    Parameters
    ----------
    meta : DataFrame
        Station metadata as returned by `_read_meteo_metadata`.

    bounds : list or None
        If specified, use only stations within the region specified as a list of
        (x_min, y_min, x_max, y_max) coordinates in the model grid CRS.

    exclude: list or None
        List of station IDs to exclude.

    include: list or None
        List of station IDs to include even if otherwise excluded via `bounds`
        or `exclude`.

    Returns
    -------
    meta : DataFrame
    """
    meta_all = meta.copy()

    if bounds is not None:
        # TODO check if using ">=" and "<" is correct here or if ">" and "<=" should be used for y
        # coordinates
        meta = meta[
            (meta.x >= bounds[0])
            & (meta.y >= bounds[1])
            & (meta.x < bounds[2])
            & (meta.y < bounds[3])
        ]

    if exclude is not None:
        meta = meta.drop(index=exclude, errors='ignore')

    if include is not None:
        for station_id in include:
            try:
                meta = pd.concat([meta, meta_all.loc[[station_id]]])
            except KeyError:
                raise errors.MeteoDataError(f'Station to be included not found: {station_id}')

    meta = meta[~meta.index.duplicated(keep='first')]  # remove potential duplicate entries
    return meta


def _aggregate_wind_dir(df, freq, resample_kwargs):
    """
    Aggregate wind direction by averaging the wind vector components.

    Parameters
    ----------
    df : DataFrame
        Original data. Must contain a "wind_dir" column.

    freq : str
        Aggregation frequency.

    resample_kwargs : dict
        Keyword arguments to be passed to resample().

    Returns
    -------
    wind_dir : Series
        Aggregated wind direction.
    """
    df = df.copy()

    try:
        ws = df['wind_speed']
    except KeyError:
        ws = df['wind_dir'] * float('nan')

    wind_us, wind_vs = oameteo.wind_to_uv(ws, df['wind_dir'])
    df['wind_u'] = wind_us
    df['wind_v'] = wind_vs
    df = df[['wind_u', 'wind_v']]

    df_res = df.resample(freq, **resample_kwargs).mean()
    _, wd_res = oameteo.wind_from_uv(df_res['wind_u'], df_res['wind_v'])
    df_res['wind_dir'] = wd_res

    return df_res['wind_dir']
