import numpy as np
from openamundsen import constants, errors, util
import pandas as pd
import xarray as xr


def make_point_dataset(
        data=None,
        dates=None,
        point_id=None,
        name=None,
        lon=None,
        lat=None,
        alt=None,
):
    """
    Create a dataset containing the meteorological forcing data for a single
    point including metadata (point ID and name, variable units).

    Parameters
    ----------
    data : xr.Dataset or pd.DataFrame, optional
        Point data. If None, a dataset containing all-nan values will be
        created.

    dates : pd.DatetimeIndex, optional
        Must be specified only if `data` is None, otherwise the dates will be
        taken from there.

    point_id : str, optional
        Point ID. Only required if the ID is not stored in `data.attrs`.

    name : str, optional
        Point name. If `None`, the name is taken from `data.attrs` if possible,
        otherwise the name will be set to the point ID.

    lon : float, optional
        Longitude (degrees). Must be specified only if `data` is None or does
        not contain a `lon` variable.

    lat : float, optional
        Latitude (degrees). Must be specified only if `data` is None or does
        not contain a `lat` variable.

    alt : float, optional
        Altitude (m). Must be specified only if `data` is None or does
        not contain an `alt` variable.

    Returns
    -------
    ds : xr.Dataset
    """
    if data is None:
        data = xr.Dataset()

        if dates is None:
            dates = pd.DatetimeIndex([])
    elif isinstance(data, xr.Dataset):
        data = data.copy(deep=True)  # required because of the potential precipitation-altering
        dates = data.indexes['time']
    elif isinstance(data, pd.DataFrame):
        data = data.copy()
        data.index.name = 'time'
        dates = data.index
        data = data.to_xarray()
    else:
        raise TypeError(f'Unsupported type: {type(data)}')

    if point_id is None:
        if data is None or 'station_id' not in data.attrs:
            raise TypeError('Point ID neither in data nor passed as keyword argument')
        point_id = data.attrs['station_id']

    if name is None:
        try:
            name = data.attrs['station_name']
        except KeyError:
            name = point_id

    if lon is None:
        if 'lon' not in data.variables:
            raise TypeError('Longitude neither in data nor passed as keyword argument')
        lon = float(data.lon)

    if lat is None:
        if 'lat' not in data.variables:
            raise TypeError('Latitude neither in data nor passed as keyword argument')
        lat = float(data.lat)

    if alt is None:
        if 'alt' not in data.variables:
            raise TypeError('Altitude neither in data nor passed as keyword argument')
        alt = float(data.alt)

    ds = make_empty_point_dataset(dates, point_id, name, lon, lat, alt)

    # Copy values (and check if the units are as expected)
    for var, meta in constants.METEO_VAR_METADATA.items():
        if var not in data:
            continue

        if var == 'precip':
            # Convert precipitation rates to sums first if required
            if 'units' in data['precip'].attrs and data['precip'].units == 'kg m-2 s-1':
                try:
                    freq = pd.infer_freq(dates)
                    dt = util.offset_to_timedelta(freq).total_seconds()
                except ValueError:  # infer_freq() needs at least 3 dates
                    dt = np.nan

                data['precip'] *= dt
                data['precip'].attrs['units'] = 'kg m-2'

        units = meta['units']
        if 'units' in data[var].attrs and data[var].units != units:
            raise errors.MeteoDataError(
                f'{var} has wrong units: expected {units}, got {ds[var].units}'
            )

        ds[var].values[:] = data[var].values

    return ds


def make_empty_point_dataset(dates, point_id, name, lon, lat, alt):
    """
    Create an empty dataset for the meteorological forcing data for a single
    point.

    The returned dataset contains the station metadata (point ID, name,
    coordinates and altitude) and the meteorological variables, which are set to
    all-nan values.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Dates.

    point_id : str
        Point ID.

    name : str
        Point name

    lon : float
        Longitude (degrees).

    lat : float
        Latitude (degrees).

    alt : float
        Altitude (m)

    Returns
    -------
    ds : xr.Dataset
    """
    data = {
        'lon': ([], float(lon)),
        'lat': ([], float(lat)),
        'alt': ([], float(alt)),
    }

    for var, meta in constants.METEO_VAR_METADATA.items():
        data[var] = (
            ['time'],
            np.full(len(dates), np.nan),
            meta,
        )

    ds = xr.Dataset(
        data,
        coords={'time': (['time'], dates)},
        attrs={
            'station_id': point_id,
            'station_name': name,
        },
    )

    return ds


def combine_point_datasets(datasets):
    """
    Combine a list of meteo datasets as returned by `make_point_dataset`.
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
