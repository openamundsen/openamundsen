import numpy as np
from openamundsen import constants, errors, util
import pandas as pd
import rasterio
import xarray as xr


_POINT_DATASET_META_VARS = [
    'station_name',
    'lon',
    'lat',
    'alt',
]
_POINT_DATASET_ALLOWED_METEO_VARS = list(constants.METEO_VAR_METADATA.keys())
_POINT_DATASET_GRID_VARS = [
    'x',
    'y',
    'col',
    'row',
    'within_grid_extent',
    'within_roi',
]
_POINT_DATASET_MINIMAL_VARS = _POINT_DATASET_META_VARS + constants.MINIMUM_REQUIRED_METEO_VARS


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

    meteo_vars = list(set(data.data_vars) & set(_POINT_DATASET_ALLOWED_METEO_VARS))
    ds = make_empty_point_dataset(
        dates,
        point_id,
        name,
        lon,
        lat,
        alt,
        meteo_vars=meteo_vars,
    )

    # Copy values (and check if the units are as expected)
    for var in meteo_vars:
        meta = constants.METEO_VAR_METADATA[var]

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


def make_empty_point_dataset(dates, point_id, name, lon, lat, alt, meteo_vars=None):
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

    meteo_vars : list, default None
        Meteorological variables to create. If None, create a dataset with the
        minimally required variables.

    Returns
    -------
    ds : xr.Dataset
    """
    data = {
        'lon': ([], float(lon)),
        'lat': ([], float(lat)),
        'alt': ([], float(alt)),
    }

    if meteo_vars is None:
        meteo_vars = constants.MINIMUM_REQUIRED_METEO_VARS

    for var in meteo_vars:
        meta = constants.METEO_VAR_METADATA[var]
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


def combine_point_datasets(datasets, add_minimum_required_vars=True):
    """
    Combine a list of meteo datasets as returned by `make_point_dataset`.
    The datasets are merged by adding an additional "station" (= station id)
    dimension.

    Parameters
    ----------
    add_minimum_required_vars : bool, default True
        Add the minimum required variables for an openAMUNDSEN model run to the
        resulting combined dataset even if the variables do not exist in any of
        the individual datasets.
    """
    datasets_processed = []

    # Create a list of all meteo variables occurring in at least one of the datasets
    meteo_vars = set()
    if add_minimum_required_vars:
        meteo_vars.update(set(constants.MINIMUM_REQUIRED_METEO_VARS))
    for ds in datasets:
        meteo_vars.update(set(ds.data_vars))
    meteo_vars = sorted(list(meteo_vars & set(_POINT_DATASET_ALLOWED_METEO_VARS)))

    for ds in datasets:
        ds = ds.copy()

        for var in meteo_vars:
            meta = constants.METEO_VAR_METADATA[var]

            # Add missing variables to the dataset
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


def prepare_point_coordinates(ds, grid, crs):
    """
    Transform the lon/lat coordinates of the meteorological stations to the
    coordinate system of the model grid. The transformed coordinates are
    stored in the `x` and `y` variables of the meteo dataset.
    Additionally, the row and column indices of the stations within the
    model grid are stored in the `row` and `col` variables, and two boolean
    variables `within_grid_extent` and `within_roi` indicate whether the
    stations lie within the model grid extent and the ROI, respectively.

    Parameters
    ----------
    ds : xr.Dataset
        Point forcing dataset as returned by `combine_point_datasets`.

    grid : ModelGrid
        Model grid.

    crs : str
        Coordinate reference system.

    Returns
    -------
    ds : xr.Dataset
    """
    ds = ds.copy()

    x, y = util.transform_coords(ds.lon, ds.lat, constants.CRS_WGS84, crs)

    x_var = ds.lon.copy()
    x_var.values = x
    x_var.attrs = {
        'standard_name': 'projection_x_coordinate',
        'units': 'm',
    }

    y_var = ds.lat.copy()
    y_var.values = y
    y_var.attrs = {
        'standard_name': 'projection_y_coordinate',
        'units': 'm',
    }

    ds['x'] = x_var
    ds['y'] = y_var

    bool_var = ds.x.copy().astype(bool)
    bool_var[:] = False
    bool_var.attrs = {}
    int_var = bool_var.copy().astype(int)
    int_var[:] = -1

    if len(x) > 0:
        rows, cols = rasterio.transform.rowcol(grid.transform, x, y)
    else:
        rows = []
        cols = []

    row_var = int_var.copy()
    col_var = int_var.copy()
    row_var.values[:] = rows
    col_var.values[:] = cols
    ds['col'] = col_var
    ds['row'] = row_var

    ds['within_grid_extent'] = (
        (col_var >= 0)
        & (col_var < grid.cols)
        & (row_var >= 0)
        & (row_var < grid.rows)
    )

    within_roi_var = bool_var.copy()
    ds['within_roi'] = within_roi_var

    for station in ds.indexes['station']:
        dss = ds.sel(station=station)

        if dss.within_grid_extent:
            row = int(dss.row)
            col = int(dss.col)
            within_roi_var.loc[station] = grid.roi[row, col]

    # reorder variables (only for aesthetic reasons)
    meteo_vars = sorted(list(set(_POINT_DATASET_ALLOWED_METEO_VARS) & set(ds.data_vars)))
    var_order = _POINT_DATASET_META_VARS + _POINT_DATASET_GRID_VARS + meteo_vars
    ds = ds[var_order]

    return ds


def strip_point_dataset(ds):
    """
    Remove the grid-specific variables from a point forcing dataset.
    """
    return ds.drop_vars(_POINT_DATASET_GRID_VARS)


def is_valid_point_dataset(ds, dates=None):
    """
    Test if the passed variable is a valid point forcing dataset.
    """
    if set(ds.dims) != set(['station', 'time']):
        return False

    for v in _POINT_DATASET_MINIMAL_VARS:
        if v not in ds:
            return False

    if dates is not None:
        if not pd.DatetimeIndex(dates).equals(ds.indexes['time']):
            return False

    return True
