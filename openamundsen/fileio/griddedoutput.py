from dataclasses import dataclass
import netCDF4
import numpy as np
from openamundsen import constants, errors, fileio, util
import pandas as pd
import pandas.tseries.frequencies
import pyproj
import xarray as xr

try:
    import dask.array
    _DASK_AVAILABLE = True
except ImportError:
    _DASK_AVAILABLE = False


_ALLOWED_OFFSETS = [
    pd.tseries.offsets.YearEnd,
    pd.tseries.offsets.YearBegin,
    pd.tseries.offsets.MonthEnd,
    pd.tseries.offsets.MonthBegin,
    pd.tseries.offsets.Day,
    pd.tseries.offsets.Hour,
    pd.tseries.offsets.Minute,
]


@dataclass
class OutputField:
    """
    Class for defining an output field, i.e., a state variable that should be
    written at specified dates.

    Parameters
    ----------
    var : str
        Name of the state variable (e.g. "meteo.temp").

    output_name : str
        Output name.

    agg : str, optional
        Aggregation function. Can be either None (if instantaneous values
        should be written), "sum" or "mean".

    write_dates : pd.DatetimeIndex
        Dates at which the field should be written.

    data : np.array, optional
        Current state of the aggregated values (only used if `agg` is not None).

    num_aggregations : int, default 0
        Current number of aggregations (required for calculating a running mean).
    """
    var: str
    output_name: str
    agg: str
    write_dates: pd.DatetimeIndex
    data: np.array = None
    num_aggregations: int = 0


def _field_key(field):
    return (tuple(field.write_dates), field.agg is None)


class GriddedOutputManager:
    """
    Class for managing and storing gridded output data which should be written at
    specified dates.

    Parameters
    ----------
    model : OpenAmundsen
        openAMUNDSEN model instance.
    """
    def __init__(self, model):
        config = model.config.output_data.grids
        fields = []

        for field_cfg in config.variables:
            try:
                output_name = field_cfg['name']
            except KeyError:
                output_name = None

            try:
                freq = field_cfg['freq']
            except KeyError:
                freq = model.dates.freqstr

            try:
                agg = field_cfg['agg']
            except KeyError:
                agg = None

            if 'dates' in field_cfg:
                write_dates = pd.to_datetime(field_cfg['dates'])
            else:
                write_dates = _freq_write_dates(model.dates, freq, agg is not None)

            write_dates = write_dates[
                (write_dates >= model.dates[0])
                & (write_dates <= model.dates[-1])
            ]
            if len(write_dates) == 0:
                model.logger.debug(f'Discarding grid output variable {field_cfg["var"]}'
                                   ' (nothing to be written)')
                continue

            if output_name is None:
                output_name = field_cfg.var.split('.')[-1]

            if output_name in [f.output_name for f in fields]:
                raise errors.ConfigurationError(f'Duplicate grid output name: {output_name}')

            fields.append(OutputField(
                var=field_cfg['var'],
                output_name=output_name,
                agg=agg,
                write_dates=write_dates,
            ))

        self.model = model
        self.fields = fields
        self.format = config.format
        self.nc_file_created = False
        self.data = None

    def update(self):
        """
        Update the output fields for the current time step, i.e., update the
        aggregated fields (if aggregation functions are used) and write the
        variables to file at the specified dates.
        """
        # If there is nothing to be written, return right away
        if len(self.fields) == 0:
            return

        self.model.logger.debug('Updating field outputs')

        date = self.model.date
        roi = self.model.grid.roi

        if self.format == 'netcdf':
            nc_file = self.model.config.results_dir / 'output_grids.nc'
            append = self.model.config.output_data.grids.append

            if not self.nc_file_created:
                ds = self._create_dataset(in_memory=(not _DASK_AVAILABLE))

                if append:
                    # Convert time values to encoded times (e.g. X hours since YYYY-MM-DD)
                    self._time_dims = [d for d in list(ds.dims) if d.startswith('time')]
                    self._encoded_times = {}
                    for coord_name in list(ds.coords):
                        if not coord_name.startswith('time'):
                            continue

                        encoded_times, _, _ = xr.coding.times.encode_cf_datetime(
                            ds.coords[coord_name],
                            units=ds[self._time_dims[0]].encoding['units'],
                            calendar=ds[self._time_dims[0]].encoding['calendar'],
                        )
                        # (taking units and calendar from the first time variable works because all
                        # time variables have the same encoding)
                        self._encoded_times[coord_name] = encoded_times

                    ds = ds.drop_sel({v: ds[v] for v in self._time_dims})
                    ds.encoding['unlimited_dims'] = self._time_dims

                ds.to_netcdf(nc_file)
                self.nc_file_created = True

            ds = netCDF4.Dataset(nc_file, 'r+')
        elif self.format == 'memory':
            if self.data is None:
                self.data = self._create_dataset(in_memory=True)

        # Loop through all fields, update aggregations where necessary and write files at the
        # specified dates
        for field in self.fields:
            if field.agg is not None:
                if field.data is None:
                    meta = self.model.state.meta(field.var)

                    if meta.dim3 == 0:
                        arr = np.full(self.model.grid.shape, np.nan)
                        arr[roi] = 0
                    else:
                        arr = np.full((meta.dim3, *self.model.grid.shape), np.nan)
                        arr[:, roi] = 0

                    field.data = arr

                data_cur = self.model.state[field.var]

                if field.agg == 'sum':
                    if field.data.ndim == 2:
                        field.data[roi] += data_cur[roi]
                    else:
                        field.data[:, roi] += data_cur[:, roi]
                elif field.agg == 'mean':
                    if field.data.ndim == 2:
                        field.data[roi] += (data_cur[roi] - field.data[roi]) / (field.num_aggregations + 1)
                    else:
                        field.data[:, roi] += (data_cur[:, roi] - field.data[:, roi]) / (field.num_aggregations + 1)

                field.num_aggregations += 1

            if date in field.write_dates:
                date_idx = np.flatnonzero(field.write_dates == date)[0]

                if field.agg is None:
                    data = self.model.state[field.var]
                else:
                    data = field.data

                if self.format == 'netcdf':
                    ds[field.output_name][date_idx, :, :] = data

                    if append:
                        field_time_dim = ds[field.output_name].dimensions[0]
                        ds[field_time_dim][date_idx] = self._encoded_times[field_time_dim][date_idx]

                        bounds_var = f'{field_time_dim}_bounds'
                        if bounds_var in self._encoded_times:
                            ds[bounds_var][date_idx, :] = self._encoded_times[bounds_var][date_idx]
                elif self.format in ('ascii', 'geotiff'):
                    if self.format == 'ascii':
                        ext = 'asc'
                        rio_meta = {'driver': 'AAIGrid'}
                        # (do not add CRS information when using AAIGrid output to avoid writing
                        # .prj files)
                    elif self.format == 'geotiff':
                        ext = 'tif'
                        rio_meta = {
                            'driver': 'GTiff',
                            'crs': self.model.grid.crs,
                        }

                    if field.agg is None:
                        date_str = f'{date:%Y-%m-%dT%H%M}'
                    else:
                        # Find the start date of the current output interval for the output file
                        # name
                        if date_idx == 0:
                            start_date = self.model.dates[0]
                        else:
                            start_date = field.write_dates[date_idx - 1] + pd.Timedelta(
                                seconds=self.model.timestep)

                        date_str = f'{start_date:%Y-%m-%dT%H%M}_{date:%Y-%m-%dT%H%M}'

                    if data.ndim == 2:
                        filename = self.model.config.results_dir / f'{field.output_name}_{date_str}.{ext}'
                        self.model.logger.debug(f'Writing field {field.var} to {filename}')
                        fileio.write_raster_file(
                            filename,
                            data,
                            self.model.grid.transform,
                            **rio_meta,
                        )
                    else:
                        # For 3-dimensional variables, write each layer as a separate file
                        for layer_num in range(data.shape[0]):
                            filename = (
                                self.model.config.results_dir
                                / f'{field.output_name}_{layer_num}_{date_str}.{ext}'
                            )
                            self.model.logger.debug(f'Writing field {field.var} (layer {layer_num})'
                                                    ' to {filename}')
                            fileio.write_raster_file(
                                filename,
                                data[layer_num, :, :],
                                self.model.grid.transform,
                                **rio_meta,
                            )
                elif self.format == 'memory':
                    self.data[field.output_name].values[date_idx, :, :] = data
                else:
                    raise NotImplementedError

                field.data = None
                field.num_aggregations = 0

        if self.format == 'netcdf':
            ds.close()

    def _create_dataset(self, in_memory=False):
        """
        Create a CF-compliant Dataset covering the specified output variables
        and dates.

        Parameters
        ----------
        in_memory : bool, default False
            If True use in-memory arrays for creating the dataset, if False use
            Dask arrays.

        Returns
        -------
        ds : xr.Dataset
        """
        # Define names of time variables - if there is only one time variable simply name it "time",
        # otherwise they are named "time1", "time2", ...
        time_var_names = {}
        num_time_vars = 0
        for field in self.fields:
            key = _field_key(field)
            if key not in time_var_names:
                num_time_vars += 1
                time_var_names[key] = f'time{num_time_vars}'

        if num_time_vars == 1:
            key = next(iter(time_var_names))
            time_var_names[key] = 'time'

        times = {}  # dict for storing times and boundaries (for aggregated variables) of the time variables
        field_time_vars = []  # contains for each field the name of the respective NetCDF time variable

        for field in self.fields:
            key = _field_key(field)
            time_var_name = time_var_names[key]
            time_vals = field.write_dates.values

            if field.agg is None:
                time_vals = field.write_dates
                time_bounds = None
            else:
                time_bounds = np.repeat(time_vals[:, np.newaxis], 2, axis=1).copy()
                time_bounds[1:, 0] = time_bounds[:-1, 1]
                time_bounds[0, 0] = self.model.dates[0]

            field_time_vars.append(time_var_name)

            if time_var_name not in times:
                times[time_var_name] = (time_vals, time_bounds)

        x_coords = self.model.grid.X[0, :]
        y_coords = self.model.grid.Y[:, 0]

        # Define coordinate variables
        coords = {}
        for time_var, (time_vals, time_bounds) in times.items():
            time_attrs = {}

            if time_bounds is not None:
                bound_var_name = f'{time_var}_bounds'
                time_attrs['bounds'] = bound_var_name
                coords[bound_var_name] = (
                    [time_var, 'nbnd'],
                    time_bounds,
                    {
                        'long_name': 'time interval endpoints',
                    }
                )

            coords[time_var] = (
                time_var,
                time_vals,
                time_attrs,
            )

        coords['x'] = (
            ['x'],
            x_coords,
            {
                'standard_name': 'projection_x_coordinate',
                'long_name': 'x coordinate of projection',
                'units': 'm',
            },
        )
        coords['y'] = (
            ['y'],
            y_coords,
            {
                'standard_name': 'projection_y_coordinate',
                'long_name': 'y coordinate of projection',
                'units': 'm',
            },
        )
        coords['crs'] = (
            [],
            np.array(0),
            pyproj.crs.CRS(self.model.grid.crs).to_cf(),
        )

        if in_memory:
            full = np.full
        else:
            full = dask.array.full

        # Define data variables
        data = {}
        three_dim_coords = {}
        for field, field_time_var in zip(self.fields, field_time_vars):
            meta = self.model.state.meta(field.var)
            attrs = {}

            for attr in ('standard_name', 'long_name', 'units'):
                attr_val = getattr(meta, attr)
                if attr_val is not None:
                    attrs[attr] = attr_val

            attrs['grid_mapping'] = 'crs'

            # Assign output data type - float-like variables are written as float32, integer
            # variables as int32 or float32 (the latter if agg == 'mean')
            if (
                np.issubdtype(self.model.state.meta(field.var).dtype, np.integer)
                and field.agg != 'mean'
            ):
                dtype = np.int32
            else:
                dtype = np.float32

            if meta.dim3 == 0:  # 2-dimensional variable
                data[field.output_name] = (
                    [field_time_var, 'y', 'x'],
                    full((len(field.write_dates), len(y_coords), len(x_coords)), np.nan, dtype=dtype),
                    attrs,
                )
            else:  # 3-dimensional variable
                category = self.model.state.parse(field.var)[0]
                coord_name = f'{category}_layer'

                if category in three_dim_coords:
                    if three_dim_coords[coord_name] != meta.dim3:
                        # We assume that all 3-dimensional variables within a category have the
                        # same shape (e.g. "soil.temp" must have the same shape as "soil.therm_cond");
                        # varying numbers of layers within a category are not supported
                        raise Exception('Inconsistent length of third variable dimension')
                else:
                    three_dim_coords[coord_name] = meta.dim3

                data[field.output_name] = (
                    [field_time_var, coord_name, 'y', 'x'],
                    full(
                        (len(field.write_dates), meta.dim3, len(y_coords), len(x_coords)),
                        np.nan,
                        dtype=dtype,
                    ),
                    attrs,
                )

        # Add 3-dimensional coordinates
        for coord_name, coord_len in three_dim_coords.items():
            coords[coord_name] = ([coord_name], np.arange(coord_len))

        ds = xr.Dataset(data, coords=coords)
        ds.attrs['Conventions'] = 'CF-1.7'

        _, datetime_units, calendar = xr.coding.times.encode_cf_datetime(self.model.dates)
        for time_var in times:
            ds[time_var].attrs['standard_name'] = 'time'

            # Set time units manually because otherwise the units of the time and the time bounds
            # variables might be different which is not recommended by CF standards
            ds[time_var].encoding['units'] = datetime_units
            ds[time_var].encoding['calendar'] = calendar

            # Store time variables as doubles for CF compliance
            ds[time_var].encoding['dtype'] = np.float64
            if f'{time_var}_bounds' in ds:
                ds[f'{time_var}_bounds'].encoding['dtype'] = np.float64

        return ds


def _freq_write_dates(dates, out_freq, agg):
    """
    Calculate output dates for gridded outputs when a frequency string is set.

    For non-aggregated fields the write dates are assigned to the start of the
    respective intervals for non-anchored and begin-anchored offsets (e.g. 'D',
    'MS', 'AS'), and to the end of the intervals for end-anchored offsets (e.g.
    'M', 'A'). For aggregated fields, the write dates are always assigned to the
    end of the intervals.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Simulation dates.

    out_freq : str
        Output frequency as a pandas offset string (e.g. '3H', 'M').

    agg : boolean
        Prepare write dates for aggregated outputs (if True) or for
        instantaneous values.

    Returns
    -------
    write_dates : pd.DatetimeIndex

    Examples
    --------
    >>> dates = pd.date_range(
    ...     start='2021-01-01 00:00',
    ...     end='2021-12-31 23:00',
    ...     freq='H',
    ... )
    ... _freq_write_dates(dates, 'A', False)
    DatetimeIndex(['2021-12-31 23:00:00'], dtype='datetime64[ns]', freq=None)

    >>> _freq_write_dates(dates, 'AS', False)
    DatetimeIndex(['2021-01-01'], dtype='datetime64[ns]', freq='AS-JAN')

    >>> _freq_write_dates(dates, 'D', False)
    DatetimeIndex(['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04',
                   '2021-01-05', '2021-01-06', '2021-01-07', '2021-01-08',
                   '2021-01-09', '2021-01-10',
                   ...
                   '2021-12-22', '2021-12-23', '2021-12-24', '2021-12-25',
                   '2021-12-26', '2021-12-27', '2021-12-28', '2021-12-29',
                   '2021-12-30', '2021-12-31'],
                  dtype='datetime64[ns]', length=365, freq='D')

    >>> _freq_write_dates(dates, 'D', True)
    DatetimeIndex(['2021-01-01 23:00:00', '2021-01-02 23:00:00',
                   '2021-01-03 23:00:00', '2021-01-04 23:00:00',
                   '2021-01-05 23:00:00', '2021-01-06 23:00:00',
                   '2021-01-07 23:00:00', '2021-01-08 23:00:00',
                   '2021-01-09 23:00:00', '2021-01-10 23:00:00',
                   ...
                   '2021-12-22 23:00:00', '2021-12-23 23:00:00',
                   '2021-12-24 23:00:00', '2021-12-25 23:00:00',
                   '2021-12-26 23:00:00', '2021-12-27 23:00:00',
                   '2021-12-28 23:00:00', '2021-12-29 23:00:00',
                   '2021-12-30 23:00:00', '2021-12-31 23:00:00'],
                  dtype='datetime64[ns]', length=365, freq='D')
    """
    model_freq = dates.freqstr
    model_freq_td = util.offset_to_timedelta(model_freq)

    try:
        out_offset = pandas.tseries.frequencies.to_offset(out_freq)
        if not any([isinstance(out_offset, o) for o in _ALLOWED_OFFSETS]):
            raise ValueError
    except ValueError:
        allowed_offsets_str = ", ".join([o().__class__.__name__ for o in _ALLOWED_OFFSETS])
        raise errors.ConfigurationError(f'Unsupported output frequency: {out_freq}. '
                                        f'Supported offsets: {allowed_offsets_str}')

    if not out_offset.is_anchored():
        # For non-anchored offsets (e.g., '3H', 'D'), the output frequency must be a multiple of
        # (and not smaller than) the model timestep
        out_freq_td = util.offset_to_timedelta(out_freq)

        if out_freq_td < model_freq_td:
            raise ValueError('Output frequency must not be smaller than the model timestep')
        elif not (out_freq_td.total_seconds() / model_freq_td.total_seconds()).is_integer():
            raise ValueError('Output frequency must be a multiple of the model timestep')

    if agg:
        if out_offset.is_anchored():  # e.g. 'M', 'A'
            if model_freq_td.total_seconds() > constants.HOURS_PER_DAY * constants.SECONDS_PER_HOUR:
                raise NotImplementedError('Aggregation of gridded outputs with anchored offsets '
                                          'not supported for timesteps > 1d')

            period_end_dates = (
                pd.period_range(
                    start=dates[0],
                    end=dates[-1],
                    freq=out_freq,
                )
                .asfreq(model_freq, how='end')
                .to_timestamp()
            )

            d0 = dates[dates <= period_end_dates[0]][-1]
            write_dates = period_end_dates + (d0 - period_end_dates[0])

            if period_end_dates[0] - write_dates[0] > pd.Timedelta('1d'):
                write_dates = write_dates.delete(0)

            # Keep the last output interval only if it is fully covered (e.g., do not write half
            # months)
            if len(write_dates) > 0 and write_dates[-1] > dates[-1]:
                write_dates = write_dates.delete(-1)
        else:
            write_dates = pd.date_range(
                start=dates[0] + out_freq_td - model_freq_td,
                end=dates[-1],
                freq=out_freq,
            )
    else:
        write_dates = pd.date_range(
            start=dates[0],
            end=dates[-1],
            freq=out_freq,
        )

        if any([isinstance(out_offset, o) for o in (
                pd.tseries.offsets.YearEnd,
                pd.tseries.offsets.MonthEnd,
        )]) and model_freq_td < pd.Timedelta(days=1):
            write_dates += pd.Timedelta(days=1) - model_freq_td

    return write_dates
