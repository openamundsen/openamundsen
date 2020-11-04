from dataclasses import dataclass
import netCDF4
import numpy as np
from openamundsen import fileio
import pandas as pd
import pyproj
import xarray as xr


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


class FieldOutputManager:
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
                # If a frequency is set, for non-aggregated fields the write dates are assigned
                # to the start of the respective intervals (e.g. if the model timestep is
                # hourly and the write frequency is 'D', the write dates are 00:00 of each
                # day).
                # For aggregated fields, the write dates are assigned to the end of the
                # intervals (e.g., in this case the write dates would be 23:00 of each day).
                if agg is None:
                    write_dates = pd.date_range(
                        start=model.dates[0],
                        end=model.dates[-1],
                        freq=freq,
                    )
                else:
                    write_dates = pd.period_range(
                        start=model.dates[0],
                        end=model.dates[-1],
                        freq=freq,
                    ).asfreq(model.config.timestep, how='E').to_timestamp()

                    # Write aggregated fields at the last model timestep even if the last
                    # aggregation is not yet finished (e.g. if the aggregation is monthly
                    # but the model run ends in the middle of the month)
                    if model.dates[-1] not in write_dates:
                        write_dates = (
                            write_dates
                            .append(pd.DatetimeIndex([model.dates[-1]]))
                            .sort_values()
                        )

            if output_name is None:
                output_name = field_cfg.var.split('.')[-1]

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

        # Open the NetCDF file in case of NetCDF output (and create it beforehand when calling the
        # method for the first time)
        if self.format == 'netcdf':
            nc_file = self.model.config.results_dir / 'output_grids.nc'

            if not self.nc_file_created:
                ds = self._create_dataset()
                ds.to_netcdf(nc_file)
                self.nc_file_created = True

            ds = netCDF4.Dataset(nc_file, 'r+')

        # Loop through all fields, update aggregations where necessary and write files at the
        # specified dates
        for field in self.fields:
            if field.agg is not None:
                if field.data is None:
                    meta = self.model.state.meta(field.var)

                    if meta.dim3 == 0:
                        arr = np.full((self.model.grid.rows, self.model.grid.cols), np.nan)
                        arr[roi] = 0
                    else:
                        arr = np.full((meta.dim3, self.model.grid.rows, self.model.grid.cols), np.nan)
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
                if field.agg is None:
                    data = self.model.state[field.var]
                else:
                    data = field.data

                if self.format == 'netcdf':
                    date_idx = np.flatnonzero(field.write_dates == date)[0]
                    ds[field.output_name][date_idx, :, :] = data
                elif self.format == 'ascii':
                    if data.ndim == 2:
                        filename = self.model.config.results_dir / f'{field.output_name}_{date:%Y-%m-%dT%H%M}.asc'
                        self.model.logger.debug(f'Writing field {field.var} to {filename}')
                        fileio.write_raster_file(filename, data, self.model.grid.transform)
                    else:
                        # For 3-dimensional variables, write each layer as a separate file
                        for layer_num in range(data.shape[0]):
                            filename = (
                                self.model.config.results_dir
                                / f'{field.output_name}_{layer_num}_{date:%Y-%m-%dT%H%M}.asc'
                            )
                            self.model.logger.debug(f'Writing field {field.var} (layer {layer_num}) to {filename}')
                            fileio.write_raster_file(filename, data[layer_num, :, :], self.model.grid.transform)
                else:
                    raise NotImplementedError

                field.data = None
                field.num_aggregations = 0

        if self.format == 'netcdf':
            ds.close()

    def _create_dataset(self):
        """
        Create a CF-compliant Dataset covering the specified output variables
        and dates.

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

            if meta.dim3 == 0:  # 2-dimensional variable
                data[field.output_name] = (
                    [field_time_var, 'y', 'x'],
                    np.full((len(field.write_dates), len(y_coords), len(x_coords)), np.nan, dtype=np.float32),
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
                    np.full(
                        (len(field.write_dates), meta.dim3, len(y_coords), len(x_coords)),
                        np.nan,
                        dtype=np.float32,
                    ),
                    attrs,
                )

        # Add 3-dimensional coordinates
        for coord_name, coord_len in three_dim_coords.items():
            coords[coord_name] = ([coord_name], np.arange(coord_len))

        ds = xr.Dataset(data, coords=coords)
        ds.attrs['Conventions'] = 'CF-1.7'

        for time_var in times:
            ds[time_var].attrs['standard_name'] = 'time'

            # Set time units manually because otherwise the units of the time and the time bounds
            # variables might be different which is not recommended by CF standards
            ds[time_var].encoding['units'] = f'hours since {self.model.dates[0]:%Y-%m-%d %H:%M}'

            # Store time variables as doubles for CF compliance
            ds[time_var].encoding['dtype'] = np.float64
            if f'{time_var}_bounds' in ds:
                ds[f'{time_var}_bounds'].encoding['dtype'] = np.float64

        return ds
