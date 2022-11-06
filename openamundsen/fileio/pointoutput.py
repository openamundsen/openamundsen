from dataclasses import dataclass
import numpy as np
from openamundsen import constants, errors, util
import pandas as pd
import pprint
import rasterio
import xarray as xr


@dataclass
class PointOutputVariable:
    """
    Class for defining a point output variable.

    Parameters
    ----------
    var_name : str
        Name of the state variable (e.g. "meteo.temp").

    output_name : str, optional
        Output name (by default the variable name, e.g. "temp" for the variable
        "meteo.temp")
    """
    var_name: str
    output_name: str = None

    def __post_init__(self):
        if self.output_name is None:
            self.output_name = self.var_name.split('.')[-1]


@dataclass
class OutputPoint:
    """
    Class for defining an output point.

    Parameters
    ----------
    name : str
        Point name.

    lon : float
        Longitude (degrees).

    lat : float
        Latitude (degrees).

    alt : float
        Altitude (m).

    x : float
        x coordinate in the model CRS (m).

    y : float
        y coordinate in the model CRS (m).

    col : int
        Column index of the point within the model grid.

    row : int
        Row index of the point within the model grid.
    """
    name: str
    lon: float
    lat: float
    alt: float
    x: float
    y: float
    col: int
    row: int


_default_output_vars = [
    PointOutputVariable('meteo.temp'),
    PointOutputVariable('meteo.precip'),
    PointOutputVariable('meteo.snowfall'),
    PointOutputVariable('meteo.rainfall'),
    PointOutputVariable('meteo.rel_hum'),
    PointOutputVariable('meteo.wind_speed'),
    PointOutputVariable('meteo.sw_in'),
    PointOutputVariable('meteo.sw_out'),
    PointOutputVariable('meteo.lw_in'),
    PointOutputVariable('meteo.lw_out'),
    PointOutputVariable('meteo.sw_in_clearsky'),
    PointOutputVariable('meteo.dir_in_clearsky'),
    PointOutputVariable('meteo.diff_in_clearsky'),
    PointOutputVariable('meteo.cloud_factor'),
    PointOutputVariable('meteo.cloud_fraction'),
    PointOutputVariable('meteo.wet_bulb_temp'),
    PointOutputVariable('meteo.dew_point_temp'),
    PointOutputVariable('meteo.atmos_press'),
    PointOutputVariable('meteo.sat_vap_press'),
    PointOutputVariable('meteo.vap_press'),
    PointOutputVariable('meteo.spec_hum'),

    PointOutputVariable('surface.temp', 'surface_temp'),
    PointOutputVariable('surface.heat_flux', 'surface_heat_flux'),
    PointOutputVariable('surface.sens_heat_flux'),
    PointOutputVariable('surface.lat_heat_flux'),
    PointOutputVariable('surface.advective_heat_flux'),
    PointOutputVariable('surface.albedo', 'surface_albedo'),

    PointOutputVariable('soil.temp', 'soil_temp'),
    PointOutputVariable('soil.heat_flux', 'soil_heat_flux'),

    PointOutputVariable('snow.swe', 'swe'),
    PointOutputVariable('snow.depth', 'snow_depth'),
    PointOutputVariable('snow.temp', 'snow_temp'),
    PointOutputVariable('snow.thickness', 'snow_thickness'),
    PointOutputVariable('snow.density', 'snow_density'),
    PointOutputVariable('snow.ice_content'),
    PointOutputVariable('snow.liquid_water_content'),
    PointOutputVariable('snow.melt'),
    PointOutputVariable('snow.runoff'),
    PointOutputVariable('snow.sublimation'),
    PointOutputVariable('snow.refreezing'),
]


class PointOutputManager:
    """
    Class for managing and storing point output data (i.e. time series for
    selected point locations). The data is stored in an xarray Dataset and
    written to file at specified intervals.

    Parameters
    ----------
    model : OpenAmundsen
        openAMUNDSEN model instance.
    """
    def __init__(self, model):
        vars = []
        points = []
        config = model.config.output_data.timeseries

        # Initialize write dates
        if config.format == 'memory':
            write_dates = []
        else:
            write_dates = pd.date_range(
                start=model.config.start_date,
                end=model.config.end_date,
                freq=config.write_freq,
            )
            write_dates = write_dates[
                (write_dates >= model.config.start_date)
                & (write_dates <= model.config.end_date)
            ]

            # Add last timestep of the model run if it is not included anyway
            if model.config.end_date not in write_dates:
                write_dates = write_dates.append(pd.DatetimeIndex([model.config.end_date]))
        self.write_dates = write_dates

        # Add default output variables
        if config.add_default_variables:
            for var in _default_output_vars:
                vars.append(var)

        # Add user defined output variables
        for var_cfg in config.variables:
            vars.append(
                PointOutputVariable(
                    var_name=var_cfg['var'],
                    output_name=var_cfg['name'] if 'name' in var_cfg else None
                )
            )

        # Check if all variable specifications are valid
        for var in vars:
            try:
                _ = model.state[var.var_name]
            except (AttributeError, KeyError):
                raise errors.ConfigurationError(f'Invalid time series output variable: {var.var_name}')

        # Check if there are any duplicate output names
        if len(set([v.output_name for v in vars])) < len(vars):
            raise errors.ConfigurationError('Duplicate output names in time series output configuration.\n'
                                            f'List of variables:\n{pprint.pformat(vars)}')

        # Add default output points (= stations within ROI)
        if config.add_default_points:
            stations = model.meteo.sel(station=model.meteo.within_roi)
            df = stations[['lon', 'lat', 'alt', 'x', 'y', 'col', 'row']].to_dataframe()

            for point in df.itertuples():
                points.append(OutputPoint(
                    name=point.Index,
                    lon=point.lon,
                    lat=point.lat,
                    alt=model.state.base.dem[point.row, point.col],
                    x=point.x,
                    y=point.y,
                    col=point.col,
                    row=point.row,
                ))

        # Add additional output points
        for point_num, point in enumerate(config.points):
            if 'name' in point:
                point_name = point['name']
            else:
                point_name = f'point{point_num + 1}'

            # Check if point name is already in use
            if point_name in [p.name for p in points]:
                raise errors.ConfigurationError(f'Duplicate point name: {point_name}')

            lon, lat = util.transform_coords(
                point['x'],
                point['y'],
                model.config.crs,
                constants.CRS_WGS84,
            )

            row, col = rasterio.transform.rowcol(model.grid.transform, point['x'], point['y'])

            # Check if point is within the grid boundaries
            if row < 0 or row >= model.grid.rows or col < 0 or col >= model.grid.cols:
                raise errors.ConfigurationError('Output point is outside of the grid boundaries: '
                                                f'{dict(point)}')

            # Check if point is within the ROI (if not, still allow it but raise a warning)
            if not model.grid.roi[row, col]:
                model.logger.warning(f'Output point is outside of the ROI: {dict(point)}')

            points.append(OutputPoint(
                name=point_name,
                lon=lon,
                lat=lat,
                alt=model.state.base.dem[row, col],
                x=point['x'],
                y=point['y'],
                col=col,
                row=row,
            ))

        self.model = model
        self.vars = vars
        self.points = points
        self.point_cols = [p.col for p in points]
        self.point_rows = [p.row for p in points]
        self.data = None
        self.format = config.format

    def _create_dataset(self, dates):
        """
        Create a Dataset covering the specified dates and all point locations
        and output variables according to the model run configuration.

        Parameters
        ----------
        dates : list of datetime-like

        Returns
        -------
        ds : xr.Dataset
        """
        data = {}
        three_dim_coords = {}
        for var in self.vars:
            meta = self.model.state.meta(var.var_name)
            attrs = {}

            if np.issubdtype(meta.dtype, np.integer):
                dtype = np.int32
            else:
                dtype = np.float32

            for attr in ('standard_name', 'long_name', 'units'):
                attr_val = getattr(meta, attr)
                if attr_val is not None:
                    attrs[attr] = attr_val

            if meta.dim3 == 0:  # 2-dimensional variable
                var_def = (
                    ['time', 'point'],
                    np.full((len(dates), len(self.points)), np.nan, dtype=dtype),
                    attrs,
                )
            else:  # 3-dimensional variable
                category = self.model.state.parse(var.var_name)[0]
                coord_name = f'{category}_layer'

                if category in three_dim_coords:
                    if three_dim_coords[coord_name] != meta.dim3:
                        # We assume that all 3-dimensional variables within a category have the
                        # same shape (e.g. "soil.temp" must have the same shape as "soil.therm_cond");
                        # varying numbers of layers within a category are not supported
                        raise Exception('Inconsistent length of third variable dimension')
                else:
                    three_dim_coords[coord_name] = meta.dim3

                var_def = (
                    ['time', coord_name, 'point'],
                    np.full((len(dates), meta.dim3, len(self.points)), np.nan, dtype=np.float32),
                    attrs,
                )

            data[var.output_name] = var_def

        coords = {
            'time': (['time'], dates),
            'point': (['point'], [p.name for p in self.points]),
            'lon': (['point'], [p.lon for p in self.points]),
            'lat': (['point'], [p.lat for p in self.points]),
            'alt': (['point'], [p.alt for p in self.points]),
            'x': (['point'], [p.x for p in self.points]),
            'y': (['point'], [p.y for p in self.points]),
        }

        # Add 3-dimensional coordinates
        for coord_name, coord_len in three_dim_coords.items():
            coords[coord_name] = ([coord_name], np.arange(coord_len))

        return xr.Dataset(data, coords=coords)

    def _current_chunk_dates(self):
        """
        Return the dates of the current "chunk", i.e., the period between the
        last date for which the output time series were written and the next
        date for which they will be written.

        Returns
        -------
        dates : pd.DatetimeIndex
        """
        if self.format == 'memory':
            dates = self.model.dates
        else:
            idxs = np.flatnonzero(self.write_dates >= self.model.date)
            next_write_date = self.write_dates[idxs[0]]

            idxs = np.flatnonzero(self.write_dates < self.model.date)
            if len(idxs) == 0:
                dates = self.model.dates[self.model.dates <= next_write_date]
            else:
                prev_write_date = self.write_dates[idxs[-1]]
                dates = self.model.dates[
                    (self.model.dates > prev_write_date)
                    & (self.model.dates <= next_write_date)
                ]

        return dates

    def update(self):
        """
        Update the point output data for the current time step, i.e., write the
        output variable values for all point locations to the internal dataset.
        """
        self.model.logger.debug('Updating point outputs')

        if self.data is None:
            self.data = self._create_dataset(self._current_chunk_dates())

        ds = self.data
        date = self.model.date

        # Get the index for the current date because directly writing to the
        # underlying numpy arrays of the xarray variables
        # (ds_var.values[date_idx, :]) is a lot faster than using xarray with
        # label-based indexing (ds_var.loc[date, :]).
        # Using np.argmax is equivalent to using np.where to get the index,
        # but is faster because it stops at the first match.
        date_idx = np.argmax(ds.time.values == date.to_datetime64())

        # Update dataset
        for var in self.vars:
            var_data = self.model.state[var.var_name]
            ds_var = ds.variables[var.output_name]  # faster than ds[var.output_name]

            if var_data.ndim == 2:
                ds_var.values[date_idx, :] = var_data[self.point_rows, self.point_cols]
            else:  # 3-dimensional variable
                ds_var.values[date_idx, :, :] = var_data[:, self.point_rows, self.point_cols]

        # Write data to file
        # If we are at the first write date, simple write the file (i.e. overwrite possibly
        # existing files), otherwise merge the already written dataset with the in-memory one.
        if self.format != 'memory' and date == ds.indexes['time'][-1]:
            self.model.logger.debug('Writing point outputs')

            if self.format == 'netcdf':
                filename = self.model.config.results_dir / 'output_timeseries.nc'

                if date == self.write_dates[0]:
                    ds.to_netcdf(filename)
                else:
                    with xr.open_dataset(filename) as old_ds:
                        # Handle an issue introduced in xarray v2022.06.0 - without this line, the
                        # time index somehow loses the time information and keeps only the date part
                        # (triggered by test_point_output.py::test_write_freq)
                        old_ds['time'] = old_ds.indexes['time']

                        ds_merge = xr.concat([old_ds, ds], 'time')

                    ds_merge.to_netcdf(filename)
            elif self.format == 'csv':
                ds_out = ds.copy()

                # If there are 3-dimensional variables, convert them to 2-dimensional variables
                # before writing the CSV files
                # (e.g. a 3-dimensional variable "soil_temp" with 4 layers is converted to 4
                # variables "soil_temp0", "soil_temp1", "soil_temp2", "soil_temp3")
                for var in self.vars:
                    meta = self.model.state.meta(var.var_name)

                    if meta.dim3 > 0:
                        for i in range(meta.dim3):
                            ds_out[f'{var.output_name}{i}'] = ds[var.output_name].loc[:, i, :]

                        ds_out = ds_out.drop_vars(var.output_name)

                for point in self.points:
                    filename = self.model.config.results_dir / f'point_{point.name}.csv'

                    ds_out_point = ds_out.sel(point=point.name)

                    # Drop all coordinate variables except "time" (i.e. "lon", "lat", etc.)
                    # so that they are not in the resulting dataframe when calling to_dataframe()
                    ds_out_point = ds_out_point.drop_vars(list(set(list(ds_out_point.coords)) - set(['time'])))

                    df = ds_out_point.to_dataframe()

                    if date == self.write_dates[0]:
                        df.to_csv(filename)
                    else:
                        df.to_csv(filename, mode='a', header=False)

            self.data = None
