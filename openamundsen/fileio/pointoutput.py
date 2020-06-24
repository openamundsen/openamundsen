from dataclasses import dataclass
import numpy as np
from openamundsen import constants, util
import pandas as pd
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
    'meteo.temp',
    'meteo.precip',
    'meteo.snow',
    'meteo.rain',
    'meteo.rel_hum',
    'meteo.wind_speed',
    'meteo.sw_in',
    'meteo.sw_out',
    'meteo.lw_in',
    'meteo.lw_out',
    'meteo.sw_in_clearsky',
    'meteo.dir_in_clearsky',
    'meteo.diff_in_clearsky',
    'meteo.cloud_factor',
    'meteo.cloud_fraction',
    'meteo.wetbulb_temp',
    'meteo.dewpoint_temp',
    'meteo.atmos_press',
    'meteo.sat_vap_press',
    'meteo.vap_press',
    'meteo.spec_hum',
]


class PointOutputManager:
    """
    Class for managing and storing point output data (i.e. time series for
    selected point locations). The data is stored in an xarray Dataset and
    written to file at specified intervals.

    Parameters
    ----------
    model : Model
    """
    def __init__(self, model):
        vars = []
        points = []
        config = model.config.output_data.points

        # Initialize write dates (+ add last time step of the model run if not included anyway)
        write_dates = pd.date_range(
            start=model.config.start_date,
            end=model.config.end_date,
            freq=config.write_freq,
        )
        write_dates = write_dates[
            (write_dates >= model.config.start_date)
            & (write_dates <= model.config.end_date)
        ]
        if model.config.end_date not in write_dates:
            write_dates = write_dates.append(pd.DatetimeIndex([model.config.end_date]))
        self.write_dates = write_dates

        # Add default output variables
        for var in _default_output_vars:
            vars.append(PointOutputVariable(var))

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

            lon, lat = util.transform_coords(
                point['x'],
                point['y'],
                model.config.crs,
                constants.CRS_WGS84,
            )

            row, col = rasterio.transform.rowcol(model.grid.transform, point['x'], point['y'])
            # TODO check if point is within the grid boundaries

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
        for var in self.vars:
            meta = self.model.state.meta(var.var_name)
            attrs = {}

            for attr in ('standard_name', 'long_name', 'units'):
                attr_val = getattr(meta, attr)
                if attr_val is not None:
                    attrs[attr] = attr_val

            data[var.output_name] = (
                ['time', 'point'],
                np.full((len(dates), len(self.points)), np.nan, dtype=np.float32),
                attrs,
            )

        coords = {
            'time': (['time'], dates),
            'point': (['point'], [p.name for p in self.points]),
            'lon': (['point'], [p.lon for p in self.points]),
            'lat': (['point'], [p.lat for p in self.points]),
            'alt': (['point'], [p.alt for p in self.points]),
            'x': (['point'], [p.x for p in self.points]),
            'y': (['point'], [p.y for p in self.points]),
        }

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

        # Update dataset
        for var in self.vars:
            var_data = self.model.state[var.var_name]
            ds[var.output_name].loc[date, :] = var_data[self.point_rows, self.point_cols]

        # Write data to file
        # If we are at the first write date, simple write the file (i.e. overwrite possibly
        # existing files), otherwise merge the already written dataset with the in-memory one.
        if date == ds.indexes['time'][-1]:
            self.model.logger.debug(f'Writing point outputs')

            if self.format == 'netcdf':
                filename = self.model.config.results_dir / 'point_outputs.nc'

                if date == self.write_dates[0]:
                    ds.to_netcdf(filename)
                else:
                    with xr.open_dataset(filename) as old_ds:
                        ds_merge = xr.concat([old_ds, ds], 'time')

                    ds_merge.to_netcdf(filename)
            elif self.format == 'csv':
                for point in self.points:
                    filename = self.model.config.results_dir / f'point_output_{point.name}.csv'
                    df = ds.sel(point=point.name).to_dataframe().drop(columns=[
                        'point',
                        'lon',
                        'lat',
                        'alt',
                        'x',
                        'y',
                    ])

                    if date == self.write_dates[0]:
                        df.to_csv(filename)
                    else:
                        df.to_csv(filename, mode='a', header=False)

            self.data = None
