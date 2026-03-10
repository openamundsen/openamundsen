from dataclasses import dataclass
from pathlib import Path, PosixPath, WindowsPath
from typing import Union

import numpy as np
import pandas as pd
import pandas.tseries.frequencies
import pyproj
import rasterio
import ruamel.yaml
import xarray as xr
from munch import Munch

from openamundsen import constants


class ConfigurationYAML(ruamel.yaml.YAML):
    def __init__(self):
        super().__init__(typ="rt")  # .indent() works only with the roundtrip dumper
        self.default_flow_style = False
        self.indent(mapping=2, sequence=4, offset=2)

        self.representer.add_representer(pd.Timestamp, self._repr_datetime)

        # Add representers for path objects (just using pathlib.Path does not
        # work, so have to add PosixPath and WindowsPath separately)
        self.representer.add_representer(PosixPath, self._repr_path)
        self.representer.add_representer(WindowsPath, self._repr_path)

    def _repr_datetime(self, representer, date):
        return representer.represent_str(str(date))

    def _repr_path(self, representer, path):
        return representer.represent_str(str(path))

    def dump(self, data, stream=None, **kw):
        inefficient = False

        if stream is None:
            inefficient = True
            stream = ruamel.yaml.compat.StringIO()

        super().dump(data, stream, **kw)

        if inefficient:
            return stream.getvalue()


yaml = ConfigurationYAML()


def read_yaml_file(filename):
    """
    Read a YAML file.

    Parameters
    ----------
    filename : str

    Returns
    -------
    result : dict
    """
    with open(filename) as f:
        return load_yaml(f.read())


def load_yaml(s):
    return yaml.load(s)


def to_yaml(d):
    return yaml.dump(d)


def raster_filename(kind, config):
    """
    Return the filename of an input raster file for a model run.

    Parameters
    ----------
    kind : str
        Type of input file, e.g. 'dem' or 'roi'.

    config : dict
        Model run configuration.

    Returns
    -------
    file : pathlib.Path
    """
    grids_dir = config["input_data"]["grids"]["dir"]
    domain = config["domain"]
    resolution = config["resolution"]
    extension = "asc"
    return Path(f"{grids_dir}/{kind}_{domain}_{resolution}.{extension}")


def convert_roi_pixel_dataset_to_grid(ds: xr.Dataset) -> xr.Dataset:
    """
    Reconstruct a gridded-output dataset with ``layout="roi_pixel"`` to ``layout="grid"``.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset in ROI-pixel layout as produced by openAMUNDSEN gridded output.

    Returns
    -------
    xr.Dataset
        Dataset with all ``pixel`` dimensions expanded to ``y``/``x``.
    """
    if ds.attrs.get("openamundsen_output_layout") != "roi_pixel":
        raise ValueError('Dataset does not use the "roi_pixel" output layout')

    nrows = int(ds.attrs["nrows"])
    ncols = int(ds.attrs["ncols"])
    resolution = ds.attrs["resolution"]
    xllcorner = ds.attrs["xllcorner"]
    yllcorner = ds.attrs["yllcorner"]
    pixel_idxs = ds.pixel.values

    x_coords = xllcorner + resolution * (np.arange(ncols) + 0.5)
    y_coords = yllcorner + resolution * (nrows - np.arange(nrows) - 0.5)

    coords = {}
    for name, coord in ds.coords.items():
        if name == "pixel":
            continue
        coords[name] = (coord.dims, coord.values, coord.attrs)

    coords["x"] = (
        ["x"],
        x_coords,
        {
            "standard_name": "projection_x_coordinate",
            "long_name": "x coordinate of projection",
            "units": "m",
        },
    )
    coords["y"] = (
        ["y"],
        y_coords,
        {
            "standard_name": "projection_y_coordinate",
            "long_name": "y coordinate of projection",
            "units": "m",
        },
    )

    data_vars = {}
    for name, da in ds.data_vars.items():
        if "pixel" not in da.dims:
            data_vars[name] = (da.dims, da.values, da.attrs)
            continue

        pixel_axis = da.get_axis_num("pixel")
        values = np.moveaxis(da.values, pixel_axis, -1)
        fill_value = da.encoding.get("_FillValue", da.attrs.get("_FillValue", np.nan))
        full_values = np.full((*values.shape[:-1], nrows, ncols), fill_value, dtype=da.dtype)
        full_values.reshape(*values.shape[:-1], -1)[..., pixel_idxs] = values

        target_order = (
            *range(pixel_axis),
            values.ndim - 1,
            values.ndim,
            *range(pixel_axis, values.ndim - 1),
        )
        full_values = np.transpose(full_values, axes=target_order)

        dims = (*da.dims[:pixel_axis], "y", "x", *da.dims[pixel_axis + 1 :])
        data_vars[name] = (dims, full_values, da.attrs)

    attrs = dict(ds.attrs)
    attrs["openamundsen_output_layout"] = "grid"
    for attr in ("nrows", "ncols", "resolution", "xllcorner", "yllcorner"):
        del attrs[attr]

    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)


def transform_coords(x, y, src_crs, dst_crs):
    """
    Transform coordinates from one coordinate reference system to another.

    Parameters
    ----------
    x : ndarray
        Source x coordinates or longitude.

    y : ndarray
        Source y coordinates or latitude.

    src_crs
        CRS of the original coordinates as a pyproj-compatible input
        (e.g. a string "epsg:xxxx").

    dst_crs
        Target CRS.

    Returns
    -------
    (x, y) tuple of ndarrays containing the transformed coordinates.
    """
    x = np.array(x)
    y = np.array(y)
    transformer = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return transformer.transform(x, y)


class ModelGrid(Munch):
    """
    Container for storing model grid related variables.
    """

    def prepare_coordinates(self):
        """
        Prepare a range of variables related to the grid coordinates:
        - xs, ys: 1d-arrays containing the x and y coordinates in the grid CRS.
        - X, Y, 2d-arrays containing the x and y coordinates for each grid point.
        - all_points: (N, 2)-array containing (x, y) coordinates of all grid points.
        - roi_xs, roi_ys: 1d-arrays containing the x and y coordinates of all ROI points.
        - roi_points: (N, 2)-array containing (x, y) coordinates of all ROI points.
        - roi_idxs: (N, 2)-array containing (row, col) indexes of all ROI points.
        - roi_idxs_flat: 1d-array containing the flattened (1d) indexes of all ROI points
        """
        transform = self.transform
        if transform.a < 0 or transform.e > 0:
            raise NotImplementedError  # we only allow left-right and top-down oriented grids

        x_min = transform.xoff
        x_max = x_min + self.cols * self.resolution
        y_max = transform.yoff
        y_min = y_max - self.rows * self.resolution

        x_range, y_range = rasterio.transform.xy(
            self.transform,
            [0, self.rows - 1],
            [0, self.cols - 1],
        )
        xs = np.linspace(x_range[0], x_range[1], self.cols)
        ys = np.linspace(y_range[0], y_range[1], self.rows)
        X, Y = np.meshgrid(xs, ys)

        center_x = xs.mean()
        center_y = ys.mean()
        center_lon, center_lat = transform_coords(
            center_x,
            center_y,
            self.crs,
            constants.CRS_WGS84,
        )

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.xs = xs
        self.ys = ys
        self.X = X
        self.Y = Y
        self.all_points = np.column_stack((X.flat, Y.flat))
        self.center_lon = center_lon
        self.center_lat = center_lat

        self.extended_grid = Munch(
            available=False,
            rows=None,
            cols=None,
            row_offset=None,
            col_offset=None,
            row_slice=None,
            col_slice=None,
            dem=None,
            svf=None,
            normal_vec=None,
            shadows=None,
        )
        # ("shadows" required because the shadows are calculated in
        # clear_sky_shortwave_irradiance(), but are required again later in shortwave_irradiance()
        # for calculating the clear sky irradiance for the extended-grid stations)

        self.prepare_roi_coordinates()

    def prepare_roi_coordinates(self):
        """
        Update the roi_points and roi_idxs variables using the ROI field.
        """
        if "roi" not in self:
            self.roi = np.ones((self.rows, self.cols), dtype=bool)

        self.roi_xs = self.X[self.roi]
        self.roi_ys = self.Y[self.roi]
        self.roi_points = np.column_stack((self.roi_xs, self.roi_ys))
        self.roi_idxs = np.array(np.where(self.roi)).T
        self.roi_idxs_flat = np.where(self.roi.flat)[0]


def to_offset(offset: Union[str, pd.offsets.BaseOffset]) -> pd.offsets.BaseOffset:
    """
    Convert a pandas-compatible offset (e.g. '3h') to a DateOffset object.
    """
    # As of pandas 2.2.0, some frequency aliases have been deprecated (see
    # https://pandas.pydata.org/docs/dev/whatsnew/v2.2.0.html). In order to avoid deprecation
    # warnings in case the old aliases are still used in openAMUNDSEN configurations, we replace
    # the deprecated aliases "H", "M" and "Y" here with their new versions.
    # Furthermore, we parse the offsets "ME" and "YE" (which have been introduced in pandas 2.2.0)
    # here manually in order to make them work also with earlier pandas versions.
    # Also, as of pandas 3.0.0, the "d" offset has been deprecated in favor of "D" and also now
    # always represents calendar days instead of 24-hour spans
    # (https://pandas.pydata.org/docs/dev/whatsnew/v3.0.0.html#changed-behavior-of-pd-offsets-day-to-always-represent-calendar-day).
    # In openAMUNDSEN we assume days to be 24-hour spans, so we replace "<n>d" offsets here with
    # "<n*24>h".
    if isinstance(offset, str):
        if offset[-1].upper() == "D":
            # Replace "<n>D" with "<n*24>h" and "D" with "24h".
            num_days_str = offset[:-1]
            num_days = int(num_days_str) if len(num_days_str) > 0 else 1
            offset = f"{24 * num_days}h"
        elif offset[-1] == "H":
            offset = offset[:-1] + "h"
        elif offset[-1] == "M":
            offset = offset[:-1] + "ME"
        elif offset[-1] == "Y":
            offset = offset[:-1] + "YE"

        if offset == "ME":
            offset = pd.offsets.MonthEnd()
        elif offset == "YE":
            offset = pd.offsets.YearEnd()

    return pandas.tseries.frequencies.to_offset(offset)


def offset_to_timedelta(offset: Union[str, pd.offsets.BaseOffset]) -> pd.Timedelta:
    """
    Convert a pandas-compatible offset (e.g. '3h') to a Timedelta object.
    """
    return pd.to_timedelta(to_offset(offset))


@dataclass(frozen=True)
class TimestepProperties:
    first_of_run: bool
    strict_first_of_year: bool
    strict_first_of_month: bool
    strict_first_of_day: bool
    first_of_year: bool
    first_of_month: bool
    first_of_day: bool
    last_of_run: bool
    strict_last_of_year: bool
    strict_last_of_month: bool
    strict_last_of_day: bool
    last_of_year: bool
    last_of_month: bool
    last_of_day: bool


def normalize_array(data, min, max):  # noqa: A002
    """
    Normalize an array within a range.

    Parameters
    ----------
    data : array-like
        Data values.

    min : numeric
       Minimum of the normalized array.

    max : numeric
       Maximum of the normalized array.

    Returns
    -------
    data_norm : np.array
        Data normalized within [min, max].
    """
    data = np.asarray(data)
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)
    scale_factor = (max - min) / (data_max - data_min)
    return min + scale_factor * (data - data_min)
