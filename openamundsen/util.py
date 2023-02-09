from dataclasses import dataclass
from munch import Munch
import numpy as np
from openamundsen import constants
import pandas as pd
import pandas.tseries.frequencies
from pathlib import Path, PosixPath, WindowsPath
import pyproj
import rasterio
import ruamel.yaml


class ConfigurationYAML(ruamel.yaml.YAML):
    def __init__(self):
        super().__init__(typ='rt')  # .indent() works only with the roundtrip dumper
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
    dir = config['input_data']['grids']['dir']
    domain = config['domain']
    resolution = config['resolution']
    extension = 'asc'
    return Path(f'{dir}/{kind}_{domain}_{resolution}.{extension}')


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

        self.extended_grid = Munch(dict(
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
        ))
        # ("shadows" required because the shadows are calculated in
        # clear_sky_shortwave_irradiance(), but are required again later in shortwave_irradiance()
        # for calculating the clear sky irradiance for the extended-grid stations)

        self.prepare_roi_coordinates()

    def prepare_roi_coordinates(self):
        """
        Update the roi_points and roi_idxs variables using the ROI field.
        """
        if 'roi' not in self:
            self.roi = np.ones((self.rows, self.cols), dtype=bool)

        roi_xs = self.X[self.roi]
        roi_ys = self.Y[self.roi]
        self.roi_points = np.column_stack((roi_xs, roi_ys))
        self.roi_idxs = np.array(np.where(self.roi)).T
        self.roi_idxs_flat = np.where(self.roi.flat)[0]


def offset_to_timedelta(offset):
    """
    Convert a pandas-compatible offset (e.g. '3H') to a Timedelta object.
    """
    return pd.to_timedelta(pandas.tseries.frequencies.to_offset(offset))


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


def normalize_array(data, min, max):
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
