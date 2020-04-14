import copy
import numpy as np
from openamundsen import errors
from pathlib import Path
import pyproj
from ruamel.yaml import YAML


def create_empty_array(shape, dtype):
    """
    Create an empty array with a given shape and dtype initialized to "no
    data". The value of "no data" depends on the dtype and is e.g. NaN for
    float, 0 for int and False for float.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g. (2, 3).

    dtype : type
        The desired data type for the array.

    Returns
    -------
    out : ndarray
    """
    dtype_init_vals = {
        float: np.nan,
        int: 0,
        bool: False,
    }

    return np.full(shape, dtype_init_vals[dtype], dtype=dtype)


def merge_data(a, b):
    """
    Recursively merge b into a and return the result.
    Based on https://stackoverflow.com/a/15836901/1307576.

    Parameters
    ----------
    a : dict, list or primitive (str, int, float)
    b : dict, list or primitive (str, int, float)

    Returns
    -------
    result : same dtype as `a`
    """
    a = copy.deepcopy(a)

    try:
        if a is None or isinstance(a, (str, int, float)):
            a = b
        elif isinstance(a, list):
            # lists are appended
            if isinstance(b, list):
                # merge lists
                a.extend(b)
            else:
                # append to list
                a.append(b)
        elif isinstance(a, dict):
            # dicts are merged
            if isinstance(b, dict):
                for key in b:
                    if key in a:
                        a[key] = merge_data(a[key], b[key])
                    else:
                        a[key] = b[key]
            else:
                raise errors.YamlReaderError(f'Cannot merge non-dict "{b}" into dict "{a}"')
        else:
            raise errors.YamlReaderError(f'Merging "{b}" into "{a}" is not implemented')
    except TypeError as e:
        raise errors.YamlReaderError(f'TypeError "{e}" in key "{key}" when merging "{b}" into "{a}"')

    return a


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
    yaml = YAML(typ='safe')

    with open(filename) as f:
        return yaml.load(f.read())


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
