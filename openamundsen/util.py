import copy
import collections
import loguru
from munch import Munch
import numpy as np
from openamundsen import errors
from ruamel.yaml import YAML
import sys
from . import dataio
from . import util


class StateVariableContainer(Munch):
    """
    Container for storing state variables. This class inherits from `Munch` so
    that attributes are accessible both using dict notation (`state['temp']`)
    as well as dot notation (`state.temp`).
    """
    pass


def initialize_logger(model):
    """Initialize the logger for a Model instance."""
    loguru.logger.remove()
    logger = copy.deepcopy(loguru.logger)
    log_format = '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>'
    logger.add(sys.stderr, format=log_format, filter='openamundsen', level='DEBUG')
    model.logger = logger


def initialize_model_grid(model):
    """
    Initialize the grid parameters (number of rows and columns, transformation
    parameters) for a Model instance by reading the associated DEM file.
    """
    model.logger.info('Initializing model grid')

    dem_file = dataio.raster_filename('dem', model.config)
    meta = dataio.read_raster_metadata(dem_file)

    model.config['rows'] = meta['rows']
    model.config['cols'] = meta['cols']
    model.config['raster_meta'] = meta

    model.logger.info(f'Grid has dimensions {meta["rows"]}x{meta["cols"]}')


def initialize_state_variables(model):
    """Initialize the state variables of a Model instance."""
    model.logger.debug('Initializing state variables')

    arr = np.full((model.config['rows'], model.config['cols']), np.nan)

    model.state = util.StateVariableContainer()
    model.state.base = util.StateVariableContainer()
    model.state.meteo = util.StateVariableContainer()
    model.state.snow = util.StateVariableContainer()
    model.state.surface = util.StateVariableContainer()

    model.state.base.dem = arr.copy()
    model.state.base.slope = arr.copy()
    model.state.base.aspect = arr.copy()
    model.state.base.roi = arr.copy().astype(bool)
    model.state.base.landcover = arr.copy().astype(int)
    model.state.base.soil = arr.copy().astype(int)
    model.state.base.catchments = arr.copy().astype(int)

    model.state.meteo.temp = arr.copy()  # air temperature (K)
    model.state.meteo.precip = arr.copy()  # precipitation (mm h-1)
    model.state.meteo.hum = arr.copy()  # relative humidity (%)
    model.state.meteo.glob = arr.copy()  # global radiation (W m-2)
    model.state.meteo.windspeed = arr.copy()  # wind speed (m s-1)

    model.state.snow.swe = arr.copy()  # snow water equivalent (kg m-2)
    model.state.snow.depth = arr.copy()  # snow depth (m)
    model.state.snow.albedo = arr.copy()  # snow albedo

    model.state.surface.temp = arr.copy()  # surface temperature (K)
    model.state.surface.albedo = arr.copy()  # surface albedo


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


