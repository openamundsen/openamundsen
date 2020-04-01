import copy
from dataclasses import dataclass
import loguru
from munch import Munch
import numpy as np
from openamundsen import errors
from ruamel.yaml import YAML
import sys
from . import dataio


class StateVariableContainer(Munch):
    """
    Container for storing state variables. This class inherits from `Munch` so
    that attributes are accessible both using dict notation (`state['temp']`)
    as well as dot notation (`state.temp`).
    """
    pass


@dataclass(frozen=True)
class StateVariableDefinition:
    """
    Class for describing metadata of a state variable.

    Parameters
    ----------
    dtype : str, default 'float'
        Data type for the variable values in `np.dtype`-compatible notation.

    standard_name : str, optional
        Standard name of the variable in CF notation
        (http://cfconventions.org/standard-names.html).

    long_name : str, optional
        Long descriptive name of the variable.

    units : str, optional
        udunits-compatible definition of the units of the given variable.

    Examples
    --------
    >>> StateVariableDefinition(
            standard_name='precipitation_flux',
            long_name='Total precipitation flux',
            units='kg m-2 s-1')
    """
    dtype: str = 'float'
    standard_name: str = None
    long_name: str = None
    units: str = None


def initialize_logger(model):
    """Create and initialize the logger for a Model instance."""
    loguru.logger.remove()
    logger = copy.deepcopy(loguru.logger)
    log_format = '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>'
    logger.add(sys.stderr, format=log_format, filter='openamundsen', level='DEBUG')
    model.logger = logger


def initialize_model_grid(model):
    """
    Initialize the grid parameters (number of rows and columns, transformation
    parameters) for a Model instance by reading the DEM file associated to the
    model run.
    """
    model.logger.info('Initializing model grid')

    dem_file = dataio.raster_filename('dem', model.config)
    meta = dataio.read_raster_metadata(dem_file)

    model.config['rows'] = meta['rows']
    model.config['cols'] = meta['cols']
    model.config['raster_meta'] = meta

    model.logger.info(f'Grid has dimensions {meta["rows"]}x{meta["cols"]}')


def add_state_variable(model, category, var_name, definition=None):
    """
    Add a state variable to a Model instance with optional metadata.
    Each state variable is associated with a category (e.g., 'base', 'meteo',
    'snow') and is subsequently mapped to the respective `model.state` entries
    (e.g., `model.state['base']`).
    The actual arrays are however not created here yet (this is done in
    `ìnitialize_state_variables`), only the categories, variable names and
    definitions are stored in the hidden `_state_variable_definitions`
    attribute of the Model instance.

    Parameters
    ----------
    category : str
        Category to which the variable is associated.

    var_name : str
        Desired variable name. Since the `model.state` entries are
        `StateVariableContainer` instances, the variables are later accessible
        both via dict notation (`model.state['meteo']['temp']`) as well as dot
        notation (`model.state.meteo.temp`).

    definition : StateVariableDefinition, optional
        Metadata (e.g., data type, units, CF-compliant standard_name)
        associated with the variable.
    """
    if definition is None:
        definition = StateVariableDefinition()  # empty definition only specifying the default dtype

    model.logger.debug(f'Adding state variable: category={category} var={var_name} definition={definition}')

    if category not in model._state_variable_definitions:
        model._state_variable_definitions[category] = {}

    model._state_variable_definitions[category][var_name] = definition


def add_default_state_variables(model):
    """
    Add all state variables to a Model instance which are required for any
    model run. Depending on which submodules are activated in the run
    configuration, further state variables might be added in other locations.
    """
    # Base variables
    model.add_state_variable(
        'base',
        'dem',
        StateVariableDefinition(standard_name='surface_altitude', units='m'),
    )
    model.add_state_variable('base', 'slope')
    model.add_state_variable('base', 'aspect')
    model.add_state_variable(
        'base',
        'roi',
        StateVariableDefinition(dtype='bool'),
    )

    # Meteorological variables
    model.add_state_variable(
        'meteo',
        'temp',
        StateVariableDefinition(standard_name='air_temperature', units='K'),
    )
    model.add_state_variable(
        'meteo',
        'precip',
        StateVariableDefinition(standard_name='precipitation_flux', units='kg m-2 s-1'),
    )
    model.add_state_variable(
        'meteo',
        'hum',
        StateVariableDefinition(standard_name='relative_humidity', units='%'),
    )
    model.add_state_variable(
        'meteo',
        'glob',
        StateVariableDefinition(
            standard_name='surface_downwelling_shortwave_flux_in_air',
            units='W m-2',
        ),
    )
    model.add_state_variable(
        'meteo',
        'wind_speed',
        StateVariableDefinition(
            standard_name='wind_speed',
            units='m s-1',
        ),
    )

    # Snow variables
    model.add_state_variable(
        'snow',
        'swe',
        StateVariableDefinition(
            standard_name='liquid_water_content_of_surface_snow',
            units='kg m-2',
        ),
    )
    model.add_state_variable(
        'snow',
        'depth',
        StateVariableDefinition(
            standard_name='surface_snow_thickness',
            units='m',
        ),
    )

    # ...


def create_empty_array(shape, dtype):
    """
    Create an empty array with a given shape and dtype initialized to "no
    data". The value of "no data" depends on the dtype and is e.g. NaN for
    float, 0 for int and False for float.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g. (2, 3).

    dtype : str
        The desired data type for the array.

    Returns
    -------
    out : ndarray
    """
    dtype_init_vals = {
        'float': np.nan,
        'int': 0,
        'bool': False,
    }

    return np.full(shape, dtype_init_vals[dtype], dtype=dtype)


def initialize_state_variables(model):
    """
    Initialize the state variables (i.e., create the actual arrays) of a Model
    instance. The arrays are created according to the variable names and data
    types specified in the respective `add_state_variable` calls.
    """
    model.logger.info('Initializing state variables')

    rows = model.config['rows']
    cols = model.config['cols']
    var_defs = model._state_variable_definitions

    model.state = StateVariableContainer()

    for category in var_defs.keys():
        model.state[category] = StateVariableContainer()

        for var_name, var_def in var_defs[category].items():
            model.state[category][var_name] = create_empty_array((rows, cols), var_def.dtype)


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
