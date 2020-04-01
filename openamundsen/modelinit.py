import copy
import sys
import loguru
import pandas as pd
from . import dataio
from .util import StateVariableDefinition, StateVariableContainer, create_empty_array


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


def prepare_time_steps(config):
    """
    Prepare the time steps for a model run according to the run configuration
    (start date, end date, and time step).

    Parameters
    ----------
    config : dict
        Model run configuration.

    Returns
    -------
    dates : pd.DatetimeIndex
    """
    return pd.date_range(
        start=config['start_date'],
        end=config['end_date'],
        freq=pd.DateOffset(seconds=config['timestep']),
    )


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