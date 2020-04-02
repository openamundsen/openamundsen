import copy
import sys
import loguru
import pandas as pd
from . import dataio
from .util import StateVariableContainer, create_empty_array


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


def initialize_state_variables(model):
    """
    Initialize the default state variables (i.e., create empty arrays) of a
    Model instance. Depending on which submodules are activated in the run
    configuration, further state variables might be added at other locations.
    """
    model.logger.info('Initializing state variables')

    rows = model.config['rows']
    cols = model.config['cols']

    def field(dtype=float):
        return create_empty_array((rows, cols), dtype)

    model.state = StateVariableContainer()

    # Base variables
    base = StateVariableContainer()
    base.dem = field()  # terrain elevation (m)
    base.slope = field()  # terrain slope
    base.aspect = field()  # terrain aspect
    base.roi = field(bool)  # region of interest

    # Meteorological variables
    meteo = StateVariableContainer()
    meteo.temp = field()  # air temperature (K)
    meteo.precip = field()  # precipitation (kg m-2 s-1)
    meteo.hum = field()  # relative humidity (%)
    meteo.glob = field()  # shortwave incoming radiation (W m-2)
    meteo.wind_speed = field()  # wind speed (m s-1)

    # Snow variables
    snow = StateVariableContainer()
    snow.swe = field()
    snow.depth = field()

    model.state.base = base
    model.state.meteo = meteo
    model.state.snow = snow
