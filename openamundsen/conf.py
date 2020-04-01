import datetime
import pandas as pd
from pathlib import Path
from . import util


def read_config(filename):
    """
    Read a config (YAML) file and return the resulting dict.

    Parameters
    ----------
    filename : str

    Returns
    -------
    config : dict
    """
    return util.read_yaml_file(filename)


def full_config(config):
    """
    Convert a configuration dict into a "full" configuration, i.e. fill unspecified
    values with the respective values from the default configuration.

    Parameters
    ----------
    config : dict

    Returns
    -------
    config : dict
    """
    return util.merge_data(DEFAULT_CONFIG, config)


def apply_config(model, config):
    model.logger.info('Checking configuration')

    infer_end_hour = isinstance(config['end_date'], datetime.date)
    config['start_date'] = pd.to_datetime(config['start_date'])
    config['end_date'] = pd.to_datetime(config['end_date'])

    # If no end hour is specified (only the date), set it to the last time step of the respective day
    # (for the start date the hour is automatically set to 0 if not explicitly specified)
    if infer_end_hour:
        config['end_date'] += pd.Timedelta(hours=24) - pd.Timedelta(seconds=config['timestep'])

    model.config = config


# read in the default configuration from the module directory
module_dir = Path(__file__).parent
DEFAULT_CONFIG = util.read_yaml_file(f'{module_dir}/data/defaultconfig.yaml')
