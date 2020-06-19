import datetime
from munch import Munch
from openamundsen import util
from openamundsen.errors import ConfigurationError
import pandas as pd
from pathlib import Path
import re


class Configuration(Munch):
    """
    Container for storing model configuration. This class inherits from `Munch`
    so that attributes are accessible both using dict notation
    (`config['start_date']`) as well as dot notation (`config.end_date`).
    """
    pass


def read_config(filename):
    """
    Read a configuration (YAML) file and return the resulting dict as a
    Configuration object.
    """
    return Configuration.fromDict(util.read_yaml_file(filename))


def full_config(config):
    """
    Convert a configuration dict into a "full" configuration, i.e. fill unspecified
    values with the respective values from the default configuration.
    """
    return Configuration.fromDict(util.merge_data(DEFAULT_CONFIG, config))


def parse_config(config):
    if config.domain is None:
        raise ConfigurationError('Domain not specified')

    if config.start_date is None:
        raise ConfigurationError('Start date not specified')

    if config.end_date is None:
        raise ConfigurationError('End date not specified')

    if config.resolution is None:
        raise ConfigurationError('Resolution not specified')

    end_date = config.end_date

    # If end_date is specified without an hour value, the end hour should be inferred
    # (i.e., set to the latest time step of the end day).
    if isinstance(end_date, datetime.date):
        infer_end_hour = True
    elif isinstance(end_date, str) and re.match(r'^\d\d\d\d-\d\d-\d\d$', end_date.strip()):
        infer_end_hour = True
    else:
        infer_end_hour = False

    config.start_date = pd.to_datetime(config.start_date)
    config.end_date = pd.to_datetime(end_date)

    # If no end hour is specified (only the date), set it to the last time step of the respective day
    # (for the start date the hour is automatically set to 0 if not explicitly specified)
    if infer_end_hour:
        config.end_date += pd.Timedelta(hours=24) - pd.Timedelta(seconds=config.timestep)

    if config.results_dir is None:
        config.results_dir = Path('.')
    else:
        config.results_dir = Path(config.results_dir)

    return config


# read in the default configuration from the module directory
module_dir = Path(__file__).parent
DEFAULT_CONFIG = util.read_yaml_file(f'{module_dir}/data/defaultconfig.yml')
