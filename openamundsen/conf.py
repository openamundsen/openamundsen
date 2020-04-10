import datetime
from openamundsen import util
import pandas as pd
from pathlib import Path
import re


def read_config(filename):
    """
    Read a configuration (YAML) file and return the resulting dict.
    """
    return util.read_yaml_file(filename)


def full_config(config):
    """
    Convert a configuration dict into a "full" configuration, i.e. fill unspecified
    values with the respective values from the default configuration.
    """
    return util.merge_data(DEFAULT_CONFIG, config)


def parse_config(config):
    end_date = config['end_date']

    # If end_date is specified without an hour value, the end hour should be inferred
    # (i.e., set to the latest time step of the end day).
    if isinstance(end_date, datetime.date):
        infer_end_hour = True
    elif isinstance(end_date, str) and re.match(r'^\d\d\d\d-\d\d-\d\d$', end_date.strip()):
        infer_end_hour = True
    else:
        infer_end_hour = False

    config['start_date'] = pd.to_datetime(config['start_date'])
    config['end_date'] = pd.to_datetime(end_date)

    # If no end hour is specified (only the date), set it to the last time step of the respective day
    # (for the start date the hour is automatically set to 0 if not explicitly specified)
    if infer_end_hour:
        config['end_date'] += pd.Timedelta(hours=24) - pd.Timedelta(seconds=config['timestep'])

    return config


# read in the default configuration from the module directory
module_dir = Path(__file__).parent
DEFAULT_CONFIG = util.read_yaml_file(f'{module_dir}/data/defaultconfig.yml')
