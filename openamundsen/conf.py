import cerberus
import datetime
import json
from munch import Munch
from openamundsen import constants, util
from openamundsen.errors import ConfigurationError
import pandas as pd
from pathlib import Path
import re


class ConfigurationValidator(cerberus.Validator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _normalize_coerce_float(self, value):
        return float(value)

    def _normalize_coerce_datetime(self, date):
        return pd.to_datetime(date)

    def _normalize_coerce_path(self, path):
        return Path(path)

    def _normalize_coerce_snowmodel(self, model):
        # Snow model name 'layers' is deprecated -> change to 'multilayer'
        if model == 'layers':
            return 'multilayer'

        return model


class ConfigurationEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, pd.Timestamp):
            return str(o)
        else:
            raise TypeError


def parse_end_date(end_date, timestep):
    # If end_date is specified without an hour value, the end hour should be inferred
    # (i.e., set to the latest time step of the end day).
    if type(end_date) is datetime.date:  # do not use isinstance() because we want to catch only datetime.date and not its subclasses (datetime.datetime, pd.Timestamp etc.)
        infer_end_hour = True
    elif isinstance(end_date, str) and re.match(r'^\d\d\d\d-\d\d-\d\d$', end_date.strip()):
        infer_end_hour = True
    else:
        infer_end_hour = False

    end_date = pd.to_datetime(end_date)

    # If no end hour is specified (only the date), set it to the last time step
    # of the respective day (for the start date the hour is automatically set
    # to 0 if not explicitly specified)
    timedelta = util.offset_to_timedelta(timestep)
    if infer_end_hour and timedelta < pd.Timedelta(days=1):
        end_date += pd.Timedelta(hours=24) - util.offset_to_timedelta(timestep)

    return end_date


class Configuration(Munch):
    """
    Container for storing model configuration. This class inherits from `Munch`
    so that attributes are accessible both using dict notation
    (`config['start_date']`) as well as dot notation (`config.end_date`).
    """
    @classmethod
    def from_dict(cls, d):
        return cls.fromDict(d)

    @classmethod
    def from_yaml(cls, s):
        return cls.from_dict(util.load_yaml(s))

    def __repr__(self):
        return util.to_yaml(self.toDict())


def read_config(filename):
    """
    Read a configuration (YAML) file and return the resulting dict as a
    Configuration object.
    """
    return Configuration.fromDict(util.read_yaml_file(filename))


def parse_config(config):
    module_dir = Path(__file__).parent
    schema = util.read_yaml_file(f'{module_dir}/data/configschema.yml')

    v = ConfigurationValidator(schema)
    valid = v.validate(config)

    if not valid:
        raise ConfigurationError('Invalid configuration\n\n' + util.to_yaml(v.errors))

    full_config = Configuration.fromDict(v.document)
    full_config['end_date'] = parse_end_date(full_config['end_date'], full_config['timestep'])
    validate_config(full_config)

    return full_config


def validate_config(config):
    """
    Perform some additional validations which are too complicated with Cerberus.
    """
    if config.start_date > config.end_date:
        raise ConfigurationError('End date must be after start date')

    # Check if timestep matches start/end dates
    dates = pd.date_range(
        start=config.start_date,
        end=config.end_date,
        freq=config.timestep,
    )
    if dates[-1] != config.end_date:
        raise ConfigurationError('Start/end date is not compatible with timestep')

    # Check if write_freq is compatible with timestep - as long as the time step is <= 1d,
    # write_freq can be an offset like 'M' or 'Y', but for larger timesteps it is not guaranteed
    # that these dates generated with pd.date_range(start=start_date, end=end_date, freq=write_freq)
    # are actually reached, so in this case write_freq must be a multiple of timestep
    # (e.g. timestep = '5D' and write_freq = '30D')
    timestep_td = util.offset_to_timedelta(config.timestep)
    write_freq = config.output_data.timeseries.write_freq
    if timestep_td > pd.Timedelta(days=1):
        try:
            write_freq_td = pd.Timedelta(write_freq)
            if write_freq_td.total_seconds() % timestep_td.total_seconds() != 0:
                raise ConfigurationError('write_freq must be a multiple of timestep')
        except ValueError:
            raise ConfigurationError('write_freq must be a multiple of timestep')

    if config.input_data.meteo.format == 'netcdf' and config.input_data.meteo.crs is not None:
        print('Warning: Ignoring CRS specification for meteo input data '
              '(CRS not required when using NetCDF format)')

    if config.snow.model == 'multilayer' and config.snow.melt.method != 'energy_balance':
        raise ConfigurationError(f'Melt method "{config.snow.melt.method}" not supported for the '
                                 f'snow model "{config.snow.model}"')

    if config.snow.melt.method == 'temperature_index':
        if config.snow.melt.degree_day_factor is None:
            raise ConfigurationError('Missing field: snow.melt.degree_day_factor')
    elif config.snow.melt.method == 'enhanced_temperature_index':
        if config.snow.melt.degree_day_factor is None:
            raise ConfigurationError('Missing field: snow.melt.degree_day_factor')
        if config.snow.melt.albedo_factor is None:
            raise ConfigurationError('Missing field: snow.melt.albedo_factor')

    if abs(config.meteo.precipitation_phase.threshold_temp) < 20:
        print('Warning: precipitation phase threshold temperature seems to be in °C, converting to K')
        config.meteo.precipitation_phase.threshold_temp += constants.T0

    ac = config.snow.albedo
    if ac.method == 'usaco':
        print('Warning: albedo method "usaco" is deprecated, please use "snow_age"')
        ac.method = 'snow_age'
        ac.cold_snow_decay_timescale = 1 / ac.k_neg * constants.HOURS_PER_DAY
        ac.melting_snow_decay_timescale = 1 / ac.k_pos * constants.HOURS_PER_DAY
        ac.decay_timescale_determination_temperature = 'air'
        ac.refresh_snowfall = ac.significant_snowfall
        ac.refresh_method = 'binary'
    elif ac.method == 'fsm':
        print('Warning: albedo method "fsm" is deprecated, please use "snow_age"')
        ac.method = 'snow_age'
        ac.decay_timescale_determination_temperature = 'surface'
        ac.refresh_method = 'continuous'
