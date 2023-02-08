import cerberus
import datetime
import json
from munch import Munch
from openamundsen import constants, util
from openamundsen.errors import ConfigurationError
import pandas as pd
from pathlib import Path
import re

try:
    import openamundsen_snowmanagement
    SNOW_MANAGEMENT_AVAILABLE = True
    SNOW_MANAGEMENT_DATA_DIR = Path(openamundsen_snowmanagement.__file__).parent / 'data'
except ImportError:
    SNOW_MANAGEMENT_AVAILABLE = False

DATA_DIR = Path(__file__).parent / 'data'


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

    def _normalize_coerce_meteo_interpolation(self, d):
        # 'wind_speed' is deprecated -> change to 'wind'
        if 'wind_speed' in d:
            d['wind'] = d.pop('wind_speed')

        return d

    def _normalize_coerce_cloudiness(self, d):
        # 'day_method' is deprecated -> change to 'method'
        if 'day_method' in d:
            d['method'] = d.pop('day_method')

        # 'night_method' is deprecated -> change to 'clear_sky_fraction_night_method'
        if 'night_method' in d:
            d['clear_sky_fraction_night_method'] = d.pop('night_method')

        return d


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
        end_date += pd.Timedelta(hours=24) - timedelta

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
    _init_schemas()
    v = ConfigurationValidator(cerberus.schema_registry.get('openamundsen_config'))
    valid = v.validate(config)

    if not valid:
        raise ConfigurationError('Invalid configuration\n\n' + util.to_yaml(v.errors))

    full_config = Configuration.fromDict(v.document)
    full_config['end_date'] = parse_end_date(full_config['end_date'], full_config['timestep'])

    full_config['land_cover']['classes'] = _merge_land_cover_params(
        full_config['land_cover']['classes']
    )

    validate_config(full_config)

    return full_config


def _init_schemas():
    """
    Read in the validation schemas and add them to the Cerberus registry.
    """
    schema_reg = cerberus.schema_registry
    schemas = schema_reg.all()

    if 'openamundsen_config' not in schemas:
        config_schema = util.read_yaml_file(f'{DATA_DIR}/configschema.yml')

        if SNOW_MANAGEMENT_AVAILABLE:
            sm_schema = util.read_yaml_file(f'{SNOW_MANAGEMENT_DATA_DIR}/configschema.yml')
            config_schema.update(sm_schema)

        schema_reg.add('openamundsen_config', config_schema)

        lcc_schema = util.read_yaml_file(f'{DATA_DIR}/land_cover_class_schema.yml')
        schema_reg.add('openamundsen_land_cover_class', lcc_schema)


def _merge_land_cover_params(class_params):
    """
    Merge the manually set land cover class parameters with the default parameters.
    """
    default_config = read_config(f'{DATA_DIR}/land_cover_class_params.yml')

    dict_schema = {
        'classes': {
            'type': 'dict',
            'valuesrules': {
                'schema': 'openamundsen_land_cover_class',
            },
        },
    }
    v = cerberus.Validator(dict_schema)
    valid = v.validate(default_config)

    if not valid:
        raise ConfigurationError('Invalid land cover configuration\n\n' + util.to_yaml(v.errors))

    default_class_params = Configuration.fromDict(v.document)['classes']
    merged_params = default_class_params.copy()

    for lcc, lcc_params in class_params.items():
        if lcc in merged_params:
            for k, v in lcc_params.items():
                merged_params[lcc][k] = v
        else:
            merged_params[lcc] = lcc_params

    return merged_params


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

    if config.input_data.meteo.format == 'memory':
        if (
            config.input_data.meteo.bounds != 'grid'
            or len(config.input_data.meteo.exclude) > 0
            or len(config.input_data.meteo.include) > 0
        ):
            print('Warning: "bounds", "exclude" and "include" are currently ignored for '
                  'format "memory"')

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

    if SNOW_MANAGEMENT_AVAILABLE:
        openamundsen_snowmanagement.validate_config(config)
