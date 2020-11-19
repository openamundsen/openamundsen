from copy import deepcopy
import openamundsen as oa
import openamundsen.errors as errors
import pytest
from ruamel.yaml import YAML
import tempfile
from textwrap import dedent


def save_and_read_yaml_dict(d):
    yaml = YAML()

    with tempfile.NamedTemporaryFile() as f:
        yaml.dump(d, f)
        return oa.read_config(f.name)


def save_and_parse_config(config):
    with tempfile.NamedTemporaryFile('w+') as f:
        if isinstance(config, dict):
            yaml = YAML()
            yaml.dump(config, f)
        elif isinstance(config, str):
            f.write(config)
            f.flush()
        else:
            raise Exception('Unsupported type')

        return oa.parse_config(oa.read_config(f.name))


@pytest.fixture(scope='function')
def minimal_config():
    return {
        'domain': 'dummy',
        'start_date': '2019-11-01',
        'end_date': '2020-04-30',
        'resolution': 50,
        'timezone': 1,
    }


def test_config_equal():
    config = {
        'domain': 'dummy',
        'start_date': '2019-11-01',
        'end_date': '2020-04-30',
        'resolution': 50,
        'timezone': 1,

        'input_data': {
            'grids': {
                'dir': 'dummydir',
            },

            'meteo': {
                'dir': 'meteodummydir',
                'format': 'netcdf',
            },
        },
    }

    config_read = save_and_read_yaml_dict(config)

    assert config == config_read


def test_read_minimal_config(minimal_config):
    save_and_parse_config(minimal_config)


def test_read_config(minimal_config):
    config = save_and_parse_config(minimal_config)
    assert config['end_date'].hour == 23

    yaml_str = dedent("""
        domain: dummy
        start_date: 2019-11-01
        end_date: 2020-12-31
        resolution: 50
        timezone: 1
    """)
    config = save_and_parse_config(yaml_str)
    assert config['end_date'].hour == 23


def test_start_end_date_order(minimal_config):
    mc = deepcopy(minimal_config)
    mc['start_date'] = '2020-01-01'
    mc['end_date'] = '2020-01-01'
    oa.parse_config(mc)
    mc['timestep'] = 'D'
    oa.parse_config(mc)

    mc = deepcopy(minimal_config)
    mc['start_date'] = '2020-01-01 00:00'
    mc['end_date'] = '2020-01-01 00:00'
    oa.parse_config(mc)

    mc = deepcopy(minimal_config)
    mc['start_date'] = '2020-01-01'
    mc['end_date'] = '2019-12-31'
    with pytest.raises(errors.ConfigurationError):
        oa.parse_config(mc)

    mc = deepcopy(minimal_config)
    mc['start_date'] = '2020-01-01 01:00'
    mc['end_date'] = '2020-01-01 00:00'
    with pytest.raises(errors.ConfigurationError):
        oa.parse_config(mc)


def test_infer_end_date(minimal_config):
    config = oa.parse_config(minimal_config)
    assert config['end_date'].hour == 23

    mc = deepcopy(minimal_config)
    mc['end_date'] = '2020-04-30 11:00'
    config = oa.parse_config(mc)
    assert config['end_date'].hour == 11

    mc = deepcopy(minimal_config)
    mc['end_date'] = ' 2020-04-30 '
    config = oa.parse_config(mc)
    assert config['end_date'].hour == 23

    mc = deepcopy(minimal_config)
    mc['timestep'] = 'D'
    config = oa.parse_config(mc)
    assert config['end_date'].hour == 0

    yaml_str = dedent("""
        domain: dummy
        start_date: 2019-11-01
        end_date: {d}
        resolution: 50
        timezone: 1
    """)

    config = save_and_parse_config(yaml_str.format(d='2020-04-30'))
    assert config['end_date'].hour == 23

    config = save_and_parse_config(yaml_str.format(d='"2020-04-30"'))
    assert config['end_date'].hour == 23

    config = save_and_parse_config(yaml_str.format(d='2020-04-30 11:00'))
    assert config['end_date'].hour == 11

    config = save_and_parse_config(yaml_str.format(d='"2020-04-30 11:00"'))
    assert config['end_date'].hour == 11


def test_timesteps(minimal_config):
    mc = deepcopy(minimal_config)
    mc['start_date'] = '2019-11-01 03:00'
    oa.parse_config(mc)

    mc = deepcopy(minimal_config)
    mc['start_date'] = '2019-11-01 00:15'
    mc['end_date'] = '2019-11-01 23:45'
    mc['timestep'] = '15min'
    oa.parse_config(mc)

    mc = deepcopy(minimal_config)
    mc['start_date'] = '2019-11-01 02:00'
    mc['end_date'] = '2019-11-01 22:00'
    mc['timestep'] = '3H'
    with pytest.raises(errors.ConfigurationError):
        oa.parse_config(mc)

    mc = deepcopy(minimal_config)
    mc['end_date'] = '2019-12-31 15:00'
    mc['timestep'] = 'D'
    with pytest.raises(errors.ConfigurationError):
        oa.parse_config(mc)

    mc = deepcopy(minimal_config)
    mc['start_date'] = '2019-12-31 01:23'
    mc['end_date'] = '2019-12-31 03:45'
    mc['timestep'] = 'H'
    with pytest.raises(errors.ConfigurationError):
        oa.parse_config(mc)


def test_missing_parameter(minimal_config):
    for key in minimal_config.keys():
        config = deepcopy(minimal_config)
        del config[key]

        with pytest.raises(errors.ConfigurationError):
            oa.parse_config(config)
