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

        return oa.conf.parse_config(oa.read_config(f.name))


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


def test_infer_end_date(minimal_config):
    config = save_and_parse_config(minimal_config)
    assert config['end_date'].hour == 23

    minimal_config['end_date'] = '2020-04-30 11:00'
    config = save_and_parse_config(minimal_config)
    assert config['end_date'].hour == 11

    minimal_config['end_date'] = ' 2020-04-30 '
    config = save_and_parse_config(minimal_config)
    assert config['end_date'].hour == 23

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


def test_missing_parameter(minimal_config):
    for key in minimal_config.keys():
        config = minimal_config.copy()
        del config[key]

        with pytest.raises(errors.ConfigurationError):
            save_and_parse_config(config)
