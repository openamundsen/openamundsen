from .conftest import base_config
from openamundsen.util import to_yaml
import subprocess


def test_cli(tmp_path):
    config = base_config()
    config.start_date = '2020-01-18 00:00'
    config.end_date = '2020-01-18 00:00'
    config.results_dir = tmp_path

    config_file = tmp_path / 'config.yml'
    with open(config_file, 'w') as f:
        f.write(to_yaml(config.toDict()))

    subprocess.check_call(['openamundsen', str(config_file)])
