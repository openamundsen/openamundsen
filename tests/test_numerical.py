from .conftest import base_config
from openamundsen.util import to_yaml
import os
import pytest
import subprocess
import textwrap
import xarray as xr


def test_numba(tmp_path):
    config = base_config()
    config.start_date = '2020-01-18'
    config.end_date = '2020-01-18'
    config.results_dir = tmp_path
    config.output_data.timeseries.format = 'netcdf'
    config.output_data.grids.format = 'netcdf'
    config.output_data.grids.variables = [
        {'var': 'meteo.temp'},
        {'var': 'snow.swe'},
        {'var': 'snow.temp', 'name': 'snow_temp'},
    ]

    config_file = tmp_path / 'config.yml'
    with open(config_file, 'w') as f:
        f.write(to_yaml(config.toDict()))

    subprocess.check_call(['openamundsen', str(config_file)])
    ds_points_numba = xr.load_dataset(tmp_path / 'output_timeseries.nc')
    ds_grids_numba = xr.load_dataset(tmp_path / 'output_grids.nc')

    env = os.environ.copy()
    env['NUMBA_DISABLE_JIT'] = '1'
    subprocess.check_call(['openamundsen', str(config_file)], env=env)
    ds_points_nonumba = xr.load_dataset(tmp_path / 'output_timeseries.nc')
    ds_grids_nonumba = xr.load_dataset(tmp_path / 'output_grids.nc')

    assert ds_points_numba.equals(ds_points_nonumba)
    assert ds_grids_numba.equals(ds_grids_nonumba)


@pytest.mark.slow
def test_floating_errors(tmp_path):
    config = base_config()
    config.results_dir = tmp_path

    config_file = tmp_path / 'config.yml'
    with open(config_file, 'w') as f:
        f.write(to_yaml(config.toDict()))

    with open(tmp_path / 'run.py', 'w') as f:
        f.write(textwrap.dedent(
            f'''
            import numpy as np
            np.seterr(all='raise')

            import openamundsen as oa

            config = oa.read_config('{config_file}')
            model = oa.OpenAmundsen(config)
            model.initialize()
            model.run()
            '''
        ))

    env = os.environ.copy()
    env['NUMBA_DISABLE_JIT'] = '1'
    subprocess.check_call(['python', tmp_path / 'run.py'], env=env)
