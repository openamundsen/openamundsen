import openamundsen as oa
import copy
import pooch
import pytest
import textwrap

version = oa.__version__
data_fetcher = pooch.create(
    path=pooch.os_cache('openamundsen-tests'),
    base_url='https://github.com/openamundsen/testdata/raw/{version}/',
    version=version,
    version_dev='main',
    registry={  # hashes generated with generate_hashes.py
        'grids/rofental/dem_rofental_1000.asc': '368ee374bfbb927bae5408a38dbafbbeab4ff59dc94ae30956b617bf28c114ec',
        'grids/rofental/roi_rofental_1000.asc': '5c495833e8465931d7aedc982ef093332572034d9cf7da5fc8b98b72111e3d25',
        'grids/rofental/srf_rofental_1000.asc': '431f0ed2181abc2a79c6ffd79b804e5c983c9c47003bce57b853db2b41abf62d',
        'grids/rofental/svf_rofental_1000.asc': 'a4b7704fad6f7c63445a76c64207ff881b73023ec7d958d4581cd7fb42d06966',
        'meteo/rofental/csv/bellavista.csv': 'fab913929aa84e04a5c47c9c4ea89ed613b8ecc14cf54194a090600de4ce9d98',
        'meteo/rofental/csv/latschbloder.csv': '2d6a8a6297464d36ff9ccfc34ce4cd44a50f398c35de1a6e32e8922871503ff0',
        'meteo/rofental/csv/proviantdepot.csv': '7a068c921127c5e0f15ed3718c3a257c70146111ae8a1c5e97f0f8db1e60d2bc',
        'meteo/rofental/csv/stations.csv': '175b420d9659d3478d4755fc4b8d8c2569400b37d84ffb9906e1ab661afe5dc4',
        'meteo/rofental/netcdf/bellavista.nc': '96e5fc1af44f269015e52c2ec2a18f2e8d729401b5998d4fdbebb3b0014c8c58',
        'meteo/rofental/netcdf/latschbloder.nc': '0a885ebd2b70818a4e6c30498346827e61cd2e983a311eb9fc5b3257f5b01bd2',
        'meteo/rofental/netcdf/proviantdepot.nc': 'ecb70cdd0bbd1d87d77fcc0bebb3f28212b2aeae687380cf3a67c1b954f29c91',
    },
)
DATA_DIR = data_fetcher.abspath


def pytest_addoption(parser):
    parser.addoption('--run-slow', action='store_true', default=False, help='run slow tests')


def pytest_configure(config):
    pytest.DATA_DIR = DATA_DIR
    config.addinivalue_line('markers', 'slow: mark test as slow to run')


def pytest_collection_modifyitems(config, items):
    if config.getoption('--run-slow'):
        return

    skip_slow = pytest.mark.skip(reason='need --run-slow option to run')

    for item in items:
        if 'slow' in item.keywords:
            item.add_marker(skip_slow)


def pytest_sessionstart(session):
    fetch_data_files()


def fetch_data_files():
    for file in data_fetcher.registry_files:
        data_fetcher.fetch(file)


@pytest.fixture(scope='session')
def base_config():
    config_yaml = textwrap.dedent(f'''
        domain: rofental
        start_date: 2020-01-15
        end_date: 2020-04-30
        resolution: 1000
        timestep: 3H
        crs: "epsg:32632"
        timezone: 1

        input_data:
          grids:
            dir: {DATA_DIR}/grids/rofental
          meteo:
            dir: {DATA_DIR}/meteo/rofental/netcdf

        output_data:
          timeseries:
            format: memory

            variables:
              - var: snow.num_layers
                name: num_snow_layers
              - var: snow.albedo
                name: snow_albedo

          grids:
            format: memory
    ''')

    config = oa.Configuration.from_yaml(config_yaml)
    full_config = oa.parse_config(config)

    return full_config


@pytest.fixture(scope='session')
def base_config_point_results(base_config):
    model = oa.OpenAmundsen(base_config)
    model.initialize()
    model.run()
    return model.point_output.data


@pytest.fixture(scope='function')
def base_config_single_point_results(base_config_point_results):
    return base_config_point_results.sel(point='proviantdepot')


@pytest.fixture(scope='function')
def single_point_results_multilayer(base_config_single_point_results):
    return base_config_single_point_results
