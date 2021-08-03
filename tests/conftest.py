import openamundsen as oa
from pathlib import Path
import pooch
import pytest
import textwrap

version = oa.__version__
if '+' in version:
    data_url = 'https://github.com/openamundsen/testdata/raw/{version}/'
else:
    data_url = 'https://github.com/openamundsen/testdata/raw/v{version}/'

data_fetcher = pooch.create(
    path=pooch.os_cache('openamundsen-tests'),
    base_url=data_url,
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

_BASE_CONFIG_YAML = textwrap.dedent(f'''
    domain: rofental
    start_date: 2020-01-15
    end_date: 2020-01-15
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

      grids:
        format: memory
''')
_BASE_CONFIG = oa.parse_config(oa.Configuration.from_yaml(_BASE_CONFIG_YAML))


def pytest_addoption(parser):
    parser.addoption('--run-slow', action='store_true', default=False, help='run slow tests')
    parser.addoption(
        '--prepare-comparison-data',
        action='store_true',
        default=False,
        help='prepare baseline model results for comparisons',
    )
    parser.addoption(
        '--comparison-data-dir',
        type=str,
        default=None,
        help='baseline model results directory',
    )
    parser.addoption(
        '--skip-comparisons',
        action='store_true',
        default=False,
        help='skip comparisons',
    )
    parser.addoption(
        '--reports-dir',
        type=str,
        default=None,
        help='directory for writing comparison reports',
    )


def pytest_configure(config):
    pytest.DATA_DIR = DATA_DIR
    config.addinivalue_line('markers', 'slow: marks test as slow')
    config.addinivalue_line('markers', 'comparison: marks test as comparing with baseline data')


def pytest_collection_modifyitems(config, items):
    run_slow = config.getoption('--run-slow')
    skip_slow = pytest.mark.skip(reason='need --run-slow option to run')
    skip_comparisons = pytest.mark.skipif(
        config.getoption('--skip-comparisons'),
        reason='skipping comparisons',
    )

    for item in items:
        if 'slow' in item.keywords and not run_slow:
            item.add_marker(skip_slow)
        elif 'comparison' in item.keywords:
            item.add_marker(skip_comparisons)


def pytest_sessionstart(session):
    fetch_data_files()


def fetch_data_files():
    for file in data_fetcher.registry_files:
        data_fetcher.fetch(file)


@pytest.fixture(scope='session')
def prepare_comparison_data(request):
    return request.config.getoption('--prepare-comparison-data')


@pytest.fixture(scope='session')
def comparison_data_dir(request, prepare_comparison_data):
    d = request.config.getoption('--comparison-data-dir')

    if d is None and prepare_comparison_data:
        raise Exception('--comparison-data-dir must be specified when '
                        '--prepare-comparison-data is set')

    if d is not None:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)

    return d


@pytest.fixture(scope='session')
def reports_dir(request):
    d = request.config.getoption('--reports-dir')

    if d is not None:
        d = Path(d)

    return d


def base_config():
    return _BASE_CONFIG.copy()
