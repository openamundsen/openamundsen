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
        'grids/rofental/lc_rofental_1000.asc': '5bde204bf9e79acdc47f9d9c70f21f20f669cfb7cbe65418ef42dd0a52676b98',
        'grids/rofental/roi_rofental_1000.asc': '5c495833e8465931d7aedc982ef093332572034d9cf7da5fc8b98b72111e3d25',
        'grids/rofental/soil_rofental_1000.asc': '48b46a8b3216a012393f36e34561473b3c1fd8dccef86d9a7e34aab9e3deb89b',
        'grids/rofental/srf_rofental_1000.asc': '431f0ed2181abc2a79c6ffd79b804e5c983c9c47003bce57b853db2b41abf62d',
        'grids/rofental/svf_rofental_1000.asc': 'a4b7704fad6f7c63445a76c64207ff881b73023ec7d958d4581cd7fb42d06966',
        'meteo/rofental/convert_csv_to_netcdf.py': 'f4e7aeb04d3eff7f1bdc6c4619600c4ab8dc560ffbc8feab83a30d200400f49f',
        'meteo/rofental/csv/bellavista.csv': '0c760f490aa7dc8d6af9da8987a6cf59807ec3e41c802a9c73b7773d269ec603',
        'meteo/rofental/csv/latschbloder.csv': 'f9e089bca0e7c931df51e1c624dc4985089c20c802d43de8b4c5175353a00931',
        'meteo/rofental/csv/proviantdepot.csv': '0e3a7eb07f9f127f0a257b17c62442e91860fb5a64f57387da547f69dcb61b6f',
        'meteo/rofental/csv/stations.csv': '175b420d9659d3478d4755fc4b8d8c2569400b37d84ffb9906e1ab661afe5dc4',
        'meteo/rofental/netcdf/bellavista.nc': '0fdeb860bce28a287bd48d2fc95f42a07ed842e9a0f0a0045f658605df2ff564',
        'meteo/rofental/netcdf/latschbloder.nc': 'cc96e0dbc4024b093fcc161d8f2c862bb8abd379a7d3dbd540d78242f17aef75',
        'meteo/rofental/netcdf/proviantdepot.nc': 'f8a276c98343c2574fb810eb4b3ee1b08bd29d9f5d0cfaa5a1104f1aa25af374',
        'results/canopy_point.nc': 'f62055c50b9ef2d73691107fb69a3ee33ce14996ca4c0372d5c7518af7131bdf',
        'results/snow_cryolayers_point.nc': '64cae07460342169425bd945afebde02b4e1e7460f2bcc6664b96ace3833178e',
        'results/snow_multilayer_point.nc': '3127dc48fa71f9096830174baefbd6fa83332c732d48f76c7df3335f223bc275',
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
        '--skip-comparisons',
        action='store_true',
        default=False,
        help='skip comparisons',
    )
    parser.addoption(
        '--prepare-comparison-data',
        action='store_true',
        default=False,
        help='prepare baseline model results for comparisons',
    )
    parser.addoption(
        '--comparison-data-dir',
        type=str,
        help='baseline model results directory',
    )
    parser.addoption(
        '--reports-dir',
        type=str,
        help='directory for writing comparison reports',
    )


def pytest_configure(config):
    config.addinivalue_line('markers', 'slow: marks test as slow')
    config.addinivalue_line('markers', 'comparison: marks test as comparing with baseline data')

    prepare_comparison_data = config.getoption('--prepare-comparison-data')

    comp_data_dir = config.getoption('--comparison-data-dir')
    if comp_data_dir is None:
        if prepare_comparison_data:
            raise Exception('--comparison-data-dir must be specified when '
                            '--prepare-comparison-data is set')
        else:
            comp_data_dir = DATA_DIR / 'results'
    if comp_data_dir is not None:
        comp_data_dir = Path(comp_data_dir)
        comp_data_dir.mkdir(parents=True, exist_ok=True)

    reports_dir = config.getoption('--reports-dir')
    if reports_dir is not None:
        reports_dir = Path(reports_dir)

        try:
            import plotly
        except ImportError:
            raise ImportError('plotly is required for report creation')

    pytest.DATA_DIR = DATA_DIR
    pytest.COMPARISON_DATA_DIR = comp_data_dir
    pytest.REPORTS_DIR = reports_dir
    pytest.PREPARE_COMPARISON_DATA = prepare_comparison_data


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


def base_config():
    return _BASE_CONFIG.copy()
