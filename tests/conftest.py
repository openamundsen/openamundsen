import openamundsen as oa
import pooch


def pytest_sessionstart(session):
    fetch_data_files()


def fetch_data_files():
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
            'meteo/rofental/netcdf/bellavista.nc': '0ae9350ec9b092724775496a0c449f27d825cb30e7059ede206ca26032a0fd6a',
            'meteo/rofental/netcdf/latschbloder.nc': '0a885ebd2b70818a4e6c30498346827e61cd2e983a311eb9fc5b3257f5b01bd2',
            'meteo/rofental/netcdf/proviantdepot.nc': 'ecb70cdd0bbd1d87d77fcc0bebb3f28212b2aeae687380cf3a67c1b954f29c91',
        },
    )

    for file in data_fetcher.registry_files:
        data_fetcher.fetch(file)
