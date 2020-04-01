from setuptools import setup, find_packages


setup(
    name='openamundsen',
    version='0.0.1',
    description='A spatially distributed snow and hydrological modeling framework',
    packages=find_packages(),
    include_package_data=True,
    scripts=['bin/openamundsen'],
    install_requires=[
        'loguru>=0.3.2',
        'munch>=2.5.0',
        'netCDF4>=1.5.2',
        'numpy>=1.17.2',
        'pandas>=0.25.1',
        'ruamel.yaml>=0.15.0',
        'rasterio>=1.1.0',
        'xarray>=0.14.0',
    ],
)
