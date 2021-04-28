from distutils.util import convert_path
from setuptools import setup, find_packages


long_description = """
openAMUNDSEN is a modular snow and hydroclimatological modeling framework written in Python.

<p align="center">
  <img src="https://user-images.githubusercontent.com/17546246/115751189-3afe4c00-a399-11eb-8bfa-87d0a86c2119.gif" />
</p>

openAMUNDSEN is a fully distributed model, designed primarily for resolving the mass and energy
balance of snow and ice covered surfaces in mountain regions.
Typically, it is applied in areas ranging from the point scale to the regional scale (i.e., up to
some hundreds to thousands of square kilometers), using a spatial resolution of 10–100 m and a
temporal resolution of 1–3 h, however its potential applications are very versatile.
"""

version_ns = {}
version_file = convert_path('openamundsen/_version.py')
with open(version_file) as f:
    exec(f.read(), version_ns)

setup(
    name='openamundsen',
    version=version_ns['__version__'],
    author='openAMUNDSEN Developers',
    author_email='florian.hanzer@gmail.com',
    description='Modular snow and hydroclimatological modeling framework',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/openamundsen/openamundsen',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    packages=find_packages(),
    include_package_data=True,
    scripts=['bin/openamundsen'],
    install_requires=[
        'cerberus',
        'loguru',
        'munch',
        'netCDF4',
        'numba>=0.50.1',
        'numpy',
        'pandas',
        'pyproj',
        'scipy',
        'ruamel.yaml>=0.15.0',
        'rasterio>=1.1.0',
        'xarray>=0.14.0',
    ],
    extras_require={
        'liveview': [
            'matplotlib>=3.0.0',
            'PyQt5>=5.12',
        ],
        'test': [
            'pytest',
            'pvlib',
        ],
        'docs': [
            'sphinx',
            'sphinx_rtd_theme',
        ],
    },
)
