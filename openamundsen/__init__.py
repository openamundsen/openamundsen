from .conf import Configuration, parse_config, read_config
from .model import OpenAmundsen, Model
from . import constants, errors, terrain

# from pkg_resources import get_distribution, DistributionNotFound
# try:
#     __version__ = get_distribution(__name__).version
# except DistributionNotFound:
#     __version__ = '0.0.0'
# del get_distribution, DistributionNotFound

from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("package-name")
except PackageNotFoundError:
    __version__ = '0.0.0'

__all__ = [
    'OpenAmundsen',
    'Configuration',
    'parse_config',
    'read_config',
]
