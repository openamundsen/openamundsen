# Check for Python version
import sys
if sys.version_info < (3, 6):
    sys.exit('openAMUNDSEN requires Python 3.6+.')
del sys


from .conf import Configuration, parse_config, read_config
from .model import OpenAmundsen, Model
from . import constants, errors, terrain
from ._version import __version__


__all__ = [
    'OpenAmundsen',
    'Configuration',
    'parse_config',
    'read_config',
]
