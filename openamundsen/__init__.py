from .conf import Configuration, parse_config, read_config
from .model import OpenAmundsen, Model
from . import constants, errors, terrain


# Get version (method as used by matplotlib: https://github.com/matplotlib/matplotlib/blob/bcc1ce8461f5b6e874baaaa02ef776d0243a4abe/lib/matplotlib/__init__.py#L133-L151)
def __getattr__(name):
    if name == '__version__':
        from pathlib import Path
        import setuptools_scm
        global __version__
        root = Path(__file__).resolve().parents[1]
        if (root / '.git').exists() and not (root / '.git/shallow').exists():
            __version__ = setuptools_scm.get_version(
                root=root,
                version_scheme='post-release',
                fallback_version='0.0.0+UNKNOWN',
            )
        else:
            try:
                from . import _version
                __version__ = _version.version
            except ImportError:
                __version__ = '0.0.0+UNKNOWN'
        return __version__
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


__all__ = [
    'OpenAmundsen',
    'Configuration',
    'parse_config',
    'read_config',
]
