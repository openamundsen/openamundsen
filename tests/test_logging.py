import io
import logging

import openamundsen as oa

from .conftest import base_config


def _default_handlers(logger):
    return [
        handler
        for handler in logger.handlers
        if getattr(handler, "_openamundsen_default_handler", False)
    ]


class _LogStateRestorer:
    def __init__(self):
        self.package_logger = logging.getLogger("openamundsen")
        self.root_logger = logging.getLogger()
        self.package_handlers = list(self.package_logger.handlers)
        self.root_handlers = list(self.root_logger.handlers)
        self.package_level = self.package_logger.level
        self.root_level = self.root_logger.level
        self.package_propagate = self.package_logger.propagate

    def restore(self):
        for handler in _default_handlers(self.package_logger):
            handler.close()

        self.package_logger.handlers[:] = self.package_handlers
        self.root_logger.handlers[:] = self.root_handlers
        self.package_logger.setLevel(self.package_level)
        self.root_logger.setLevel(self.root_level)
        self.package_logger.propagate = self.package_propagate


def test_default_logging_adds_single_handler():
    state = _LogStateRestorer()
    try:
        package_logger = logging.getLogger("openamundsen")
        root_logger = logging.getLogger()
        package_logger.handlers[:] = [
            handler
            for handler in package_logger.handlers
            if isinstance(handler, logging.NullHandler)
        ]
        root_logger.handlers[:] = []

        config = base_config()
        config.enable_default_logging = True
        model = oa.OpenAmundsen(config)

        model.configure_logger()
        assert len(_default_handlers(package_logger)) == 1
        assert not package_logger.propagate

        model.configure_logger()
        assert len(_default_handlers(package_logger)) == 1
    finally:
        state.restore()


def test_disabled_default_logging_keeps_propagation():
    state = _LogStateRestorer()
    try:
        package_logger = logging.getLogger("openamundsen")
        root_logger = logging.getLogger()
        package_logger.handlers[:] = [
            handler
            for handler in package_logger.handlers
            if isinstance(handler, logging.NullHandler)
        ]
        root_logger.handlers[:] = []

        config = base_config()
        config.enable_default_logging = False
        model = oa.OpenAmundsen(config)
        model.configure_logger()

        assert _default_handlers(package_logger) == []
        assert package_logger.propagate

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        package_logger.addHandler(handler)
        package_logger.propagate = False
        logging.getLogger("openamundsen.test").info("plain log message")
        assert "\033[" not in stream.getvalue()
    finally:
        state.restore()


def test_host_logging_configuration_disables_default_logging():
    state = _LogStateRestorer()
    try:
        package_logger = logging.getLogger("openamundsen")
        root_logger = logging.getLogger()
        package_logger.handlers[:] = [
            handler
            for handler in package_logger.handlers
            if isinstance(handler, logging.NullHandler)
        ]
        root_logger.handlers[:] = []

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)

        config = base_config()
        config.enable_default_logging = True
        model = oa.OpenAmundsen(config)
        model.configure_logger()

        assert _default_handlers(package_logger) == []
        assert package_logger.propagate

        logging.getLogger("openamundsen.test").info("host-managed log message")
        assert "host-managed log message" in stream.getvalue()
        assert "\033[" not in stream.getvalue()
    finally:
        state.restore()
