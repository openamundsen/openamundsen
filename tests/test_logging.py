import io
import logging
import logging.handlers
import multiprocessing
import queue

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


def _run_model_with_log_capture(args):
    model, log_queue = args

    def create_queue_handler(stream=None):
        handler = logging.handlers.QueueHandler(log_queue)
        handler._openamundsen_default_handler = True
        return handler

    oa.logformat.create_default_stream_handler = create_queue_handler
    model.run()


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
        assert package_logger.level == logging.NOTSET

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        package_logger.addHandler(handler)
        logging.getLogger("openamundsen.test").info("plain log message")
        assert "\033[" not in stream.getvalue()
    finally:
        state.restore()


def test_package_logging_configuration_disables_default_logging():
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
        package_logger.addHandler(handler)
        package_logger.setLevel(logging.INFO)

        config = base_config()
        config.enable_default_logging = True
        model = oa.OpenAmundsen(config)
        model.configure_logger()

        assert _default_handlers(package_logger) == []
        assert package_logger.propagate
        assert package_logger.level == logging.INFO

        logging.getLogger("openamundsen.test").info("host-managed log message")
        assert "host-managed log message" in stream.getvalue()
        assert "\033[" not in stream.getvalue()
    finally:
        state.restore()


def test_root_logging_configuration_keeps_root_log_level():
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
        root_logger.setLevel(logging.WARNING)

        config = base_config()
        config.enable_default_logging = False
        config.log_level = "INFO"
        model = oa.OpenAmundsen(config)
        model.configure_logger()

        assert package_logger.level == logging.NOTSET

        logging.getLogger("openamundsen.test").info("suppressed info message")
        logging.getLogger("openamundsen.test").warning("visible warning message")

        assert "suppressed info message" not in stream.getvalue()
        assert "visible warning message" in stream.getvalue()
    finally:
        state.restore()


def test_multiprocessing_logging():
    config = base_config()
    model = oa.OpenAmundsen(config)
    model.initialize()

    spawn_context = multiprocessing.get_context("spawn")
    log_queue = spawn_context.Queue()
    process = spawn_context.Process(target=_run_model_with_log_capture, args=((model, log_queue),))
    process.start()
    process.join()

    assert process.exitcode == 0

    messages = []
    while True:
        try:
            record = log_queue.get_nowait()
        except queue.Empty:
            break
        else:
            messages.append(record.getMessage())

    assert messages.count("Starting model run") == 1
    assert sum(message.startswith("Processing time step ") for message in messages) == len(
        model.dates
    )
    assert sum(message.startswith("Model run finished. Runtime: ") for message in messages) == 1
