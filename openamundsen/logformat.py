import logging

_LEVEL_COLORS = {
    "DEBUG": "\033[34m",  # blue
    "INFO": "\033[1m",  # bold
    "WARNING": "\033[33m",  # yellow
    "ERROR": "\033[31m",  # red
    "CRITICAL": "\033[31;1m",  # red bold
}
_RESET = "\033[0m"
_GREEN = "\033[32m"
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class ColoredFormatter(logging.Formatter):
    """Formatter that replicates the loguru-style colored output."""

    def format(self, record):
        timestamp = self.formatTime(record, _DATE_FORMAT)
        level_color = _LEVEL_COLORS.get(record.levelname, "")
        level = f"{record.levelname: <8}"
        message = record.getMessage()
        text = (
            f"{_GREEN}{timestamp}{_RESET} | "
            f"{level_color}{level}{_RESET} | "
            f"{level_color}{message}{_RESET}"
        )

        if record.exc_info:
            text += "\n" + self.formatException(record.exc_info)

        if record.stack_info:
            text += "\n" + self.formatStack(record.stack_info)

        return text


def _use_colors(stream):
    return hasattr(stream, "isatty") and stream.isatty()


def create_default_stream_handler(stream=None):
    handler = logging.StreamHandler(stream)
    if _use_colors(handler.stream):
        formatter = ColoredFormatter()
    else:
        formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    handler.setFormatter(formatter)
    handler._openamundsen_default_handler = True
    return handler
