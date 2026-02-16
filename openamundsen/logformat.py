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


class ColoredFormatter(logging.Formatter):
    """Formatter that replicates the loguru-style colored output."""

    def format(self, record):
        timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        level_color = _LEVEL_COLORS.get(record.levelname, "")
        level = f"{record.levelname: <8}"
        message = record.getMessage()
        return (
            f"{_GREEN}{timestamp}{_RESET} | "
            f"{level_color}{level}{_RESET} | "
            f"{level_color}{message}{_RESET}"
        )
