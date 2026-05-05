"""Helper utilities."""

import logging


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get a configured logger with console output.

    Args:
        name: Logger name, typically __name__ from the calling module.
        level: Logging level (default: INFO).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level)
    return logger
