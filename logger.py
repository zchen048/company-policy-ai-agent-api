import logging
import sys
from pathlib import Path

# Directory for log files
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(exist_ok=True)  

def get_logger(name: str) -> logging.Logger:
    """
    Creates and configures a logger for a given module.

    Args:
        name (str): Usually __name__, so logs show the module/file they come from.

    Returns:
        logging.Logger: Configured logger instance.
    """

    logger = logging.getLogger(name)

    # Prevent duplicate handlers if get_logger is called multiple times
    if not logger.handlers:
        fmt = "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"

        # Console handler (prints logs to terminal)
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            fmt=fmt,
            datefmt=datefmt,
        )
        console_handler.setFormatter(console_formatter)

        # File handler (saves logs to file)
        file_handler = logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8")
        file_formatter = logging.Formatter(
            fmt=fmt,
            datefmt=datefmt,
        )
        file_handler.setFormatter(file_formatter)

        # Add both handlers to logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        # Default logging level: DEBUG
        logger.setLevel(logging.DEBUG)

    return logger
