import logging
import os


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger that writes to disk and stdout.

    Ensures a consistent format across agents, reuses handlers when possible,
    and stores logs in ``logs/system.log`` so downstream reviewers can audit
    runs.

    Parameters
    ----------
    name:
        Logical identifier for the component requesting logging.

    Returns
    -------
    logging.Logger
        A logger primed with file and console handlers.
    """

    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger(name)
    if not logger.handlers:  # avoid duplicate handlers


        file_handler = logging.FileHandler("logs/system.log")
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)


        logger.setLevel(logging.INFO)
    return logger
