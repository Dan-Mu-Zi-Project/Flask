import logging
import sys

def setup_logger(name: str, level=None):
    try:
        from flask import current_app
        if level is None:
            level = current_app.config.get("LOG_LEVEL", logging.INFO)
    except Exception:
        level = level or logging.INFO

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        handler.setLevel(level)
        logger.addHandler(handler)

    logger.propagate = False
    return logger
