import time
from app.logger_config import setup_logger
from contextlib import contextmanager
import logging

logger = setup_logger(__name__, logging.DEBUG)

@contextmanager
def time_block(label: str = "⏱️ elapsed"):
    start = time.time()
    yield
    end = time.time()
    elapsed = end - start
    logger.debug(f"[{label}] {elapsed:.4f} seconds")

def time_function(label: str = None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            name = label or func.__name__
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            logger.debug(f"[⏱️ {name}] {elapsed:.4f} seconds")
            return result
        return wrapper
    return decorator