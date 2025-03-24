import time
from contextlib import contextmanager

@contextmanager
def time_block(label: str = "⏱️ elapsed"):
    start = time.time()
    yield
    end = time.time()
    elapsed = end - start
    print(f"[{label}] {elapsed:.4f} seconds")

def time_function(label: str = None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            name = label or func.__name__
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            print(f"[⏱️ {name}] {elapsed:.4f} seconds")
            return result
        return wrapper
    return decorator