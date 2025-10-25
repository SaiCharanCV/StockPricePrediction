"""Performance monitoring utilities."""
import time
import logging
from functools import wraps
from typing import Any, Callable
from contextlib import contextmanager

logger = logging.getLogger(__name__)

def log_execution_time(func: Callable) -> Callable:
    """
    Decorator to log function execution time.
    
    Args:
        func: Function to be decorated
        
    Returns:
        Wrapped function with execution time logging
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
        return result
    return wrapper

@contextmanager
def timer(description: str = "Operation"):
    """
    Context manager for timing code blocks.
    
    Args:
        description: Description of the operation being timed
    """
    start = time.time()
    yield
    elapsed_time = time.time() - start
    logger.info(f"{description} completed in {elapsed_time:.2f} seconds")