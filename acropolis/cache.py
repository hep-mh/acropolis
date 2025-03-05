# functools
from functools import wraps


def cached_member(clear_by_index=-1):
    def decorator(f):
        # Define the cache as a dictionary
        cache = {}

        xc = None
        # Define the wrapper function
        @wraps(f)
        def wrapper(*args):
            nonlocal xc

            x     = args[clear_by_index]
            # Drop the first argument 'self'
            # ! only for member functions !
            pargs = args[1:]

            # Define a key for the cache
            cache_key = (f.__name__, pargs)

            # For each new temperature,
            # clear the cache and start over
            if x != xc:
                xc = x
                # -->
                cache.clear()

            if cache_key not in cache:
                cache[cache_key] = f(*args)

            return cache[cache_key]

        return wrapper
    
    return decorator
