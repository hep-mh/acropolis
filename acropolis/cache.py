# functools
from functools import wraps


def cached(f):
    # Define the cache as a dictionary
    cache = {}
    Tc = {"_": -1.}

    # Define the wrapper function
    @wraps(f)
    def f_cached(*args):
        T     = args[-1]
        # Drop the first argument 'self'
        # ! only for member functions !
        pargs = args[1:]

        # For each new temperature,
        # clear the cache and start over
        if T != Tc["_"]:
            Tc["_"] = T
            cache.clear()

        if pargs not in cache:
            cache[pargs] = f(*args)

        return cache[pargs]

    return f_cached
