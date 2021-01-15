# functools
from functools import wraps


def cached_member(f_uncached):
    # Define the cache as a dictionary
    cache = {}
    cT = {"_": -1.}

    # Define the wrapper function
    @wraps(f_uncached)
    def f_cached(*args):
        T     = args[-1]
        # Drop the first argument 'self'
        # ! only for member functions !
        pargs = args[1:]

        # For each new temperature,
        # clear the cache and start over
        if T != cT["_"]:
            cT["_"] = T
            cache.clear()

        if pargs not in cache:
            cache[pargs] = f_uncached(*args)

        return cache[pargs]

    return f_cached
