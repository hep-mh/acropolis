# functools
from functools import wraps


# ATTENTION: Only use on member functions with the
# argument structure (self,...,T, enforce_thomson)
def cached_rate_or_kernel(f_uncached):
    # Define the cache as a dictionary
    cache = {}
    Tc = {"_": -1.}

    # Define the wrapper function
    @wraps(f_uncached)
    def f_cached(*args):
        N = len(args)

        # Calculate the offset based on the position
        # if 'T': If 'enforce_thomson' is present,
        # it needs to be dropped
        offset = 1 if type(args[-1]) == bool else 0

        T     = args[-1-offset]
        # Drop the first argument 'self', as well
        # as the last argument 'enforce_thomson'
        # if necessary
        pargs = args[1:N-offset]

        # For each new temperature,
        # clear the cache and start over
        if T != Tc["_"]:
            Tc["_"] = T
            cache.clear()

        if pargs not in cache:
            cache[pargs] = f_uncached(*args)

        return cache[pargs]

    return f_cached
