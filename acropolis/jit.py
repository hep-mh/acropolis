_use_numba_jit = True

# numba
try:
    import numba as nb
except ImportError:
    _use_numba_jit = False


def _null_decorator(func):
    return func


jit_decorator = nb.njit(cache=True) if _use_numba_jit else _null_decorator