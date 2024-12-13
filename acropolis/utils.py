# math
from math import log, pow, exp
# numpy
import numpy as np
# scipy
from scipy.integrate import cumulative_simpson

# jit
from acropolis.jit import jit


class LogInterp(object):

    def __init__(self, x_grid, y_grid, base=np.e, fill_value=None):
        if not ( np.all(y_grid < 0) or np.all(y_grid > 0) ):
            raise ValueError(
                "The values in y_grid need to be either all positive or all negative."
            )
        if not np.all(x_grid > 0):
            raise ValueError(
                "The values in x_grid need to be all positive."
            )
        
        self._sSign = 1 if y_grid[0] > 0 else -1

        self._sBase    = base
        self._sLogBase = log(self._sBase)

        self._sFillValue = fill_value

        self._sXLog = np.log( np.abs(x_grid) )/self._sLogBase
        self._sYLog = np.log( np.abs(y_grid) )/self._sLogBase

        self._sXminLog = self._sXLog[ 0]
        self._sXmaxLog = self._sXLog[-1]

        xdiff = np.diff(self._sXLog)
        if not np.all( xdiff >= 0 ):
            raise ValueError(
                "The values in x_grid need to be in ascending order."
            )
        if not np.allclose( xdiff, xdiff[0] ):
            raise ValueError(
                "The values in x_grid need to be equidistant in log space."
            )

        self._sN = len(self._sXLog)

        self._sCache = {}


    def _perform_interp(self, x):
        x_log = log(x)/self._sLogBase

        if not (self._sXminLog <= x_log <= self._sXmaxLog):
            if type(self._sFillValue) is int:
                return self._sFillValue
            
            if type(self._sFillValue) in [list, tuple] and len(self._sFillValue) == 2:
                if x_log < self._sXminLog:
                    return self._sFillValue[0]
                
                if x_log > self._sXmaxLog:
                    return self._sFillValue[1]
            
            raise ValueError(
                    "The given value does not lie within the interpolation range."
                )

        ix = int( ( x_log - self._sXminLog )*( self._sN - 1 )/( self._sXmaxLog - self._sXminLog ) )

        # Handle the case for which ix+1 is out-of-bounds
        if ix == self._sN - 1:
            ix -= 1

        x1_log, x2_log = self._sXLog[ix], self._sXLog[ix+1]
        y1_log, y2_log = self._sYLog[ix], self._sYLog[ix+1]

        m = ( y2_log - y1_log )/( x2_log - x1_log )
        b = y2_log - m*x2_log

        return self._sSign * pow( self._sBase, m*x_log + b )


    def __call__(self, x):
        if x not in self._sCache:
            self._sCache[x] = self._perform_interp(x)

        return self._sCache[x]


# Cummulative numerical Simpson integration
def cumsimp(x_grid, y_grid):
    return cumulative_simpson(x_grid*y_grid, x=np.log(x_grid), initial=0.)


@jit
def _cumsimp(x_grid, y_grid):
    n = len(x_grid)

    dz = log(x_grid[-1]/x_grid[0])/(n-1)

    g_grid = x_grid*y_grid
    i_grid = np.zeros(n)

    last_even_int = 0.
    for i in range(1, int(n/2 + 1)):
        ie = 2 * i
        io = 2 * i - 1

        i_grid[io] = last_even_int + .5 * dz * (g_grid[io-1] + g_grid[io])
        if ie < n:
            i_grid[ie] = last_even_int + dz * (g_grid[ie-2] + 4.*g_grid[ie-1] + g_grid[ie])/3.
            last_even_int = i_grid[ie]

    return i_grid


@jit
def flipped_cumsimp(x_grid, y_grid):
     return -np.flip(_cumsimp(np.flip(x_grid), np.flip(y_grid)))


def mavg(x_grid, n, use_log=False):
    if n <= 3:
        return x_grid

    N = len(x_grid)

    z_grid = np.log(x_grid) if use_log else x_grid

    # Initialize the return array
    xa_grid = np.zeros(N)

    # -->
    xa_grid[ 0] = x_grid[ 0]
    xa_grid[-1] = x_grid[-1]

    # Loop over all elements
    for i in range(1, N-1):
        l = i - n
        h = i + n

        d = 0
        # Handle the edge cases
        if l < 0:
            d = abs(l)
        elif h > N - 1:
            d = abs(N - 1 - h)
        
        # Adjust the window size
        h -= d
        l += d
        
        # Calculate the mean of the symmetric window
        mean = np.mean(z_grid[l:h])

        # -->
        xa_grid[i] = exp(mean) if use_log else mean

    return xa_grid