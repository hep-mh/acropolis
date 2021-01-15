# math
from math import log, pow
# numpy
import numpy as np


class LogInterp(object):

    def __init__(self, x_grid, y_grid, base=np.e, fill_value=None):
        self._sBase    = base
        self._sLogBase = log(self._sBase)

        self._sFillValue = fill_value

        self._sXLog = np.log(x_grid)/self._sLogBase
        self._sYLog = np.log(y_grid)/self._sLogBase

        self._sXminLog = self._sXLog[ 0]
        self._sXmaxLog = self._sXLog[-1]

        self._sN = len(self._sXLog)

        self._sCache = {}


    def _perform_interp(self, x):
        x_log = log(x)/self._sLogBase

        if not (self._sXminLog <= x_log <= self._sXmaxLog):
            if self._sFillValue is None:
                raise ValueError(
                    "The given value does not lie within the interpolation range."
                )

            return self._sFillValue

        ix = int( ( x_log - self._sXminLog )*( self._sN - 1 )/( self._sXmaxLog - self._sXminLog ) )

        x1_log, x2_log = self._sXLog[ix], self._sXLog[ix+1]
        y1_log, y2_log = self._sYLog[ix], self._sYLog[ix+1]

        m = ( y2_log - y1_log )/( x2_log - x1_log )
        b = y2_log - m*x2_log

        return pow( self._sBase, m*x_log + b )


    def __call__(self, x):
        if x not in self._sCache:
            self._sCache[x] = self._perform_interp(x)

        return self._sCache[x]


def cumsimp(x_grid, y_grid):
    n = len(x_grid)
    delta_z = log(x_grid[-1] / x_grid[0])/(n-1)
    g_grid = x_grid*y_grid

    integral = np.zeros(n)

    last_even_int = 0.
    for i in range(1, int(n/2 + 1)):
        ie = 2 * i
        io = 2 * i - 1

        integral[io] = last_even_int + 0.5 * delta_z * (g_grid[io-1] + g_grid[io])
        if ie < n:
            integral[ie] = last_even_int + delta_z * (g_grid[ie-2] + 4.*g_grid[ie-1] + g_grid[ie])/3.
            last_even_int = integral[ie]

    return integral
