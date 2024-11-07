# numpy
import numpy as np

# input
from acropolis.input import locate_data_file
# util
from acropolis.utils import LogInterp
# params
from acropolis.params import zeta3, pi2
from acropolis.params import me


_f_data = np.loadtxt( locate_data_file("f.dat") )
# -->
_f = LogInterp(_f_data[:,0], _f_data[:,1], fill_value=(3.*zeta3/2., 0.))


def na(T):
    return 2. * zeta3 * (T**3.) / pi2


def nb(T, eta):
    return eta * na(T)


def nee(T, Y, eta):
    xe = me/T

    # 1. EQUILIBRIUM WITH VANISHING CHEM. POTENTIAL
    nee_1 = 4. * _f(xe) * (T**3.) / ( 2. * pi2 )

    # 2. EQUILIBRIUM WITH NON-VANISHING CHEM. POTENTIAL
    nee_2 = ( Y[1] + 2.*Y[5] ) * nb(T, eta)

    return max(nee_1, nee_2)


def nH(T, Y, eta):
    return Y[1] * nb(T, eta) 


def nHe4(T, Y, eta):
    return Y[5] * nb(T, eta)