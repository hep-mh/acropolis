# math
from math import pi, log, exp, sqrt
# numba
import numba as nb
# scipy
from scipy.integrate import quad

# params
from acropolis.params import alpha, me, me2
from acropolis.params import pi2
from acropolis.params import Ephb_T_max, eps
# -->
E_T_max = Ephb_T_max

@nb.jit(cache=True)
def _JIT_phi(x):
    a = [
         0.8048,
         0.1459,
         1.1370e-3,
        -3.8790e-6
    ]
    b = [
        -86.07,
         50.96,
        -14.45,
         8./3.,
    ]
    c = [
        2.910,
        78.35,
        1.837e3,
    ]

    if x <= 25:
        asum = 0
        for i in range(4): asum += a[i]*( (x-2.)**(i+1) )

        return (pi/12.)*(x-2.)**4./( 1. + asum )

    bsum, csum = 0, 0
    for j in range(4): bsum += b[j]*(log(x)**j)
    for k in range(3): csum += c[k]/( x**(k+1) )

    return x*bsum/( 1. - csum )


@nb.jit(cache=True)
def _JIT_eloss_bethe_heitler(logx, E, T, M):
    x = exp(logx) # kappa

    # Calculate gamma
    ga = E/M
    # Calculate nu (https://journals.aps.org/prd/pdf/10.1103/PhysRevD.1.1596)
    nu = me/(2*ga*T)

    #      log
    return x * _JIT_phi(x)/( exp(nu*x) - 1. )


# CONTINUOUS ENERGY LOSS ##################################################
# E is the energy of the energy-loosing particle
# T is the temperature of the background photons

@nb.jit(cache=True)
def dEdt_thomson(E, T, M, Q=1):
    # The gamma factor of the charged particle
    ga = E/M

    # This relation also remains for non-relativistic
    # particles, in which case gamma^2-1 = 0
    # For the comparison with the White et al. paper
    # use zeta(4) = p^4/90 and ga^2 - 1 --> ga^2
    return -32.*(pi**3.)*(alpha**2.)*(ga**2. - 1)*(T**4.)*(Q**4.)/135./(M**2.)


def dEdt_bethe_heitler(E, T, M, Q=1):
    # The gamma factor of the charged particle
    ga = E/M

    # The velocity of the charged particle
    v = sqrt(1. - 1./ga**2.) if ga > 1 else 0

    # Define the prefactor
    pref = (alpha**3.)*(Q**2.)*me2*v/( 4.*(ga**2.)*pi2 )

    # Calculate the appropriate integration limits
    Emax = E_T_max*T
    xmax = 2*ga*Emax/me
    # -->
    xmin_log, xmax_log = log(2), log(xmax)

    # Perform the integration
    I = quad(_JIT_eloss_bethe_heitler, xmin_log, xmax_log,
                epsabs=0, epsrel=eps, args=(E, T, M))

    return -pref*I[0]