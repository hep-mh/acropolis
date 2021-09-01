# math
from math import log, exp, sqrt
# numba
import numba as nb
# scipy
from scipy.special import erf

# params
from acropolis.params import pi, pi2, zeta3
from acropolis.params import alpha, me

"""
#@nb.jit(cache=True)
def _JIT_fa(E, T):
    return (E/pi)**2. / (exp(E/T) - 1 )


#@nb.jit(cache=True)
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
        asum = sum( a[i]*( (x-2.)**(i+1) ) for i in range(4) )

        return (pi/12.)*(x-2.)**4./( 1. + asum )

    bsum = sum( b[j]*(log(x)**j) for j in range(4) )
    csum = sum( c[k]/( x**(k+1) ) for k in range(3) )

    return x*bsum/( 1. - csum )


#@nb.jit(cache=True)
def _JIT_eloss_bethe_heitler(logx, ga, T):
    x = exp(logx) # kappa

    return x * _JIT_fa( x/(2*ga), T ) * _JIT_phi(x) / x**2.
"""

class InteractingParticle(object):

    def __init__(self, m, q):
        # The mass of the particle
        self._sM = m # in MeV
        # The charge of the particle
        self._sQ = q # in units of e


    # TODO: Interface correctly with ACROPOLIS
    def _ne(self, T):
        # The number density of photons
        na = 2.*zeta3*(T**3.)/pi2

        if T >= me:
            return 1.5*na

        if me > T >= me/26.:
            return 4.*exp(-me/T)*( me*T/(2.*pi) )**1.5

        # The baryon-to-photon ratio
        eta = 6.137e-10
        # The abundance of helium-4
        Y = 0.2474

        return (1. - Y/2.)*eta*na


    # CHARGED PARTICLES #######################################################

    def _dEdt_coulomb(self, E, T):
        # The plasma frequency squared...
        wp2 = 4.*pi*self._ne(T)*alpha/me
        # ... and also not squared
        wp  = sqrt(wp2)

        # The gamma factor of the charged particle
        ga = E/self._sM

        # The velocity of the charged particle
        v = sqrt( 1. - 1./ga**2. )

        if v < 2*T/me:
            # TODO
            return 0.

        Z = self._sQ
        # The 'b-factor'
        b = max( 1, Z*alpha/v )/( ga*me*v )

        return -(Z**2.)*alpha*wp2*( log( 0.76*v/(wp*b) ) + v**2./2. )/v


    def _dEdt_thompson(self, E, T):
        # The gamma factor of the charged particle
        ga = E/self._sM

        Z = self._sQ
        # This holds true also for non-relativistic
        # particles, in which case gamma^2-1 = 0
        return -32.*(pi**3.)*(alpha**2.)*(ga**2. - 1)*(T**4.)*(Z**4.)/(135.*self._sM**2.)


    def _dEdt_bethe_heitler(self, E, T):
        return 0.


    def _dEdt_charged(self, E, T):
        return self._dEdt_thompson(E, T)

        #return self._dEdt_coulomb(E, T) + self._dEdt_thompson(E, T) + self._dEdt_bethe_heitler(E, T)


    # NEUTRAL PARTICLES #######################################################

    def _dEdt_magnetic_moment(self, E, T):
        return 0.


    def _dEdt_neutral(self, E, T):
        return self._dEdt_magnetic_moment(E, T)


    # COMBINED ################################################################

    def dEdt(self, E, T):
        if self._sQ == 0:
            return self._dEdt_neutral(E, T)

        return self._dEdt_charged(E, T)
