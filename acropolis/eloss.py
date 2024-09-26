# math
from math import log, exp, sqrt, erf
# scipy
from scipy.integrate import quad

# cosmo
from acropolis.cosmo import na, nee
# jit
from acropolis.jit import jit
# params
from acropolis.params import pi, pi2
from acropolis.params import alpha, me, me2
from acropolis.params import Ephb_T_max, eps

E_T_max = Ephb_T_max


# HELPER FUNCTIONS ############################################################

@jit
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
        for i in range(4):
            asum += a[i]*( (x-2.)**(i+1) )

        return (pi/12.)*(x-2.)**4./( 1. + asum )

    bsum, csum = 0., 0.
    for j in range(4):
        bsum += b[j]*(log(x)**j)

    for k in range(3):
        csum += c[k]/( x**(k+1) )

    return x*bsum/( 1. - csum )


@jit
def _JIT_eloss_bethe_heitler(logx, T, E, M):
    x = exp(logx) # kappa

    # Calculate gamma
    ga = E/M
    # Calculate nu (https://journals.aps.org/prd/pdf/10.1103/PhysRevD.1.1596)
    nu = me/(2.*ga*T)

    #      log
    return x * _JIT_phi(x) * exp(-nu*x)/ ( 1. - exp(-nu*x) )


# MAIN FUNCTIONS ##############################################################
# E is the energy of the energy-loosing particle
# T is the temperature of the background photons

# N + e- > N + e-
def _dEdt_coulomb(T, E, M, Q, Y, eta):
    if E <= M:
        return 0.

    # Calculate the gamma factor of the charged particle
    ga = E/M # > 1

    # Calculate the velocity of the charged particle
    v = sqrt(1. - 1./ga**2.) # > 0

    # Calculate the plasma frequency
    wp2 = 4. * pi * alpha * nee(T, Y, eta) / me
    # -->
    wp = sqrt(wp2)

    # Calculate the b parameter
    b = max( Q*alpha/( ga*me*(v**2.) ), 1./( ga*me*v ) )

    # Calculate the Lambda parameter
    Lambda = log( 0.76 * v / wp / b )

    # Calculate the y parameter
    y = sqrt( me*v*v/(2.*T) )

    # Calculate the two summands
    s1 = erf(y) * ( (Lambda/v) * ( 1. - T/(2.*me) ) + .5 * ( v - T/(v*me) ) )
    s2 = -2. * exp(-y**2.) * y * ( Lambda * ( 1. - T/(2.*me) + (v**2.)/8. + me*(v**4.)/(8.*T) ) - T/(2.*me) ) / sqrt(pi) / v

    return -(Q**2.) * alpha * wp2 * ( s1 + s2 )


# N + a > N + a
def _dEdt_thomson(T, E, M, Q):
    if E <= M:
        return 0.

    # Calculate the gamma factor of the charged particle
    ga = E/M # > 1

    # This relation also remains true for non-relativistic
    # particles, in which case gamma^2-1 = 0
    # For the comparison with the White et al. paper
    # use zeta(4) = p^4/90 and ga^2 - 1 --> ga^2
    return -32. * (pi**3.) * (alpha**2.) * (ga**2. - 1.) * (T**4.) * (Q**4.) / 135. / (M**2.)


# N + a > N + e+ + e-
def _dEdt_bethe_heitler(T, E, M, Q):
    if E <= M:
        return 0.
    
    if E <= 2.*me: # Below electron-positron threshold
        return 0.

    # Calculate the gamma factor of the charged particle
    ga = E/M # > 1

    # Calculate the velocity of the charged particle
    v = sqrt(1. - 1./ga**2.) # > 0

    # Calculate the prefactor
    pref = (alpha**3.) * (Q**2.) * me2 * v / ( 4. * (ga**2.) * pi2 )

    Emax = E_T_max*T
    # Calculate the appropriate integration limits
    xmin = 2.
    xmax = xmin + 2.*ga*Emax/me
    # -->
    xmin_log, xmax_log = log(xmin), log(xmax)

    I = quad(
            _JIT_eloss_bethe_heitler, xmin_log, xmax_log, epsabs=0., epsrel=eps, args=(T, E, M)
        )[0]

    return -pref*I


# n + e- > n + e-
def _dEdt_magnetic_moment(T, E, M, G, Y, eta):
    if E <= M:
        return 0.

    # In our scenario, this should always be the case
    # For larger temperatures, the formulae are different
    assert T <= me

    # Calculate the kinetic energy of the neutral particle
    Ekin = E - M # > 0

    # Calculate the number density of electrons and positrons
    ne = nee(T, Y, eta)

    # In the low-temperature regime, use the formula
    # from the footnote of 'astro-ph/0408426v2'
    if ne == ( 1. - Y/2. ) * eta * na(T):
        return -4. * (alpha**2.) * (G**2.) * ne * (2.*pi/me/T)**1.5 * (me**3.) * T * Ekin / ( 3. * pi * (M**3.) )

    if Ekin > M:
        return -3. * pi * (alpha**2.) * (G**2.) * me * ne * (Ekin**2.) / (M**4.)
    
    x = me/T
    # -->
    Ge = (x**3.) + 3.*(x**2.) + 6.*x + 6.

    # Ekin <= M
    return -16. * (alpha**2.) * (G**2.) * (T**4.) * exp(-x) * Ge * Ekin / ( 3. * pi * (M**3.) )


# p/n + a > p/n + pi0
def _dEdt_nucleon_pion(T, E, M):
    if E <= M:
        return 0.

    # ATTENTION: The below formulae are
    # only correct for neutral pions

    mpi0 = 134.9768                  # MeV
    sigp = 6.8e-36*2.5681899885e+27  # 1/MeV³
    eps0 = mpi0*( 1. + mpi0/(2.*M) ) # MeV

    if E < eps0*M/T:
        return -2. * sigp * (eps0**2.) * (T**3.) * E * exp( -eps0*M/(2.*E*T) ) / ( pi2 * M )
    
    # E >= eps0*M/T
    #                | 1/(yr K³) --> 1/MeV²
    return -1.8e-8 * 3.2616628389e+01 * E * (T**3.) / (2.7**3.)