# math
from math import log, exp, sqrt, erf
# numpy
import numpy as np
# scipy
from scipy.integrate import quad
from scipy.interpolate import interp1d

# cosmo
from acropolis.cosmo import nee
# hrates
from acropolis.hrates import get_mean_free_path
# jit
from acropolis.jit import jit
# util
from acropolis.utils import flipped_cumsimp
# params
from acropolis.params import pi, pi2
from acropolis.params import alpha, me, me2
from acropolis.params import Ephb_T_max, eps, approx_zero
# particles
from acropolis.particles import ParticleSpectrum
from acropolis.particles import is_projectile, is_nucleus, is_unstable
from acropolis.particles import mass, lifetime, charge, dipole
# -->
E_T_max = Ephb_T_max


# HELPER FUNCTIONS ##################################################

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


# ENERGY-LOSS FUNCTIONS #############################################
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

    # Calculate the number density of electrons and positrons
    ne = nee(T, Y, eta)

    # Calculate the gamma factor of the neutral particle
    ga = E/M
    
    # Calculate the velocity of the neutral particle
    v  = sqrt(1. - 1./ga**2.)

    # -->
    return -3. * pi * (alpha**2.) * (G**2.) * me * ne * ga**2 * v**3 / (M**2.)


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


# All reactions
def _dEdt(particle, K, T, Y, eta): # = dKdt
    # Extract the mass of the particle
    M = mass[particle]

    # Extract the charge of the particle
    Q = charge[particle]

    # Extract the magnetic moment of the particle
    G = 0. # for nuclei
    if is_projectile(particle):
        G = dipole[particle]

    # Calculate the total energy of the particle
    E = K + M

    # Initialize the differential energy loss
    dEdt = 0.
    # -->
    if Q != 0.:
        dEdt += _dEdt_coulomb(T, E, M, Q, Y, eta)
        dEdt += _dEdt_thomson(T, E, M, Q)
        dEdt += _dEdt_bethe_heitler(T, E, M, Q)
    elif G != 0.: # Q == 0.
        dEdt += _dEdt_magnetic_moment(T, E, M, G, Y, eta)
    
    return dEdt


# MAIN FUNCTIONS ####################################################

def _eloss_kernel(particle, Ki, T, Y, eta):
    # Calculate the mean free path of the particle
    lN = get_mean_free_path(particle, Ki, T, Y, eta)

    # Calculate the energy loss of the particle
    dEdt = _dEdt(particle, Ki, T, Y, eta)

    # -->
    return -1./( lN * dEdt )


def _decays(particle, Ki, T, Y, eta):
    if not is_unstable(particle):
        return False
    
    # Extract the mass and the lifetime of the particle
    m, tau = mass[particle], lifetime[particle]

    # Calculate the reaction rates for scattering and decay
    Rs = 1. / get_mean_free_path(particle, Ki, T, Y, eta)
    Rd = m / ( (Ki+m) * tau )

    return (Rd > Rs)


def process(egrid, particle, T, Y, eta, fallback=None):
    if not ( is_projectile(particle) or is_nucleus(particle) ):
        raise ValueError("The given particle must be a projectile or a nucleus")

    N = egrid.nbins()

    # Extract the central energy values
    Ki_grid = egrid.central_values()

    # Handle the special case of unstable particles
    if is_unstable(particle) and (fallback is None):
        raise ValueError("Unstable particles require a fallback")

    # Calculate the integral kernel
    Ik_grid = np.array([
        _eloss_kernel(particle, Ki, T, Y, eta) for Ki in Ki_grid
    ])
    
    # Perform a cummulative Simpson integration
    # over the given integral kernel
    C_grid = flipped_cumsimp(Ki_grid, Ik_grid)
    # -->
    C_grid[C_grid < approx_zero] = approx_zero

    # Perform an interpolation of Kf(C)
    Ki_grid_log, C_grid_log = np.log(Ki_grid), np.log(C_grid)
    # -->
    Cf_log = interp1d(C_grid_log, Ki_grid_log, kind="linear", bounds_error=False, fill_value=-np.inf)
    # Use fill_value=-np.inf (Kf = 0) for Kf < egrid[0]
    # This avoids adding the particle to the spectrum

    spectra, raw = [], []
    for i in range(N):
        Ki = Ki_grid[i]

        # Calculate the final energy ################################
        if _decays(particle, Ki, T, Y, eta):
            # Use the fallback
            remnant, Kf = fallback[i]
        else:
            # Calculate from scratch
            val = log(1. + C_grid[i]) # C(Kf) - C(Ki) = R = 1
            # -->
            remnant, Kf = particle, exp( Cf_log(val) )
        
        # -->
        raw.append( (remnant, Kf) )
        
        # Create and fill the spectrum ##############################
        spectra.append( ParticleSpectrum(egrid) )

        # -->
        spectra[i].add(remnant ,  1., Kf)
        spectra[i].add(particle, -1.)

    return spectra, raw
    
