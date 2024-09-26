# math
from math import log, exp, sqrt
# cmath
from cmath import exp as cexp
# numpy
import numpy as np

# hcascade
from acropolis.hcascade import mass, Projectiles, Targets, ParticleSpectrum
# params
from acropolis.params import pi
from acropolis.params import mpi0, mpic, Kt, mp, mn


mb_to_iMeV2 = 2.5681899885e-06


def _K_to_s(projectile, target, K):
    mN, mA = mass[projectile], mass[target]

    return mN**2. + mA**2. + 2.*(K + mN)*mA # MeV


def _K_to_E(particle, K):
    return K + mass[particle]


# HELPER FUNCTIONS ##################################################


# s in MeV²
def _bp(s):
    C = 10.94 * 1e-6 # 1/MeV²
    D = -0.09 * 1e-6 # 1/MeV²
    E = 0.044 * 1e-6 # 1/MeV²

    return C + D*log(s/1e6) + E*(log(s/1e6)**2.) # 1/MeV²


# s in MeV²
def _bm(s):
    C = 23.2 * 1e-6 # 1/MeV²
    D = 0.94 * 1e-6 # 1/MeV²

    return C + D*log(s/1e6) # 1/MeV²


# s in MeV²
def _Mp(s):
    A  = 41.8 * mb_to_iMeV2 # 1/MeV²
    B  = 0.68 * mb_to_iMeV2 # 1/MeV²
    s0 = 343e6              # MeV²
    
    return -1j * s * ( A + B*(log(s/s0) - 1.j*pi/2.)**2. ) # unitless


# s in MeV²
def _Mm(s):
    D  = -39e6*mb_to_iMeV2 # GeV^{-2al}
    al = 0.48

    return D * (s/1e6)**al * cexp(1j*pi*(1.-al)/2.) # unitless


# s in MeV²
def _Bsl(projectile, target, Ki):
    # Calculate the com-energy squared
    s = _K_to_s(projectile, target, Ki)

    if target == Targets.ALPHA:
        # The same expression is used for any type of projectile
        
        return 28 * 1e-6 # 1/MeV²
        #         | 1/GeV² --> 1/MeV²
    
    if target == Targets.PROTON:
        bp, bm = _bp(s), _bm(s)
        Mp, Mm = _Mp(s), _Mm(s)
        # -->
        ReMp, ImMp = Mp.real, Mp.imag
        ReMm, ImMm = Mm.real, Mm.imag
        # -->
        ReMp2, ImMp2 = ReMp**2., ImMp**2.
        ReMm2, ImMm2 = ReMm**2., ImMm**2.

        # Bsl = d ln(dsig / dt) /dt = (d²sig / dt²) / (dsig / dt) @ t = tref
        def _Bsl(k):
            tref = -0.02e6 # MeV²

            num = bm*exp(bm*tref)*ImMm2 + exp(bp*tref)*ImMp2*bp + bm*exp(bm*tref)*ReMm2 \
                + bp*exp(bp*tref)*ReMp2 + (bm + bp)*exp(((bm + bp)*tref)/2.)*k*(ImMm*ImMp + ReMm*ReMp)
            
            den = exp(bm*tref)*(ImMm2 + ReMm2) + \
                + 2.*exp(((bm + bp)*tref)/2.)*k*(ImMm*ImMp + ReMm*ReMp) + exp(bp*tref)*(ImMp2 + ReMp2)

            return num/den # 1/MeV²
        
        if projectile == Projectiles.PROTON \
        or projectile == Projectiles.NEUTRON:
            return _Bsl(-1) # 1/MeV²

        if projectile == Projectiles.ANTI_PROTON \
        or projectile == Projectiles.ANTI_NEUTRON:
            return _Bsl(+1) # 1/MeV²

    # Invalid projectile or target
    return np.nan


def _gcm(projectile, target, Ki):
    mN, mA = mass[projectile], mass[target]

    return ( (mN + mA) + Ki ) / sqrt( (mN + mA)**2. + 2.*mA*Ki )


#####################################################################


# TODO
def _any_elastic(egrid, projectile, target, Ki):
    if projectile == Projectiles.ANTI_PROTON \
    or projectile == Projectiles.ANTI_NEUTRON:
        raise ValueError(
            "Scattering of anti-protons(neutrons) is currently not supported"
        )
    
    # Initialize the spectrum
    spectrum = ParticleSpectrum( egrid )
    
    mN, mA = mass[projectile], mass[target]
    # Calculate the maximal value of Kj'
    Kj_p_max = 2.*mA*Ki*(Ki + 2.*mN) / ( (mN + mA)**2. + 2.*mA*Ki )

    # Calculate the slope parameter Bsl
    Bsl = _Bsl(projectile, target, Ki)

    # Calculate the prefactor of the distribution
    pref = 1./( 1. - exp(-2.*mA*Bsl*Kj_p_max) )

    def _integrate_fj_over_bin(i):
        Kj_p_l, Kj_p_h = np.minimum( egrid.bin_range(i), Kj_p_max )

        return pref * ( exp( -2.*mA*Bsl*Kj_p_l ) - exp( -2.*mA*Bsl*Kj_p_h ) )

    def _integrate_fi_over_bin(i):
        return 0. # TODO

    # Loop over all bins
    for i in range( egrid.nbins() ):
        # Handle the scattered projectile particle
        spectrum.addp(projectile, _integrate_fj_over_bin(i), egrid[i])

        # Handle the scattered target particle
        spectrum.addp(target    , _integrate_fi_over_bin(i), egrid[i])

    return spectrum


# Reactions (i,p,2) and (i,p,3)
#
# convert_target = False:
# p + p_bg -> p + p + pi0
# n + p_bg -> n + p + pi0
#
# convert_target = True:
# p + p_bg -> p + n + pi+
# n + p_bg -> n + n + pi+
def _proton_inelastic(egrid, projectile, Ki, convert_target):
    if projectile == Projectiles.ANTI_PROTON \
    or projectile == Projectiles.ANTI_NEUTRON:
        raise ValueError(
            "Scattering of anti-protons(neutrons) is currently not supported"
        )
    
    # Initialize the spectrum
    spectrum = ParticleSpectrum( egrid )

    # Specify the target particle
    target = Targets.PROTON

    # Specify the daughter particle
    daughter = Projectiles.PROTON if not convert_target else Projectiles.NEUTRON
    # -->
    (md, mpi) = (mp, mpi0) if not convert_target else (mn, mpic)

    # Calculate the gamma factor for transforming
    # from the com frame to the target rest frame
    gcm = _gcm(projectile, target, Ki)

    # Estimate the kinetic energy of the
    # scattered projectile particle
    Ki_p = .5*Ki

    # Estimate the kinetic energy of the
    # scattered target particle
    Kj_p = gcm*Kt + (gcm - 1.)*md

    # Estimate the kinetic energy of the
    # produced neutral/charged pion
    Kpi_p = gcm*Kt + (gcm - 1.)*mpi

    # Ensure energy conservation
    # TODO
    if Ki + mp < Ki_p + Kj_p + md + Kpi_p + mpi:
        pass

    spectrum.addp(projectile, 1, Ki_p)
    spectrum.addp(daughter  , 1, Kj_p)
    # pions can be ignored

    return spectrum


# Reaction (i,p,1)
# p + p_bg -> p + p
# n + p_bg -> n + p
def proton_r1(egrid, projectile, Ki):
    return _any_elastic(egrid, projectile, Targets.PROTON, Ki)



# Reaction (i,p,2)
# p + p_bg -> p + p + pi0
# n + p_bg -> n + p + pi0
def proton_r2(egrid, projectile, Ki):
    return _proton_inelastic(egrid, projectile, Ki, convert_target=False)


# Reaction (i,p,3)
# p + p_bg -> p + n + pi+
# n + p_bg -> n + n + pi+
def proton_r3(egrid, projectile, Ki):
    return _proton_inelastic(egrid, projectile, Ki, convert_target=True)


# Any reaction of the form
# p + he4_bg -> p + X
# n + he4_bg -> p + X
# with X not including T/He3
# TODO
#def _alpha_inelastic(energy_grid, projectile, Ki, daughter_nuclei):