# math
from math import log, exp, sqrt
# cmath
from cmath import exp as cexp
# numpy
import numpy as np

# hcascade
from acropolis.hcascade import mass
from acropolis.hcascade import Particles, is_pion
from acropolis.hcascade import ParticleSpectrum
# params
from acropolis.params import pi
from acropolis.params import mb_to_iMeV2
from acropolis.params import Kt


# HELPER FUNCTIONS ##################################################

# K in MeV
def _K_to_s(projectile, target, K):
    mN, mA = mass[projectile], mass[target]

    return mN**2. + mA**2. + 2.*(K + mN)*mA # MeV


# K in MeV
def _K_to_E(particle, K):
    return K + mass[particle]


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
def _Bsl(projectile, target, s):
    if target == Particles.HELIUM4:
        # The same expression is used for any type of projectile
        
        return 28 * 1e-6 # 1/MeV²
        #         | 1/GeV² --> 1/MeV²
    
    if target == Particles.PROTON:
        bp, bm = _bp(s), _bm(s)
        Mp, Mm = _Mp(s), _Mm(s)
        # -->
        ReMp, ImMp = Mp.real, Mp.imag
        ReMm, ImMm = Mm.real, Mm.imag
        # -->
        ReMp2, ImMp2 = ReMp**2., ImMp**2.
        ReMm2, ImMm2 = ReMm**2., ImMm**2.

        # Bsl = d ln(dsig / dt) /dt = (d²sig / dt²) / (dsig / dt) @ t = tref
        def _Bsl_pm(k):
            tref = -0.02e6 # MeV²

            num = bm*exp(bm*tref)*ImMm2 + exp(bp*tref)*ImMp2*bp + bm*exp(bm*tref)*ReMm2 \
                + bp*exp(bp*tref)*ReMp2 + (bm + bp)*exp(((bm + bp)*tref)/2.)*k*(ImMm*ImMp + ReMm*ReMp)
            
            den = exp(bm*tref)*(ImMm2 + ReMm2) + \
                + 2.*exp(((bm + bp)*tref)/2.)*k*(ImMm*ImMp + ReMm*ReMp) + exp(bp*tref)*(ImMp2 + ReMp2)

            return num/den # 1/MeV²
        
        return _Bsl_pm(-1) # 1/MeV²

    # Invalid target
    return np.nan


# K in MeV
def _gcm(projectile, target, K):
    mN, mA = mass[projectile], mass[target]

    return ( (mN + mA) + K ) / sqrt( (mN + mA)**2. + 2.*mA*K )


# GENERIC FUNCTIONS #################################################

# Reactions of the form
# p + X_bg  -> p + X
# n + X_bg  -> n + X
# with X = p, He4
def _elastic(egrid, projectile, target, Ki):
    # Initialize the spectrum
    spectrum = ParticleSpectrum(egrid)
    
    mN, mA = mass[projectile], mass[target]
    # Calculate the maximal value of Kj'
    Kj_p_max = 2.*mA*Ki*(Ki + 2.*mN) / ( (mN + mA)**2. + 2.*mA*Ki )

    # Calculate the slope parameter
    Bsl = _Bsl(projectile, target, s=_K_to_s(projectile, target, Ki))

    # Calculate the prefactor of the distribution
    pref = 1./( 1. - exp(-2.*mA*Bsl*Kj_p_max) )

    def _integrate_fj_over_bin(i):
        Kj_p_l, Kj_p_h = np.minimum( egrid.bin_range(i), Kj_p_max )

        if Kj_p_l == Kj_p_h:
            return 0.

        return pref * ( exp( -2.*mA*Bsl*Kj_p_l ) - exp( -2.*mA*Bsl*Kj_p_h ) )

    def _integrate_fi_over_bin(i):
        Ki_p_l, Ki_p_h = np.maximum(
            np.minimum( egrid.bin_range(i), Ki ), Ki - Kj_p_max
        )
        # DEBUG
        assert Ki_p_l <= Ki_p_h

        if Ki_p_l == Ki_p_h:
            return 0.

        return pref * ( exp( -2.*mA*Bsl*(Ki-Ki_p_h) ) - exp( -2.*mA*Bsl*(Ki-Ki_p_l) ) )

    # Fill the spectrum by looping over all bins
    for i in range( egrid.nbins() ):
        # Handle the scattered projectile particle
        spectrum.add(projectile, _integrate_fi_over_bin(i), egrid[i])

        # Handle the scattered target particle
        spectrum.add(target    , _integrate_fj_over_bin(i), egrid[i])

    return spectrum


# Reactions of the form
# p + X_bg -> p + [...]
# n + X_bg -> p + [...]
# with T, He3 ∉ [...]
# TODO: convert_projectile???
def _inelastic(egrid, projectile, target, Ki, daughters):
    # Initialize the spectrum
    spectrum = ParticleSpectrum(egrid)

    # Calculate the gamma factor for transforming
    # from the com frame to the target rest frame
    gcm = _gcm(projectile, target, Ki)

    # Estimate the kinetic energy of the
    # scattered projectile particle
    Ki_p = .5*Ki

    # Initialize a variable to store
    # the mass difference of the reaction
    dM = mass[target]
    
    Kj_p = []
    # Estimate the kinetic energies of
    # the various daughter particles
    for daughter in daughters:
        md = mass[daughter]
        
        # Estimate the energy and append it
        Kj_p.append( gcm*Kt + (gcm - 1.)*md )

        # Update the mass difference
        dM -= md
    
    # Ensure energy conservation
    # TODO: Implement + handle pions

    # Fill the spectrum
    spectrum.add(projectile, 1., Ki_p)
    for i, daughter in enumerate(daughters):
        if not is_pion(daughter): # ignore pions
            spectrum.add(daughter, 1., Kj_p[i])
    
    return spectrum


# INDIVIDUAL FUNCTIONS ##############################################

# Reaction (i,p,1)
# p + p_bg -> p + p
# n + p_bg -> n + p
def _r1_proton(egrid, projectile, Ki):
    return _elastic(egrid, projectile, Particles.PROTON, Ki)


# Reaction (i,p,2)
# p + p_bg -> p + p + pi0
# n + p_bg -> n + p + pi0
def _r2_proton(egrid, projectile, Ki):
    daughters = [Particles.PROTON, Particles.NEUTRAL_PION]
    # -->
    return _inelastic(
        egrid, projectile, Particles.PROTON, Ki, daughters
    )


# Reaction (i,p,3)
# p + p_bg -> p + n + pi+
# n + p_bg -> n + n + pi+
def _r3_proton(egrid, projectile, Ki):
    daughters = [Particles.NEUTRON, Particles.CHARGED_PION]
    # -->
    return _inelastic(
        egrid, projectile, Particles.PROTON, Ki, daughters
    )


# Reaction (i,al,1)
# p + he4_bg -> p + he4_bg
# n + he4_bg -> n + he4_bg
def _r1_alpha(egrid, projectile, Ki):
    return _elastic(egrid, projectile, Particles.HELIUM4, Ki)


# TODO 2 - 4


# Reaction (i,al,5)
# p + he4_bg -> p + 2D
# n + he4_bg -> n + 2D
def _r5_alpha(egrid, projectile, Ki):
    daughters = [Particles.DEUTERIUM, Particles.DEUTERIUM]
    # -->
    return _inelastic(
        egrid, projectile, Particles.HELIUM4, Ki, daughters
    )


# TODO: Add to public function
# if projectile == Projectiles.ANTI_PROTON \
# or projectile == Projectiles.ANTI_NEUTRON:
#     raise NotImplementedError(
#         "Scattering of anti-nucleons is currently not implemented"
#     )