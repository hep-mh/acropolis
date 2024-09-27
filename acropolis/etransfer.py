# math
from math import log, exp, sqrt
# cmath
from cmath import exp as cexp
# numpy
import numpy as np

# hcascade
from acropolis.hcascade import mass
from acropolis.hcascade import Particles, is_pion, convert_nucleon
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
def _Bsl(target, s):
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
# p + X(bg) -> p + X
# n + X(bg) -> n + X
# with X = p, He4
def _elastic(egrid, projectile, Ki, target):
    # Initialize the spectrum
    spectrum = ParticleSpectrum(egrid)
    
    mN, mA = mass[projectile], mass[target]
    # Calculate the maximal value of Kj'
    Kj_p_max = 2.*mA*Ki*(Ki + 2.*mN) / ( (mN + mA)**2. + 2.*mA*Ki )

    # Calculate the slope parameter
    Bsl = _Bsl(target, s=_K_to_s(projectile, target, Ki))

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
# p + X(bg) -> p + Y
# n + X(bg) -> p + Y
# with X = p, He4 and arbitrary Y
def _inelastic(egrid, projectile, Ki, target, daughters, convert_projectile):
    # Initialize the spectrum
    spectrum = ParticleSpectrum(egrid)

    # Determine the type of the projectile after
    # the inelastic scattering process
    if convert_projectile:
        projectile_remnant = convert_nucleon(projectile)
    else:
        projectile_remnant = projectile

    # Calculate the gamma factor between the
    # com frame and the target rest frame
    gcm = _gcm(projectile, target, Ki)

    # Estimate the kinetic energy of the
    # scattered projectile particle
    Ki_p = .5*Ki

    # Initialize a variable to store
    # the mass difference of the reaction
    dM = mass[target] + (mass[projectile] - mass[projectile_remnant])
    
    Kj_p_L = []
    # Loop over the various daughter particles
    for daughter in daughters:
        md = mass[daughter]
        
        # Estimate the kinetic energy of the daughter particle
        if daughter not in [Particles.TRITIUM, Particles.HELIUM3]:
            Kj_p_L.append( gcm*Kt + (gcm - 1.)*md ) 
        else: # T and He3 act as spectator particles
            Kj_p_L.append( 5.789 ) # MeV

        # Update the mass difference
        dM -= md
    
    # Ensure energy conservation
    # TODO: Implement (careful with pions)

    # Fill the spectrum
    spectrum.add(projectile_remnant, 1., Ki_p)
    for i, daughter in enumerate(daughters):
        if not is_pion(daughter): # ignore pions
            spectrum.add(daughter, 1., Kj_p_L[i])
    
    return spectrum


# INDIVIDUAL FUNCTIONS ##############################################

# Reaction (i,p,1)
# p + p(bg) -> p + p
# n + p(bg) -> n + p
def _r1_proton(egrid, projectile, Ki):
    return _elastic(
        egrid, projectile, Ki,
        target=Particles.PROTON
    )


# Reaction (i,p,2)
# p + p(bg) -> p + p + pi0
# n + p(bg) -> n + p + pi0
def _r2_proton(egrid, projectile, Ki):
    return _inelastic(
        egrid, projectile, Ki,
        target=Particles.PROTON,
        convert_projectile=False,
        daughters=[
            Particles.PROTON,
            Particles.NEUTRAL_PION
        ]
    )


# Reaction (i,p,3)
# p + p(bg) -> p + n + pi+
# n + p(bg) -> n + n + pi+
def _r3_proton(egrid, projectile, Ki):
    return _inelastic(
        egrid, projectile, Ki,
        target=Particles.PROTON,
        convert_projectile=False,
        daughters=[
            Particles.NEUTRON,
            Particles.CHARGED_PION
        ]
    )


# Reaction (i,al,1)
# p + He4(bg) -> p + He4
# n + He4(bg) -> n + He4
def _r1_alpha(egrid, projectile, Ki):
    return _elastic(
        egrid, projectile, Ki,
        target=Particles.HELIUM4
    )


# Reaction (i,al,5)
# p + He4(bg) -> p + 2D
# n + He4(bg) -> n + 2D
def _r5_alpha(egrid, projectile, Ki):
    return _inelastic(
        egrid, projectile, Ki,
        target=Particles.HELIUM4,
        convert_projectile=False,
        daughters=[
            Particles.DEUTERIUM,
            Particles.DEUTERIUM
        ]
    )
