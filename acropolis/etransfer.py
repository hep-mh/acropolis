# math
from math import log, exp, sqrt
# cmath
from cmath import exp as cexp
# numpy
import numpy as np
# scipy
from scipy.optimize import root
# enum
from enum import Enum

# pprint
from acropolis.pprint import print_error
# hcascade
from acropolis.hcascade import mass
from acropolis.hcascade import Particles, is_pion, is_spectator, convert_nucleon
from acropolis.hcascade import ParticleSpectrum
# params
from acropolis.params import pi
from acropolis.params import mb_to_iMeV2
from acropolis.params import Kt, mp, mn


class _Actions(Enum):
    KEEP    = 0
    DESTROY = 1
    CONVERT = 2


# HELPER FUNCTIONS ##################################################

# K in MeV
def _s(projectile, target, K):
    mN, mA = mass[projectile], mass[target]

    return mN**2. + mA**2. + 2.*(K + mN)*mA # MeV

"""
# K in MeV
def _boost_projectile(particle, K, gcm, vcm):
    m = mass[particle]

    # -->
    E = K + m
    p = sqrt( K**2. + 2.*K*m )

    return gcm * ( E - vcm*p ) - m
"""

# Ecm in MeV
def _equip(particles, Ecm):
    # Extract the masses of all particles
    masses = np.array([mass[particle] for particle in particles])

    def _root_fct(p):
        return Ecm - np.sum(
            sqrt(p**2 + m**2) for m in masses
        )
    
    # Run the optimizer (start with non-rel limit)
    p0 = sqrt( 2. * ( Ecm - np.sum(masses) ) / np.sum(1./masses) )
    # -->
    res = root(_root_fct, p0)

    # Check if the optimizer was successful
    if not res.success or len(res.x) != 1:
        return np.nan
    
    return res.x[0]


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
def _Ecm(projectile, target, K):
    mN, mA = mass[projectile], mass[target]

    return sqrt( (mN + mA)**2. + 2.*mA*K )


# K in MeV
def _gcm(projectile, target, K):
    mN, mA = mass[projectile], mass[target]

    return ( (mN + mA) + K ) / sqrt( (mN + mA)**2. + 2.*mA*K )

"""
# K in MeV
def _vcm(projectile, target, K):
    mN, mA = mass[projectile], mass[target]

    return sqrt( K**2. + 2*mN*K )/( K + mN + mA )
"""

# GENERIC FUNCTIONS #################################################

# Reactions of the form
# p -> p
# n -> p + [...]
# TODO: Implement energy distribution
def _decay(egrid, projectile, Ki):
    # Initialize the spectrum
    spectrum = ParticleSpectrum(egrid)

    # Fill the spectrum
    if   projectile == Particles.PROTON:
        Ki_p = Ki
    elif projectile == Particles.NEUTRON:
        Ki_p = Ki + mn - mp

    # Fill the spectrum
    spectrum.add(Particles.PROTON, 1., Ki_p)

    return spectrum


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
    Bsl = _Bsl(target, s=_s(projectile, target, Ki))

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
# p + X(bg) -> Y
# n + X(bg) -> Y
# with X = p, He4 and arbitrary Y
def _inelastic(egrid, projectile, Ki, target, daughters, projectile_action):
    # Initialize the spectrum
    spectrum = ParticleSpectrum(egrid)

    # Calculate the COM energy
    Ecm = _Ecm(projectile, target, Ki)

    # Calculate the Lorentz factor
    gcm = _gcm(projectile, target, Ki)

    # Determine the projectile remnant
    projectile_remnant = {
        _Actions.KEEP   : projectile,
        _Actions.DESTROY: Particles.NULL,
        _Actions.CONVERT: convert_nucleon(projectile)
    }[projectile_action]

    # Initialize the amount of COM energy that
    # is available when employing equipartition
    Ecm_equip = Ecm

    # Estimate the kinetic energy of the remnant
    Ki_p = .5*Ki if projectile_remnant != Particles.NULL else 0.

    # Initialize the mass difference of the reaction
    dM = mass[target] + (mass[projectile] - mass[projectile_remnant])
    #                                       = 0 for Particles.NULL

    # Initialize the list of particles that are
    # considered when employing equipartition
    particles_equip = []
    if projectile_remnant != Particles.NULL:
        particles_equip.append(projectile_remnant)

    Kj_p_L= []
    # Loop over the various daughter particles
    for i, daughter in enumerate(daughters):
        md = mass[daughter]
        
        # Estimate the kinetic energy of the daughter particle
        if is_spectator(daughter):
            Kj_p_L.append( 5.789 ) # MeV

            # Update the available COM energy
            # (K ~ 5.789 MeV is fixed by the distribution)
            Ecm_equip -= gcm * ( Kj_p_L[-1] + md ) # vm*p ~ 0
        else:
            # Kj_p_cm ~ Kt
            Kj_p_L.append( gcm * ( Kt + md ) - md )

            # -->
            particles_equip.append(daughter)

        # Update the mass difference
        dM -= md
    
    # Ensure energy conservation
    # NOTE: In the '<' case, we assume that the remaining
    # energy is carried away by additional pions
    if Ki + dM > Ki_p + sum(Kj_p_L): # Energy too large
        # Assume equipartition of momenta
        pe = _equip(particles_equip, Ecm_equip)
        
        # Update the energy of the remnant
        if projectile_remnant != Particles.NULL:
            mr = mass[projectile_remnant]
            # -->
            Ei_p_cm = sqrt( pe**2. + mr**2. )
            # -->
            Ki_p = gcm * Ei_p_cm - mr

        # Update the energies of the daughter particles
        for i, daughter in enumerate(daughters):
            if is_spectator(daughter):
                continue

            md = mass[daughter]
            # -->
            Ej_p_cm = sqrt( pe**2. + md**2. ) 
            # -->
            Kj_p_L[i] = gcm * Ej_p_cm - md

    # Fill the spectrum
    if projectile_remnant != Particles.NULL:
        spectrum.add(projectile_remnant, 1., Ki_p)
    
    for i, daughter in enumerate(daughters):
        if not is_pion(daughter): # ignore pions
            spectrum.add(daughter, 1., Kj_p_L[i])
    
    return spectrum


# INDIVIDUAL FUNCTIONS ##############################################

# Reaction
# n -> p* + [...]
def _r0_null(egrid, projectile, Ki):
    return _decay(egrid, projectile, Ki)


# Reaction (i,p,1)
# p + p(bg) -> p + [p]
# n + p(bg) -> n + [p]
def _r1_proton(egrid, projectile, Ki):
    return _elastic(
        egrid, projectile, Ki,
        target=Particles.PROTON
    )


# Reaction (i,p,2)
# p + p(bg) -> p + [p + pi0]
# n + p(bg) -> n + [p + pi0]
def _r2_proton(egrid, projectile, Ki):
    return _inelastic(
        egrid, projectile, Ki,
        target=Particles.PROTON,
        projectile_action=_Actions.KEEP,
        daughters=[
            Particles.PROTON,
            Particles.NEUTRAL_PION
        ]
    )


# Reaction (i,p,3)
# p + p(bg) -> p + [n + pi+]
# n + p(bg) -> n + [n + pi+]
def _r3_proton(egrid, projectile, Ki):
    return _inelastic(
        egrid, projectile, Ki,
        target=Particles.PROTON,
        projectile_action=_Actions.KEEP,
        daughters=[
            Particles.NEUTRON,
            Particles.CHARGED_PION
        ]
    )


# Reaction (i,p,4)
# p + p(bg) -> n* + [n + 2pi+]
# n + p(bg) -> p* + [n + 2pi0]
def _r4_proton(egrid, projectile, Ki):
    Particles_PION = Particles.CHARGED_PION if projectile == Particles.PROTON else Particles.NEUTRAL_PION

    return _inelastic(
        egrid, projectile, Ki,
        target=Particles.PROTON,
        projectile_action=_Actions.CONVERT,
        daughters=[
            Particles.NEUTRON,
            Particles_PION,
            Particles_PION
        ]
    )


# Reaction (i,p,5)
# p + p(bg) -> n* + [p + pi+]
# n + p(bg) -> p* + [p + pi-]
def _r5_proton(egrid, projectile, Ki):
    return _inelastic(
        egrid, projectile, Ki,
        target=Particles.PROTON,
        projectile_action=_Actions.CONVERT,
        daughters=[
            Particles.PROTON,
            Particles.CHARGED_PION
        ]
    )


# Reaction (i,al,1)
# p + He4(bg) -> p + [He4]
# n + He4(bg) -> n + [He4]
def _r1_alpha(egrid, projectile, Ki):
    return _elastic(
        egrid, projectile, Ki,
        target=Particles.HELIUM4
    )


# Reaction (i,al,2)
# p + He4(bg) -> _ + [D + He3]
# n + He4(bg) -> _ + [D + T]
def _r2_alpha(egrid, projectile, Ki):
    Particles_A3 = Particles.HELIUM3 if projectile == Particles.PROTON else Particles.TRITIUM

    return _inelastic(
        egrid, projectile, Ki,
        target=Particles.HELIUM4,
        projectile_action=_Actions.DESTROY,
        daughters=[
            Particles.DEUTERIUM,
            Particles_A3
        ]
    )


# Reaction (i,al,3)
# p + He4(bg) -> p + [n + He3]
# n + He4(bg) -> n + [n + He3]
def _r3_alpha(egrid, projectile, Ki):
    return _inelastic(
        egrid, projectile, Ki,
        target=Particles.HELIUM4,
        projectile_action=_Actions.KEEP,
        daughters=[
            Particles.NEUTRON,
            Particles.HELIUM3
        ]
    )


# Reaction (i,al,4)
# p + He4(bg) -> p + [p + T]
# n + He4(bg) -> n + [p + T]
def _r4_alpha(egrid, projectile, Ki):
    return _inelastic(
        egrid, projectile, Ki,
        target=Particles.HELIUM4,
        projectile_action=_Actions.KEEP,
        daughters=[
            Particles.PROTON,
            Particles.TRITIUM
        ]
    )


# Reaction (i,al,5)
# p + He4(bg) -> p + [2D]
# n + He4(bg) -> n + [2D]
def _r5_alpha(egrid, projectile, Ki):
    return _inelastic(
        egrid, projectile, Ki,
        target=Particles.HELIUM4,
        projectile_action=_Actions.KEEP,
        daughters=[
            Particles.DEUTERIUM,
            Particles.DEUTERIUM
        ]
    )


# Reaction (i,al,6)
# p + He4(bg) -> p + [p + n + D]
# n + He4(bg) -> n + [p + n + D]
def _r6_alpha(egrid, projectile, Ki):
    return _inelastic(
        egrid, projectile, Ki,
        target=Particles.HELIUM4,
        projectile_action=_Actions.KEEP,
        daughters=[
            Particles.PROTON,
            Particles.NEUTRON,
            Particles.DEUTERIUM
        ]
    )


# Reaction (i,al,7)
# p + He4(bg) -> p + [2p + 2n]
# n + He4(bg) -> n + [2p + 2n]
def _r7_alpha(egrid, projectile, Ki):
    return _inelastic(
        egrid, projectile, Ki,
        target=Particles.HELIUM4,
        projectile_action=_Actions.KEEP,
        daughters=[
            Particles.PROTON,
            Particles.PROTON,
            Particles.NEUTRON,
            Particles.NEUTRON
        ]
    )


# Reaction (i,al,8)
# p + He4(bg) -> p + [He4 + pi0]
# n + He4(bg) -> n + [He4 + pi0]
def _r8_alpha(egrid, projectile, Ki):
    return _inelastic(
        egrid, projectile, Ki,
        target=Particles.HELIUM4,
        projectile_action=_Actions.KEEP,
        daughters=[
            Particles.HELIUM4,
            Particles.NEUTRAL_PION
        ]
    )


# MAIN FUNCTIONS ####################################################

def get_fs_spectrum(egrid, projectile, Ki, rid):
    params = (egrid, projectile, Ki)

    if rid == 0:
        # n -> p* + [...]
        return _r0_null(*params)

    if rid == 1: # {pp_pp, np_np}
        # p + p(bg) -> p + [p]
        # n + p(bg) -> n + [p]
        return _r1_proton(*params)

    if rid == 2: # {pp_inel, np_inel}
        # p + p(bg) -> p + [p + pi0]
        # n + p(bg) -> n + [p + pi0]
        return _r2_proton(*params)
    
    if rid == 3: # {pp_inel, np_inel}
        # p + p(bg) -> p + [n + pi+]
        # n + p(bg) -> n + [n + pi+]
        return _r3_proton(*params)
    
    if rid == 4: # {pp_inel, np_inel}
        # p + p(bg) -> n* + [n + 2pi+]
        # n + p(bg) -> p* + [n + 2pi0]
        return _r4_proton(*params)
    
    if rid == 5: # {pp_inel, np_inel}
        # p + p(bg) -> n* + [p + pi+]
        # n + p(bg) -> p* + [p + pi-]
        return _r5_proton(*params)
    
    if rid == 6: # {pHe4_pHe4}
        # p + He4(bg) -> p + [He4]
        # n + He4(bg) -> n + [He4]
        return _r1_alpha(*params)
    
    if rid == 7: # {pHe4_DHe3}
        # p + He4(bg) -> _ + [D + He3]
        # n + He4(bg) -> _ + [D + T]
        return _r2_alpha(*params)
    
    if rid == 8: # {pHe4_pnHe3}
        # p + He4(bg) -> p + [n + He3]
        # n + He4(bg) -> n + [n + He3]
        return _r3_alpha(*params)
    
    if rid == 9: # {pHe4_2pT}
        # p + He4(bg) -> p + [p + T]
        # n + He4(bg) -> n + [p + T]
        return _r4_alpha(*params)
    
    if rid == 10: # {pHe4_p2D}
        # p + He4(bg) -> p + [2D]
        # n + He4(bg) -> n + [2D]
        return _r5_alpha(*params)
    
    if rid == 11: # {pHe4_2pnD}
        # p + He4(bg) -> p + [p + n + D]
        # n + He4(bg) -> n + [p + n + D]
        return _r6_alpha(*params)
    
    if rid == 12: # {pHe4_3p2n}
        # p + He4(bg) -> p + [2p + 2n]
        # n + He4(bg) -> n + [2p + 2n]
        return _r7_alpha(*params)
    
    if rid == 13: # {pHe4_pHe4pi}
        # p + He4(bg) -> p + [He4 + pi0]
        # n + He4(bg) -> n + [He4 + pi0]
        return _r8_alpha(*params)
    
    # Invalid reaction id
    print_error(
        "Reaction with rid = " + str(rid) + " does not exist.",
        "acropolis.etransfer.get_fs_spectrum"
    )