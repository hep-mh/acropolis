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

# eloss
import acropolis.eloss as eloss
# flags
import acropolis.flags as flags
# hrates
from acropolis.hrates import get_all_probs
# params
from acropolis.params import pi
from acropolis.params import mb_to_iMeV2
from acropolis.params import Kt, mn, mp, mpi0
# particles
from acropolis.particles import Particles, ParticleSpectrum
from acropolis.particles import eth, mass, za
from acropolis.particles import is_projectile, is_pion, is_spectator, is_nucleus
from acropolis.particles import convert
# utils
from acropolis.utils import LogInterp


class _Actions(Enum):
    KEEP    = 0
    DESTROY = 1
    CONVERT = 2


class _Background(object):
    def __init__(self, T, Y, eta):
        self.T   = T
        self.Y   = Y
        self.eta = eta
    
    def get(self):
        return (self.T, self.Y, self.eta)


# HELPER FUNCTIONS ##################################################

# K in MeV
def _s(projectile, target, K):
    mN, mA = mass[projectile], mass[target]

    return mN**2. + mA**2. + 2.*(K + mN)*mA # MeV


# K in MeV
def _Ecm(projectile, target, K):
    mN, mA = mass[projectile], mass[target]

    return sqrt( (mN + mA)**2. + 2.*mA*K )


# K in MeV
def _gcm(projectile, target, K):
    mN, mA = mass[projectile], mass[target]

    return ( (mN + mA) + K ) / sqrt( (mN + mA)**2. + 2.*mA*K )


_interp_cache = {}
def _survives(egrid, nucleus, Ki, bg):
    global _interp_cache

    if not is_nucleus(nucleus):
        return (True, Ki)

    if flags.all_nuclei_survive:
        return (True, Ki)

    # Generate a unique dictionary key
    key = nucleus.value

    # PHOTODISINTEGRATION VIA CMB PHOTONS ###########################

    Z, A = za[nucleus]

    # Calculate the threshold energy for
    # photodisintegration
    Eth_pdi = Z*mp + (A-Z)*mn - mass[nucleus]

    # -->
    if sqrt( 3*Ki*bg.T ) > Eth_pdi:
        return (False, Ki)

    
    # HADRODISINTEGRATION VIA BACKGROUND PROTONS ####################
    
    if key not in _interp_cache:
        _, raw = eloss.process(egrid, nucleus, *bg.get())

        # Extract the grids for the initial and final energies
        Ki_grid = egrid.central_values()
        Kf_grid = np.array([Kf for _, Kf in raw])
        # TODO: Check case of small n: cumsimp problem?
        # -->
        Ki_grid = Ki_grid[Kf_grid != 0.]
        Kf_grid = Kf_grid[Kf_grid != 0.]

        data_valid = (len(Kf_grid) >= 2)
        # Create and store the interpolation function
        _interp_cache[key] = LogInterp(
            Ki_grid, Kf_grid, fill_value=(0.,np.nan)
        ) if data_valid else None

    # Calculate the energy before scattering
    Kf = 0.
    if _interp_cache[key] is not None:
        Kf = _interp_cache[key](Ki)

    # DEBUG
    assert ( not np.isnan(Kf) )

    # Calculate the theshold energy for hadrodisintegration
    Eth_hdi = eth[nucleus]

    # -->
    if Kf > Eth_hdi:
        return (False, Kf)

    # Otherwise (no pdi or hdi)
    return (True, Ki)


def _fragments(egrid, particle, Ki, bg):
    survives, Kf = _survives(egrid, particle, Ki, bg)

    if survives:
        return [(particle, Kf)]

    # Extract the mass number and the
    # atomic number of the particle
    Z, A = za[particle]
    
    if not flags.reinject_fragments:
        return [(Particles.PROTON, 0.)]*Z + [(Particles.NEUTRON, 0.)]*(A-Z)

    # Extract the mass of the particle
    m = mass[particle]

    # Calculate the kinetic energy of the
    # resulting protons and neutrons
    Kp_p = (Kf + m)/A - mp
    Kn_p = (Kf + m)/A - mn
    
    # -->
    return [(Particles.PROTON, Kp_p)]*Z + [(Particles.NEUTRON, Kn_p)]*(A-Z)


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
    success = False
    for method in ["lm"]: # NOTE: Edit to add fallback methods
        res = root(_root_fct, p0, method=method)

        if res.success and len(res.x) == 1:
            success = True
            break

    # Check if the optimizer was successful
    if not success:
        return np.nan
    
    return res.x[0] # MeV


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


# GENERIC FUNCTIONS #################################################

# Reactions of the form
# p -> p
# n -> p + [...]
def _decay(spectrum, projectile, Ki, prob):
    # Determine the projectile remnant after
    # the decay
    projectile_remnant = {
        Particles.PROTON : Particles.PROTON,
        Particles.NEUTRON: Particles.PROTON
    }[projectile]

    # Estimate the kinetic energy of the remnant
    # Ep ~ En
    Ki_p = Ki + mass[projectile] - mass[projectile_remnant]

    # UPDATE THE SPECTRUM #################################
    # _survives(projectile_remnant) == True
    spectrum.add(projectile_remnant, prob, Ki_p)

    # Account for the destruction of the
    # initial-state projectile
    spectrum.add(projectile, -prob)


# Reactions of the form
# p + X(bg) -> p + X
# n + X(bg) -> n + X
# with X = p, He4
def _elastic(spectrum, projectile, Ki, prob, bg, target):
    # Extract the energy grid
    egrid = spectrum.egrid()

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
        # -->
        assert Ki_p_l <= Ki_p_h

        if Ki_p_l == Ki_p_h:
            return 0.

        return pref * ( exp( -2.*mA*Bsl*(Ki-Ki_p_h) ) - exp( -2.*mA*Bsl*(Ki-Ki_p_l) ) )

    # UPDATE THE SPECTRUM #################################
    sum_Fi, sum_Fj = 0., 0.
    for i in range( egrid.nbins() ):
        Fi = _integrate_fi_over_bin(i)
        Fj = _integrate_fj_over_bin(i)
        # ->
        sum_Fi += Fi
        sum_Fj += Fj

        # Extract the current energy
        Ki_p = Kj_p = egrid[i]

        # Handle the scattered PROJECTILE particle
        # _survives(projectile) == True
        spectrum.add(projectile, prob*Fi, Ki_p)

        # Handle the scattered TARGET particle
        for fragment, Kj_f in _fragments(egrid, target, Kj_p, bg):
            spectrum.add(fragment, prob*Fj, Kj_f)
        
    # Account for (1) the destruction of the
    # initial-state particles, as well as (2)
    # the creation of particles below Kmin
    #                        = -1 + (1 - sum_Fx)
    spectrum.add(projectile, -prob*sum_Fi)
    spectrum.add(target    , -prob*sum_Fj)


# Reactions of the form
# p + X(bg) -> Y
# n + X(bg) -> Y
# with X = p, He4 and arbitrary Y
def _inelastic(spectrum, projectile, Ki, prob, bg, target, daughters, projectile_action):
    # Extract the energy grid
    egrid = spectrum.egrid()
    
    # Calculate the COM energy
    Ecm = _Ecm(projectile, target, Ki)

    # Calculate the Lorentz factor
    gcm = _gcm(projectile, target, Ki)

    # Determine the projectile remnant
    # final_state = [projectile_remnant, *daughters]
    projectile_remnant = {
        _Actions.KEEP   : projectile,
        _Actions.DESTROY: Particles.NULL,
        _Actions.CONVERT: convert(projectile)
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

    Kj_p_L = []
    # Loop over the various daughter particles
    for i, daughter in enumerate(daughters):
        md = mass[daughter]
        
        # Estimate the kinetic energy of the daughter particle
        if is_spectator(daughter):
            Kj_p_L.append( 5.789 ) # MeV

            # Update the available COM energy
            # (K ~ 5.789 MeV is fixed by the distribution)
            Ecm_equip -= ( Kj_p_L[-1] + md ) / gcm
            # This substraction ensures energy conservation
            # in the cosmic frame after equipartition
        else:
            # Kj_p_cm ~ Kt
            Kj_p_L.append( gcm * ( Kt + md ) - md )

            # -->
            particles_equip.append(daughter)

        # Update the mass difference
        dM -= md
    
    # DEBUG
    if False:
        print("Final-state particles :")
        _l1 = [Ki_p, *Kj_p_L]
        _l2 = [projectile_remnant, *daughters]
        for _K, _d in zip(_l1, _l2):
            print(f"  • {_d.name:<12}   {_K:.3e}MeV")
        print(f"Gamma factor          :  {gcm:.5e}")
        print(f"Mass difference       : {dM:+.5e}MeV")
        print(f"Energy balance        :  {Ki:.5e}MeV (in)")
        print(f"                         {Ki_p+sum(Kj_p_L)-dM:.5e}MeV (out)")
    
    # Check if the reaction is kinematically allowed
    M_equip = sum(mass[peq] for peq in particles_equip)
    # -->
    if Ecm_equip <= M_equip:
        raise ValueError(
            "Insufficient energy to create final-state particles"
        )

    # Ensure energy conservation
    # NOTE:
    # In the '>' case, we assume that the remaining
    # energy is carried away by additional pions
    if Ki + dM < Ki_p + sum(Kj_p_L) + gcm*mpi0: # Outgoing energy too large
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
        
        # Check if energy is conserved
        assert np.isclose(Ki+dM, Ki_p+sum(Kj_p_L))

    # UPDATE THE SPECTRUM ###########################################
    if projectile_remnant != Particles.NULL:
        # _survives(projectile_remnant) == True
        spectrum.add(projectile_remnant, prob, Ki_p)
    
    for i, daughter in enumerate(daughters):
        if is_pion(daughter):
            continue # ignore pions

        for fragment, Kj_f in _fragments(egrid, daughter, Kj_p_L[i], bg):
            spectrum.add(fragment, prob, Kj_f)
    
    # Account for the destruction of the
    # initial-state particles
    spectrum.add(projectile, -prob)
    spectrum.add(target    , -prob)


# INDIVIDUAL FUNCTIONS ##############################################

# Reaction
# n -> p* + [...]
def _r0_null(spectrum, projectile, Ki, prob):
    _decay(spectrum, projectile, Ki, prob)


# Reaction (i,p,1)
# p + p(bg) -> p + [p]
# n + p(bg) -> n + [p]
def _r1_proton(spectrum, projectile, Ki, prob, bg):
    _elastic(
        spectrum, projectile, Ki, prob, bg,
        target=Particles.PROTON
    )


# Reaction (i,p,2)
# p + p(bg) -> p + [p + pi0]
# n + p(bg) -> n + [p + pi0]
def _r2_proton(spectrum, projectile, Ki, prob, bg):
    _inelastic(
        spectrum, projectile, Ki, prob, bg,
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
def _r3_proton(spectrum, projectile, Ki, prob, bg):
    _inelastic(
        spectrum, projectile, Ki, prob, bg,
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
def _r4_proton(spectrum, projectile, Ki, prob, bg):
    Particles_PION = {
        Particles.PROTON : Particles.CHARGED_PION,
        Particles.NEUTRON: Particles.NEUTRAL_PION
    }[projectile]

    # -->
    _inelastic(
        spectrum, projectile, Ki, prob, bg,
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
def _r5_proton(spectrum, projectile, Ki, prob, bg):
    _inelastic(
        spectrum, projectile, Ki, prob, bg,
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
def _r1_alpha(spectrum, projectile, Ki, prob, bg):
    _elastic(
        spectrum, projectile, Ki, prob, bg,
        target=Particles.HELIUM4
    )


# Reaction (i,al,2)
# p + He4(bg) -> _ + [D + He3]
# n + He4(bg) -> _ + [D + T]
def _r2_alpha(spectrum, projectile, Ki, prob, bg):
    Particles_A3 = {
        Particles.PROTON : Particles.HELIUM3,
        Particles.NEUTRON: Particles.TRITIUM
    }[projectile]

    # -->
    _inelastic(
        spectrum, projectile, Ki, prob, bg,
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
def _r3_alpha(spectrum, projectile, Ki, prob, bg):
    _inelastic(
        spectrum, projectile, Ki, prob, bg,
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
def _r4_alpha(spectrum, projectile, Ki, prob, bg):
    _inelastic(
        spectrum, projectile, Ki, prob, bg,
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
def _r5_alpha(spectrum, projectile, Ki, prob, bg):
    _inelastic(
        spectrum, projectile, Ki, prob, bg,
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
def _r6_alpha(spectrum, projectile, Ki, prob, bg):
    _inelastic(
        spectrum, projectile, Ki, prob, bg,
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
def _r7_alpha(spectrum, projectile, Ki, prob, bg):
    _inelastic(
        spectrum, projectile, Ki, prob, bg,
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
def _r8_alpha(spectrum, projectile, Ki, prob, bg):
    _inelastic(
        spectrum, projectile, Ki, prob, bg,
        target=Particles.HELIUM4,
        projectile_action=_Actions.KEEP,
        daughters=[
            Particles.HELIUM4,
            Particles.NEUTRAL_PION
        ]
    )


# MAIN FUNCTIONS ####################################################

def _update_spectrum(spectrum, rid, projectile, Ki, prob, bg):
    params = (spectrum, projectile, Ki, prob)

    if   rid == 0:
        # n -> p* + [...]
        _r0_null(*params)

    elif rid == 1: # {pp_pp, np_np}
        # p + p(bg) -> p + [p]
        # n + p(bg) -> n + [p]
        _r1_proton(*params, bg)

    elif rid == 2: # {pp_inel, np_inel}
        # p + p(bg) -> p + [p + pi0]
        # n + p(bg) -> n + [p + pi0]
        _r2_proton(*params, bg)
    
    elif rid == 3: # {pp_inel, np_inel}
        # p + p(bg) -> p + [n + pi+]
        # n + p(bg) -> n + [n + pi+]
        _r3_proton(*params, bg)
    
    elif rid == 4: # {pp_inel, np_inel}
        # p + p(bg) -> n* + [n + 2pi+]
        # n + p(bg) -> p* + [n + 2pi0]
        _r4_proton(*params, bg)
    
    elif rid == 5: # {pp_inel, np_inel}
        # p + p(bg) -> n* + [p + pi+]
        # n + p(bg) -> p* + [p + pi-]
        _r5_proton(*params, bg)
    
    elif rid == 6: # {pHe4_pHe4}
        # p + He4(bg) -> p + [He4]
        # n + He4(bg) -> n + [He4]
        _r1_alpha(*params, bg)
    
    elif rid == 7: # {pHe4_DHe3}
        # p + He4(bg) -> _ + [D + He3]
        # n + He4(bg) -> _ + [D + T]
        _r2_alpha(*params, bg)
    
    elif rid == 8: # {pHe4_pnHe3}
        # p + He4(bg) -> p + [n + He3]
        # n + He4(bg) -> n + [n + He3]
        _r3_alpha(*params, bg)
    
    elif rid == 9: # {pHe4_2pT}
        # p + He4(bg) -> p + [p + T]
        # n + He4(bg) -> n + [p + T]
        _r4_alpha(*params, bg)
    
    elif rid == 10: # {pHe4_p2D}
        # p + He4(bg) -> p + [2D]
        # n + He4(bg) -> n + [2D]
        _r5_alpha(*params, bg)
    
    elif rid == 11: # {pHe4_2pnD}
        # p + He4(bg) -> p + [p + n + D]
        # n + He4(bg) -> n + [p + n + D]
        _r6_alpha(*params, bg)
    
    elif rid == 12: # {pHe4_3p2n}
        # p + He4(bg) -> p + [2p + 2n]
        # n + He4(bg) -> n + [2p + 2n]
        _r7_alpha(*params, bg)
    
    elif rid == 13: # {pHe4_pHe4pi}
        # p + He4(bg) -> p + [He4 + pi0]
        # n + He4(bg) -> n + [He4 + pi0]
        _r8_alpha(*params, bg)


def _process_single_energy(egrid, projectile, Ki, T, Y, eta):
    # Initialize the spectrum
    spectrum = ParticleSpectrum(egrid)

    # Calculate the scattering probabilities
    probs = get_all_probs(projectile, Ki, T, Y, eta)

    # Initialize the background object
    bg = _Background(T, Y, eta)

    # Loop over all reactions and update the
    # spectrum based on the given probability
    for rid, prob in enumerate(probs):
        if prob == 0:
            continue
        
        _update_spectrum(spectrum, rid, projectile, Ki, prob, bg)
    
    # Return the final spectrum
    return spectrum


def process(egrid, projectile, T, Y, eta):
    if not is_projectile(projectile):
        raise ValueError("The given particle must be a projectile")

    spectra = []
    for i in range(egrid.nbins()):
        spectra.append(
            _process_single_energy(egrid, projectile, egrid[i], T, Y, eta)
        )
    
    return spectra