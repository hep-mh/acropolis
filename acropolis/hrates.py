# math
from math import sqrt, log, exp
# numpy
import numpy as np

# input
from acropolis.input import locate_data_file
# particles
from acropolis.particles import Particles, mass, label
from acropolis.particles import is_projectile
# params
from acropolis.params import zeta3, pi2
from acropolis.params import mb_to_iMeV2


# TODO: Move
NR = 14

# K ≘ kinetic energy, E ≘ total energy


def _nH(T, Y, eta):
    return 2. * zeta3 * (T**3.) * eta * (1.-Y) / pi2


def _nHe4(T, Y, eta):
    return 2. * zeta3 * (T**3.) * eta * (Y/4.) / pi2


# READ REACTION DATA ################################################

_reaction_labels = [
    "pp_pp",
    "pn_pn",
    "pp_inel",
    "pn_inel",
    "pp_tot",
    "pn_tot",
    "pD_tot",
    "pT_tot",
    "pHe3_tot",
    "pHe4_tot",
    "pHe4_pHe4",
    "pHe4_2pnD",
    "pHe4_2pT",
    "pHe4_pnHe3",
    "pHe4_p2D",
    "pHe4_DHe3",
    "pHe4_3p2n",
    "pHe4_pHe4pi"
]


def _load_log_reaction_data(label):
    filename = f"cross_sections/{label}.dat"

    reaction_data = np.loadtxt(f"{locate_data_file(filename)}")
    # -->
    reaction_data[:,1] *= mb_to_iMeV2

    return np.log(reaction_data)


_log_reaction_data = {
    label: _load_log_reaction_data(label) for label in _reaction_labels
}


# EVALUATE REACTION RATES ###########################################

def _interp_reaction_data(label, K):
    logK = log(K)

    log_reaction_data = _log_reaction_data[label]

    if logK < log_reaction_data[0,0]:
        return 0.

    return exp(
        np.interp( logK, log_reaction_data[:,0], log_reaction_data[:,1] )
    )


# Threshold energies for two pion production
_Kth_r4 = {
    Particles.PROTON : 605.7509051898966,
    Particles.NEUTRON: 579.1139640875533
} # MeV


def get_all_rates(projectile, Ki, T, Y, eta):
    # PREPARE #######################################################

    # Initialize an array for storing the rates
    rates = np.zeros(NR)
    
    # Extract the projectile label
    x = label[projectile]

    # Extract the mass of the projectile
    m = mass[projectile]

    # Calculate the velocity of the projectile
    v = sqrt( Ki * (Ki + 2.*m) ) / ( Ki + m ) # = p/E

    # Calculate the target densities
    nH, nHe4 = _nH(T, Y, eta), _nHe4(T, Y, eta)

    # Determine how many channels contribute to
    # inelastic projectile-proton scattering
    N_p_pi = 4 if ( Ki > _Kth_r4[projectile] ) else 3 

    # FILL ##########################################################

    # rid = 0
    # if projectile == Particles.NEUTRON:
    #     rates[0] = sqrt(1. - v**2.) * hbar / tau_n
    
    # rid = 1
    rates[1]  = nH * _interp_reaction_data(f"p{x}_p{x}", Ki) * v

    # rid = 2
    rates[2]  = nH * _interp_reaction_data(f"p{x}_inel", Ki) * v / N_p_pi

    # rid = 3
    rates[3]  = rates[2]

    # rid = 4
    if N_p_pi == 4:
        rates[4]  = rates[2]

    # rid = 5
    rates[5]  = rates[2]

    # rid = 6
    rates[6]  = nHe4 * _interp_reaction_data("pHe4_pHe4", Ki) * v

    # rid = 7
    rates[7]  = nHe4 * _interp_reaction_data("pHe4_DHe3", Ki) * v

    # rid = 8
    rates[8]  = nHe4 * _interp_reaction_data("pHe4_pnHe3", Ki) * v

    # rid = 9
    rates[9]  = nHe4 * _interp_reaction_data("pHe4_2pT", Ki) * v

    # rid = 10
    rates[10] = nHe4 * _interp_reaction_data("pHe4_p2D", Ki) * v

    # rid = 11
    rates[11] = nHe4 * _interp_reaction_data("pHe4_2pnD", Ki) * v

    # rid = 12
    rates[12] = nHe4 * _interp_reaction_data("pHe4_3p2n", Ki) * v

    # rid = 13
    rates[13] = nHe4 * _interp_reaction_data("pHe4_pHe4pi", Ki) * v

    # RETURN ########################################################

    return rates


def get_all_probs(projectile, Ki, T, Y, eta):
    rates = get_all_rates(projectile, Ki, T, Y, eta)

    return rates/np.sum(rates)


def get_mean_free_path(particle, Ki, T, Y, eta):
    rate = 0.

    # Extract the particle label
    x = label[particle]

    # Extract the mass of the particle
    m = mass[particle]

    # Calculate the gamma factor of the particle
    ga = (Ki+m)/m

    # Calculate the velocity of the particle
    v = sqrt(1. - 1./ga**2.)

    # Handle scattering on protons
    rate += _nH(T, Y, eta) * _interp_reaction_data(f"p{x}_tot", Ki) * v

    # Handle scattering on helium-4
    if is_projectile(particle): # p ~ n in this case
        rate += _nHe4(T, Y, eta) * _interp_reaction_data("pHe4_tot", Ki) * v

    return 1./rate
