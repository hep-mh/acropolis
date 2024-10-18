# math
from math import log10, sqrt, log, exp
# numpy
import numpy as np
# enum
from enum import Enum

# etransfer
from acropolis.etransfer import get_fs_spectrum
# input
from acropolis.input import locate_data_file
# pprint
from acropolis.pprint import print_error
# params
from acropolis.params import zeta3, pi2
from acropolis.params import mb_to_iMeV2
from acropolis.params import mp, mn, mD, mT, mHe3, mHe4, mpi0, mpic
from acropolis.params import hbar, tau_n
from acropolis.params import approx_zero


# TODO: Move
NR = 14

# K ≘ kinetic energy, E ≘ total energy


def _nH(T, Y, eta):
    return 2. * zeta3 * (T**3.) * eta * (1.-Y) / pi2


def _nHe4(T, Y, eta):
    return 2. * zeta3 * (T**3.) * eta * (Y/4.) / pi2


#####################################################################

_reaction_labels = [
    "pp_pp",
    "np_np",
    "pp_inel",
    "np_inel",
    "pp_tot",
    "np_tot",
    "pHe3_tot",
    "pD_tot",
    "pHe4_pHe4",
    "pHe4_tot",
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


def _interp_reaction_data(label, K):
    logK = log(K)

    log_reaction_data = _log_reaction_data[label]

    if logK < log_reaction_data[0,0]:
        return approx_zero

    return exp(
        np.interp( logK, log_reaction_data[:,0], log_reaction_data[:,1] )
    )
 

#####################################################################


class Particles(Enum):
    PROTON  =  0
    NEUTRON =  1

    DEUTERIUM = -4
    TRITIUM   = -3
    HELIUM3   = -2
    HELIUM4   = -1

    CHARGED_PION = -6
    NEUTRAL_PION = -5

    NULL = None


def is_nucleon(particle):
    if particle not in Particles:
        return False

    return (0 <= particle.value <= 1)


def is_nucleus(particle):
    if particle not in Particles:
        return False
    
    return (-4 <= particle.value <= -1)


def is_pion(particle):
    if particle not in Particles:
        return False
    
    return (-6 <= particle.value <= -5)


def is_spectator(particle):
    if particle not in Particles:
        return False
    
    # T and He3 act as spectator particles
    return (-3 <= particle.value <= -2)


def is_valid_projectile(particle):
    return is_nucleon(particle)


def is_valid_target(particle):
    if particle not in Particles:
        return False

    return (particle.value in [0, -1] )


def convert(particle):
    if not is_nucleon(particle):
        return particle
    
    if particle == Particles.PROTON:
        return Particles.NEUTRON
    
    if particle == Particles.NEUTRON:
        return Particles.PROTON


# All masses in MeV
mass = {
    Particles.PROTON : mp,
    Particles.NEUTRON: mn,

    Particles.DEUTERIUM: mD,
    Particles.TRITIUM  : mT,
    Particles.HELIUM3  : mHe3,
    Particles.HELIUM4  : mHe4,

    Particles.NEUTRAL_PION: mpi0,
    Particles.CHARGED_PION: mpic,

    Particles.NULL: 0.
}

charge = {
    Particles.PROTON : 1,
    Particles.NEUTRON: 0,

    Particles.DEUTERIUM: 1,
    Particles.TRITIUM  : 1,
    Particles.HELIUM3  : 2,
    Particles.HELIUM4  : 2,

    #Particles.NEUTRAL_PION: 0,
    #Particles.CHARGED_PION: 1,

    Particles.NULL: 0
}


#####################################################################


def _get_all_rates(projectile, Ki, T, Y, eta):
    # PREPARE #######################################################

    # Initialize an array for storing the rates
    rates = np.zeros(NR)

    if not is_valid_projectile(projectile):
        print_error(
            "The given particles is not a valid projectile.",
            "acropolis.hcascade.get_all_rates"
        )
    
    x = {
        Particles.PROTON : "p",
        Particles.NEUTRON: "n"
    }[projectile]

    # Extract the mass of the projectile
    m = mass[projectile]

    # Calculate the velocity of the projectile
    v = sqrt( Ki * (Ki + 2.*m) ) / ( Ki + m ) # = p/E

    # Calculate the target densities
    nH, nHe4 = _nH(T, Y, eta), _nHe4(T, Y, eta)

    # FILL ##########################################################

    # rid = 0
    if projectile == Particles.NEUTRON:
        rates[0] = sqrt(1. - v**2.) *hbar / tau_n
    
    # rid = 1
    rates[1]  = nH * _interp_reaction_data(f"{x}p_{x}p", Ki) * v

    # rid = 2
    rates[2]  = nH * _interp_reaction_data(f"{x}p_inel", Ki) * v / 4.

    # rid = 3
    rates[3]  = rates[2]

    # rid = 4
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


def _get_all_probs(projectile, Ki, T, Y, eta):
    rates = _get_all_rates(projectile, Ki, T, Y, eta)

    return rates/np.sum(rates)


#####################################################################


class EnergyGrid(object):

    def __init__(self, Kmin, Kmax, N):
        if Kmax <= Kmin:
            raise ValueError(
                "The value of Kmax must be larger than the one of Kmin"
            )
        
        self._sN = N

        self._sKmin      = Kmin
        self._sKmax      = Kmax
        # -->
        self._sKminLog10 = log10(Kmin)
        self._sKmaxLog10 = log10(Kmax)

        # Generate the different energy bins, i.e. the bin edges
        self._sKbins = np.logspace(
            self._sKminLog10, self._sKmaxLog10, N+1
        ) # +1 in order to have N central values |
        
        # Calculate the central values of neighbouring bins
        self._sKcent = self._sKbins[:-1] * sqrt(self._sKbins[1]/self._sKbins[0])
        #                           | N elements


    def index_of(self, K):
        if not self._sKmin <= K <= self._sKmax:
            raise ValueError(
                "The given energy does not lie within the grid"
            )
        
        # Handle the special case K = Kmax
        if K == self._sKmax:
            return self._sN - 1
        
        # log10(K) = log10(Kmin) + log10(Kmax/Kmin) * i / ( [N + 1] - 1 )
        return int(
            self._sN * ( log10(K) - self._sKminLog10 )  / ( self._sKmaxLog10 - self._sKminLog10 )
        )


    def lower_edge(self):
        return self._sKmin


    def upper_edge(self):
        return self._sKmax


    def nbins(self):
        return self._sN


    def bin_range(self, i):
        return (self._sKbins[i], self._sKbins[i+1])


    def __getitem__(self, i):
        return self._sKcent[i]


class ParticleSpectrum(object):

    def __init__(self, energy_grid):
        self._sEnergyGrid = energy_grid

        # Extract the number of bins
        self._sN = energy_grid.nbins()

        # Initialize a list containing all non-zero entries
        self._sEntries = {}


    def _increment(self, index, increment, acc=1e-5):
        if increment <= acc/self._sN:
            return

        if index in self._sEntries:
            self._sEntries[index] += increment
        else:
            self._sEntries[index] = increment


    def _add_nucleon(self, nucleon, increment, K):
        if K < self._sEnergyGrid.lower_edge():
            return

        index = nucleon.value*self._sN + self._sEnergyGrid.index_of(K)
        # -->
        self._increment(index, increment)


    def _add_nucleus(self, nucleus, increment, K):
        # TODO: Check if the nucleus survives

        index = nucleus.value
        # -->
        self._increment(index, increment)


    def add(self, particle, increment, K):
        # DEBUG
        print(f"Adding {particle} with energy {K:.5e}MeV")
        
        if K > self._sEnergyGrid.upper_edge():
            raise ValueError(
                "The given value of K lies ouside the energy grid"
            )

        # Protons & Neutrons
        if is_nucleon(particle):
            self._add_nucleon(particle, increment, K)
        
        # Nuclei
        elif is_nucleus(particle):
            self._add_nucleus(particle, increment, K)
        
        else:
            raise ValueError(
                "The given particles cannot be added to the spectrum"
            )


    def non_zero(self):
        for index in self._sEntries:
            yield ( index, self._sEntries[index] )


    def at(self, index):
        if index in self._sEntries:
            return self._sEntries[index]
        
        return 0.


    def rescale(self, factor):
        for index in self._sEntries:
            self._sEntries[index] *= factor


    def egrid(self):
        return self._sEnergyGrid


    def __repr__(self):
        str_repr = ""

        for i in range(self._sN):
            str_repr += f"{self._sEnergyGrid[i]:.3e} |"

            for j in range(0, 2): # nucleons
                str_repr += f" {self.at(j*self._sN + i):.3e}"
            
            str_repr += "\n"
        
        str_repr += "----------x\n"

        for k in range(-4, 0): # nuclei
            str_repr += f"{self.at(k):.3e} | \n"
        
        return str_repr

#####################################################################

def _get_etransfer_matrix(egrid, T, Y, eta):
    # Extract the number of bins
    N = egrid.nbins()

    # Initialize the matrix
    matrix = np.identity(N)

    # Loop over all possible projectiles
    for projectile in [Particles.PROTON, Particles.NEUTRON]:
        # Loop over all possible energies
        for i in len(N):
            Ki = egrid[i]

            # Calculate the scattering probabilities
            probs = _get_all_probs(projectile, Ki, T, Y, eta)

            # Calculate the final-state spectrum
            spectrum = get_fs_spectrum(egrid, projectile, Ki, probs)

            # Loop over the spectrum and fill the matrix
            for (j, val) in spectrum.non_zero():
                matrix[j,i] = val
    
    return matrix

