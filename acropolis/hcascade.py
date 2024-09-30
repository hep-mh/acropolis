# math
from math import log10, sqrt
# numpy
import numpy as np
# enum
from enum import Enum

# params
from acropolis.params import zeta3, pi2
from acropolis.params import mp, mn, mD, mT, mHe3, mHe4, mpi0, mpic


# K ≘ kinetic energy, E ≘ total energy


def _nH(T, Y, eta):
    return 2. * zeta3 * (T**3.) * eta * (1.-Y) / pi2


def _nHe4(T, Y, eta):
    return 2. * zeta3 * (T**3.) * eta * (Y/4.) / pi2


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


def convert_nucleon(particle):
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


    def _increment(self, index, increment):
        if increment == 0:
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
                "The given particles cannot be included in the spectrum"
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


class TransitionMatrix(object):

    def __init__(self, Kmin, Kmax, N):
        self._sEnergyGrid = EnergyGrid(Kmin, Kmax, N)