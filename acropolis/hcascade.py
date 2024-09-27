# math
from math import log10, sqrt
# numpy
import numpy as np
# enum
from enum import Enum

# params
from acropolis.params import zeta3, pi2
from acropolis.params import mp, mn, mD, mT, mHe3, mHe4


# K ≘ kinetic energy, E ≘ total energy


def _nH(T, Y, eta):
    return 2. * zeta3 * (T**3.) * eta * (1.-Y) / pi2



def _nHe4(T, Y, eta):
    return 2. * zeta3 * (T**3.) * eta * (Y/4.) / pi2


class Projectiles(Enum):
    PROTON       = 0
    NEUTRON      = 1
    ANTI_PROTON  = 2
    ANTI_NEUTRON = 3


class Targets(Enum):
    PROTON = 0
    ALPHA  = 1


class Nuclei(Enum):
    DEUTERIUM = -4
    TRITIUM   = -3
    HELIUM3   = -2
    HELIUM4   = -1


def is_particle(particle):
    return (particle in Projectiles) or (particle in Targets) or (particle in Nuclei)


# All masses in MeV
mass = {
    Projectiles.PROTON      : mp,
    Projectiles.NEUTRON     : mn,
    Projectiles.ANTI_PROTON : mp,
    Projectiles.ANTI_NEUTRON: mn,

    Targets.PROTON: mp,
    Targets.ALPHA : mHe4,

    Nuclei.DEUTERIUM: mD,
    Nuclei.TRITIUM  : mT,
    Nuclei.HELIUM3  : mHe3,
    Nuclei.HELIUM4  : mHe4
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


    def _add_projectile(self, projectile, increment, K):
        if projectile not in Projectiles:
            raise ValueError(
                "Invalid projectile"
            )

        if K < self._sEnergyGrid.lower_edge():
            return

        index = projectile.value*self._sN + self._sEnergyGrid.index_of(K)
        # -->
        self._increment(index, increment)


    def _add_nucleus(self, nucleus, increment):
        if nucleus not in Nuclei:
            raise ValueError(
                "Invalid nucleus"
            )
        
        # TODO Check survival

        index = nucleus.value
        # -->
        self._increment(index, increment)


    def add(self, particle, increment, Ki):
        # Projectiles
        if   particle in Projectiles:
            self._add_projectile(particle, increment, Ki)
        
        # Nuclei
        elif particle in Nuclei:
            self._add_nucleus(particle, increment)
        
        # Targets
        elif particle == Targets.PROTON:
            self._add_projectile(Projectiles.PROTON, increment, Ki)

        elif particle == Targets.ALPHA:
            self._add_nucleus(Nuclei.HELIUM4, increment)


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

            for projectile in Projectiles:
                j = projectile.value

                str_repr += f" {self.at(j*self._sN + i):.3e}"
            
            str_repr += "\n"
        
        str_repr += "----------x\n"

        for nucleus in Nuclei:
            str_repr += f"{self.at(nucleus.value):.3e} | \n"
        
        return str_repr


class TransitionMatrix(object):

    def __init__(self, Kmin, Kmax, N):
        self._sEnergyGrid = EnergyGrid(Kmin, Kmax, N)