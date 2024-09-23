# math
from math import log10, sqrt
# numpy
import numpy as np
# enum
from enum import Enum

# K ≘ kinetic energy, E ≘ total energy

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


    def bin_edges(self):
        return self._sKbins
    

    def central_values(self):
        return self._sKcent


    def nbins(self):
        return self._sN


class FinalStateSpectrum(object):

    def __init__(self, energy_grid):
        self._sEnergyGrid = energy_grid

        # Extract the number of bins
        self._sN = energy_grid.nbins()

        # Initialize the spectrum
        self._sSpectrum = np.zeros(
            len(Projectiles) * self._sN + len(Nuclei)
        )

        # Initialize a list containing all non-zero entries
        self._sNonZero = set()


    def addp(self, projectile, increment, K):
        index = projectile.value*self._sN + self._sEnergyGrid.index_of(K)
        # -->
        self._sSpectrum[index] += increment

        if increment != 0:
            self._sNonZero.add( index )


    def addn(self, nucleus, increment):
        index = nucleus.value
        # -->
        self._sSpectrum[index] += increment

        if increment != 0:
            self._sNonZero.add( index )


    def non_zero(self):
        for index in self._sNonZero:
            yield ( index, float(self._sSpectrum[index]) )


    def __repr__(self):
        Kcent = self._sEnergyGrid.central_values()

        str_repr = ""
        for i in range(self._sN):
            str_repr += f"{Kcent[i]:.3e} |"

            for projectile in Projectiles:
                j = projectile.value

                str_repr += f" {self._sSpectrum[j*self._sN + i]:.3e}"
            
            str_repr += "\n"
        
        str_repr += "----------x\n"

        for nucleus in Nuclei:
            str_repr += f"{self._sSpectrum[nucleus.value]:.3e} | \n"
        
        return str_repr


class TransitionMatrix(object):

    def __init__(self, Kmin, Kmax, N):
        self._sEnergyGrid = EnergyGrid(Kmin, Kmax, N)