# math
from math import log10, sqrt
# numpy
import numpy as np
# enum
from enum import Enum


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


class ParticleSpectra(object):

    def __init__(self, Emin, Emax, N):
        if Emax <= Emin:
            raise ValueError(
                "The value of Emax must be larger than the one of Emin"
            )

        self._sND   = len(Projectiles) * N + len(Nuclei)

        # Save the number of bins
        self._sN = N

        # Save the minimal and maximal energy
        self._sEmin      = Emin
        self._sEmax      = Emax
        # -->
        self._sEminLog10 = log10(Emin)
        self._sEmaxLog10 = log10(Emax)

        # Generate the different energy bins
        self._sEbins = np.logspace( log10(Emin), log10(Emax), N+1 )
        
        # Calculate the central value of each bin
        fac = sqrt(self._sEbins[1]/self._sEbins[0])
        # -->
        self._sEcent = self._sEbins[:-1] * fac

        #print(self._sEbins, self._sEcent, fac, self._sEcent[496])

        # Initialize the spectrum
        self._sSpectrum = np.zeros(self._sND)
    

    def _find_index(self, energy):
        if not self._sEmin <= energy <= self._sEmax:
            raise ValueError(
                "The given energy does not lie within the appropriate range"
            )
        
        if energy == self._sEmax:
            return self._sN - 1
        
        return int(
            self._sN * ( log10(energy) - self._sEminLog10 )  / ( self._sEmaxLog10 - self._sEminLog10 )
        )


    def add_to_projectile(self, projectile, increment, energy):
        j = int( self._find_index(energy) )

        #print(j)

        self._sSpectrum[projectile.value*self._sN + j] += increment


    def add_to_nucleus(self, nucleus, increment):
        self._sSpectrum[nucleus.value] += increment
    

    def __repr__(self):
        pass