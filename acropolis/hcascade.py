# math
from math import log10, sqrt
# numpy
import numpy as np

# eloss
import acropolis.eloss as eloss
# etransfer
import acropolis.etransfer as etransfer
# particles
from acropolis.particles import Particles
from acropolis.particles import Np, Nn


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


    def central_values(self):
        return self._sKcent


    def __getitem__(self, i):
        return self._sKcent[i]


def _get_etransfer_matrix(egrid, T, Y, eta):
    # Extract the number of bins
    N = egrid.nbins()

    # Calculate the dimension of the matrix
    d = Np*N + Nn

    # Initialize the matrix
    matrix = np.zeros( (d, d) )
    for i in range(Nn):
        matrix[-(i+1),-(i+1)] = 1.

    # Loop over all possible projectiles
    for projectile in [Particles.PROTON, Particles.NEUTRON]:
        spectra = etransfer.process(egrid, projectile, T, Y, eta)

        for i, spectrum in enumerate(spectra):
            # DEBUG
            assert np.isclose(spectrum.baryon_number(), 0.)

            k = N*projectile.value + i
            # Loop over the spectrum and fill the matrix
            for (j, val) in spectrum.non_zero():
                matrix[j,k] = val
    
    return matrix


def _get_eloss_matrix(egrid, T, Y, eta):
    # Extract the number of bins
    N = egrid.nbins()

    # Calculate the dimension of the matrix
    d = Np*N + Nn

    # Initialize the matrix
    matrix = np.zeros( (d, d) )
    for i in range(Nn):
        matrix[-(i+1),-(i+1)] = 1.

    fallback = None
    # Loop over all possible projectiles
    for projectile in [Particles.PROTON, Particles.NEUTRON]:
        spectra, fallback = eloss.process(egrid, projectile, T, Y, eta, fallback)

        for i, spectrum in enumerate(spectra):
            # DEBUG
            assert np.isclose(spectrum.baryon_number(), 0.)

            k = N*projectile.value + i
            # Loop over the spectrum and fill the matrix
            for (j, val) in spectrum.non_zero():
                matrix[j,k] = val
    
    return matrix


def get_transition_matrix(egrid, T, Y, eta, n):
    pass