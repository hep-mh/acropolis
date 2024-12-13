# math
from math import log, log10, sqrt
# numpy
import numpy as np
# scipy
from scipy.interpolate import interp1d
# time
from time import time

# cosmo
from acropolis.cosmo import nb
# eloss
import acropolis.eloss as eloss
# etransfer
import acropolis.etransfer as etransfer
# particles
from acropolis.particles import Particles, _nuceq
from acropolis.particles import Np, Nn, projectiles, nuclei
# pprint
from acropolis.pprint import print_info
# utils
from acropolis.utils import mavg
# params
from acropolis.params import NY
from acropolis.params import approx_zero


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


def _etransfer_matrix(egrid, T, Y, eta):
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


def _eloss_matrix(egrid, T, Y, eta):
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


def _transition_matrix(egrid, T, Y, eta, eps=1e-5, max_iter=30):
    M1 = _eloss_matrix    (egrid, T, Y, eta)
    M2 = _etransfer_matrix(egrid, T, Y, eta)

    A = M2 @ M1
    B = A

    # Break after a maximum of
    # 2^max_iter iteration steps
    for n in range(max_iter):
        # Square the previous matrix
        # After n iterations, we have
        # done 2^n cascade steps
        A = A @ A

        # Check for convergence
        Sa, Sb = A[-Nn:,:-Nn], B[-Nn:,:-Nn]

        diff = 1
        if np.array_equal(Sa == 0, Sb == 0):
            Sa = Sa[Sb != 0]
            Sb = Sb[Sb != 0]

            # -->
            diff = np.max( np.abs(Sa - Sb)/Sa )

        # Break if convergence has been
        # archieved
        if diff < eps:
            break

        # Store the current matrix for
        # comparison in the next step
        B = A
    
    return A, n


def _xi_interpolators(egrid, T, Y, eta, eps=1e-5, max_iter=30):
    # Extract the number of bins
    N = egrid.nbins()

    # -->
    window_size = N//20

    # Calculate the transition matrix
    M, n = _transition_matrix(egrid, T, Y, eta, eps, max_iter)

    # DEBUG
    assert (n <= max_iter - 1)

    # Extract the grid of kinetic energies
    logKi = np.log( egrid.central_values() )
    # -->
    logKi_avg = mavg(logKi, window_size)

    xi_ip_log = {}
    # Loop over all projectiles and nuclei
    for projectile in projectiles:
        for nucleus in nuclei:
            # Extract the xi grid
            xi = np.array([
                M[nucleus.value, projectile.value*N + i] for i in range(N)
            ]) + ( 1. if nucleus == _nuceq(projectile) else 0. )
            # -->
            xi_avg = mavg(xi, window_size)

            # DEBUG (check lowest energy bin)
            assert (xi[0] < 1e-4 or xi[0] - 1 < 1e-4)

            # Perform the interpolation
            key = (projectile.value, nucleus.value)
            # -->
            xi_ip_log[key] = interp1d(
                logKi_avg, xi_avg, kind="linear"
            )
    
    return xi_ip_log


# TEMP
def get_Xhdi(E0, K0, temp_grid, Y, eta, dndt, eps=1e-5, max_iter=30):
    # TODO: Move this to params.py?
    NK_pd, Kmin = 50, 3
    
    logK0 = log( max(K0, Kmin) )
    # Use K0 = Kmin for K0 < Kmin

    # Extract the size of the temperature grid
    NT = len(temp_grid)

    # Calculate the size of the energy grid
    Kmax = 2.*E0
    # -->
    NK = int( NK_pd * log10(Kmax/Kmin) )
    
    # Construct the energy grid
    egrid = EnergyGrid(Kmin, Kmax, NK)

    start_time = time()
    print_info(
        "Calculating ξ parameters for hadrodisintegration.",
        "acropolis.hcascade.get_Xhdi",
        verbose_level=1
    )

    Xhdi_grids = {nid: np.full(NT, approx_zero) for nid in range(NY)}

    # TEMP: Uncomment to run without hdi
    #return Xhdi_grids

    # Loop over all temperatures
    for i, T in enumerate(temp_grid):
        progress = 100*i/NT
        print_info(
            "Progress: {:.1f}%".format(progress),
            "acropolis.hcascade.get_Xhdi",
            eol="\r", verbose_level=1
        )

        # Construct the xi interpolators
        xi_ip_log = _xi_interpolators(egrid, T, Y, eta, eps, max_iter)

        # Calculate the baryon-to-photon ratio
        _nb = nb(T, eta)

        # Loop over all nuclei
        for nucleus in nuclei:
            nid = nucleus.value + 6 # adapt to the values in nucl.py

            # TODO: Generalize
            Yref = Y[1] if nid in [0, 1] else Y[5]

            # -->
            pref = dndt / ( Yref * _nb )

            # Loop over all projectiles (sum)
            for projectile in projectiles:
                key = (projectile.value, nucleus.value)

                Xhdi_grids[nid][i] += pref * xi_ip_log[key]( logK0 )
        
    end_time = time()
    print_info(
        "Finished after {:.1f}s.".format(end_time - start_time),
        "acropolis.hcascade.get_Xhdi",
        verbose_level=1
    )
        
    return Xhdi_grids # one grid for each nucleus

