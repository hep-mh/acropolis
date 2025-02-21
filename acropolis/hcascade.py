# math
from math import log10, sqrt
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
from acropolis.utils import mavg, all_zero
# params
from acropolis.params import NY
from acropolis.params import approx_zero
from acropolis.params import Kmin, NK_pd


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
            tol = {"rtol": 1e-4, "atol": 0.}
            # -->
            assert np.any( np.isclose([xi[0], xi[0]+1], 1, **tol) )

            # Perform the interpolation
            key = (projectile.value, nucleus.value)
            # -->
            xi_ip_log[key] = interp1d(
                logKi_avg, xi_avg, kind="linear", bounds_error=False, fill_value=(xi_avg[0], np.nan)
            )
    
    return xi_ip_log


# TEMP
def get_Xhdi(temp_grid, k0_grids, dndt_grids, E0, Y, eta, eps=1e-5, max_iter=30):
    start_time = time()
    print_info(
        "Calculating ξ parameters.",
        "acropolis.hcascade.get_Xhdi",
        verbose_level=1
    )

    # Extract the size of the temperature grid
    NT = len(temp_grid)

    # Initialize the result dictionary
    Xhdi_grids = {nid: np.full(NT, approx_zero) for nid in range(NY)}
    
    # Check if hadrodisintegration can be skipped
    if np.all(k0_grids <= Kmin) or all_zero(dndt_grids):
        print_info(
            "Skipped.",
            "acropolis.hcascade.get_Xhdi",
            verbose_level=1
        )

        return Xhdi_grids

    # Calculate the maximal kinetic energy
    Kmax = 2.*np.max(k0_grids) # > Kmin, since np.max(k0_grids) > Kmin

    # Construct the energy grid
    NK = int( NK_pd * log10(Kmax/Kmin) )
    # -->
    egrid = EnergyGrid(Kmin, Kmax, NK)

    # Loop over all temperatures
    for i, T in enumerate(temp_grid):
        progress = 100*i/NT
        print_info(
            "Progress: {:.1f}%".format(progress),
            "acropolis.hcascade.get_Xhdi",
            eol="\r", verbose_level=1
        )

        # Extract the lists of dndt and K0 values
        # corresponding to the current temperature
        dndt_list, K0_list = dndt_grids[i,:], k0_grids[i,:]

        # DEBUG
        assert len(dndt_list) == len(K0_list)

        # Check if the current temperature can be skipped
        # Here: Add 1e-6 to account for floating-point errors
        if all_zero(dndt_list) or np.all(K0_list <= Kmin):
            continue

        # -->
        logK0_list = np.log(K0_list)

        # Construct the xi interpolators
        xi_ip_log = _xi_interpolators(egrid, T, Y, eta, eps, max_iter)

        # Calculate the common prefactor
        pref = 1. / nb(T, eta)

        # Loop over all nuclei
        for nucleus in nuclei:
            nid = nucleus.value + 6 # adapt to the values in nucl.py

            # Calculate the normalization factor
            # TODO: Find a more robust implementation
            Yref = Y[1] if nid in [0, 1] else Y[5]

            # Loop over all projectiles (sum)
            for projectile in projectiles:
                key = (projectile.value, nucleus.value)

                # Loop over all values of dndt and K0
                for dndt, logK0 in zip(dndt_list, logK0_list):
                    Xhdi_grids[nid][i] += pref * dndt * xi_ip_log[key]( logK0 ) / Yref

    end_time = time()
    print_info(
        "Finished after {:.1f}s.".format(end_time - start_time),
        "acropolis.hcascade.get_Xhdi",
        verbose_level=1
    )
        
    return Xhdi_grids # one grid for each nucleus

