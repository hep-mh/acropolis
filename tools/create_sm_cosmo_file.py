#! /usr/bin/env python3

# sys
import sys; sys.path.append('..')
# math
from math import sqrt
# numpy
import numpy as np
# scipy
from scipy.integrate import quad
# matplotlib
import matplotlib.pyplot as plt

# params
from acropolis.params import pi
from acropolis.params import hbar, GN


def dm_energy_density(T):
    rho_d0 = 0.12*8.095894680377574e-35   # DM density today in MeV^4
    T0     = 2.72548*8.6173324e-11        # CMB temperature today in MeV

    return 0.


# Only consider changes in g* due to
# electron-positron annihilation
def sm_energy_density(T):
    return 0.


def hubble_rate(T):
    # Get the total energy density
    rho_tot = sm_energy_density(T) + dm_energy_density(T)
    # Calculate the prefactor
    pref = 8.*pi*GN/3.

    return sqrt( pref * rho_tot )


print( hubble_rate(1) )
