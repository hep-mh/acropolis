# math
from math import pi
# scipy
from scipy.special import zeta


# Only parameters that specify a default value are
# meant to be changed by the user, i.e. everything
# under ALGORITHM-SPECIFIC PARAMETERS


# PHYSICAL CONSTANTS ################################################

# The fine-structure constant
alpha = 1./137.036

# The electron mass (in MeV)
me = 0.511
# -->
me2 = me**2.

# The classical electron radius (in 1/MeV)
re = alpha/me

# The gravitational constant (in 1/MeV^2)
GN = 6.70861e-45

# The reduced Planck constant (in MeV*s)
hbar = 6.582119514e-22

# The speed of light (in cm/s)
c_si = 2.99792458e10

# The neutron lifetime (in s)
tau_n = 8.794e2 # pre PDG2020: 8.802e2

# The tritium lifetime (in s)
# T_(1/2) = 3.885e8
tau_t = 5.605e8


# MATHEMATICAL CONSTANTS ############################################

# The Riemann-Zeta function at point 3
zeta3 = zeta(3.)

# The beautiful value of pi, squared
pi2 = pi**2.


# INTERPOLATION-SPECIFIC PARAMETERS #################################

# Boundery values
Emin_log, Tmin_log = 0, -6
Emax_log, Tmax_log = 3, -1
# Number of entries...
num_pd = 150 # ...per decade
Enum = (Emax_log - Emin_log)*num_pd
Tnum = (Tmax_log - Tmin_log)*num_pd


# ALGORITHM-SPECIFIC PARAMETERS #####################################

# NOTE: The default parameters below are
# optimized for parameter scans that cover
# several orders of magnitude, in which
# case a precision > 0.1% is justified
# To instead archieve < 0.1% precision,
# we recommend setting, which comes at 
# a runtime penalty
# NE_pd = 150
# NT_pd =  30

# The number of nuclei that are
# considered in the Boltzmann equation
# for non-thermal nucleosynthesis
# Default: 9
NY = 9

# The number of mandatory columns in
# 'cosmo_file.dat'
# Default: 5
NC = 5

# The minimum energy for the different
# spectra (in MeV) originating from the
# electromagnetic cascade
# This value should not be larger than
# the minimal nucleon-interaction threshold
# of 1.586627 MeV
# (reaction_id: 15 in 'astro-ph/0211258')
# Default: 1.5
Emin = 1.5

# The value that is used for ~ 0.
# Default: 1e-200
approx_zero = 1e-200

# The relative accuracy for each integral
# Default: 1e-3
eps = 1e-3

# The maximal value of x = Ephb/T to avoid
# overflow in the exponential function
# Default: 200
Ephb_T_max = 200.

# The maximal value of E/EC up to which the
# integration is performed when using the
# full spectrum with exponential suppression
E_EC_max = 10.

# The number of points per decade for
# the energy grid, which is used within
# the solution of the cascade equation
# Default: 120
NE_pd  = 120
# The minimal number of points for the
# energy grid
# Default: 10
NE_min = 10

# The number of points per decade for
# the temperature grid, which is used
# for the interpolation of the nuclear
# reaction rates
# Default: 20
NT_pd = 20
