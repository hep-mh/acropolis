# math
from math import pi
# scipy
from scipy.special import zeta


# FLAGS #############################################################

# If this flag is set to 'True',
# the pregenerated databases
# will be used to interpolate
# the different reaction rates
# Default: True
usedb = True

# If this flag is set to 'True',
# additional output is printed
# to the screen
# Default: True
verbose = True

# If this flag is set to 'True',
# additional debug information
# is printed to the screen
# Default: False
debug = False


# PHYSICAL CONSTANTS ################################################

# The fine-structure constant
alpha = 1./137.036

# The electron mass (in MeV)
me = 0.511

# The electron mass squared (in MeV^2)
me2 = me**2.

# The classical electron radius (in 1/MeV)
re = alpha/me

# The gravitational constant (in 1/MeV^2)
GN = 6.70861e-45

# The reduced Planck constant (in MeV*s)
hbar = 6.582119514e-22

# The speed of light (in cm/s)
c_si = 2.99792458e10

# The muon lifetime (in s)
tau_m = 2.1969811e-6

# The neutron lifetime (in s)
tau_n = 8.802e2

# The lifetime of tritium (in s)
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

# The number of elements/isotops that
# are considered in the calculation
NY = 9

# The number of mandatory columns in
# 'cosmo_file.dat'
NC = 5

# Minimum energy for the different spectra (in MeV)
# This value should not be larger than the minimal
# nucleon-interaction threshold of 1.586627 MeV
# (reaction_id: 15 in 'astro-ph/0211258')
Emin = 1.5

# The value that is used for 'approximately' zero
# Default: 1e-200
approx_zero = 1e-200

# The relative accuracy for each integral
# Default: 1e-3
eps = 1e-3

# The maximal value of x = Ephb/T to avoid
# overflow in the exponential function
# Default: 200
Ephb_T_max = 200.

# The energy in units of EC at which to
# cutoff strongly suppressed spectra
# Default: 500.
E_EC_cut = 500.

# The number of points per decade for
# the energy grid, which is used within
# the solution of the cascade equation
# Default: 150
NE_pd  = 150
# The minimal number of points for
# the energy grid
# Default: 10
NE_min = 10

# The number of points per decade for
# the temperature grid, which us used
# for the interpolation of the thermal
# nuclear rates
# Default: 50
NT_pd = 50
