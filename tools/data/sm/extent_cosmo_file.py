# sys
import sys; sys.path.append("../../../")
# numpy
import numpy as np
# scipy
from scipy.special import zeta

# utils
from acropolis.utils import cumsimp
# params
from acropolis.params import hbar


# Load the data without nb_etaf
data = np.loadtxt("cosmo_file.dat~")

# Extract the temperature
T = data[:,1]
# -->
T0 = T[-1]

# Calculate the scale factor
sf = np.exp( cumsimp(data[:,0]/hbar, data[:,4]) )
# -->
sf0 = sf[-1]

# Calculate nb_etaf
nb_etaf = 2. * zeta(3.) * (T0**3) * (sf0/sf)**3. / (np.pi**2.)

# -->
np.savetxt("cosmo_file.dat", np.column_stack([data, nb_etaf]))
