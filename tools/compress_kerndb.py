#! /usr/bin/python3

# sys
import sys; sys.path.append('..')
# numpy
import numpy as np

# params
from acropolis.params import approx_zero

# Read the full dataset
data_ls = np.loadtxt('kernels.db.ls')

data_ls[data_ls==0] = approx_zero

# Drop the first three colums and log
data = np.log10(data_ls[:,3:6])

np.savetxt('kernels.db', data)
