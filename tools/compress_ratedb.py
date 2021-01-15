#! /usr/bin/python3

# sys
import sys; sys.path.append('..')
# numpy
import numpy as np

# params
from acropolis.params import approx_zero

# Read the full dataset
data_ls = np.loadtxt('rates.db.ls')

data_ls[data_ls==0] = approx_zero

# Drop the first two colums and log
data = np.log10(data_ls[:,2:4])

np.savetxt('rates.db', data)
