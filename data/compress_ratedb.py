#! /usr/bin/python3

# sys
import sys; sys.path.append('..')
# numpy
import numpy as np
# pickle
import pickle

# params
from acropolis.params import approx_zero

# Read the full dataset
data_ls = no.loadtxt('rates.db.ls')
# with open('rates.db.ls', 'rb') as f:
#     data_ls = pickle.load(f)

data_ls[data_ls==0] = approx_zero

# Drop the first to colums and log
data = np.log10(data_ls[:,2:4])

np.savetxt('rates.db', data)

# Save the compressed data as a pickle file
# with open('rates.db', 'wb') as rf:
#     pickle.dump(data, rf)
