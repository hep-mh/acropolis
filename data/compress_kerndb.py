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
with open('kernels.db.ls', 'rb') as f:
    data_ls = pickle.load(f)

data_ls[data_ls==0] = approx_zero

# Drop the first two colums and log
data = np.log10(data_ls[:,3:6])

# Save the compressed data as a pickle file
with open('kernels.db', 'wb') as rf:
    pickle.dump(data, rf)
