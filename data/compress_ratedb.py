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
with open('rates.db.ls', 'rb') as f:
    data_ls = pickle.load(f)

data_ls[data_ls==0] = approx_zero

# Drop the first to colums and log
data = np.log10(data_ls[:,2:4])

# Save the compressed data as a pickle file
with open('rates.db', 'wb') as rf:
    pickle.dump(data, rf)

# Save the compressed data
# np.savetxt('rates.db', data, fmt='%.8g')
#
# # Compress the file
# with open('rates.db', 'rb') as db:
#     with gzip.open('rates.db' + '.gz', 'wb') as dbgz:
#         shutil.copyfileobj(db, dbgz)
