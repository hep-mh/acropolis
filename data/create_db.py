#! /usr/bin/python3

# sys
import sys; sys.path.append('..')
# multiprocessing
from multiprocessing import Pool, cpu_count
# numpy
import numpy as np

# cascade
from acropolis.cascade import _PhotonReactionWrapper, _ElectronReactionWrapper
# params
from acropolis.params import Emin_log, Emax_log, Enum, Tmin_log, Tmax_log, Tnum


cores = cpu_count()

phg = _PhotonReactionWrapper  ([], 0., None)
elg = _ElectronReactionWrapper([], 0., None)

Tg = np.logspace(Tmin_log, Tmax_log, Tnum)
Eg = np.logspace(Emin_log, Emax_log, Enum)

ratedb = np.zeros((Tnum*Enum            , 4))
kerndb = np.zeros((Tnum*Enum*(Enum+1)//2, 6))

# Get the mode as a command-line argument
# 0: rates, 1: kernels
mode = int( sys.argv[1] ) if len(sys.argv) != 1 else 0


# RATES #######################################################################

def get_rates(params):
    (T, E) = params

    ph_rpc = phg._rate_pair_creation  (E, T)
    el_ric = elg._rate_inverse_compton(E, T)

    return [T, E, ph_rpc, el_ric]

if mode == 0:
    for i, T in enumerate(Tg):
        print( str(i+1) + "/" + str(Tnum) )

        params = [(T, E) for E in Eg]

        with Pool(processes=cores) as pool:
            parallel_results = pool.map(get_rates, params, 1)
            pool.terminate()

        off = ratedb.shape[0]//Tnum
        ratedb[i*off:(i+1)*off,:] = parallel_results

    np.savetxt('rates.db.ls', ratedb)


# KERNELS #####################################################################

def get_kernels(params):
    (T, E, Ep) = params

    ph_kic = phg._kernel_inverse_compton(E, Ep, T)
    el_kpc = elg._kernel_pair_creation  (E, Ep, T)
    el_kic = elg._kernel_inverse_compton(E, Ep, T)

    return [T, E, Ep, ph_kic, el_kpc, el_kic]

if mode == 1:
    for i, T in enumerate(Tg):
        print( str(i+1) + "/" + str(Tnum) )

        params = [(T, E, Ep) for E in Eg for Ep in Eg if Ep >= E]

        with Pool(processes=cores) as pool:
            parallel_results = pool.map(get_kernels, params, 1)
            pool.terminate()

        off = kerndb.shape[0]//Tnum
        kerndb[i*off:(i+1)*off,:] = parallel_results

    np.savetxt('kernels.db.ls', kerndb)
