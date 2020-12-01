# gzip
import gzip
# pickle
import pickle
# os
from os import path
# numba
import numba as nb
# time
from time import time

# pprint
from acropolis.pprint import print_info
# params
from acropolis.params import usedb
from acropolis.params import Emin_log, Emax_log, Enum
from acropolis.params import Tmin_log, Tmax_log, Tnum


def import_data_from_db():
    pkg_dir, _ = path.split(__file__)
    db_file    = path.join(pkg_dir, "data", "rates.db.gz")

    ratedb = None
    if not usedb or not path.exists(db_file):
        return ratedb

    start_time = time()
    print_info(
        "Extracting and reading database files.",
        "acropolis.db.import_data_from_db"
    )

    ratefl = gzip.open(db_file, "rb")
    ratedb = pickle.load(ratefl)
    ratefl.close()

    end_time = time()
    print_info(
        "Finished after " + str( int( (end_time - start_time)*1e4 )/10 ) + "ms."
    )

    return ratedb


def in_rate_db(E_log, T_log):
    if (Emin_log <= E_log <= Emax_log) and (Tmin_log <= T_log <= Tmax_log):
        return True

    return False


def in_kernel_db(E_log, Ep_log, T_log):
    if (Emin_log <= E_log <= Emax_log) \
      and (Emin_log <= Ep_log <= Emax_log) \
      and (Tmin_log <= T_log <= Tmax_log):
        return True

    return False


@nb.jit(cache=True)
def _get_E_log(i):
    return Emin_log + (Emax_log - Emin_log)*i/(Enum - 1)


@nb.jit(cache=True)
def _get_T_log(i):
    return Tmin_log + (Tmax_log - Tmin_log)*i/(Tnum - 1)


@nb.jit(cache=True)
def _get_E_index(E_log):
    index = int( ( Enum - 1 ) * ( E_log - Emin_log ) / ( Emax_log - Emin_log ) )

    # For points at the upper boundary, i+1 does not exist
    return index if index != Enum -1 else index - 1


@nb.jit(cache=True)
def _get_T_index(T_log):
    index = int( ( Tnum - 1 ) * ( T_log - Tmin_log ) / ( Tmax_log - Tmin_log ) )

    # For points at the upper boundary, i+1 does not exist
    return index if index != Tnum - 1 else index - 1


@nb.jit(cache=True)
def interp_rate_db(rate_db, id, E_log, T_log):
    # Extract the correct index for the datafile
    c = {
        'ph:rate_pair_creation'  : 0,
        'el:rate_inverse_compton': 1
    }[id]

    # Calculate the respective indices in the interpolation file
    iE, iT  = _get_E_index(E_log), _get_T_index(T_log)

    # Perform the interpolation according to the wikipedia page:
    # https://en.wikipedia.org/wiki/Bilinear_interpolation
    x , y = T_log, E_log
    x0, y0 = _get_T_log(iT  ), _get_E_log(iE  )
    x1, y1 = _get_T_log(iT+1), _get_E_log(iE+1)
    xd, yd = (x-x0)/(x1-x0), (y-y0)/(y1-y0)

    # Define the index function
    k = lambda jT, jE: jT*Enum + jE

    c00 = rate_db[ k(iT  , iE  ) ][c]
    c10 = rate_db[ k(iT+1, iE  ) ][c]
    c01 = rate_db[ k(iT  , iE+1) ][c]
    c11 = rate_db[ k(iT+1, iE+1) ][c]

    d = (x0-x1)*(y0-y1)

    a0 = ( c00*x1*y1 - c01*x1*y0 - c10*x0*y1 + c11*x0*y0 )/d
    a1 = ( -c00*y1 + c01*y0 + c10*y1 - c11*y0 )/d
    a2 = ( -c00*x1 + c01*x1 + c10*x0 - c11*x0 )/d
    a3 = ( c00 - c01 - c10 + c11 )/d

    return 10.**( a0 + a1*x + a2*y + a3*x*y )


@nb.jit(cache=True)
def interp_kernel_db(kernel_db, id, E_log, Ep_log, T_log):
    c = {
        'ph:kernel_inverse_compton': 0,
        'el:kernel_pair_creation'  : 1,
        'el:kernel_inverse_compton': 2
    }[id]

    # Calculate the respective indices in the interpolation file
    iE, iEp, iT  = _get_E_index(E_log), _get_E_index(Ep_log), _get_T_index(T_log)

    # Perform the interpolation according to the wikipedia page:
    # https://en.wikipedia.org/wiki/Trilinear_interpolation
    x , y , z = T_log, E_log, Ep_log
    x0, y0, z0, = _get_T_log(iT  ), _get_E_log(iE  ), _get_E_log(iEp  )
    x1, y1, z1, = _get_T_log(iT+1), _get_E_log(iE+1), _get_E_log(iEp+1)
    xd, yd, zd = (x-x0)/(x1-x0), (y-y0)/(y1-y0), (z-z0)/(z1-z0)

    # Define the index function
    k = lambda jT, jE, jEp: jT*Enum*(Enum+1)//2 + jE*Enum - (jE-1)*jE//2 + (jEp - jE)

    c000 = kernel_db[ k(iT  , iE  , iEp  ) ][c]
    c100 = kernel_db[ k(iT+1, iE  , iEp  ) ][c]
    c010 = kernel_db[ k(iT  , iE+1, iEp  ) ][c]
    c001 = kernel_db[ k(iT  , iE  , iEp+1) ][c]
    c110 = kernel_db[ k(iT+1, iE+1, iEp  ) ][c]
    c101 = kernel_db[ k(iT+1, iE  , iEp+1) ][c]
    c111 = kernel_db[ k(iT+1, iE+1, iEp+1) ][c]
    c011 = kernel_db[ k(iT  , iE+1, iEp+1) ][c]

    c00 = c000*(1.-xd) + c100*xd
    c01 = c001*(1.-xd) + c101*xd
    c10 = c010*(1.-xd) + c110*xd
    c11 = c011*(1.-xd) + c111*xd

    c0 = c00*(1.-yd) + c10*yd
    c1 = c01*(1.-yd) + c11*yd

    c = c0*(1.-zd) + c1*zd

    d = (x0-x1)*(y0-y1)*(z0-z1)

    a0 = ( -c000*x1*y1*z1 + c001*x1*y1*z0 + c010*x1*y0*z1 - c011*x1*y0*z0 + \
           c100*x0*y1*z1 - c101*x0*y1*z0 - c110*x0*y0*z1 + c111*x0*y0*z0)/d
    a1 = ( c000*y1*z1 - c001*y1*z0 - c010*y0*z1 + c011*y0*z0 + \
           -c100*y1*z1 + c101*y1*z0 + c110*y0*z1 - c111*y0*z0)/d
    a2 = ( c000*x1*z1 - c001*x1*z0 - c010*x1*z1 + c011*x1*z0 + \
           -c100*x0*z1 + c101*x0*z0 + c110*x0*z1 - c111*x0*z0)/d
    a3 = ( c000*x1*y1 - c001*x1*y1 - c010*x1*y0 + c011*x1*y0 + \
           -c100*x0*y1 + c101*x0*y1 + c110*x0*y0 - c111*x0*y0)/d
    a4 = ( -c000*z1 + c001*z0 + c010*z1 - c011*z0 + c100*z1 - c101*z0 - c110*z1 + c111*z0 )/d
    a5 = ( -c000*y1 + c001*y1 + c010*y0 - c011*y0 + c100*y1 - c101*y1 - c110*y0 + c111*y0 )/d
    a6 = ( -c000*x1 + c001*x1 + c010*x1 - c011*x1 + c100*x0 - c101*x0 - c110*x0 + c111*x0 )/d
    a7 = ( c000 - c001 - c010 + c011 - c100 + c101 + c110 - c111 )/d

    return 10.**( a0 + a1*x + a2*y + a3*z + a4*x*y + a5*x*z + a6*y*z + a7*x*y*z )
