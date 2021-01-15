# os
from os import path
# math
from math import log10
# numpy
import numpy as np
# scipy
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
# tarfilfe
import tarfile

# util
from acropolis.utils import cumsimp
# pprint
from acropolis.pprint import print_error
# params
from acropolis.params import hbar
from acropolis.params import NY, NC


def locate_sm_file():
    pkg_dir, _  = path.split(__file__)
    sm_file     = path.join(pkg_dir, "data", "sm.tar.gz")

    return sm_file


class InputInterface(object):

    def __init__(self, input_file):
        # Read the input file
        tf, tc = tarfile.open(input_file, "r:gz"), {}

        # Extract the different files and
        # store them in a dictionary
        for m in tf.getmembers(): tc[m.name] = tf.extractfile(m)

        # READ THE PREVIOUSLY GENERATED DATA
        self._sCosmoData = np.genfromtxt(tc["cosmo_file.dat"]    )
        self._sAbundData = np.genfromtxt(tc["abundance_file.dat"])
        self._sParamData = np.genfromtxt(tc["param_file.dat"],
                                            delimiter="=",
                                            dtype=None,
                                            encoding=None
                                         )

        # Calculate the scale factor and add it
        sf = np.exp( cumsimp(self._sCosmoData[:,0]/hbar, self._sCosmoData[:,4]) )
        self._sCosmoData   = np.column_stack( [self._sCosmoData, sf] )

        # Log the cosmo data for the interpolation
        # ATTENTION: At this point we have to take the
        # absolute value, because dT/dt is negative
        self._sCosmoDataLog = np.log10( np.abs(self._sCosmoData) )
        self._sCosmoDataShp = self._sCosmoData.shape

        # Reshape the abundance data
        self._sAbundData = self._sAbundData.reshape(
                                    (NY, self._sAbundData.size//NY)
                                )
        # Reshape the param data and
        # turn it into a dictionary
        self._sParamData = dict( self._sParamData.reshape(self._sParamData.size) )

        # Check if the data is consistent
        self._check_data()


    def _check_data(self):
        # Check if param_file.dat includes the required parameters
        req_param   = ( "eta" in self._sParamData )
        if not req_param:
            print_error(
                "The mandatory variable 'eta' could not be found in 'param_file.dat'",
                "acropolis.input.InputInterface::_check_data"
            )

        # Check if abundance_file.dat has the correct dimensions
        abund_shape = ( self._sAbundData.shape[0] == NY )
        if not abund_shape:
            print_error(
                "The content of 'abundance_file.dat' does not have the required shape.",
                "acropolis.input.InputInterface::_check_data"
            )

        # Check if cosmo_file.dat has the correct number of columns
        cosmo_shape = ( self._sCosmoDataShp[1] >= NC )
        if not cosmo_shape:
            print_error(
                "The content of 'cosmo_file.dat' does not have the required shape.",
                "acropolis.input.InputInterface::_check_data"
            )


    # 1. COSMO_DATA ###########################################################

    def _interp_cosmo_data(self, val, xc, yc):
        x = self._sCosmoDataLog[:,xc]
        y = self._sCosmoDataLog[:,yc]
        N = self._sCosmoDataShp[0]

        val_log = log10(val)
        # Extract the index corresponding to
        # the data entries above and below 'val'
        ix = np.argmin( np.abs( x - val_log ) )
        if ix == N - 1:
            ix -= 1

        m = (y[ix+1] - y[ix])/(x[ix+1] - x[ix])
        b = y[ix] - m*x[ix]

        return 10**(m*val_log + b)


    def temperature(self, t):
        return self._interp_cosmo_data(t, 0, 1)


    def time(self, T):
        return self._interp_cosmo_data(T, 1, 0)


    def dTdt(self, T):
        return -self._interp_cosmo_data(T, 1, 2)


    def neutrino_temperature(self, T):
        return self._interp_cosmo_data(T, 1, 3)


    def hubble_rate(self, T):
        return self._interp_cosmo_data(T, 1, 4)


    def scale_factor(self, T):
        return self._interp_cosmo_data(T, 1, -1)


    def cosmo_column(self, yc, val, xc=1):
        return self._interp_cosmo_data(val, xc, yc)


    # 2. ABUNDANCE_DATA #######################################################

    def bbn_abundances(self):
        return self._sAbundData

    def bbn_abundances_0(self):
        return self._sAbundData[:,0]


    # 3. PARAM_DATA ###########################################################

    def parameter(self, key):
        return self._sParamData[key]
