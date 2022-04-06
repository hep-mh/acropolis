# os
from os import path
# math
from math import log10
# numpy
import numpy as np
# tarfilfe
import tarfile
# abc
from abc import ABC, abstractmethod

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


def data_from_file(filename):
    # Read the input file
    tf, tc = tarfile.open(filename, "r:gz"), {}

    # Extract the different files and
    # store them in a dictionary
    for m in tf.getmembers(): tc[m.name] = tf.extractfile(m)

    # READ THE PREVIOUSLY GENERATED DATA
    cosmo_data = np.genfromtxt(tc["cosmo_file.dat"]    )
    abund_data = np.genfromtxt(tc["abundance_file.dat"])
    param_data = np.genfromtxt(tc["param_file.dat"],
                                        delimiter="=",
                                        dtype=None,
                                        encoding=None
                                     )

    return InputData(cosmo_data, abund_data, param_data)


class AbstractData(ABC):

    @abstractmethod
    def get_cosmo_data(self):
        pass

    @abstractmethod
    def get_abund_data(self):
        pass

    @abstractmethod
    def get_param_data(self):
        pass


class InputData(AbstractData):

    def __init__(self, cosmo_data, abund_data, param_data):
        self._sCosmoData = cosmo_data
        self._sAbundData = abund_data
        self._sParamData = param_data


    def get_cosmo_data(self):
        return self._sCosmoData


    def get_abund_data(self):
        return self._sAbundData


    def get_param_data(self):
        return self._sParamData


class InputInterface(object):

    def __init__(self, input_data):
        # If input_data is a filename, extract the data first
        if type(input_data) == str:
            input_data = data_from_file(input_data)

        # Extract the provided input data
        self._sCosmoData = input_data.get_cosmo_data()
        self._sAbundData = input_data.get_abund_data()
        self._sParamData = input_data.get_param_data()

        # Calculate the scale factor and add it
        sf = np.exp( cumsimp(self._sCosmoData[:,0]/hbar, self._sCosmoData[:,4]) )
        self._sCosmoData = np.column_stack( [self._sCosmoData, sf] )

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

    def _find_index(self, x, x0):
        # Returns an index ix such that x0
        # lies between x[ix] and x[ix+1]
        ix = np.argmin( np.abs( x - x0 ) )

        # Check the edge of the array
        if ix == self._sCosmoDataShp[0] - 1:
            # In this case, the condition
            # below is always False
            # --> No additional -1
            ix -= 1

        # If x0 is not between ix and ix+1,...
        if not (x[ix] <= x0 <= x[ix+1] or x[ix] >= x0 >= x[ix+1]):
            # ...it must be between ix-1 and ix
            ix -= 1

        return ix


    def _interp_cosmo_data(self, val, xc, yc):
        # ATTENTION: To ensure maximal performance,
        # it is assumed that x is already sorted in
        # either increasing or decreasing order
        x = self._sCosmoDataLog[:,xc]
        y = self._sCosmoDataLog[:,yc]

        val_log = log10(val)

        # Extract the index closest to 'val_log'
        ix = self._find_index(x, val_log)

        m = (y[ix+1] - y[ix])/(x[ix+1] - x[ix])
        b = y[ix] - m*x[ix]

        return 10.**(m*val_log + b)


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


    def cosmo_range(self):
        return ( min(self._sCosmoData[:,1]), max(self._sCosmoData[:,1]) )


    # 2. ABUNDANCE_DATA #######################################################

    def bbn_abundances(self):
        return self._sAbundData

    def bbn_abundances_0(self):
        return self._sAbundData[:,0]


    # 3. PARAM_DATA ###########################################################

    def parameter(self, key):
        return self._sParamData[key]
