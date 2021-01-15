# numpy
import numpy as np
# itertools
from itertools import product
# multiprocessing
from multiprocessing import Pool, cpu_count

# pprint
from acropolis.pprint import print_error
# params
from acropolis.params import NY
# models
from acropolis.models import AbstractModel


class ScanParameter(object):

    def __init__(self, ivalue, fvalue, num, spacing="log", fast=False):
        self._sInitialValue  = ivalue
        self._sFinalValue    = fvalue
        self._sSpacingMode   = spacing
        self._sFastParameter = fast
        self._sNumPoints     = num


    def get_range(self):
        if   self._sSpacingMode == "log":
            return np.logspace( self._sInitialValue, self._sFinalValue, self._sNumPoints )
        elif self._sSpacingMode == "lin":
            return np.linspace( self._sInitialValue, self._sFinalValue, self._sNumPoints )


    def is_fast(self):
        return self._sFastParameter


class BufferedScanner(object):

    def __init__(self, model, **kwargs):
        # Store the requested model
        # self._sWrapper(...) creates
        # a new instance of this class
        if not issubclass(model, AbstractModel):
            print_error(str(model) + " is not a subclass of 'AbstractModel'")

        self._sModel = model

        # Define the various sets
        self._sFixed = {}      # Fixed parameter
        self._sScanp = {}      # Scan parameters...
        self._sFastf = {}      # ...that allow for fast scanning

        # Define the number of scan parameters
        self._sNp = 0

        # Parse the keyword arguments and build up the
        # sets 'self._sFixed' and 'self._sScanp'
        self._parse_arguments(**kwargs)

        #######################################################################

        # Generate the keys for the scan parameters
        self._sScanp_id = list( self._sScanp.keys() )

        # Determine the parameter for the parallelisation
        # In case there is a 'fast' parameter, this whould be
        # one of the 'non-fast' parameters
        #
        # Sort the keys in order for the fast parameter
        # to be at position 0
        list.sort( self._sScanp_id, key=lambda id: self._sFastf[id], reverse=True )
        # Choose the last parameter, which in any case is not the
        # 'fast' parameter and therefore can be calculated in parallel
        self._sPP_id = self._sScanp_id[-1]

        # Determine the 'fast' parameter
        self._sFP_id = self._sScanp_id[ 0]


    def _parse_arguments(self, **kwargs):
        # Loop over the different parameters
        for key in kwargs.keys():
            param = kwargs[key]

            # Extract the fixed values
            if   type(param) in [int, float]:
                self._sFixed[key] = float(param)
            # Extract the scan parameters
            elif isinstance(param, ScanParameter):
                self._sNp += 1

                # Save the 'is_fast' status of all parameters
                self._sFastf[key] = param.is_fast()
                # Save the relevant range of all paremeters
                self._sScanp[key] = param.get_range()
            else:
                print_error(
                    "All parameters must either be 'int', 'float' or an instance of 'ScanParameter'",
                    "BufferedScanner._parse_arguments"
                )

        if list( self._sFastf.values() ).count(True) > 1:
            print_error(
                "Using more than one 'fast' parameter is not yet supported",
                "BufferedScanner._parse_arguments"
            )


    def rescale_matp_buffer(self, buffer, factor):
        return (factor*buffer[0], buffer[1])


    def _perform_non_parallel_scan(self, pp):
        # Generate all possible parameter combinations, thereby
        # NOT! including the parameter used for the parallelisation
        scanp_ls = product( *[self._sScanp[id] for id in self._sScanp_id[:-1]] )

        # TODO: Extend to more than two parameters
        dx = len( self._sScanp[self._sFP_id] )
        dy = len( self._sScanp_id ) + 3*NY # + 1
        results = np.zeros( ( dx, dy ) )

        matpb, matpf = None, False
        print(pp)
        # Loop over the non-parallel parameter(s)
        for count, scanp in enumerate(scanp_ls):
            # Define the set that contains only scan parameters
            scanp_set = dict( zip(self._sScanp_id, scanp) )
            scanp_set.update( {self._sPP_id: pp} )

            # Define the set that contains all parameters
            fullp_set = scanp_set.copy()
            fullp_set.update( self._sFixed )

            # Initialize the model wrapper of choice
            model = self._sModel(**fullp_set)

            scanp_set_id_0 = scanp_set[self._sScanp_id[0]]
            # Rescale the rates with the 'fast' parameter
            # TODO: Only do this if a fast parameter exists
            if count != 0 and matpf == True:
                factor = scanp_set_id_0/fastp
                model.set_matp_buffer( self.rescale_matp_buffer(matpb, factor) )

            ##############################################################
            Yb = model.run_disintegration()
            ##############################################################

            # Rescale the nuclear-rate buffer existent
            if count == 0:
                matpb = model.get_matp_buffer()
                matpf = matpb is not None
                # matpb might still be None if E0 < Emin

                fastp = scanp_set_id_0

            # For the output, use the following format
            # 1. The 'parallel' parameter
            # 2. All 'non fast/parallel' parameters
            # 3. The 'fast' parameter
            sortp_ls = list( zip( scanp_set.keys(), scanp_set.values() ) )
            list.sort(sortp_ls, key=lambda el: self._sFastf[ el[0] ])
            sortp_ls = [ el[1] for el in sortp_ls ]

            results[count] = [*sortp_ls, *Yb.transpose().reshape(Yb.size)]

        return results


    def perform_scan(self, cores=1):
        num_cpus = cpu_count() if cores == -1 else cores
        with Pool(processes=num_cpus) as pool:
            # Loop over all possible combinations, by...
            #   ...1. looping over the 'parallel' parameter (map)
            #   ...2. looping over all parameter combinations,
            #   thereby exclusing the 'parallel' parameter (perform_non_parallel_scan)
            parallel_results = pool.map(self._perform_non_parallel_scan, self._sScanp[self._sPP_id], 1)
            pool.terminate()

        parallel_results = np.array(parallel_results)
        old_shape = parallel_results.shape
        parallel_results.shape = (old_shape[0]*old_shape[1], len( self._sScanp_id ) + 3*NY) # + 1)

        return parallel_results
