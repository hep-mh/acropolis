# numpy
import numpy as np
# time
from time import time, sleep
# itertools
from itertools import product
# multiprocessing
from multiprocessing import Pool, cpu_count

# pprint
from acropolis.pprint import print_info, print_error
# params
from acropolis.params import NY
# models
from acropolis.models import AbstractModel


class _Batch(object):

    def __init__(self, length, is_fast):
        self.length  = length
        self.is_fast = is_fast


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
        # self._sModel(...) afterwards creates
        # a new instance of the requested model
        if not issubclass(model, AbstractModel):
            print_error(
                model.__name__ + " is not a subclass of AbstractModel",
                "acropolis.scans.BufferedScanner.__init__"
            )

        self._sModel = model

        #######################################################################

        # Initialize the various sets
        self._sFixed = {}      # Fixed parameter
        self._sScanp = {}      # Scan parameters...
        self._sFastf = {}      # ...w/o fast scanning

        # Initialize the number of scan parameters...
        self._sNP      = 0 # (all)
        self._sNP_fast = 0 # (only fast)

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
        # Sort the keys in order for the fast parameters
        # to be at he beginning of the array
        list.sort( self._sScanp_id, key=lambda id: self._sFastf[id], reverse=True )
        # Choose the last parameter, which in any case is not the
        # 'fast' parameter and therefore can be calculated in parallel
        self._sId_pp = self._sScanp_id[-1]

        #######################################################################

        # Extract the dimension of parallel/sequential jobs
        self._sDp, self._sDs = 0, 0
        for id in self._sScanp_id:
            if id == self._sId_pp:
                self._sDp += len( self._sScanp[id] )
            else:
                self._sDs += len( self._sScanp[id] )


    def _parse_arguments(self, **kwargs):
        # Loop over the different parameters
        for key in kwargs.keys():
            param = kwargs[key]

            # Extract the fixed values
            if   type(param) in [int, float]:
                self._sFixed[key] = float(param)
            # Extract the scan parameters
            elif isinstance(param, ScanParameter):
                self._sNP += 1

                # Save the relevant range of all paremeters
                self._sScanp[key] = param.get_range()
                # Save the 'is_fast' status of all parameters
                self._sFastf[key] = param.is_fast()
            else:
                print_error(
                    "All parameters must either be 'int', 'float' or an instance of 'ScanParameter'",
                    "acropolis.scans.BufferedScanner._parse_arguments"
                )

        # Get the number of 'fast' parameters (Np_fast <= Np - 1)
        self._sNP_fast = list( self._sFastf.values() ).count(True)

        # ERRORS for not-yet-implemented features (TODO) ################################
        if self._sNP_fast > 1 or self._sNP != 2:
            print_error(
                "Currently only exactly 2 scan parameters with <= 1 fast parameter are supported!",
                "acropolis.scans.BufferedScanner._parse_arguments"
            )


    # TODO!!!
    def _build_batches(self):
        # Generate all possible parameter combinations, thereby
        # NOT! including the parameter used for the parallelisation
        scanp_ls = product( *[self._sScanp[id] for id in self._sScanp_id[:-1]] )
        # Right now: One sequential parameter, which is either fast or not
        scanp_bt = [ _Batch(self._sDs, self._sNP_fast != 0), ]

        return scanp_ls, scanp_bt


    def rescale_matp_buffer(self, buffer, factor):
        return (factor*buffer[0], buffer[1])


    def _perform_non_parallel_scan(self, pp):
        # Build the relevant batches
        scanp_ls, scanp_bt = self._build_batches()

        # Determine the dimensions of the 'result grid'
        dx = self._sDs                    # rows
        dy = self._sNP + 3*NY             # columns
        results = np.zeros( ( dx, dy ) )

        # Initialize the buffer
        matpb = None

        nb, ib = 0, 0
        # Loop over the non-parallel parameter(s)
        for i, scanp in enumerate(scanp_ls):
            # Store the current batch
            batch = scanp_bt[nb]

            # Check if a reset is required
            reset_required = (ib == 0)

            # Define the set that contains only scan parameters
            scanp_set = dict( zip(self._sScanp_id, scanp) )
            scanp_set.update( {self._sId_pp: pp} )
            # Define the set that contains all parameters
            fullp_set = scanp_set.copy()
            fullp_set.update( self._sFixed )

            # Initialize the model of choice
            model = self._sModel(**fullp_set)

            scanp_set_id_0 = scanp_set[self._sScanp_id[0]]
            # Rescale the rates with the 'fast' parameter
            # but only if the current parameter is 'fast'
            if batch.is_fast and (not reset_required):
                if matpb is not None:
                # matpb might still be None if E0 < Emin
                # save, since parameters determining the
                # injection energy, should never be fast
                    factor = scanp_set_id_0/fastp
                    model.set_matp_buffer( self.rescale_matp_buffer(matpb, factor) )

            ##############################################################
            Yb = model.run_disintegration()
            ##############################################################

            # Reset the buffer/rescaling
            if batch.is_fast and reset_required:
                matpb = model.get_matp_buffer()

                fastp = scanp_set_id_0

            # For the output, use the following format
            # 1. The 'non fast' parameters
            # 3. The 'fast' parameters
            sortp_ls = list( zip( scanp_set.keys(), scanp_set.values() ) )
            list.sort(sortp_ls, key=lambda el: self._sFastf[ el[0] ]) # False...True
            sortp_ls = [ el[1] for el in sortp_ls ]

            results[i] = [*sortp_ls, *Yb.transpose().reshape(Yb.size)]

            # Update the batch index
            if ib == batch.length - 1: # next batch
                ib =  0
                nb += 1
            else:
                ib += 1

        return results


    def perform_scan(self, cores=1):
        num_cpus = cpu_count() if cores == -1 else cores

        start_time = time()
        print_info(
            "Running scan for {} on {} cores.".format(self._sModel.__name__, num_cpus),
            "acropolis.scans.BufferedScanner.perform_scan",
            verbose_level=3
        )

        with Pool(processes=num_cpus) as pool:
            # Loop over all possible combinations, by...
            #   ...1. looping over the 'parallel' parameter (map)
            #   ...2. looping over all parameter combinations,
            #   thereby exclusing the 'parallel' parameter (perform_non_parallel_scan)
            async_results = pool.map_async(
                self._perform_non_parallel_scan, self._sScanp[self._sId_pp], 1
            )

            progress = 0
            while ( progress < 100 ) or ( not async_results.ready() ):
                progress = 100*( self._sDp - async_results._number_left )/self._sDp
                print_info(
                    "Progress: {:.1f}%".format(progress),
                    "acropolis.scans.BufferedScanner.perform_scan",
                    eol="\r", verbose_level=3
                )

                sleep(1)

            parallel_results = async_results.get()
            pool.terminate()

        parallel_results = np.array(parallel_results)
        old_shape = parallel_results.shape
        parallel_results.shape = (old_shape[0]*old_shape[1], len( self._sScanp_id ) + 3*NY) # + 1)

        end_time = time()
        print_info(
            "Finished after {:.1f}min.".format( (end_time - start_time)/60 ),
            "acropolis.scans.BufferedScanner.perform_scan",
            verbose_level=3
        )

        return parallel_results
