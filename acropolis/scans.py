# functools
from functools import partial
# numpy
import numpy as np
# time
from time import time, sleep
# multiprocessing
from multiprocessing import Pool, cpu_count

# pprint
from acropolis.pprint import print_info, print_error
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
        self._sModelClass = model.func if type(model) is partial else model
        
        # Store the requested model
        # self._sModel(...) afterwards creates
        # a new instance of the requested model
        if not issubclass(self._sModelClass, AbstractModel):
            print_error(
                self._sModelClass.__name__ + " is not a subclass of AbstractModel",
                "acropolis.scans.BufferedScanner.__init__"
            )

        self._sModel = model
        
        #######################################################################

        # Initialize the sets that store the verious parameters
        self._sFixed = {}  # Fixed parameters
        self._sScanp = {}  # Scan parameters

        # Initialize the number of scan parameters
        self._sNP      = 0 # (all)
        self._sNP_fast = 0 # (only fast)

        # Initialize the key of the 'fast' parameter
        self._sFast_id = None

        # Parse the keyword arguments and build up the
        # sets 'self._sFixed' and 'self._sScanp'
        self._parse_arguments(**kwargs)

        #######################################################################

        # Extract the keys of the scan parameters
        self._sScanp_id = list( self._sScanp.keys() )


    def _parse_arguments(self, **kwargs):
        # Loop over the different parameters
        for key in kwargs.keys():
            param = kwargs[key]

            # Extract the scan parameters
            if isinstance(param, ScanParameter):
                self._sNP += 1

                # Save the relevant parameter range
                self._sScanp[key] = param.get_range()

                # Check if the parameter is 'fast'
                if param.is_fast():
                    self._sNP_fast += 1

                    # Save the corresponding id
                    self._sFast_id = key
            # Extract the fixed parameters
            else:
                self._sFixed[key] = param

        # Exit in case of unsupported scenarios
        if self._sNP_fast > 1 or self._sNP != 2:
            print_error(
                "The class BufferedScanner only supports 2d parameter scans with <= 1 fast parameters!",
                "acropolis.scans.BufferedScanner._parse_arguments"
            )


    def _rescale_matp_buffer(self, buffer, factor):
        return (factor*buffer[0], buffer[1])


    def _run_single(self, scanp_set):
        assert self._sFast_id is None and self._sNP_fast == 0 and self._sNP == 2
        
        # Define the set that contains all parameters
        fullp_set = scanp_set.copy()
        fullp_set.update( self._sFixed )

        ##############################################################
        Yb = self._sModel(**fullp_set).run_disintegration()
        ##############################################################

        # Sort the scan parameters to guarantee a consistent output
        sorted_scanp = [scanp_set[key] for key in sorted(scanp_set)]

        return [*sorted_scanp, *Yb.transpose().reshape(Yb.size)]


    def _run_batch(self, pp_set):
        assert self._sFast_id is not None and self._sNP_fast == 1 and self._sNP == 2

        # Extract the range of 'fast' parameters to loop over
        fast_params = self._sScanp[self._sFast_id]

        # Initialize the results array
        results = []

        # Initialize the buffer
        matpb = None

        # Initialize fast_param_0
        fast_param_0 = fast_params[0]

        for fast_param in fast_params:
            # Define the set that contains all parameters
            scanp_set = pp_set.copy()
            scanp_set.update( {self._sFast_id: fast_param} )
            # -->
            fullp_set = scanp_set.copy()
            fullp_set.update( self._sFixed )

            # Initialize the model of choice
            model = self._sModel(**fullp_set)

            if matpb is not None:
                # matpb might still be None if E0 < Emin

                factor = fast_param/fast_param_0
                model.set_matp_buffer( self._rescale_matp_buffer(matpb, factor) )

            ##############################################################
            Yb = model.run_disintegration()
            ##############################################################

            # Set up the rescaling
            if matpb is None:
                matpb = model.get_matp_buffer()

                fast_param_0 = fast_param
            
            # Sort the scan parameters to guarantee a consistent output
            sorted_scanp = [scanp_set[key] for key in sorted(scanp_set)]
            # -->
            results.append(
                [*sorted_scanp, *Yb.transpose().reshape(Yb.size)]
            )
        
        return results


    def perform_scan(self, cores=1):
        assert self._sNP_fast <= 1 and self._sNP == 2

        num_cpus = cpu_count() if cores == -1 else cores

        start_time = time()
        print_info(
            "Running scan for {} on {} cores.".format(self._sModelClass.__name__, num_cpus),
            "acropolis.scans.BufferedScanner.perform_scan",
            verbose_level=3
        )

        key1, key2 = self._sScanp_id
        # Generate all possible parameter combinations
        if self._sNP_fast == 0:
            map_params = [
                {key1: val1, key2: val2} for val1 in self._sScanp[key1] \
                                         for val2 in self._sScanp[key2]
            ]

            map_func = self._run_single
        else: # self._sNP_fast == 1
            # Extract the non-fast, i.e. parallel, parameter
            keyp = key1 if key1 != self._sFast_id else key2

            map_params = [
                {keyp: valp} for valp in self._sScanp[keyp]
            ]

            map_func = self._run_batch

        with Pool(processes=num_cpus) as pool:
            # Loop over all possible parameter combinations
            async_results = pool.map_async(map_func, map_params, 1)

            progress = 0
            # Track and print the progress
            while ( progress < 100 ) or ( not async_results.ready() ):
                progress = 100*( len(map_params) - async_results._number_left )/len(map_params)
                print_info(
                    "Progress: {:.1f}%".format(progress),
                    "acropolis.scans.BufferedScanner.perform_scan",
                    eol="\r", verbose_level=3
                )

                sleep(1)

            parallel_results = async_results.get()
            pool.terminate()

        end_time = time()
        print_info(
            "Finished after {:.1f}min.".format( (end_time - start_time)/60 ),
            "acropolis.scans.BufferedScanner.perform_scan",
            verbose_level=3
        )

        parallel_results = np.array(parallel_results)

        # Reshape the array and return
        old_shape = parallel_results.shape
        # -->
        new_shape = (
            np.prod(old_shape[:-1]), old_shape[-1]
        )

        return parallel_results.reshape(new_shape)
