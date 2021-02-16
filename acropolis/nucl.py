# functools
from functools import wraps
# math
from math import sqrt, log10, log, exp
# numpy
import numpy as np
# scipy
from scipy.integrate import quad
from scipy.integrate import IntegrationWarning
from scipy.linalg import expm
# time
from time import time
# warnings
import warnings

# util
from acropolis.utils import LogInterp
# pprint
from acropolis.pprint import print_error, print_warning, print_info
# params
from acropolis.params import me, me2, hbar, tau_n, tau_t
from acropolis.params import approx_zero, eps
from acropolis.params import NT_pd
# cascade
from acropolis.cascade import SpectrumGenerator


# A dictionary containing all relevant nuclei, or more precisely
# all nuclei that appear in the reactions specified in '_reactions'
# or in the decays specified in '_decays'
_nuclei = {
    "n"  : 0,
    "p"  : 1,
    "d"  : 2,
    "t"  : 3,
    "He3": 4,
    "He4": 5,
    "Li6": 6,
    "Li7": 7,
    "Be7": 8
}


# A dictionary containing all relevant pdi reactions
# This dict can be modified if new reactions are added
# In this case, also remember to modify the function
# 'NuclearReactor.get_cross_section(reaction_id, E)'
_reactions = {
    1 : "d+a>n+p",
    2 : "t+a>n+d",
    3 : "t+a>n+p+n",
    4 : "He3+a>p+d",
    5 : "He3+a>n+p+p",
    6 : "He4+a>p+t",
    7 : "He4+a>n+He3",
    8 : "He4+a>d+d",
    9 : "He4+a>n+p+d",
    10: "Li6+a>n+p+He4",
    11: "Li6+a>X",
    12: "Li7+a>t+He4",
    13: "Li7+a>n+Li6",
    14: "Li7+a>n+n+p+He4",
    15: "Be7+a>He3+He4",
    16: "Be7+a>p+Li6",
    17: "Be7+a>p+p+n+He4"
}
# A dictionary containing all accociated threshold
# energies. All energies are given in MeV
_eth = {
    1 :  2.224573,
    2 :  6.257248,
    3 :  8.481821,
    4 :  5.493485,
    5 :  7.718058,
    6 : 19.813852,
    7 : 20.577615,
    8 : 23.846527,
    9 : 26.071100,
    10:  3.698892,
    11: 15.794685,
    12:  2.467032,
    13:  7.249962,
    14: 10.948850,
    15:  1.586627,
    16:  5.605794,
    17:  9.304680
}


# A dictionary containing all relevant decays
_decays = {
    1: "n>p",
    2: "t>He3"
    #3: "Be7>Li7"
}
# A dictionary containing all accociated lifetimes.
# All lifetimes are given in s
_tau = {
    1: tau_n,
    2: tau_t
    #3: 6.634e6         # T_(1/2) = 4.598e6
}


# The number of relevant nucleons
_nnuc = len( _nuclei )
# The number of relevant reaction
_nrec = len( _reactions )
# The number of relevant decays
_ndec = len( _decays )


# A list containing all reactions id's
_lrid = list( _reactions.keys() )
# A list containing all decay id's
_ldid = list( _decays.keys() )


def _extract_signature(reaction_str):
    sp = reaction_str.split(">")

    # Extract the inital state
    istate = _nuclei[ sp[0].split("+")[0] ]

    # Extract the final state
    #
    # Set up a default dictionary (to store the final nucleons)
    fstate = { i:0 for i in range( _nnuc ) }

    # Fill the dictionary; Afterwards this variable stores
    # the number of all nuleids in the final state
    # The result looks somewhat like
    # {0:<number_of_n_in_fs>, 1:<number_of_p_in_fs, 3:...}
    for Nstr in sp[1].split("+"):
        if Nstr in _nuclei: # Do not consider particles like X in reaction 11
            fstate[ _nuclei[Nstr] ] += 1

    return istate, fstate


# A dictionary containing the signatures
# for the different pdi reactions
_rsig = { rid:_extract_signature( _reactions[rid] ) for rid in _lrid }
# A dictionary containing the signatures
# for the different decays
_dsig = { did:_extract_signature( _decays[ did]   ) for did in _ldid }


def _convert_mb_to_iMeV2(f_in_mb):
    # Define the conversion factor
    cf = 2.56819e-6

    # Define the wrapper function
    @wraps(f_in_mb)
    def f_in_iMeV2(*args, **kwargs):
        return cf * f_in_mb(*args, **kwargs)

    return f_in_iMeV2


class NuclearReactor(object):

    def __init__(self, s0, sc, temp_rg, e0, ii):
        self._sII = ii

        # A dictionary containing the BBN parameters
        self._sY0 = self._sII.bbn_abundances_0()

        # The injection energy
        self._sE0 = e0

        # The baryon-to-photon ratio at the time of the CMB
        self._sEta = self._sII.parameter("eta")

        # The source terms without the detla function
        self._sS0 = s0

        # The FSR source terms
        self._sSc = sc

        # The approximate decay temperature of the mediator
        self._sTrg  = temp_rg

        # An instance of 'Spectrum_Generator' in order to calculate
        # the photon spectrum in the function 'get_reaction_rate(reaction_id, T)'
        self._sGen = SpectrumGenerator(self._sY0, self._sEta)

    # BEGIN REACTIONS ###############################################

    def _generic_expr(self, E, Q, N, p1, p2, p3):
        # Below threshold, return 0
        if E < Q:
            return 0.

        return N * (Q**p1) * (E-Q)**p2 / (E**p3)


    def _da_np(self, E):
        Q = _eth[1]

        # Below threshold, return 0.
        if E < Q:
            return 0.

        return 18.75 * ( ( sqrt( Q*(E-Q) )/E )**3. + 0.007947*( sqrt( Q*(E-Q) )/E )**2. * ( (sqrt(Q) - sqrt(0.037))**2./( E - Q + 0.037 ) ) )


    def _ta_nd(self, E):
        return self._generic_expr(E, _eth[2], 9.8, 1.95, 1.65, 3.6)


    def _ta_npn(self, E):
        return self._generic_expr(E, _eth[3], 26.0, 2.6, 2.3, 4.9)


    def _He3a_pd(self, E):
        return self._generic_expr(E, _eth[4], 8.88, 1.75, 1.65, 3.4)


    def _He3a_npp(self, E):
        return self._generic_expr(E, _eth[5], 16.7, 1.95, 2.3, 4.25)


    def _He4a_pt(self, E):
        return self._generic_expr(E, _eth[6], 19.5, 3.5, 1.0, 4.5)


    def _He4a_nHe3(self, E):
        # Prefactor changed from 17.1mb to 20.7mb to account for EXFOR data
        # cf. 'hep-ph/0604251 [38]' for more details
        return self._generic_expr(E, _eth[7], 20.7, 3.5, 1.0, 4.5)


    def _He4a_dd(self, E):
        return self._generic_expr(E, _eth[8], 10.7, 10.2, 3.4, 13.6)


    def _He4a_npd(self, E):
        return self._generic_expr(E, _eth[9], 21.7, 4.0, 3.0, 7.0)


    def _Li6a_npHe4(self, E):
        # Prefactor changed from 104.0mb to 143.0mb to account for EXFOR data
        # cf. 'hep-ph/0604251 [39]' for more details
        return self._generic_expr(E, _eth[10], 143.0, 2.3, 4.7, 7.0)


    def _Li6a_XA3(self, E):
        Q = _eth[11]

        # Template for the exponential term of the form
        # -- N * exp( -1/2*( (E - Eb)/Ed )^2 ) --
        # E, Eb, Ed in MeV, N unitless, result unitless
        def exp_term(E, N, Eb, Ed):
            return N * exp( -(1./2.)*( (E - Eb)/Ed )**2. )

        # __genereic_expr returns 0. below threshold
        return self._generic_expr(E, Q, 38.1, 3.0, 2.0, 5.0) * ( exp_term(E, 3.7, 19.0, 3.5) + exp_term(E, 2.75, 30.0, 3.0) + exp_term(E, 2.2, 43.0, 5.0) )


    def _Li7a_tHe4(self, E):
        Q = _eth[12]

        # Below threshold, return 0.
        if E < Q:
            return 0.

        # Define the excess energy relative to the threshold
        Ecm = E - Q

        # Define the closing polynomial in Ecm...
        pEcm = 1. + 2.2875*(Ecm**2.) - 1.1798*(Ecm**3.) + 2.5279*(Ecm**4.)

        # ...and for pEcm < 0, return 0. (This should be a continuous transition)
        # For this reaction, however, there should not be any problems
        # The roots of pEcm are all imaginary according to 'WolframAlpha'
        # Therefore, do not perform the check in order not to loose performance
        #if pEcm < 0.:
        #    return 0.

        return 0.105 * ( 2371./(E**2) ) * exp( -2.5954/sqrt(Ecm) ) * exp(-2.056*Ecm) * pEcm


    def _Li7a_nLi6(self, E):
        Q = _eth[13]

        # Below threshold, return 0.
        if E < Q:
            return 0.

        return self._generic_expr(E, Q, 0.176, 1.51, 0.49, 2.0) + self._generic_expr(E, Q, 1205.0, 5.5, 5.0, 10.5) + 0.06/( 1. + ( (E - Q - 7.46)/0.188 )**2. )


    def _Li7a_nnpHe4(self, E):
        return self._generic_expr(E, _eth[14], 122.0, 4.0, 3.0, 7.0)


    def _Be7a_He3He4(self, E):
        Q = _eth[15]

        # Below threshold, return 0.
        if E < Q:
            return 0.

        # Define the excess energy relative to the threshold
        Ecm = E - Q

        # Define the closing polynomial in Ecm...
        pEcm = 1. - 0.428*(Ecm**2.) + 0.534*(Ecm**3.) - 0.115*(Ecm**4.)

        # ... and for pEcm < 0, return 0. (This should be a continuous transition)
        # In fact, pEcm has a root at Ecm ~ 3.92599MeV > Q according to 'WolframAlpha',
        # This root lies above threshold, which is why we have to explicitly perform
        # the corresponding check for this reaction
        if pEcm < 0.:
            return 0.

        return 0.504 * ( 2371./(E**2.) ) * exp( -5.1909/sqrt(Ecm) ) * exp(-0.548*Ecm) * pEcm


    def _Be7a_pLi6(self, E):
        Q = _eth[16]

        return self._generic_expr(E, Q, 32.6, 10.0, 2.0, 12.0) + self._generic_expr(E, Q, 2.27e6, 8.8335, 13.0, 21.8335)


    def _Be7a_ppnHe4(self, E):
        return self._generic_expr(E, _eth[17], 133.0, 4.0, 3.0, 7.0)

    # END REACTIONS #################################################


    @_convert_mb_to_iMeV2
    def get_cross_section(self, reaction_id, E):
        # There is no switch statement in python :(
        if reaction_id ==  1: return self._da_np(E)              #  1. d + a -> n + p
        if reaction_id ==  2: return self._ta_nd(E)              #  2. t + a -> n + d
        if reaction_id ==  3: return self._ta_npn(E)             #  3. t + a -> 2n + p
        if reaction_id ==  4: return self._He3a_pd(E)            #  4. He3 + a -> p + d
        if reaction_id ==  5: return self._He3a_npp(E)           #  5. He3 + a -> n + 2p
        if reaction_id ==  6: return self._He4a_pt(E)            #  6. He4 + a -> p + t
        if reaction_id ==  7: return self._He4a_nHe3(E)          #  7. He4 + a -> n + He3
        if reaction_id ==  8: return self._He4a_dd(E)            #  8. He4 + a -> 2d
        if reaction_id ==  9: return self._He4a_npd(E)           #  9. He4 + a -> n + p + d
        if reaction_id == 10: return self._Li6a_npHe4(E)         # 10. Li6 + a -> n + p + He4
        if reaction_id == 11: return self._Li6a_XA3(E)           # 11. Li7 + a -> X + A3
        if reaction_id == 12: return self._Li7a_tHe4(E)          # 12. Li7 + a -> t + He4
        if reaction_id == 13: return self._Li7a_nLi6(E)          # 13. Li7 + a -> n + Li6
        if reaction_id == 14: return self._Li7a_nnpHe4(E)        # 14. Li7 + a -> 2n + p + He4
        if reaction_id == 15: return self._Be7a_He3He4(E)        # 15. Be7 + a -> He3 + He4
        if reaction_id == 16: return self._Be7a_pLi6(E)          # 16. Be7 + a -> p + Li6
        if reaction_id == 17: return self._Be7a_ppnHe4(E)        # 17. Be7 + a -> 2p + n + He4


        # If no match is found, return 0.
        print_error(
            "Reaction with reaction_id" + str(reaction_id) + "does not exist.",
            "acropolis.nucl.NuclearReactor.get_cross_section"
        )


    def _pdi_rates(self, T):
        EC = me2/(22.*T)
        # Calculate the maximal energy
        Emax = min( self._sE0, 10.*EC )
        # For E > me2/T >> EC, the spectrum
        # is strongly suppressed

        # Define a dict containing the rates
        # for the photodisintegration reactions
        # key = reaction_id (from _reactions)
        pdi_rates = {rid:approx_zero for rid in _lrid}

        # Calculate the spectra for the given temperature
        xsp, ysp = self._sGen.nonuniversal_spectrum(
                            self._sE0, self._sS0, self._sSc, T
                        )

        # Interpolate the photon spectrum (in log-log space)
        # With this procedure it should be sufficient to perform
        # a linear interpolation, which also has less side effects
        # lgFph = interp1d( np.log(xsp), np.log(ysp), kind='linear' )
        Fph = LogInterp(xsp, ysp)
        # Calculate the kernel for the integration in log-space
        def Fph_s(log_E, rid):
            E = exp( log_E ); return Fph( E ) * E * self.get_cross_section(rid, E)

        # Define the total rate of interactions altering the photon spectrum,
        # evaluated at the relevant injection energy E0
        rate_photon_E0 = self._sGen.rate_photon(self._sE0, T)

        # Calculate the different rates by looping over all available reaction_id's
        for rid in _lrid:
            # Do not perform the integral for energies below
            # threshold or for strongly suppressed spectra
            if _eth[rid] > Emax:
                continue

            # Perform the integration from the threshold energy to Emax
            with warnings.catch_warnings(record=True) as w:
                log_Emin, log_Emax = log(_eth[rid]), log(Emax)
                I_Fs = quad(Fph_s, log_Emin, log_Emax, epsrel=eps, epsabs=0, args=(rid,))

                if len(w) == 1 and issubclass(w[0].category, IntegrationWarning):
                    print_warning(
                        "Slow convergence when calculating the pdi rates " +
                        "@ rid = %i, T = %.3e, E0 = %.3e, Eth = %.3e" % (rid, T, self._sE0, _eth[rid]),
                        "acropolis.nucl.NuclearReactor._thermal_rates_at"
                    )

            # Calculate the 'delta-term'
            I_dt = self._sS0[0](T)*self.get_cross_section(rid, self._sE0)/rate_photon_E0

            # Add the delta term and save the result
            pdi_rates[rid] = I_dt + I_Fs[0]

        # Go home and play
        return pdi_rates


    def get_pdi_grids(self):
        (Tmin, Tmax) = self._sTrg

        NT = int(log10(Tmax/Tmin)*NT_pd)

        # Create an array containing all
        # temperature points ('log spacing')
        Tr = np.logspace( log10(Tmin), log10(Tmax), NT )

        # Create a dictionary to store the pdi
        # rates for all reactions and temperatures
        pdi_grids = {rid:np.zeros(NT) for rid in _lrid}

        start_time = time()
        print_info(
            "Calculating non-thermal spectra and reaction rates.",
            "acropolis.nucl.NuclearReactor.get_thermal_rates"
        )

        # Loop over all the temperatures and
        # calculate the corresponding thermal rates
        for i, Ti in enumerate(Tr):
            print_info(
                "Progress: " + str( int( 1e3*i/NT )/10 ) + "%",
                "acropolis.nucl.NuclearReactor.get_thermal_rates",
                eol="\r"
            )
            rates_at_i = self._pdi_rates(Ti)
            # Loop over the different reactions
            for rid in _lrid:
                pdi_grids[rid][i] = rates_at_i[rid]

        end_time = time()
        print_info(
            "Finished after " + str( int( (end_time - start_time)*10 )/10 ) + "s."
        )

        # Go get some sun
        return (Tr, pdi_grids)


class MatrixGenerator(object):

    def __init__(self, temp, pdi_grids, ii):
        self._sII = ii

        # Save the thermal rates
        self._sTemp     = temp
        self._sPdiGrids = pdi_grids

        # Save the appropriate temperature range
        (self._sTmin, self._sTmax) = temp[0], temp[-1]

        # Interpolate the thermal rates
        self._sPdiIp = self._interp_pdi_grids()


    def _interp_pdi_grids(self):
        # A dict containing all interp. rates; key = reaction_id (from _sReactions)
        interp_grids = {}
        for rid in _lrid:
            # Interpolate the rates between
            # Tmin and Tmax in log-log space
            interp_grids[rid] = LogInterp(
                self._sTemp, self._sPdiGrids[rid], base=10. # fill_value=0.
            )

        return interp_grids


    def _pref_ij(self, state, i, j):
        ris = state[0] # initial state of the reaction
        rfs = state[1] # final state of the reaction

        # Find reactions/decays that have
        # 1. the nucleid 'nr=i' in the final state
        # 2. the nucleid 'nc=j' in the initial state
        if ris == j and rfs[i] != 0:
            return rfs[i]

        # Find reactions/decays that have
        # nc = nr in the initial state
        # (diagonal entries)
        if i == j and ris == j:
            return -1.

        return 0.


    def _pdi_kernel_ij(self, i, j, T):
        matij = 0.

        for rid in _lrid:
            matij += self._pref_ij(_rsig[rid], i, j) * self._sPdiIp[rid](T)

        # Incorporate the time-temperature relation and return
        return matij/( self._sII.dTdt(T) )


    def _dcy_kernel_ij(self, i, j, T):
        matij = 0.

        for did in _ldid:
            matij += self._pref_ij(_dsig[did], i, j) * hbar/_tau[did]

        # Incorporate the time-temperature relation and return
        return matij/( self._sII.dTdt(T) )


    def get_matp(self, T):
        # Generate empty matrices
        mpdi, mdcy = np.zeros( (_nnuc, _nnuc) ), np.zeros( (_nnuc, _nnuc) )

        start_time = time()
        print_info(
            "Running non-thermal nucleosynthesis.",
            "acropolis.nucl.MatrixGenerator.get_matp"
        )

        nt = 0
        # Rows: Loop over all relevant nuclei
        for nr in range(_nnuc):
            # Columns: Loop over all relevant nuclei
            for nc in range(_nnuc):
                nt += 1
                print_info(
                    "Progress: " + str( int( 1e3*nt/_nnuc**2 )/10 ) + "%",
                    "acropolis.nucl.MatrixGenerator.get_matp",
                    eol="\r"
                )

                # Define the kernels for the integration in log-log space
                Ik_pdi = lambda y: self._pdi_kernel_ij( nr, nc, exp(y) ) * exp(y)
                Ik_dcy = lambda y: self._dcy_kernel_ij( nr, nc, exp(y) ) * exp(y)

                # Perform the integration (in log-log space)
                mpdi[nr, nc] = quad(Ik_pdi, log(self._sTmax), log(T), epsrel=eps, epsabs=0)[0]
                mdcy[nr, nc] = quad(Ik_dcy, log(self._sTmax), log(T), epsrel=eps, epsabs=0)[0]

        end_time = time()
        print_info(
            "Finished after " + str( int( (end_time - start_time)*1e4 )/10 ) + "ms."
        )

        return (mpdi, mdcy)


    def get_final_matp(self):
        return self.get_matp( self._sTmin )
