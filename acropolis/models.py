# math
from math import pi, exp, log, log10
# numpy
import numpy as np
# scipy
from scipy.linalg import expm
# abc
from abc import ABC, abstractmethod

# input
from acropolis.input import InputInterface, locate_sm_file
# hcascade
from acropolis.hcascade import get_Xhdi
# nucl
from acropolis.nucl import NuclearReactor, MatrixGenerator
# params
from acropolis.params import zeta3
from acropolis.params import hbar, c_si, me2, alpha, tau_t, mp, mn
from acropolis.params import Emin, NY, NT_pd
# flags
import acropolis.flags as flags
# pprint
from acropolis.pprint import print_info, print_warning


class AbstractModel(ABC):

    def __init__(self, E0, ii):
        # Initialize the input interface
        self._sII  = ii

        # The injection energy
        self._sE0 = E0

        # The baryon-to-photon ratio
        self._sEta = ii.parameter("eta")

        # The temperature range that is used for the calculation
        self._sTrg = self._temperature_range()

        # The temperature grid
        self._sT = self._temperature_grid()

        # TEMP
        self._sdndt = np.array([
            self._dndt_proton(T) for T in self._sT
        ])
        self._sK0 = np.array([
            self._K0_proton(T) for T in self._sT
        ])

        # The relevant source terms
        (self._sS0, self._sSc) = self._source_terms()

        # A buffer for high-performance scans
        self._sMatpBuffer = None


    def run_disintegration(self):
        Y0 = self._sII.bbn_abundances()

        # DEBUG
        assert np.all( Y0[0,:]/Y0[1,:] < 1e-3 )

        # Print a warning if the injection energy
        # is larger than 1GeV, as this might lead
        # to wrong results
        if not flags.universal and int( self._sE0 ) > 1e3:
            print_warning(
                "Injection energy > 1 GeV. Results cannot be trusted.",
                "acropolis.models.AbstractMode.run_disintegration"
            )

        # Print a warning if the temperature range
        # of the model is not covered by the data
        # in cosmo_file.dat
        cf_temp_rg = self._sII.temperature_range()
        if not (cf_temp_rg[0] <= self._sTrg[0] <= self._sTrg[1] <= cf_temp_rg[1]):
            print_warning(
                "Temperature range not covered by input data. Results cannot be trusted.",
                "acropolis.models.AbstractMode.run_disintegration"
            )

        # If the energy is below all thresholds,
        # simply return the initial abundances
        if self._sE0 <= Emin:
            print_info(
                "Injection energy is below all thresholds. No calculation required.",
                "acropolis.models.AbstractModel.run_disintegration",
                verbose_level=1
            )

            return self._postd_matrix() @ Y0

        # Calculate the different transfer matrices
        ###########################################

        # 1. Decays before Tmax
        pred_mat  = self._pred_matrix()
        # 2. Photodisintegration + Decay
        pdi_mat   = self._pdi_matrix()
        # 3. Decays after Tmin
        postd_mat = self._postd_matrix()

        # Calculate the total transfer matrix
        transf_mat = postd_mat @ pdi_mat @ pred_mat

        # -->
        return transf_mat @ Y0


    def _source_terms(self):
        # Collect the different source terms, i.e. ...
        # ...the 'delta' source terms and...
        s0 = [
            self._source_photon_0  ,
            self._source_electron_0,
            self._source_positron_0
        ]
        # ...the continous source terms
        sc = [
            self._source_photon_c  ,
            self._source_electron_c,
            self._source_positron_c
        ]

        return (s0, sc)


    def _temperature_grid(self):
        # Extract the temperature range
        (Tmin, Tmax) = self._temperature_range()
        
        # Generate a temperature grid in log-space
        NT = int( log10(Tmax/Tmin)*NT_pd )

        # -->
        return np.logspace( log10(Tmin), log10(Tmax), NT )


    def _pdi_matrix(self):
        # TODO: Use as an argument
        Y0 = self._sII.bbn_abundances_0()

        if self._sMatpBuffer is None:
            # Calculate the thermal photodisintegration rates
            Gpdi_grids = NuclearReactor(
                self._sS0, self._sSc, self._sT, self._sE0, Y0, self._sEta
            ).get_pdi_grids()
            # One grid for each reaction

            # TEMP
            Xhdi_grids = get_Xhdi(
                self._sT, self._sK0, self._sdndt, self._sE0, Y0, self._sEta
            )

            # Calculate the final matrices and set the buffer
            self._sMatpBuffer = MatrixGenerator(
                self._sT, Gpdi_grids, Xhdi_grids, self._sII.dTdt
            ).get_final_matp()

        # Calculate the final matrices
        matp = self._sMatpBuffer

        # Calculate the final matrix and return
        return expm( sum(m for m in matp) )


    def _pred_matrix(self):
        # NOTE:
        # Until disintegration reactions become
        # relevant at t ~ 1e4s, most neutrons
        # have already decayed

        dmat = np.identity(NY)

        # n > p
        dmat[0,0], dmat[1,0] = 0., 1.

        # t > He3
        tmax = self._sII.time( max(self._sTrg) )
        # -->
        expf = exp( -tmax/tau_t )
        # -->
        dmat[3,3], dmat[4,3] = expf, 1. - expf

        return dmat


    def _postd_matrix(self):
        dmat = np.identity(NY)

        dmat[0,0], dmat[1,0] = 0., 1. # n   > p
        dmat[3,3], dmat[4,3] = 0., 1. # t   > He3
        dmat[8,8], dmat[7,8] = 0., 1. # Be7 > Li7

        return dmat


    def get_matp_buffer(self):
        return self._sMatpBuffer


    def set_matp_buffer(self, matp):
        self._sMatpBuffer = matp


    # ABSTRACT METHODS ##############################################

    @abstractmethod
    def _temperature_range(self):
        pass


    def _source_photon_0(self, T):
        return 0.


    def _source_electron_0(self, T):
        return 0.


    def _source_positron_0(self, T):
        return self._source_electron_0(T)


    def _source_photon_c(self, E, T):
        return 0.


    def _source_electron_c(self, E, T):
        return 0.


    def _source_positron_c(self, E, T):
        return self._source_electron_c(E, T)


    def _dndt_proton(self, T):
        return 0.
    

    def _dndt_neutron(self, T):
        return self._dndt_proton(T)
    

    def _K0_proton(self, T):
        return self._sE0 - mp

    
    def _K0_neutron(self, T):
        return self._sE0 - mn


class DecayModel(AbstractModel):

    def __init__(self, mphi, tau, temp0, n0a, bree, braa):
        # Initialize the Input_Interface
        self._sII   = InputInterface( locate_sm_file() )

        # The mass of the decaying particle
        self._sMphi = mphi            # in MeV
        # The lifetime of the decaying particle
        self._sTau  = tau             # in s
        # The injection energy
        self._sE0   = self._sMphi/2.  # in MeV

        # The number density of the mediator
        # (relative to photons) ...
        self._sN0a  = n0a
        # ... at T = temp0 ...
        self._sT0   = temp0           # in MeV
        # ... corresponding to t = t(temp0)
        self._st0   = self._sII.time(self._sT0)

        # The branching ratio into electron-positron pairs
        self._sBRee = bree
        # The branching ratio into two photons
        self._sBRaa = braa

        # Call the super constructor
        super(DecayModel, self).__init__(self._sE0, self._sII)


    # DEPENDENT QUANTITIES ##############################################################

    def _number_density(self, T):
        sf_ratio = self._sII.scale_factor(self._sT0)/self._sII.scale_factor(T)

        delta_t = self._sII.time(T) - self._st0
        n_gamma = (2.*zeta3)*(self._sT0**3.)/(pi**2.)

        return self._sN0a * n_gamma * sf_ratio**3. * exp( -delta_t/self._sTau )


    # ABSTRACT METHODS ##################################################################

    def _temperature_range(self):
        # The number of degrees-of-freedom to span
        mag = 2.
        # Calculate the approximate decay temperature
        Td = self._sII.temperature( self._sTau )
        # Calculate Tmin and Tmax from Td
        Td_ofm = log10(Td)
        # Here we choose -1.5 (+0.5) orders of magnitude
        # below (above) the approx. decay temperature,
        # since the main part happens after t = \tau
        Tmin = 10.**(Td_ofm - 3.*mag/4.)
        Tmax = 10.**(Td_ofm + 1.*mag/4.)

        return (Tmin, Tmax)


    def _source_photon_0(self, T):
        return self._sBRaa * 2. * self._number_density(T) * (hbar/self._sTau)


    def _source_electron_0(self, T):
        return self._sBRee * self._number_density(T) * (hbar/self._sTau)


    def _source_photon_c(self, E, T):
        EX = self._sE0

        x = E/EX
        y = me2/(4.*EX**2.)

        if 1. - y < x:
            return 0.

        _sp = self._source_electron_0(T)

        return (_sp/EX) * (alpha/pi) * ( 1. + (1.-x)**2. )/x * log( (1.-x)/y )


class AnnihilationModel(AbstractModel):

    def __init__(self, mchi, a, b, tempkd, bree, braa, omegah2=0.12):
        # Initialize the Input_Interface
        self._sII    = InputInterface( locate_sm_file() )

        # The mass of the dark-matter particle
        self._sMchi  = mchi         # in MeV
        # The s-wave and p-wave parts of <sigma v>
        self._sSwave = a            # in cm^3/s
        self._sPwave = b            # in cm^3/s
        # The dark-matter decoupling temperature. For
        # Tkd=0, the dark matter partices stays in
        # kinetic equilibrium with the SM heat bath
        self._sTkd   = tempkd       # in MeV
        # The injection energy
        self._sE0    = self._sMchi  # in MeV

        # The branching ratio into electron-positron pairs
        self._sBRee  = bree
        # The branching ratio into two photons
        self._sBRaa  = braa

        # The density parameter of dark matter
        self._sOmgh2 = omegah2

        # Call the super constructor
        super(AnnihilationModel, self).__init__(self._sE0, self._sII)


    # DEPENDENT QUANTITIES ##############################################################

    def _number_density(self, T):
        rho_d0 = 8.095894680377574e-35 * self._sOmgh2  # DM density today in MeV^4
        T0     = 2.72548*8.6173324e-11                 # CMB temperature today in MeV

        sf_ratio = self._sII.scale_factor(T0) / self._sII.scale_factor(T)

        return rho_d0 * sf_ratio**3. / self._sMchi


    def _dm_temperature(self, T):
        if T >= self._sTkd:
            return T

        sf_ratio = self._sII.scale_factor(self._sTkd) / self._sII.scale_factor(T)

        return self._sTkd * sf_ratio**2.


    def _sigma_v(self, T):
        swave_nu = self._sSwave/( (hbar**2.)*(c_si**3.) )
        pwave_nu = self._sPwave/( (hbar**2.)*(c_si**3.) )

        v2 = 6.*self._dm_temperature(T)/self._sMchi

        return swave_nu + pwave_nu*v2


    # ABSTRACT METHODS ##################################################################

    def _temperature_range(self):
        # The number of degrees-of-freedom to span
        mag = 4.
        # Tmax is determined by the Be7 threshold
        # (The factor 0.5 takes into account effects
        # of the high-energy tail)
        Tmax = me2/(22.*.5*Emin)
        # For smaller T the annihilation rate is suppressed
        # --> falls off at least with T^(-6)
        Tmin = 10.**(log10(Tmax) - mag)

        return (Tmin, Tmax)


    def _source_photon_0(self, T):
        return self._sBRaa * (self._number_density(T)**2.) * self._sigma_v(T)


    def _source_electron_0(self, T):
        return self._sBRee * .5 * (self._number_density(T)**2.) * self._sigma_v(T)


    def _source_photon_c(self, E, T):
        EX = self._sE0

        x = E/EX
        y = me2/(4.*EX**2.)

        if 1. - y < x:
            return 0.

        _sp = self._source_electron_0(T)

        return (_sp/EX) * (alpha/pi) * ( 1. + (1.-x)**2. )/x * log( (1.-x)/y )