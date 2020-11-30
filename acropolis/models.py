# math
from math import pi, exp, log, log10
# numpy
import numpy as np
# scipy
from scipy.linalg import expm
# abc
from abc import ABC, abstractmethod

# input
from acropolis.input import InputInterface
# nucl
from acropolis.nucl import NuclearReactor, MatrixGenerator
# params
from acropolis.params import hbar, c_si, me2, alpha
from acropolis.params import Emin
# pprint
from acropolis.pprint import print_info, print_warning


class AbstractModel(ABC):

    def __init__(self, e0, ii):
        # Initialize the input interface
        self._sII  = ii

        # The injection energy
        self._sE0 = e0

        # The temperature range that is used for the calculation
        self._sTrg = self._temperature_range()

        # Calculate the relevant physical quantities, i.e. ...
        # ...the 'delta' source terms and...
        self._sS0 = [
            self._source_photon_0  ,
            self._source_electron_0,
            self._source_positron_0
        ]
        # ...the ISR source terms
        self._sSc = [
            self._source_photon_c  ,
            self._source_electron_c,
            self._source_positron_c
        ]

        # A buffer for high-performance scans
        self._sMatpBuffer = None


    def run_disintegration(self):
        # Print a warning of the injection energy
        # is larger than 1GeV, as this might lead
        # to wrong results
        if self._sE0 > 1e3:
            print_warning(
                "Injection energy > 1 GeV. Results cannot be trusted.",
                "acropolis.models.AbstractMode.run_disintegration"
            )

        # If the energy is below all thresholds,
        # simply return the initial abundances
        if self._sE0 <= Emin:
            print_info(
                "Injection energy is below all thresholds. No calculation required.",
                "acropolis.models.AbstractMode.run_disintegration"
            )
            return self._squeeze_decays( self._sII.bbn_abundances() )

        if self._sMatpBuffer is not None:
            matp = self._sMatpBuffer
        else:
            # Initialize the nuclear reactor
            nr = NuclearReactor(self._sS0, self._sSc, self._sTrg, self._sE0, self._sII)

            # Calculate the thermal rates
            (temp, pdi_grids) = nr.get_pdi_grids()

            # Initialize the linear system solver
            mg = MatrixGenerator(temp, pdi_grids, self._sII)

            # Generate the final matrix power...
            matp = mg.get_final_matp()
            # ...and buffer the result
            self._sMatpBuffer = matp

        # Calculate the transfer matrix
        fmat = expm(matp)

        # Calculate the final abundances
        Yf = np.column_stack(
            list( fmat.dot( Y0i ) for Y0i in self._sII.bbn_abundances().transpose() )
        )
        # Perform the full (T>Tmax) decay of
        # all unstable particles and return
        return self._squeeze_decays(Yf)


    def _squeeze_decays(self, Yf0):
        Yf = Yf0.copy()

        NYf = Yf.shape[1]
        for i in range(NYf):
            # n > p
            Yf[0,i], Yf[1,i] = 0., Yf[0,i] + Yf[1,i]
            # t > He3
            Yf[3,i], Yf[4,i] = 0., Yf[3,i] + Yf[4,i]
            # Be7 > Li7
            Yf[8,i], Yf[7,i] = 0., Yf[8,i] + Yf[7,i]

        return Yf


    def set_matp_buffer(self, matp):
        self._sMatpBuffer = matp


    def get_matp_buffer(self):
        return self._sMatpBuffer


    # ABSTRACT METHODS ##############################################

    @abstractmethod
    def _temperature_range(self):
        pass


    @abstractmethod
    def _source_photon_0(self, T):
        pass


    @abstractmethod
    def _source_electron_0(self, T):
        pass


    def _source_positron_0(self, T):
        return self._source_electron_0(T)


    @abstractmethod
    def _source_photon_c(self, E, T):
        pass


    def _source_electron_c(self, E, T):
        return 0.


    def _source_positron_c(self, E, T):
        return self._source_electron_c(E, T)


class DecayModel(AbstractModel):

    def __init__(self, mphi, tau, temp0, n0a, bree, braa):
        # Initialize the Input_Interface
        self._sII   = InputInterface("data/sm.tar.gz")

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
        n_gamma = (pi**2.)*(self._sT0**3.)/15.

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

        # Divide by 2. since only one photon is produced
        return (_sp/EX) * (alpha/pi) * ( 1. + (1.-x)**2. )/x * log( (1.-x)/y )


class AnnihilationModel(AbstractModel):

    def __init__(self, mchi, a, b, tempkd, bree, braa):
        # Initialize the Input_Interface
        self._sII    = InputInterface("data/sm.tar.gz")

        # The mass of the dark-matter particle
        self._sMchi  = mchi         # in MeV
        # The s-wave and p-wave parts of <sigma v>
        self._sSwave = a            # in cm^3/s
        self._sPwave = b            # in cm^3/s
        # The dark matter decoupling temperature in MeV
        # For Tkd=0, the dark matter partices stays in
        # kinetic equilibrium with the SM heat bath
        self._sTkd   = tempkd       # in MeV
        # The injection energy
        self._sE0    = self._sMchi  # in MeV

        # The branching ratio into electron-positron pairs
        self._sBRee  = bree
        # The branching ratio into two photons
        self._sBRaa  = braa

        # Call the super constructor
        super(AnnihilationModel, self).__init__(self._sE0, self._sII)

    # DEPENDENT QUANTITIES ##############################################################

    def _number_density(self, T):
        rho_d0 = 0.12*8.095894680377574e-35   # DM density today in MeV^4
        T0     = 2.72548*8.6173324e-11        # CMB temperature today in MeV

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

        # Divide by 2. since only one photon is produced
        return (_sp/EX) * (alpha/pi) * ( 1. + (1.-x)**2. )/x * log( (1.-x)/y )
