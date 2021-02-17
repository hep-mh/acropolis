# math
from math import pi, log10

# input
from acropolis.input import InputInterface
# model
from acropolis.model import AbstractModel
# params
from acropolis.params import hbar, alpha

class EmModel(AbstractModel):

    def __init__(self, input_file):
        # Initialize the Input_Interface
        self._sII = InputInterface(input_file)

        # The mass of the decaying particle
        self._sMphi = self._sII.parameter("mphi") # in MeV
        # The lifetime of the decaying particle
        self._sTau  = self._sII.parameter("tau")  # in s
        # The injection energy
        self._sE0   = self._sMphi/2.

        # The number density of the decaying particle
        # as a function of temperature
        self._number_density = lambda T: self._sII.cosmo_column(5, T)

        # The branching ratio into electron-positron pairs
        self._sBRee = self._sII.parameter("bree")
        # The branching ratio into two photons
        self._sBRaa = self._sII.parameter("braa")

        # Call the super constructor
        super(EmModel, self).__init__(self._sE0, self._sII)

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
