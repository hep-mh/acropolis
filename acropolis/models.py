# math
from math import pi, exp, log, log10, sqrt
# numpy
import numpy as np
# scipy
from scipy.linalg import expm
from scipy.integrate import quad
from scipy.optimize import root
# abc
from abc import ABC, abstractmethod

# input
from acropolis.input import InputInterface, locate_sm_file
# nucl
from acropolis.nucl import NuclearReactor, MatrixGenerator
# params
from acropolis.params import zeta3, pi2
from acropolis.params import hbar, c_si, me, me2, alpha, tau_t
from acropolis.params import Emin, NY
from acropolis.params import approx_zero, eps
from acropolis.params import universal
# pprint
from acropolis.pprint import print_info, print_warning, print_error


# https://stackoverflow.com/questions/1167617/in-python-how-do-i-indicate-im-overriding-a-method
def overrides(interface_class):
    def overrider(method):
        assert(method.__name__ in dir(interface_class))
        return method
    return overrider


class AbstractModel(ABC):

    def __init__(self, e0, ii):
        # Initialize the input interface
        self._sII  = ii

        # The injection energy
        self._sE0 = e0

        # The temperature range that is used for the calculation
        self._sTrg = self._temperature_range()

        # The relevant source terms
        (self._sS0, self._sSc) = self.get_source_terms()

        # A buffer for high-performance scans
        self._sMatpBuffer = None


    def run_disintegration(self):
        # Print a warning if the injection energy
        # is larger than 1GeV, as this might lead
        # to wrong results
        if not universal and int( self._sE0 ) > 1e3:
            print_warning(
                "Injection energy > 1 GeV. Results cannot be trusted.",
                "acropolis.models.AbstractMode.run_disintegration"
            )

        # Print a warning if the temperature range
        # of the model is not covered by the data
        # in cosmo_file.dat
        cf_temp_rg = self._sII.cosmo_range()
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
            return self._squeeze_decays( self._sII.bbn_abundances() )

        # Calculate the different transfer matrices
        ###########################################

        # 1. pre-decay
        pred_mat   = self._pred_matrix()
        # 2. photodisintegration
        pdi_mat    = self._pdi_matrix()
        # 3. post-decay
        postd_mat  = self._postd_matrix()

        # Combine
        transf_mat = postd_mat.dot( pdi_mat.dot( pred_mat ) )

        # Calculate the final abundances
        Yf = np.column_stack(
            list( transf_mat.dot( Y0i ) for Y0i in self._sII.bbn_abundances().transpose() )
        )

        return Yf


    def get_source_terms(self):
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


    def _pdi_matrix(self):
        if self._sMatpBuffer is None:
            # Initialize the NuclearReactor
            nr = NuclearReactor(self._sS0, self._sSc, self._sTrg, self._sE0, self._sII)

            # Calculate the thermal rates
            (temp, pdi_grids) = nr.get_pdi_grids()

            # Initialize the MatrixGenerator
            mg = MatrixGenerator(temp, pdi_grids, self._sII)

            # Calculate the final matrices and set the buffer
            self._sMatpBuffer = mg.get_final_matp()

        # Calculate the final matrices
        matp = self._sMatpBuffer

        # Calculate the final matrix and return
        return expm( sum(m for m in matp) )


    def _pred_matrix(self):
        dmat = np.identity(NY)

        # n   > p
        dmat[0,0], dmat[1,0] = 0., 1.

        # t > He3
        Tmax = max( self._sTrg )
        tmax = self._sII.time( Tmax )

        expf = exp( -tmax/tau_t )

        dmat[3,3], dmat[4, 3] = expf, 1. - expf

        return dmat


    def _postd_matrix(self):
        dmat = np.identity(NY)

        dmat[0,0], dmat[1,0] = 0., 1. # n   > p
        dmat[3,3], dmat[4,3] = 0., 1. # t   > He3
        dmat[8,8], dmat[7,8] = 0., 1. # Be7 > Li7

        return dmat


    def _squeeze_decays(self, Yf):
        dmat = self._postd_matrix()

        return np.column_stack(
            list( dmat.dot( Yi ) for Yi in Yf.transpose() )
        )


    def get_matp_buffer(self):
        return self._sMatpBuffer


    def set_matp_buffer(self, matp):
        self._sMatpBuffer = matp


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


    def _source_photon_c(self, E, T):
        return 0.


    def _source_electron_c(self, E, T):
        return 0.


    def _source_positron_c(self, E, T):
        return self._source_electron_c(E, T)


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


# This model has been contributed by Pieter Braat (pbraat@nikhef.nl)
# When using this model, please cite arXiv:2310:XXXX
class ResonanceModel(AnnihilationModel):
    
    def __init__(self, mchi, delta, gammad, gammav, nd, tempkd=None, C=None, S=1, omegah2=0.12):

        # CALL THE SUPER CONSTRUCTOR (OF ANNIHILATION_MODEL) ##################
        #######################################################################

        super(ResonanceModel, self).__init__(
        #   mchi, a   , b   , tempkd, bree, braa, omegah2
            mchi, None, None, tempkd, 1   , 0   , omegah2
        )


        # SPECIFY THE NEW PARAMETERS ##########################################
        #######################################################################

        # The mass splitting between the resonant
        # particle and the dark-matter particle
        # mr = mchi * ( 2 + delta )
        self._sDelta  = delta
        # The couplings to the dark and to the
        # visible sector
        self._sGammad = gammad
        self._sGammav = gammav
        # The parameter to distinguish between
        # s- (nd=0) and p-wave (nd=1) processes
        self._sNd     = nd
        self._sNv     = 1
        # The symmetry factor for the annihilation cross-section
        self._sS      = S

        # The mass of the resonant particle
        self._sMR     = self._sMchi * ( 2. + self._sDelta ) # in MeV
        # The resonance momentum
        self._sPR     = self._sMchi * sqrt(self._sDelta)    # in MeV
        # The total decay width of the resonant particle
        self._sWidth  = self._decay_width(self._sPR)        # in MeV
            

        # UPDATE TKD (IF NECESARRY) ###########################################
        #######################################################################

        if tempkd is None:
            if C is None:
                print_error(
                    "When requesting to calculate the decoupling temperature with " +
                    "'tempkd = None', the parameter C must not be 'None' as well!",
                    "acropolis.models.ResonanceModel._init__"
                )
            
            self._sTkd = self._estimate_tempkd_ee( C )


        # CHECK THE INPUT PARAMATERS ##########################################
        #######################################################################
        
        self._check_input_parameters()


    def _estimate_tempkd_ee(self, C):

        def _tempkd_ee_root(logT):
            T = exp( logT )

            return log( self._sigma_v_xe_xe(C, T) * self._nee(T) / self._sII.hubble_rate(T) )
        
        # H ~ sig_v * _nee
        tempk = exp( root(_tempkd_ee_root, 0.).x )

        if tempk >= self._sMchi:
            print_error(
                "The kinetik decoupling temperature is larger than the DM mass. The calculation cannot be trusted.",
                "acropolis.models.ResonanceModel._estimate_tempkd_ee"
            )
        
        return tempk


    def _nee(self, T):
        x = me/T

        # 1. EQUILIBRIUM WITH VANISHING CHEM. POTENTIAL
        def _f_kernel(y, x): # ne = ge * T^3 * quad(f) / 2 pi^2
            if x > y:
                return 0.

            return y * sqrt(y**2. - x**2.) / ( exp(y) + 1 )
        # -->
        f = quad(_f_kernel, x, 200, epsabs=0, epsrel=eps, args=(x,))[0]
        # -->
        nee_1 = 4. * f * (T**3.) / ( 2. * pi2 )

        # 1. EQUILIBRIUM WITH NON-VANISHING CHEM. POTENTIAL
        Y = self._sII.bbn_abundances_0()
        # -->
        nee_2 = ( Y[0] + 2.*Y[5] ) * self._sII.parameter('eta') * ( 2.*zeta3/pi2 ) * (T**3.)

        return max(nee_1, nee_2)


    # TODO
    def _sigma_v_xe_xe(self, C, T):
        mchi = self._sMchi
        # -->
        x = mchi/T

        return 0 #16. * C * self._sGammav * self._sGammad * sqrt( me/(mchi + me) ) / ( sqrt(2*pi*x) * x * (mchi**2) )


    def _check_nwa(self, eps=0.1):
        y = self._sWidth / self._sMR
        # In the NWA limit, we have y -> 0, i.e. y < eps

        if y < eps and self._sNd != 0:
            return True

        return False
    

    def _check_input_parameters(self):
        if self._sDelta > 1:
            print_error(
                "The mass splitting must be < 1. The calculation cannot be trusted.",
                "acropolis.models.ResonanceModel._check_input_parameters"
            )
        
        if (self._sGammad >= 1 or self._sGammav >= 1): 
            print_error(
                "The couplings must be small. The calculation cannot be trusted.",
                "acropolis.model.ResonanceModel._check_input_parameters"
            )
        
        if self._sNd not in [0, 1]:
            print_error(
                "Currently only s-wave annihilations with 'nd = 0' and p-wave " + \
                "annihilations with 'nd = 1' are supported.",
                "acropolis.models.ResonanceModel._check_input_parameters"
            )



    # DEPENDENT QUANTITIES ##############################################################


    # The total decay width of the resonant 
    # particle into dark-sector states
    def _decay_width_d(self, p):
        return self._sGammad * self._sMR * (p / self._sMchi)**(2.*self._sNd + 1.)
    

    # The total decay width of the resonant 
    # particle into visible-sector states
    def _decay_width_v(self, p):
        # p_f ~ m_\chi
        return self._sGammav * self._sMR * (p / self._sMchi)**(2.*self._sNv + 1.)

    
    # The total decay width of the resonant
    # particle
    def _decay_width(self, p):
        return self._decay_width_d(p) + self._decay_width_v(p)


    # The thermally average annihilation
    # cross-section in the resonant regime
    def _sigma_v_res(self, T):
        x = self._sMchi/T

        return 8. * self._sS * (pi*x)**1.5 * self._sGammad * self._sGammav * self._sMR**2. * self._sDelta**(self._sNd+.5) * exp(-self._sDelta*x) \
                / self._sWidth / self._sMchi**3.

    
    # The thermally averaged annihilation
    # cross-section in the non-resonant regime
    def _sigma_v_non_res(self, T):
        x = self._sMchi/T

        # Speed up the calculation: only nd = 0, 1 are allowed
        # gamma(self._sNd+1.5)
        gamma = {
            0:      sqrt(pi) / 2., # \gamma(1.5)
            1: 3. * sqrt(pi) / 4.  # \gamma(2.5)
        }[self._sNd]

        return 4. * self._sS * sqrt(pi) * x**(-self._sNd) * self._sGammad * self._sGammav * self._sMR**2. * gamma \
                / ( self._sMchi**4. * self._sDelta**2. )
    

    # The full (non-approximate) thermally
    # averaged annihilation cross section
    def _sigma_v_full(self, T):
        x = self._sMchi/T

        # The maximal exponent in the integral that
        # still leads to relevant contributions
        exp_cutoff = 200.

        # Define the prefactor
        pref = 4. * x * sqrt(x*pi) * self._sS * self._sGammad * self._sGammav * (self._sMR/self._sMchi)**2.

        # Define the upper integration limit
        umax = exp_cutoff * self._sMchi**2. / x

        def _sigma_v_full_kernel(log_u):
            u      = exp(log_u) # = p^2
            sqrt_u = sqrt(u)    # = p

            uR = self._sPR**2.
            m2 = self._sMchi**2.

            width_u = self._decay_width(sqrt_u)

            return u * exp( -u*x/m2 ) * ( u/m2 )**(self._sNd+.5) / ( (u - uR)**2. + ( self._sMchi*width_u/2. )**2. )

        # Perform the integration
        I = quad(
            _sigma_v_full_kernel, log(approx_zero), log(umax), epsrel=eps, epsabs=0, points=(2.*log(self._sPR),)
        )

        return pref*I[0]


    @overrides(AnnihilationModel)
    def _sigma_v(self, T):
        # Calculate the dark-matter temperature
        Tdm  = self._dm_temperature(T)

        return self._sigma_v_full(Tdm)