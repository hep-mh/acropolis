# math
from math import sqrt, exp, log
# numpy
import numpy as np
# scipy
from scipy.integrate import quad
from scipy.optimize import root

# models
from acropolis.models import AnnihilationModel
# params
from acropolis.params import me
from acropolis.params import pi, pi2, zeta3
from acropolis.params import eps
# pprint
from acropolis.pprint import print_warning, print_error


# https://stackoverflow.com/questions/1167617/in-python-how-do-i-indicate-im-overriding-a-method
def overrides(interface_class):
    def overrider(method):
        assert(method.__name__ in dir(interface_class))
        return method
    return overrider


# Required keyword arguments:
# mchi, delta, gammad, gammav, nd, S, ii
def estimate_tempkd_ee(mchi, delta, gammad, gammav, nd, S, ii, C=2*pi):
    # The cross-section for \chi e^\pm -> \chi e^\pm scattering
    def _sigma_v_xe_xe(T):
        return 12.*C*gammad*gammav*(T**2.)/(mchi**4.)

    # The number density of e^\pm
    def _nee(T):
        xe = me/T

        # 1. EQUILIBRIUM WITH VANISHING CHEM. POTENTIAL
        f_kernel = lambda y, x: y * sqrt(y**2. - x**2.) / ( exp(y) + 1 ) if x <= y else 0. # x = me/T, y = Ee/T
        # -->
        f = quad(f_kernel, xe, 200, epsabs=0, epsrel=eps, args=(xe,))[0]
        # -->
        nee_1 = 4. * f * (T**3.) / ( 2. * pi2 )
        # ne = ge * T^3 * quad(f) / 2 pi^2

        # 1. EQUILIBRIUM WITH NON-VANISHING CHEM. POTENTIAL
        Y = ii.bbn_abundances_0()
        # -->
        nee_2 = ( Y[0] + 2.*Y[5] ) * ii.parameter('eta') * ( 2.*zeta3/pi2 ) * (T**3.)

        return max(nee_1, nee_2)

    # H ~ sig_v * _nee
    def _tempkd_ee_root(logT):
        T = exp( logT )

        return log( _sigma_v_xe_xe(T) * _nee(T) / ii.hubble_rate(T) )
    
    # -->
    tempkd = exp( root(_tempkd_ee_root, 0.).x )

    # if tempkd >= mchi:
    #     print_warning(
    #         "The kinetic decoupling temperature is larger than the DM mass. The calculation cannot be trusted.",
    #         "acropolis.models.estimate_tempkd_ee"
    #     )
    
    if tempkd <= me:
        print_warning(
            "The kinetic decoupling temperature is smaller than the electron mass. The calculation cannot be trusted.",
            "acropolis.models.estimate_tempkd_ee"
        )
    
    return min(tempkd, mchi/20)


# This model has been contributed by Pieter Braat (pbraat@nikhef.nl)
# When using this model, please cite arXiv:2405:XXXX
class ResonanceModel(AnnihilationModel):
    def __init__(self, mchi, delta, gammad, gammav, nd, tempkd, S=1, omegah2=0.12):

        # CALL THE SUPER CONSTRUCTOR (OF ANNIHILATION_MODEL) ##################
        #######################################################################

        super(ResonanceModel, self).__init__(
        #   mchi, a   , b   , tempkd, bree, braa, omegah2
            mchi, None, None, None  , 1   , 0   , omegah2
        #                     | tempkd will be set below
        )

        # CALCULATE TKD FROM THE GIVEN FUNCTION IF REQUESTED ##################
        #######################################################################
        if callable(tempkd):
            self._sTkd = tempkd(
                mchi=mchi, delta=delta, gammad=gammad, gammav=gammav, nd=nd, S=S, ii=self._sII
            )
        else:
            self._sTkd = tempkd

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
        # The parameters to distinguish between
        # s- (nd=0) and p-wave (nd=1) processes
        self._sNd     = nd
        self._sNv     = 1 # NOTE: Only electrons are allowed in the final state for now
        # The symmetry factor for the annihilation cross-section
        self._sS      = S

        # The mass of the resonant particle
        self._sMR     = self._sMchi * ( 2. + self._sDelta ) # in MeV
        # The resonance momentum
        self._sPR     = self._sMchi * sqrt(self._sDelta)    # in MeV
        # The total decay width of the resonant particle
        self._sWidth  = self._decay_width(self._sPR)        # in MeV


        # CHECK THE INPUT PARAMATERS ##########################################
        #######################################################################
        
        self._check_input_parameters()


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
                "The couplings must be small (< 1). The calculation cannot be trusted.",
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
        x  = self._sMchi/T

        uR = self._sPR**2.
        m2 = self._sMchi**2.

        # Set the maximal exponent in the integral
        # beyond which the integrand is cut off
        exp_cutoff = 200.

        # Define the prefactor
        pref = 4. * x * sqrt(x*pi) * self._sS * self._sGammad * self._sGammav * (self._sMR/self._sMchi)**2.    

        # Define the integration kernel
        def _sigma_v_full_kernel(log_u):
            u      = exp(log_u) # = p^2
            sqrt_u = sqrt(u)    # = p

            width_u = self._decay_width(sqrt_u)

            #      | from log integration
            return u * exp( -u*x/m2 ) * ( u/m2 )**(self._sNd+.5) / ( (u - uR)**2. + ( self._sMchi*width_u/2. )**2. )

        # Calculate the upper integration limit
        umax = exp_cutoff * m2 / x

        uR_l, uR_h = uR*(1-eps), uR*(1+eps) 

        # uR_h/uR_lt ~ 1.002
        # This is still good with exp_cutoff
        uR_l = min(uR_l, umax)

        # Perform the integration in three steps
        # BELOW RESONANCE
        I1 = quad(
            _sigma_v_full_kernel, -np.inf  , log(uR_l), epsrel=eps, epsabs=0
        )[0]
        # AROUND RESONANCE
        I2 = quad(
            _sigma_v_full_kernel, log(uR_l), log(uR_h), epsrel=eps, epsabs=0, limit=100, points=(log(uR),)
        )[0] if uR_l < umax else 0.
        # ABOVE RESONANCE
        I3 = quad(
            _sigma_v_full_kernel, log(uR_h), log(umax), epsrel=eps, epsabs=0
        )[0] if uR_l < umax else 0.

        return pref*(I1+I2+I3)


    @overrides(AnnihilationModel)
    def _sigma_v(self, T):
        return self._sigma_v_full(
            self._dm_temperature(T)
        )
        
        """
        # Calculate the resonance peak
        x_res_peak = 1.5/self._sDelta

        # Define a cutoff factor for discriminating
        # between the resonant and non-resonant region
        cutoff = 100

        # Calculate the dark-matter temperature
        Tdm  = self._dm_temperature(T)
        # -->
        x = self._sMchi/Tdm
        
        # RESONANT APPROXIMATION
        if self._check_nwa() and x < x_res_peak/cutoff:
            return self._sigma_v_res(Tdm)
        
        # NON-RESONANT APPROXIMATION
        if x > (cutoff**2.) * x_res_peak:
            return self._sigma_v_non_res(Tdm)
        

        return self._sigma_v_full(Tdm)
        """