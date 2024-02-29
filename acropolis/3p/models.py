# math
from math import sqrt, exp, log
# scipy
from scipy.integrate import quad

# models
from acropolis.models import AnnihilationModel
# pprint
from acropolis.pprint import print_error


# https://stackoverflow.com/questions/1167617/in-python-how-do-i-indicate-im-overriding-a-method
def overrides(interface_class):
    def overrider(method):
        assert(method.__name__ in dir(interface_class))
        return method
    return overrider


# This model has been contributed by Pieter Braat (pbraat@nikhef.nl)
# When using this model, please cite arXiv:2310:XXXX
class ResonanceModel(AnnihilationModel):
    
    def __init__(self, mchi, delta, gammad, gammav, nd, tempkd, S=1, omegah2=0.12):

        if callable(tempkd):
            tempkd = tempkd(
                mchi=mchi, delta=delta, gammad=gammad, gammav=gammav, nd=nd, S=S
            )

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


        # CHECK THE INPUT PARAMATERS ##########################################
        #######################################################################
        
        self._check_input_parameters()
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