# math
from math import pi, log, log10, exp, sqrt
# numpy
import numpy as np
# scipy
from scipy.integrate import quad, dblquad
# abc
from abc import ABCMeta, abstractmethod

# db
from acropolis.db import import_data_from_db
from acropolis.db import in_rate_db, interp_rate_db
# cache
from acropolis.cache import cached_member
# pprint
from acropolis.pprint import print_error
# params
from acropolis.params import me, me2, alpha, re
from acropolis.params import zeta3, pi2
from acropolis.params import Emin, approx_zero, eps, Ephb_T_max

# aot.cascade
from acropolis.aot.cascade import ph_rate_pair_creation_ae, ph_kernel_inverse_compton
from acropolis.aot.cascade import el_kernel_pair_creation_ae, el_rate_inverse_compton, el_kernel_inverse_compton
from acropolis.aot.cascade import dsdE_Z2, solve_cascade_equation


class _ReactionWrapperScaffold(object):

    def __init__(self, Y0, eta, db):
        self._sY0     = Y0
        self._sEta    = eta

        self._sRateDb = db


    # NUMBER DENSITIES of baryons, electrons and nucleons #####################

    def _nb(self, T):
        # gs does not change anymore for the relevant temperature,
        # hence (R0/R)^3 = gs(T)T^3/( gs(T0)T0^3) = (T/T0)^3
        return self._sEta * ( 2.*zeta3/pi2 ) * (T**3.)


    def _ne(self, T):
        # 1: p; 5: He4 (see 'NuclearReactor._nuclei' for all identifiers)
        return ( self._sY0[1] + 2.*self._sY0[5] ) * self._nb(T)


    def _nNZ2(self, T):
        # 1: p; 5: He4 (see 'NuclearReactor._nuclei' for all identifiers)
        return ( self._sY0[1] + 4.*self._sY0[5] ) * self._nb(T)


class _PhotonReactionWrapper(_ReactionWrapperScaffold):

    def __init__(self, Y0, eta, db):
        super(_PhotonReactionWrapper, self).__init__(Y0, eta, db)


    # RATES ###################################################################
    # E is the energy of the incoming particle
    # T is the temperature of the background photons

    # PHOTON-PHOTON SCATTERING ################################################
    def _rate_photon_photon(self, E, T):
        #if E > me2/T:
        #    return 0.
        expf = exp( -E*T/me2 )

        return 0.151348 * (alpha**4.) * me * (E/me)**3. * (T/me)**6. * expf


    # COMPTON SCATTERING ######################################################
    def _rate_compton(self, E, T):
        x = 2.*E/me

        return ( 2.*pi*(re**2.)/x ) * self._ne(T) * ( (1. - 4./x - 8./(x**2.))*log(1.+x) + .5 + 8./x - 1./(2.*(1.+x)**2.) )


    # BETHE-HEITLER PAIR CREATION #############################################
    def _rate_bethe_heitler(self, E, T):
        # For small energies, the rate can be approximated by a constant
        # (cf. 'hep-ph/0604251') --- NOT USED HERE
        #if E < 4.: E = 4.

        k = E/me

        # Below threshold, the rate vanishes
        # This case never happens since Emin = 1.5 > 2me
        # (see 'acropolis.params')
        if k < 2:
            return 0.

        # Approximation for SMALL energies
        if 2 <= k <= 4:
            r = ( 2.*k - 4. )/( k + 2. + 2.*sqrt(2.*k) )

            return ( alpha**3./me2 ) * self._nNZ2(T) * (2.*pi/3.) * ( (k-2.)/k )**3. * ( \
                     1 + r/2. + (23./40.)*(r**2.) + (11./60.)*(r**3.) + (29./960.)*(r**4.) \
                   )


        # Approximation for LARGE energies
        log2k = log(2.*k)
        # We implement corrections up to order (2./k)**6 ('astro-ph/9412055')
        # This is relevant in order to ensure a smooth transition at k = 4
        return ( alpha**3./me2 ) * self._nNZ2(T) * ( \
                   (28./9.)*log2k - 218./27. \
                 + (2./k)**2. * ( (2./3.)*log2k**3. - log2k**2. + (6. - pi2/3.)*log2k + 2.*zeta3 + pi2/6. - 7./2. ) \
                 - (2./k)**4. * ( (3./16.)*log2k + 1./8. ) \
                 - (2./k)**6. * ( (29./2304.)*log2k - 77./13824. ) \
               )


    # DOUBLE PHOTON TO ELECTRON POSITRON PAIR CREATION ########################
    def _rate_pair_creation_ae(self, E, T):
        # In general, the threshold is E ~ me^2/(22*T)
        # However, here we use a slighlty smaller threshold
        # in order to guarantee a smooth transition
        if E < me2/(50.*T):
            return 0.

        # Define the integration limits from the
        # constraint on the center-of-mass energy
        llim = me2/E            # <  50*T (see above)
        ulim = Ephb_T_max*T     # ~ 200*T
        # ulim > llim, since me2/E < 50*T
        # CHECKED!

        # Perform the integration in log-log space
        # The limits for s are always in ascending order,
        # i.e. 4*me2 < 4*E*x, since x > me2/E
        I_fso_E2 = dblquad(ph_rate_pair_creation_ae, log(llim), log(ulim), \
                             lambda logx: log(4.*me2), lambda logx: log(4.*E) + logx, \
                             epsrel=eps, epsabs=0, args=(T, me, re)
                          )

        return I_fso_E2[0]/( 8.*E**2. )


    def _rate_pair_creation_ae_db(self, E, T):
        if E < me2/(50.*T):
            return 0.

        E_log, T_log = log10(E), log10(T)
        if ( self._sRateDb is None ) or ( not in_rate_db(E_log, T_log) ):
            return self._rate_pair_creation_ae(E, T)

        return interp_rate_db(self._sRateDb, 'ph:rate_pair_creation_ae', E_log, T_log)


    # TOTAL RATE ##############################################################
    def total_rate(self, E, T):
        return self._rate_photon_photon(E, T) + self._rate_compton(E, T) + self._rate_bethe_heitler(E, T) + self._rate_pair_creation_ae_db(E, T)


    # INTEGRAL KERNELS ########################################################
    # E  is the energy of the outgoing particle
    # Ep is the energy of the incoming particle
    # T  is the temperature of the background photons

    # PHOTON-PHOTON SCATTERING ################################################
    def _kernel_photon_photon(self, E, Ep, T):
        #if Ep > me2/T:
        #    return 0.
        expf = exp( -Ep*T/me2 )

        return 1112./(10125.*pi) * (alpha**4.)/(me**8.) * 8.*(pi**4.)*(T**6.)/63. \
               * Ep**2. * ( 1. - E/Ep + (E/Ep)**2. )**2. * expf


    # COMPTON SCATTERING ######################################################
    def _kernel_compton(self, E, Ep, T):
        # Check that the energies do not execeed the 'Compton edge'
        # ATTENTION: This constraint is missing in '1503.04852'
        if Ep/(1. + 2.*Ep/me) > E:
            return 0.

        # ATTENTION:
        # If the last term is + 2.*me*(1./E - 1./Ep) , Serpico
        # If the last term is - 2.*me*(1./E - 1./Ep) , correct
        return pi*(re**2.)*me/(Ep**2.) * self._ne(T) * ( Ep/E + E/Ep + (me/E - me/Ep)**2. - 2.*me*(1./E - 1./Ep) )


    # INVERSE COMPTON SCATTERING ##############################################
    @cached_member()
    def _kernel_inverse_compton(self, E, Ep, T):
        # Incorporate the non-generic integration limit as
        # the algorithm requires Ep > E and not Ep > E + me
        if Ep < E + me:
            return 0.
        # This also ensures that Ep != E (!!!)

        # Define the integration limits from
        # the range that is specified in '_JIT_F'
        llim = .25*me2*E/( Ep*Ep - Ep*E ) # with Ep != E (see above)
        ulim = min( E, Ep-me2/(4.*Ep), Ephb_T_max*T )
        # Here, the second condition is redundant, since
        # Ep-me2/(4.*Ep) > E for Ep > E + me (see above)
        # However, we include it anyways in order to
        # provide a better documentation
        # CHECKED!

        # If the lower limit exceeds the upper limit,
        # simply return 0. This also helps to avoid
        # overflow if llim > Ephb_Tmax*T
        if ulim <= llim:
            return 0.

        # Perform the integration in log space
        I_fF_E = quad(ph_kernel_inverse_compton, log(llim), log(ulim), epsrel=eps, epsabs=0, args=(E, Ep, T, me))

        # ATTENTION: Kawasaki considers a combined e^+/e^- spectrum
        # Therefore the factor 2 should not be there in our case
        return 2.*pi*(alpha**2.)*I_fF_E[0]/(Ep**2.)


    # TOTAL INTEGRAL KERNEL ####################################################
    def total_kernel_x(self, E, Ep, T, X):
        if X == 0:
            return self._kernel_photon_photon(E, Ep, T) + self._kernel_compton(E, Ep, T)
        # Photon -> Photon

        if X == 1:
            return self._kernel_inverse_compton(E, Ep, T)
        # Electron -> Photon

        if X == 2:
            return self._kernel_inverse_compton(E, Ep, T)
        # Positron -> Photon

        print_error(
            "Particle with identifier X =" + str(X) + "does not exist.",
            "acropolis.cascade._PhotonReactionWrapper.total_kernel_x"
        )


class _AbstractElectronReactionWrapper(_ReactionWrapperScaffold, metaclass=ABCMeta):

    def __init__(self, Y0, eta, db):
        super(_AbstractElectronReactionWrapper, self).__init__(Y0, eta, db)


    # RATES ###################################################################
    # E is the energy of the incoming particle
    # T is the temperature of the background photons

    # INVERSE COMPTON SCATTERING ##############################################
    @cached_member()
    def _rate_inverse_compton(self, E, T):
        # Define the upper limit for the integration over x
        ulim = min( E - me2/(4.*E), Ephb_T_max*T )
        # The condition x <= E-me2/(4.*E) ensures
        # E- <= E+ in E- <= E <= E+ (range for y)
        # CHECKED!

        # Perform the two-dimensional integration
        # with limits that are calculated from the
        # range that is specified in '_JIT_F'
        # ATTENTION:
        # The integral over \epsilon_\gamma should start at 0.
        # In fact, for \epsilon_\gamma > \epsilon_e, we have q < 0.
        I_fF_E = dblquad(el_rate_inverse_compton, 0., ulim, lambda x: x, lambda x: 4.*x*E*E/( me2 + 4.*x*E ), epsrel=eps, epsabs=0, args=(E, T, me))

        return 2.*pi*(alpha**2.)*I_fF_E[0]/(E**2.)


    def _rate_inverse_compton_db(self, E, T):
        E_log, T_log = log10(E), log10(T)
        if ( self._sRateDb is None ) or ( not in_rate_db(E_log, T_log) ):
            return self._rate_inverse_compton(E, T)

        return interp_rate_db(self._sRateDb, 'el:rate_inverse_compton', E_log, T_log)


    # TOTAL RATE ##############################################################
    def total_rate(self, E, T):
        return self._rate_inverse_compton_db(E, T)


    # INTEGRAL KERNELS ########################################################
    # E  is the energy of the outgoing particle
    # Ep is the energy of the incoming particle
    # T  is the temperature of the background photons

    # INVERSE COMPTON SCATTERING ##############################################
    @cached_member()
    def _kernel_inverse_compton(self, E, Ep, T):
        # E == Ep leads to a divergence in
        # the Bose-Einstein distribution
        # TODO: Check if this can be handled any better
        if E == Ep:
            return 0.

        # Calculate appropriate integration limits
        pf = .25*me2/Ep - E               # <= 0.
        qf = .25*me2*(Ep-E)/Ep            # >= 0.

        sqrt_d = sqrt( (pf/2.)**2. - qf )
        z1 = -pf/2. - sqrt_d              # smaller
        z2 = -pf/2. + sqrt_d              # larger

        # Define the integration limits from
        # the range that is specified in '_JIT_F'
        llim = z1
        ulim = min( z2, Ep - me2/(4.*Ep), Ephb_T_max*T )
        # CHECKED!
        # For the check, remember to use the correct
        # '_JIT_F', i.e. '_JIT_F(Ep+x-E, Ep, x)'

        # If the lower limit exceeds the upper limit,
        # simply return 0. This also helps to avoid
        # overflow if llim > Ephb_Tmax*T
        if ulim <= llim:
            return 0.

        # Perform the integration in log space
        I_fF_E = quad(el_kernel_inverse_compton, log(llim), log(ulim), epsrel=eps, epsabs=0, args=(E, Ep, T, me))

        return 2.*pi*(alpha**2.)*I_fF_E[0]/(Ep**2.)


    # COMPTON SCATTERING ######################################################
    @abstractmethod
    def _kernel_compton(self, E, Ep, T):
        pass


    # BETHE_HEITLER PAIR CREATION #############################################
    @cached_member()
    def _kernel_bethe_heitler(self, E, Ep, T):
        # Incorporate the non-generic integration limit as
        # the algorithm requires Ep > E and not Ep > E + me
        if Ep < E + me:
            return 0.

        # Multiply by the nucleon density and return
        return self._nNZ2(T)*dsdE_Z2(E, Ep, me, re, alpha)


    # DOUBLE PHOTON TO ELECTRON POSITRON PAIR CREATION ########################
    @cached_member()
    def _kernel_pair_creation_ae(self, E, Ep, T):
        # In general, the threshold is Ep >~ me^2/(22*T)
        # However, here we use a slighlty smaller threshold
        # in acordance with the implementation we use in
        # '_PhotonReactionWrapper._rate_pair_creation'
        if Ep < me2/(50.*T):
            return 0.
        # Ep is the incoming(!) energy

        dE, E2 = Ep - E, E**2.
        z1 = Ep*( me2 - 2.*dE*( sqrt(E2 - me2) - E ) )/( 4*Ep*dE + me2 )
        z2 = Ep*( me2 + 2.*dE*( sqrt(E2 - me2) + E ) )/( 4*Ep*dE + me2 )

        # Define the integration limits from
        # the range that is specified in '_JIT_G'
        # and the constraint on the center-of-mass
        # energy, i.e. Eph*Ephb > me^2
        llim = max( me2/Ep, z1 )
        ulim = min( z2, Ep, Ephb_T_max*T )
        # The me < ... condition is fulfiled by
        # default since all energies are larger
        # than Emin > 2me
        # The reference paper also states that
        # x < Ep, which is also incorporated here
        # CHECKED!

        # If the lower limit exceeds the upper limit,
        # simply return 0. This also helps to avoid
        # overflow if llim > Ephb_Tmax*T
        if ulim <= llim:
            return 0.

        # Perform the integration in log space
        I_fG_E2 = quad(el_kernel_pair_creation_ae, log(llim), log(ulim), epsrel=eps, epsabs=0, args=(E, Ep, T, me))

        return 0.25*pi*(alpha**2.)*me2*I_fG_E2[0]/(Ep**3.)


    # TOTAL INTEGRAL KERNEL ####################################################
    @abstractmethod
    def total_kernel_x(self, E, Ep, T, X):
        pass


class _ElectronReactionWrapper(_AbstractElectronReactionWrapper):

    def __init__(self, Y0, eta, db):
        super(_ElectronReactionWrapper, self).__init__(Y0, eta, db)


    # INTEGRAL KERNELS ########################################################
    # E  is the energy of the outgoing particle
    # Ep is the energy of the incoming particle
    # T  is the temperature of the background photons

    # [...]

    # COMPTON SCATTERING ######################################################
    def _kernel_compton(self, E, Ep, T):
        # Perform a subsitution of the parameters.
        # Compared to the formula for photons, only
        # the arguments of the cross-section are different
        E_s  = Ep + me - E   # E , substituted
        Ep_s = Ep            # Ep, substituted

        # Use the same formula as in case of photons with
        # E  -> E_s
        # Ep -> Ep_s
        # Check that the energies do not exceed the 'Compton edge'
        # ATTENTION: This condition is missing in some other papers
        if Ep_s/(1. + 2.*Ep_s/me) > E_s:
            return 0.

        # ATTENTION:
        # If the last term is + 2.*me*(1./E_s - 1./Ep_s), Serpico
        # If the last term is - 2.*me*(1./E_s - 1./Ep_s), correct
        return pi*(re**2.)*me/(Ep_s**2.) * self._ne(T) * ( Ep_s/E_s + E_s/Ep_s + (me/E_s - me/Ep_s)**2. - 2.*me*(1./E_s - 1./Ep_s) )

    # [...]

    # TOTAL INTEGRAL KERNEL ####################################################
    def total_kernel_x(self, E, Ep, T, X):
        if X == 0:
            return self._kernel_compton(E, Ep, T) + self._kernel_bethe_heitler(E, Ep, T) + self._kernel_pair_creation_ae(E, Ep, T)
        # Photon -> Electron

        if X == 1:
            return self._kernel_inverse_compton(E, Ep, T)
        # Electron -> Electron

        if X == 2:
            return 0.
        # Positron -> Electron

        print_error(
            "Particle with identifier X =" + str(X) + "does not exist.",
            "acropolis.cascade._ElectronReactionWrapper.total_kernel_x"
        )


class _PositronReactionWrapper(_AbstractElectronReactionWrapper):

    def __init__(self, Y0, eta, db):
        super(_PositronReactionWrapper, self).__init__(Y0, eta, db)


    # INTEGRAL KERNELS ########################################################
    # E  is the energy of the outgoing particle
    # Ep is the energy of the incoming particle
    # T  is the temperature of the background photons

    # [...]

    # COMPTON SCATTERING ######################################################
    def _kernel_compton(self, E, Ep, T):
        # There are (almost) no thermal positrons
        return 0.

    # [...]

    # TOTAL INTEGRAL KERNEL ####################################################
    def total_kernel_x(self, E, Ep, T, X):
        if X == 0:
            return self._kernel_compton(E, Ep, T) + self._kernel_bethe_heitler(E, Ep, T) + self._kernel_pair_creation_ae(E, Ep, T)
        # Photon -> Positron

        if X == 1:
            return 0.
        # Electron -> Positron

        if X == 2:
            return self._kernel_inverse_compton(E, Ep, T)
        # Positron -> Positron

        print_error(
            "Particle with identifier X =" + str(X) + "does not exist.",
            "acropolis.cascade._PositronReactionWrapper.total_kernel_x"
        )


class SpectrumGenerator(object):

    def __init__(self, Y0, eta):
        # Extract the data from the databases; If there is
        # no data in the folder 'data/', db = (None, None)
        db = import_data_from_db()

        # Define a dictionary containing all reaction wrappers
        self._sRW = {
            0: _PhotonReactionWrapper  (Y0, eta, db),
            1: _ElectronReactionWrapper(Y0, eta, db),
            2: _PositronReactionWrapper(Y0, eta, db)
        }

         # Set the number of particle species (in the cascade)
        self._sNX = 3


    def _rate_x(self, X, E, T):
        return self._sRW[X].total_rate(E, T)


    def _kernel_x_xp(self, X, Xp, E, Ep, T):
        return self._sRW[X].total_kernel_x(E, Ep, T, Xp)


    def rate_photon(self, E, T):
        return self._rate_x(0, E, T)


    def get_spectrum(self, E_grid, S0_grid, SC_grid, T, allX=False):
        # Save the dimension of the species grid
        NX = self._sNX

        # Generate the grid for the different species
        X_grid = np.arange(NX)

        # DEBUG
        assert len(S0_grid) == NX

        # Generate the grid for the rates
        # Indices: 1: X, 2: E
        G_grid = np.array([[self._rate_x(X, E, T) for E in E_grid] for X in X_grid])

        # Generate the grid for the kernels
        # Indices: 1: X, 2: Xp, 3: E, 4: Ep
        # For Ep < E, the kernel is simply 0.
        K_grid = np.array([[[[self._kernel_x_xp(X, Xp, E, Ep, T) if Ep >= E else 0. for Ep in E_grid] for E in E_grid] for Xp in X_grid] for X in X_grid])

        # Calculate the spectra by solving
        # the cascade equation
        sol = solve_cascade_equation(
            E_grid, G_grid, K_grid, S0_grid, SC_grid, T, Emin, approx_zero
        )

        # 'sol' always has at least two columns
        return sol[0:2,:] if not allX else sol


    def get_universal_spectrum(self, E_grid, S0_grid, SC_grid, T, offset=0.):
        E0 = E_grid[-1]

        # Extract the size of the energy grid
        NE = len(E_grid)

        # Define EC and EX as in 'astro-ph/0211258'
        EC = me2/(22.*T)
        EX = me2/(80.*T)

        # Define the normalization K0 as in 'astro-ph/0211258'
        K0 = E0/( (EX**2.) * ( 2. + log( EC/EX ) ) )

        # Generate the grid for the photon spectrum
        F_grid = np.zeros(NE)

        # Calculate the spectrum for the different energies
        # TODO: Incoporate the continuous source terms in the
        #       normalization by integrating it over the energy
        SN = sum(S0_grid) # Normalization
        for i, E in enumerate(E_grid):
            if E < EX:
                F_grid[i] = SN * K0 * (EX/E)**1.5/self.rate_photon(E, T)
            elif E >= EX and E <= (1. + offset)*EC: # an offset enables better interpolation
                F_grid[i] = SN * K0 * (EX/E)**2.0/self.rate_photon(E, T)

        # Remove potential zeros
        F_grid[F_grid < approx_zero] = approx_zero

        # Define the output array...
        sol = np.zeros( (2, NE) )
        # ...and fill it
        sol[0, :] = E_grid
        sol[1, :] = F_grid

        return sol
