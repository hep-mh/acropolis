# math
from math import log, exp, sqrt, pi
# numba
from numba.pycc import CC
from numba import njit
# numpy
import numpy as np


# acropolis.aot.cascade #######################################################

cascade = CC("cascade")


@njit
def _F(Eph, Ee, Ephb, me):
    me2 = me**2.
    
    # ATTENTION: Here we use the range given in '10.1103/PhysRev.167.1159',
    # because the translation to 0 < q < 1 is questionable
    if not ( Ephb <= Eph <= 4.*Ephb*Ee*Ee/( me2 + 4.*Ephb*Ee ) ):
        # CHECKED to never happen, since the intergration
        # limits are always chosen appropriately (below)
        return 0.

    G = 4.*Ephb*Ee/me2         # \Gamma_\epsilon
    q = Eph/( G*(Ee - Eph) )   # q

    # ATTENTION:
    # If the last term is (2.-2.*G*q) , Kawasaki
    # If the last term is (2.+2.*G*q) , correct
    return 2.*q*log(q) + (1.+2.*q)*(1.-q) + (G*q)**2. * (1.-q)/(2.+2.*G*q)


@njit
def _G(Ee, Eph, Ephb, me):
    me2 = me**2.

    # Define the energy of the positron
    Eep = Eph + Ephb - Ee

    # Calculate the valid range for Ee
    # ATTENTION: This range is absent in 'astro-ph/9412055'
    # Here we adopt the original result from
    # 'link.springer.com/content/pdf/10.1007/BF01005624.pdf'
    dE_sqrt  = (Eph - Ephb)*sqrt( 1. - me2/( Eph*Ephb ) )
    Ee_lim_m = ( Eph + Ephb - dE_sqrt )/2.
    Ee_lim_p = ( Eph + Ephb + dE_sqrt )/2.
    # ATTENTION: White et al. impose the range in the soft
    # photon limit, which is more difficult to handle but
    # should lead to the same results, since the pair production
    # kernel ensures that Ephb ~ T << Eph ~ O(MeV)

    if not ( me < Ee_lim_m <= Ee <= Ee_lim_p ):
        # CHECKED to never happen, since the intergration
        # limits are always chosen appropriately (below)
        return 0.

    # Split the function into four summands
    # and calculate all of them separately
    # Ee + Eep = Eph + Ephb
    sud  = 0.
    sud += 4.*( (Ee + Eep)**2. )*log( (4.*Ephb*Ee*Eep)/( me2*(Ee + Eep) ) )/( Ee*Eep )
    sud += ( me2/( Ephb*(Ee + Eep) ) - 1. ) * ( (Ee + Eep)**4. )/( (Ee**2.)*(Eep**2.) )
    # ATTENTION: no additional minus sign in sud[2]
    # It is unclear whether it is a typo or an artifact
    # of scanning the original document
    sud += 2.*( 2.*Ephb*(Ee + Eep) - me2 ) * ( (Ee + Eep)**2. )/( me2*Ee*Eep )
    sud += -8.*Ephb*(Ee + Eep)/me2

    return sud


@cascade.export("ph_rate_pair_creation_ae", "f8(f8, f8, f8, f8, f8)")
def ph_rate_pair_creation_ae(logy, logx, T, me, re):
    me2 = me**2.

    # Return the integrand for the 2d integral in log-space
    x, y = exp(logx), exp(logy)

    # Define beta as a function of y
    b = sqrt(1. - 4.*me2/y)

    # Define the kernel for the 2d-integral; y = s, x = epsilon_bar
    #                     f/E^2                              s   \sigma_DP
    # ATTENTION: There is an error in 'astro-ph/9412055.pdf'
    # In the integration for \bar{\epsilon}_\gamma the lower
    # limit of integration should be me^2/\epsilon_\gamma
    # (the written limit is unitless, which must be wrong)
    # This limit is a consequence of the constraint on
    # the center-of-mass energy
    sig_pc = .5*pi*(re**2.)*(1.-b**2.)*( (3.-b**4.)*log( (1.+b)/(1.-b) ) - 2.*b*(2.-b**2.) )

    return ( 1./(pi**2) )/( exp(x/T) - 1. ) * y * sig_pc * (x*y)


@cascade.export("ph_kernel_inverse_compton", "f8(f8, f8, f8, f8, f8)")
def ph_kernel_inverse_compton(logx, E, Ep, T, me):
    # Return the integrand for the 1d-integral in log-space; x = Ephb
    x = exp(logx)

    return _F(E, Ep, x, me)*x/( (pi**2.)*(exp(x/T) - 1.) ) * x


@cascade.export("el_kernel_pair_creation_ae", "f8(f8, f8, f8, f8, f8)")
def el_kernel_pair_creation_ae(logx, E, Ep, T, me):
    # Define the integrand for the 1d-integral in log-space; x = Ephb
    x = exp(logx)

    return _G(E, Ep, x, me)/( (pi**2.)*(exp(x/T) - 1.) ) * x


@cascade.export("el_rate_inverse_compton", "f8(f8, f8, f8, f8, f8)")
def el_rate_inverse_compton(y, x, E, T, me):
    # Return the integrand for the 2d-integral; y = Eph, x = Ephb
    return _F(y, E, x, me)*x/( (pi**2.)*(exp(x/T) - 1.) )


@cascade.export("el_kernel_inverse_compton", "f8(f8, f8, f8, f8, f8)")
def el_kernel_inverse_compton(logx, E, Ep, T, me):
    # Define the integrand for the 1d-integral in log-space; x = Ephb
    x = exp(logx)

    return _F(Ep+x-E, Ep, x, me)*( x/(pi**2) )/( exp(x/T) - 1. ) * x


@cascade.export("dsdE_Z2", "f8(f8, f8, f8, f8, f8)")
def dsdE_Z2(Ee, Eph, me, re, alpha):
    me2 = me**2.

    # Define the energies (here: nucleon is at rest)
    Em = Ee                                                      # E_-
    Ep = Eph - Ee                                                # E_+

    # Define the various parameters that enter the cross-section
    pm = sqrt(Em*Em - me2)                                       # p_-
    pp = sqrt(Ep*Ep - me2)                                       # p_+

    L  = log( (Ep*Em + pp*pm + me2)/(Ep*Em - pp*pm + me2) )      # L

    lm = log( (Em + pm)/(Em - pm) )                              # l_-
    lp = log( (Ep + pp)/(Ep - pp) )                              # l_+

    # Define the prefactor
    pref = alpha*(re**2.)*pp*pm/(Eph**3.)

    # Calculate the infamous 'lengthy expression'
    # Therefore, split the sum into four summands
    sud  = 0.
    sud += -4./3. - 2.*Ep*Em*(pp*pp + pm*pm)/( (pp**2.)*(pm**2.) )
    sud += me2*( lm*Ep/(pm**3.) + lp*Em/(pp**3.) - lp*lm/(pp*pm) )
    sud += L*( -8.*Ep*Em/(3.*pp*pm) + Eph*Eph*((Ep*Em)**2. + (pp*pm)**2. - me2*Ep*Em)/( (pp**3.)*(pm**3.) ) )
    sud += -L*me2*Eph*( lp*(Ep*Em - pp*pp)/(pp**3.) + lm*(Ep*Em - pm*pm)/(pm**3.) )/(2.*pp*pm)

    return pref * sud


@cascade.export("solve_cascade_equation", "f8[:,:](f8[:], f8[:,:], f8[:,:,:,:], f8[:], f8[:,:], f8, f8, f8)")
def solve_cascade_equation(E_grid, G, K, S0, SC, T, Emin, approx_zero):
    # Extract the number of particle species...
    NX = len(G)
    # ...and the number of energy points
    NE = len(E_grid)

    dy = log(E_grid[-1]/Emin)/(NE-1)

    # Generate the grid for the different spectra
    # 1. index: X = photon, electron, positron
    # 2. index: Position in the energy grid
    F_grid = np.zeros( (NX, NE) )

    # Calculate F_X(E_0), last index NE-1
    FX_E0 = np.array([
        SC[X,-1]/G[X,-1] + np.sum(K[X,:,-1,-1]*S0[:]/(G[:,-1]*G[X,-1])) for X in range(NX)
    ])
    # -->
    F_grid[:, -1] = FX_E0

    # Loop over all energies
    i = (NE - 1) - 1 # start at the second to last index, NE-2
    while i >= 0: # Counting down
        B = np.zeros( (NX, NX) )
        a = np.zeros( (NX,   ) )

        I = np.identity(NX)
        # Calculate the matrix B and the vector a
        for X in range(NX):
            # Calculate B, : <--> Xp
            B[X,:] = -.5*dy*E_grid[i]*K[X,:,i,i] + G[X,i]*I[X,:]

            # Calculate a
            a[X] = SC[X,i]
            for Xp in range(NX):
                a[X] += K[X,Xp,i,-1]*S0[Xp]/G[Xp,-1] + .5*dy*E_grid[-1]*K[X,Xp,i,-1]*F_grid[Xp,-1]
                for j in range(i+1, NE-1): # Goes from i+1 to NE-2
                    a[X] += dy*E_grid[j]*K[X,Xp,i,j]*F_grid[Xp,j]

        # Solve the system of linear equations of the form BF = a
        F_grid[:, i] = np.linalg.solve(B, a)

        i -= 1

    # Remove potential zeros
    F_grid = F_grid.reshape( NX*NE )
    for i, f in enumerate(F_grid):
        if f < approx_zero:
            F_grid[i] = approx_zero
    F_grid = F_grid.reshape( (NX, NE) )

    # Define the output array...
    sol = np.zeros( (NX+1, NE) )
    # ...and fill it
    sol[0     , :] = E_grid
    sol[1:NX+1, :] = F_grid

    return sol


if __name__ == "__main__":
    for module in [cascade]:
        module.compile()