# params
from acropolis.params import alpha


# CONTINUOUS ENERGY LOSS ##################################################
# E is the energy of the energy-loosing particle
# T is the temperature of the background photons

def eloss_thompson(E, T, M, Q):
    # The gamma factor of the charged particle
    ga = E/M

    Z = Q
    # This relation also remains for non-relativistic
    # particles, in which case gamma^2-1 = 0
    # For the comparison with the White et al. paper
    # use zeta(4) = p^4/90 and ga^2 - 1 --> ga^2
    return -32.*(pi**3.)*(alpha**2.)*(ga**2. - 1)*(T**4.)*(Z**4.)/135./(M**2.)
