# math
from math import log, exp
# numpy
import numpy as np

# params
from paramsd import pi
# hcascade
from acropolis.hcascade import mass_dict, Projectiles, Targets


mb_to_iMeV2 = 2.5681899885e-06


def _K_to_s(projectile, target, K):
    m_proj = mass_dict[projectile]
    m_targ = mass_dict[target]

    return m_proj**2. + m_targ**2. + 2.*(K + m_proj)*m_targ # MeV


def _K_to_E(particle, K):
    return K + mass_dict[particle]


#####################################################################


def enforce_energy_conservation():
    pass


#####################################################################


# s in MeV²
def _bp(s):
    return 10.94 - 0.09*log(s/1e6) + 0.044*(log(s/1e6)**2.)


# s in MeV²
def _bm(s):
    return 23.2 + 0.94*log(s/1e6)


# s in MeV²
def _Mp(s):
    A  = 41.8 * mb_to_iMeV2 # 1/MeV²
    B  = 0.68 * mb_to_iMeV2 # 1/MeV²
    s0 = 343e6              # MeV²
    
    return -1j * s * ( A + B*(log(s/s0) - 1.j*pi/2.)**2. )


# s in MeV²
def _Mm(s):
    D  = -39e6*mb_to_iMeV2 # GeV^{-2al}
    al = 0.48

    return D * (s/1e6)**al * exp(1j*pi*(1.-al)/2.)


# s in MeV²
def Bsl(projectile, target, s):
    if target == Targets.ALPHA:
        # The same expression is used for any type of projectile
        
        return 28 * 1e-6
        #         | 1/GeV² --> 1/MeV²
    
    bp = _bp(s)
    bm = _bm(s)

    Mp = _Mp(s)
    Mm = _Mm(s)
    
    # target == Targets.PROTON
    if projectile == Projectiles.PROTON \
    or projectile == Projectiles.NEUTRON:
        pass

    if projectile == Projectiles.ANTI_PROTON \
    or projectile == Projectiles.ANTI_NEUTRON:
        pass

    return np.nan
