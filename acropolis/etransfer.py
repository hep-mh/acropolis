# math
from math import log, exp
# cmath
from cmath import exp as cexp
# numpy
import numpy as np

# params
from acropolis.params import pi
# hcascade
from acropolis.hcascade import mass_dict, Projectiles, Targets
# jit
from acropolis.jit import jit


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
@jit
def _bp(s):
    C = 10.94 * 1e-6 # 1/MeV²
    D = -0.09 * 1e-6 # 1/MeV²
    E = 0.044 * 1e-6 # 1/MeV²

    return C + D*log(s/1e6) + E*(log(s/1e6)**2.) # 1/MeV²


# s in MeV²
@jit
def _bm(s):
    C = 23.2 * 1e-6 # 1/MeV²
    D = 0.94 * 1e-6 # 1/MeV²

    return C + D*log(s/1e6) # 1/MeV²


# s in MeV²
@jit
def _Mp(s):
    A  = 41.8 * mb_to_iMeV2 # 1/MeV²
    B  = 0.68 * mb_to_iMeV2 # 1/MeV²
    s0 = 343e6              # MeV²
    
    return -1j * s * ( A + B*(log(s/s0) - 1.j*pi/2.)**2. ) # unitless


# s in MeV²
@jit
def _Mm(s):
    D  = -39e6*mb_to_iMeV2 # GeV^{-2al}
    al = 0.48

    return D * (s/1e6)**al * cexp(1j*pi*(1.-al)/2.) # unitless


# s in MeV²
@jit
def Bsl(projectile, target, s):
    if target == Targets.ALPHA:
        # The same expression is used for any type of projectile
        
        return 28 * 1e-6 # 1/MeV²
        #         | 1/GeV² --> 1/MeV²
    
    bp = _bp(s)
    bm = _bm(s)

    Mp = _Mp(s)
    Mm = _Mm(s)
    # -->
    ReMp, ImMp = Mp.real, Mp.imag
    ReMm, ImMm = Mm.real, Mm.imag
    # -->
    ReMp2, ImMp2 = ReMp**2., ImMp**2.
    ReMm2, ImMm2 = ReMm**2., ImMm**2.

    # Bsl = d ln(dsig / dt) /dt = (d²sig / dt²) / (dsig / dt)
    def _Bsl(k):
        tref = -0.02e6 # MeV²

        num = bm*exp(bm*tref)*ImMm2 + exp(bp*tref)*ImMp2*bp + bm*exp(bm*tref)*ReMm2 \
            + bp*exp(bp*tref)*ReMp2 + (bm + bp)*exp(((bm + bp)*tref)/2.)*k*(ImMm*ImMp + ReMm*ReMp)
        
        den = exp(bm*tref)*(ImMm2 + ReMm2) + \
            + 2.*exp(((bm + bp)*tref)/2.)*k*(ImMm*ImMp + ReMm*ReMp) + exp(bp*tref)*(ImMp2 + ReMp2)

        #num = (ReMp*bp + k*ReMm*bm)*(ReMp + k*ReMm) + (ImMp*bp + k*ImMm*bm)*(ImMp + k*ImMm)
        #den = (ReMp + k*ReMm)**2. + (ImMp + k*ImMm)**2.

        return num/den # 1/MeV²
    
    # target == Targets.PROTON
    if projectile == Projectiles.PROTON \
    or projectile == Projectiles.NEUTRON:
        return _Bsl(-1) # 1/MeV²

    if projectile == Projectiles.ANTI_PROTON \
    or projectile == Projectiles.ANTI_NEUTRON:
        return _Bsl(+1) # 1/MeV²

    # Invalid projectile or target
    return np.nan
