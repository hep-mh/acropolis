# enum
from enum import Enum

# flags
import acropolis.flags as flags
# params
from acropolis.params import mp, mn, mD, mT, mHe3, mHe4, mpi0, mpic
from acropolis.params import sp_acc


# NOTE: _NEUTRON and _PROTON are 
# not meant to be used directly
class Particles(Enum):
    NULL = None

    # nucleons
    PROTON  =  0
    NEUTRON =  1

    # pions
    CHARGED_PION = 2
    NEUTRAL_PION = 3

    # nuclei
    _NEUTRON  = -6
    _PROTON   = -5

    DEUTERIUM = -4
    TRITIUM   = -3
    HELIUM3   = -2
    HELIUM4   = -1


# PROTON, NEUTRON
def is_nucleon(particle):
    if particle not in Particles or particle == Particles.NULL:
        return False

    return (0 <= particle.value <= 1)


# _NEUTRON, _PROTON, DEUTRIUM, TRITIUM, HELIUM3, HELIUM4
def is_nucleus(particle):
    if particle not in Particles or particle == Particles.NULL:
        return False
    
    return (-6 <= particle.value <= -1)


# NEUTRAL_PION, CHARGED_PION
def is_pion(particle):
    if particle not in Particles or particle == Particles.NULL:
        return False
    
    return (2 <= particle.value <= 3)


# TRITIUM, HELIUM3
def is_spectator(particle):
    if not flags.A3_is_spectator:
        return False

    if particle not in Particles or particle == Particles.NULL:
        return False
    
    # T and He3 act as spectator particles
    return (-3 <= particle.value <= -2)


# PROTON, NEUTRON
def is_projectile(particle):
    return is_nucleon(particle)


# PROTON, HELIUM4
def is_target(particle):
    if particle not in Particles or particle == Particles.NULL:
        return False

    return (particle.value in [0, -1] )


# PROTON, NEUTRON
def has_nuceq(particle):
    if particle not in Particles or particle == Particles.NULL:
        return False
    
    return (0 <= particle.value <= 1)


def convert(nucleon):
    if not is_nucleon(nucleon):
        raise ValueError(
            "The given nucleon cannot be identified as such"
        )
    
    if nucleon == Particles.PROTON:
        return Particles.NEUTRON
    
    if nucleon == Particles.NEUTRON:
        return Particles.PROTON


def nuceq(nucleon):
    if not has_nuceq(nucleon):
        raise ValueError(
            "The given nucleon does not have a nucleus equivalent"
        )
    
    if nucleon == Particles.PROTON:
        return Particles._PROTON
    
    if nucleon == Particles.NEUTRON:
        return Particles._NEUTRON


# All masses in MeV
mass = {
    Particles.PROTON : mp,
    Particles.NEUTRON: mn,

    Particles.DEUTERIUM: mD,
    Particles.TRITIUM  : mT,
    Particles.HELIUM3  : mHe3,
    Particles.HELIUM4  : mHe4,

    Particles.NEUTRAL_PION: mpi0,
    Particles.CHARGED_PION: mpic,

    Particles.NULL: 0.
}

charge = {
    Particles.PROTON : 1,
    Particles.NEUTRON: 0,

    Particles.DEUTERIUM: 1,
    Particles.TRITIUM  : 1,
    Particles.HELIUM3  : 2,
    Particles.HELIUM4  : 2,

    Particles.NULL: 0
}

za = {
    Particles._NEUTRON : (0, 1),
    Particles._PROTON  : (1, 1),
    Particles.DEUTERIUM: (1, 2),
    Particles.TRITIUM  : (1, 3),
    Particles.HELIUM3  : (2, 3),
    Particles.HELIUM4  : (2, 4)
}


# For NUCLEONS (PROTON, NEUTRON), this class
# counts the number of "active" nucleons per
# energy bin
# For NUCLEI (DEUTERIUM, ...), this class
# counts the total number of destroyed/created
# background nuclei
class ParticleSpectrum(object):

    def __init__(self, energy_grid):
        self._sEnergyGrid = energy_grid

        # Extract the number of bins
        self._sN = energy_grid.nbins()

        # Initialize a list containing all non-zero entries
        self._sEntries = {}


    def _increment(self, index, increment, acc=0.):
        if abs(increment) <= acc/self._sN:
            return

        if index in self._sEntries:
            self._sEntries[index] += increment
        else:
            self._sEntries[index] = increment


    def _add_projectile(self, projectile, increment, K, acc=0.):
        if K < self._sEnergyGrid.lower_edge():
            return

        if K > self._sEnergyGrid.upper_edge():
            raise ValueError("The given value of K lies outside the energy range")
        
        if not is_projectile(projectile):
            raise ValueError("The given particle is not a projectile")
        
        index = projectile.value*self._sN + self._sEnergyGrid.index_of(K)
        # -->
        self._increment(index, increment, acc)


    def _add_nucleus(self, nucleus, increment):
        if has_nuceq(nucleus):
            nucleus = nuceq(nucleus)
        
        if not is_nucleus(nucleus):
            raise ValueError("The given particle is not a nucleus")
        
        index = nucleus.value
        # -->
        self._increment(index, increment)


    def add(self, particle, increment, K=0):
        # DEBUG
        print(f"▶ Adding {particle.name:<9} ({increment:+.2e}) with kinetic energy {K:.2e}MeV")

        if is_projectile(particle):
            self._add_projectile(particle, increment, K, acc=sp_acc)
        
        if is_nucleus(particle) or has_nuceq(particle):
            self._add_nucleus(particle, increment)


    def non_zero(self):
        for index in self._sEntries:
            yield ( index, self._sEntries[index] )


    def at(self, index):
        if index in self._sEntries:
            return self._sEntries[index]
        
        return 0.


    def rescale(self, factor):
        for index in self._sEntries:
            self._sEntries[index] *= factor


    def egrid(self):
        return self._sEnergyGrid


    def baryon_number(self):
        Nb = 0

        for particle in Particles:
            if not is_nucleus(particle):
                continue

            _, A = za[particle]
            # -->
            Nb += self.at(particle.value)*A
        
        return Nb


    def __repr__(self):
        id_str = {
            -6: "n  ",
            -5: "p  ",
            -4: "H2 ",
            -3: "H3 ",
            -2: "He3",
            -1: "He4"
        } 

        str_repr = ""

        for i in range(self._sN):
            str_repr += f"      {self._sEnergyGrid[i]:.3e} |"

            for j in range(0, 2): # nucleons
                str_repr += f" {self.at(j*self._sN + i):.3e}"
            
            str_repr += "\n"
        
        str_repr += "----------------x\n"

        for k in range(-6, 0): # nuclei
            str_repr += f"Δ{id_str[k]} {self.at(k):+.3e} | \n"
        
        return str_repr
