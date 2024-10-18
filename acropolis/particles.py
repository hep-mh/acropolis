# enum
from enum import Enum

# params
from acropolis.params import mp, mn, mD, mT, mHe3, mHe4, mpi0, mpic


class Particles(Enum):
    PROTON  =  0
    NEUTRON =  1

    DEUTERIUM = -4
    TRITIUM   = -3
    HELIUM3   = -2
    HELIUM4   = -1

    CHARGED_PION = -6
    NEUTRAL_PION = -5

    NULL = None


# PROTON, NEUTRON
def is_nucleon(particle):
    if particle not in Particles:
        return False

    return (0 <= particle.value <= 1)

# DEUTRIUM, TRITIUM, HELIUM3, HELIUM4
def is_nucleus(particle):
    if particle not in Particles:
        return False
    
    return (-4 <= particle.value <= -1)

# NEUTRAL_PION, CHARGED_PION
def is_pion(particle):
    if particle not in Particles:
        return False
    
    return (-6 <= particle.value <= -5)

# TRITIUM, HELIUM3
def is_spectator(particle):
    if particle not in Particles:
        return False
    
    # T and He3 act as spectator particles
    return (-3 <= particle.value <= -2)

# PROTON, NEUTRON
def is_projectile(particle):
    return is_nucleon(particle)

# PROTON, HELIUM4
def is_target(particle):
    if particle not in Particles:
        return False

    return (particle.value in [0, -1] )


def convert(particle):
    if not is_nucleon(particle):
        return particle
    
    if particle == Particles.PROTON:
        return Particles.NEUTRON
    
    if particle == Particles.NEUTRON:
        return Particles.PROTON


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

# All threshold energies in MeV
eth_pdi = {
    Particles.DEUTERIUM: mD   - 1*mp - 1*mn,
    Particles.TRITIUM  : mT   - 1*mp - 2*mn,
    Particles.HELIUM3  : mHe3 - 2*mp - 1*mn,
    Particles.HELIUM4  : mHe4 - 2*mp - 2*mn,
}


class ParticleSpectrum(object):

    def __init__(self, energy_grid):
        self._sEnergyGrid = energy_grid

        # Extract the number of bins
        self._sN = energy_grid.nbins()

        # Initialize a list containing all non-zero entries
        self._sEntries = {}


    def _increment(self, index, increment, acc=1e-5):
        if abs(increment) <= acc/self._sN:
            return

        if index in self._sEntries:
            self._sEntries[index] += increment
        else:
            self._sEntries[index] = increment


    def _add_nucleon(self, nucleon, increment, K):
        if K < self._sEnergyGrid.lower_edge():
            return

        index = nucleon.value*self._sN + self._sEnergyGrid.index_of(K)
        # -->
        self._increment(index, increment)


    def _add_nucleus(self, nucleus, increment, K):
        # TODO: Check if the nucleus survives

        index = nucleus.value
        # -->
        self._increment(index, increment)


    def add(self, particle, increment, K):
        # DEBUG
        print(f"Adding {particle} with energy {K:.5e}MeV")
        
        if K > self._sEnergyGrid.upper_edge():
            raise ValueError(
                "The given value of K lies ouside the energy grid"
            )

        # Protons & Neutrons
        if is_nucleon(particle):
            self._add_nucleon(particle, increment, K)
        
        # Nuclei
        elif is_nucleus(particle):
            self._add_nucleus(particle, increment, K)
        
        else:
            raise ValueError(
                "The given particles cannot be added to the spectrum"
            )


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


    def __repr__(self):
        str_repr = ""

        for i in range(self._sN):
            str_repr += f"{self._sEnergyGrid[i]:.3e} |"

            for j in range(0, 2): # nucleons
                str_repr += f" {self.at(j*self._sN + i):.3e}"
            
            str_repr += "\n"
        
        str_repr += "----------x\n"

        for k in range(-4, 0): # nuclei
            str_repr += f"{self.at(k):.3e} | \n"
        
        return str_repr
