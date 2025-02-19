# FLAGS #############################################################

# If this flag is set to 'True',
# the pregenerated databases
# will be used to interpolate
# the different reaction rates
# Default: True
usedb = True

# If this flag is set to 'True',
# additional output is printed
# to the screen
# Default: True
verbose = True

# If this flag is set to 'True',
# additional debug information
# is printed to the screen
# Default: False
debug = False

# If this flag is set to 'True',
# the universal spectrum is used
# for all points in parameter space
# Default: False
universal = False

# If this flag is set to 'True',
# H3 and He3 are treated as spec-
# tators with O(1)MeV energy in
# the reactions driving the
# hadronic cascade
# Default: False
A3_is_spectator = False
# TODO: Fix errors if set to 'True'

# If this flag is set to 'True',
# no survival check is performed
# on any of the nuclei that are
# injected during the hadronic
# cascade
# Default: False
all_nuclei_survive = False

# If this flag is set to 'True',
# nucleons that result from
# disintegration of nuclei during
# the hadronic cascade are getting
# reinjected
# Default: True
reinject_fragments = True

# If this flag is set to 'True',
# the neutron decay is handled
# during eloss. Otherwise it is
# handled during etransfer
# Default: False
decay_during_eloss = False


# FUNCTIONS #########################################################

def set_universal(value):
    global universal

    universal = value