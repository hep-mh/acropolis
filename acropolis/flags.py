# Only parameters that specify a default value are
# meant to be changed by the user, i.e. everything
# under FLAGS and ALGORITHM-SPECIFIC PARAMETERS


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
# ATTENTION:
# Change with caution and only if
# you know what you are doing.
# Default: False
universal = False

# If this flag is set to 'True',
# H3 and He3 are treated as spec-
# tators with O(1)MeV energy in
# the reactions driving the
# hadronic cascade
# Default: False
A3_is_spectator = False

# If this flag is set to 'True',
# nucleons that result from
# disintegration of nuclei during
# the hadronic cascade are getting
# reinjected
# Default: True
reinject_fragments = True


# FUNCTIONS #########################################################

def set_universal(value):
    global universal

    universal = value