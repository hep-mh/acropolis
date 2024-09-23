# ATTENTION !!!!!111elf
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
# World destruction possible!
# Default: False
universal = False


def set_universal(value):
    global universal

    universal = value