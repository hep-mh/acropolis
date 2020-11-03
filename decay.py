#! /usr/bin/env python3

# sys
import sys

# pprint
from acropolis.pprint import print_error
# Dependent classes
from acropolis.models import DecayModel

# Extact the number of command line arguments...
N = len(sys.argv)

# ...and check if there are exactly six
if N != 7:
    print_error("Would you kindly specify the following six command-line arguments:\n"
                + "       1. The mass of the decaying particle              [in MeV]\n"
                + "       2. The lifetime of the decaying particle          [in s]\n"
                + "       3. Some reference temperature T0                  [in MeV]\n"
                + "       4. The number density of the decaying particle\n"
                + "          (relative to photons) at T0\n"
                + "       5. The branching ratio into electron-positron pairs\n"
                + "       6. The branching ratio into two photons.\n",
                "")

# Extract the input parameters
[mphi, tau, temp0, n0a, bree, braa] = [float(arg) for arg in sys.argv[1:]]

# Run the code...
Yf = DecayModel(mphi, tau, temp0, n0a, bree, braa).run_disintegration()
# ... and print the result
print(Yf)
