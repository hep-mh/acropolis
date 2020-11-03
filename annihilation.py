#! /usr/bin/env python3

# sys
import sys

# pprint
from acropolis.pprint import print_error
# Dependent classes
from acropolis.models import AnnihilationModel

# Extact the number of command line arguments...
N = len(sys.argv)

# ...and check if there are exactly six
if N != 7:
    print_error("Would you kindly specify the following six command-line arguments:\n"
                + "       1. The mass of the dark-matter particle           [in MeV]\n"
                + "       2. The s-wave constribution to the cross-section  [in cm^3/s]\n"
                + "       3. The p-wave constribution to the cross-section  [in cm^3/s]\n"
                + "       4. The kinetic decoupling temperature of the\n"
                + "          dark-matter particle                           [in MeV]\n"
                + "       5. The branching ratio into electron-positron pairs\n"
                + "       6. The branching ratio into two photons.\n",
                "")

# Extract the input parameters
[mchi, a, b, tempkd, bree, braa] = [float(arg) for arg in sys.argv[1:]]

# Run the code...
Yf = AnnihilationModel(mchi, a, b, tempkd, bree, braa).run_disintegration()
# ... and print the result
print(Yf)
