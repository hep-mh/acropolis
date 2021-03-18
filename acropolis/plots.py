# numpy
import numpy as np
# matplotlib
import matplotlib.pyplot as plt

# params
from acropolis.params import NY


# Set the general style of the plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)
# Include additiona latex packages
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{mathpazo}'


_plot_number = 0


def _get_abundances(data, i):
    i0 = i + 2

    # Extract the different abundances...
    mean, high, low = data[:,i0], data[:,i0+NY], data[:,i0+2*NY]
    # ...and calculate an estimate for the error
    diff = np.minimum( np.abs( mean - high ), np.abs( mean - low ) )

    return mean, diff


def _get_deviations(data):
    # Extract and sum up neutrons and protons
    mn, en     = _get_abundances(data, 0)
    mp, ep     = _get_abundances(data, 1)
    mH, eH     = mn + mp, np.sqrt( en**2. + ep**2. )

    # Extract and sum up lithium-7 and berylium-7
    mLi7, eLi7 = _get_abundances(data, 7)
    mBe7, eBe7 = _get_abundances(data, 8)
    m7,   e7   = mLi7 + mBe7, np.sqrt( eLi7**2. + eBe7**2. )

    # Extract deuterium
    mD  , eD   = _get_abundances(data, 2)

    # Extract and sum up tritium and helium-3
    mT  , eT   = _get_abundances(data, 3)
    mHe3, eHe3 = _get_abundances(data, 4)
    m3  , e3   = mT + mHe3, np.sqrt( eT**2. + eHe3**2. )

    # Extract helium-4
    mHe4, eHe4 = _get_abundances(data, 5)

    # Calculate the relevant abundance ratios
    mYp , eYp  = 4.*mHe4, 4.*eHe4
    mDH , eDH  = mD/mH, (mD/mH)*np.sqrt( (eD/mD)**2. + (eH/mH)**2. )
    mHeD, eHeD = m3/mD, (m3/mD)*np.sqrt( (e3/m3)**2. + (eD/mD)**2. )
    mLiH, eLiH = m7/mH, (m7/mH)*np.sqrt( (e7/m7)**2. + (eH/mH)**2. )

    # Calculate the corresponding deviations
    Yp  = (mYp  -  2.45e-1) / np.sqrt( ( 0.03e-1)**2. +  eYp**2. )
    DH  = (mDH  - 2.547e-5) / np.sqrt( (0.035e-5)**2. +  eDH**2. )
    HeD = (mHeD -   8.3e-1) / np.sqrt( (  1.5e-1)**2. + eHeD**2. )
    LiH = (mLiH -  1.6e-10) / np.sqrt( ( 0.3e-10)**2. + eLiH**2. )

    # Take care of potential NaNs
    HeD[mDH<0.035e-5] =  10
    DH [np.isnan(DH)] = -10

    # Return (without reshaping)
    return Yp, DH, HeD, LiH


def plot_scan_results(data, output_file=None, save_pdf=True, show_fig=False):
    global _plot_number

    # If data is a filename, load the data first
    if type(data) == str:
        data = np.loadtxt(data)

    # Get the set of input parameters
    x, y = data[:,0], data[:,1]

    # Calculate the corresponding deviations
    Yp, DH, HeD, LiH = _get_deviations(data)

    # TODO

    if save_pdf == True:
        # If no name for the output file is given
        # simply enumerate the different plots
        if output_file is None:
            output_file = 'acropolis_plot_{}.pdf'.format(_plot_number)
            _plot_number += 1

        plt.savefig(output_file)

    if show_fig == True:
        plt.show()
